"""W512 graph-backed append-only text sessions, with all capture at startup.

One shared engine eagerly captures exact lengths 1..32 on an independent
synthetic 512-token seed. Real requests never warm or capture a graph. The
existing text state machine continues to own transactional session caches,
64-token snapshots, BPE rollback and the strict 8192-token limit.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import time

from graph_stream import (
    MAX_CHUNK, MAX_TOKENS, MIN_CHUNK, WINDOW, WindowGraphRunner,
    state_tensors, storage_pointers, unique_storage_bytes, validate_parameters,
)
from text_stream_runtime import (
    TextStreamRuntime, _encode_kwargs, _reject_configured_truncation, serialize,
)

CAPTURE_LENGTHS = tuple(range(MIN_CHUNK, MAX_CHUNK + 1))
SEED_PHRASE = "公开资料用于一般阅读和目录整理。 Public reference text supports neutral reading and organizing headings.\n"
SEED_TEXT = serialize([{"role": "user", "content": SEED_PHRASE * 128}])


def select_execution(prefix_tokens, new_tokens):
    """Pure scheduling rule; unsupported short lengths never get an eager fallback."""
    if (isinstance(prefix_tokens, bool) or not isinstance(prefix_tokens, int) or prefix_tokens < 0
            or isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 1):
        raise ValueError("Invalid logical context or input length")
    if prefix_tokens + new_tokens > MAX_TOKENS:
        raise ValueError("Text-session forward would exceed 8192 tokens; no truncation")
    return "graph" if prefix_tokens >= WINDOW and MIN_CHUNK <= new_tokens <= MAX_CHUNK else "eager"


def seed_token_ids(tokenizer):
    _reject_configured_truncation(tokenizer)
    encoded = tokenizer.encode(SEED_TEXT, **_encode_kwargs(tokenizer))
    if getattr(encoded, "overflowing", None):
        raise ValueError("Startup seed tokenizer returned overflowing segments")
    ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    if len(ids) < WINDOW + MAX_CHUNK:
        raise ValueError("Fixed synthetic seed did not provide 544 native tokens")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
        raise ValueError("Invalid synthetic seed token IDs")
    return ids[:WINDOW], ids[WINDOW:WINDOW + MAX_CHUNK]


class GraphTextStreamRuntime(TextStreamRuntime):
    """Shared W512 CUDA Graph engine; use new_session() exactly as before.

    The caller must already load/freeze the intended window checkpoint. This
    runtime never changes architecture, weights, tokenization or risk labels.
    engine_metadata is a copied startup record; stats() counts request calls
    only. close() drops graph state without modifying caller-owned sessions or
    model weights. Rebuild the engine after a graph error; it fails closed.
    """

    def __init__(self, model, tokenizer, *, chunk_tokens=32, prefill_chunk_tokens=2048):
        for name, value in (("chunk_tokens", chunk_tokens), ("prefill_chunk_tokens", prefill_chunk_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        started = time.perf_counter()
        super().__init__(model, tokenizer, chunk_tokens=chunk_tokens, prefill_chunk_tokens=prefill_chunk_tokens)
        self._phase = "initializing"
        self._runners = {}
        self._engine_metadata = {}
        self._request_stats = {"graph_calls": 0, "graph_forward_tokens": 0,
                               "eager_calls": 0, "eager_forward_tokens": 0,
                               "graph_attempts": 0, "graph_failures": 0, "eager_failures": 0}
        backbone = model.backbone
        if (getattr(backbone, "guard_window", None) != WINDOW
                or not hasattr(backbone, "guard_original_forward")
                or hasattr(backbone, "memory_original_forward")):
            raise ValueError("Graph text runtime supports only the existing W512 window model")
        if getattr(backbone.rotary_emb, "rope_type", None) != "default":
            raise ValueError("Graph text runtime supports only default RoPE")
        self._vocab_size = backbone.config.vocab_size
        torch = self._torch
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        seed_ids, suffix_ids = seed_token_ids(tokenizer)
        arenas = []
        seed_cache = None
        try:
            # Bypass this subclass's request dispatcher/counters: initialization
            # is explicit, isolated synthetic work before any session exists.
            seed_cache, _ = super()._forward(seed_ids, None)
            arena_pointers = set()
            for length in CAPTURE_LENGTHS:
                runner = WindowGraphRunner(model, seed_cache, suffix_ids[:length], chunk_tokens=length)
                self._runners[length] = runner
                pointers = storage_pointers(runner._arena)
                if pointers & arena_pointers or pointers & storage_pointers(seed_cache):
                    raise RuntimeError("Captured graph arenas share mutable cache storage")
                arena_pointers.update(pointers)
                arenas.append(runner.storage_metrics())
            # The synthetic seed cache is never installed into a real session.
            del seed_cache
            seed_cache = None
            gc.collect()
            torch.cuda.synchronize()
            total_calls = 1 + sum(row["setup_forward_calls"] for row in arenas)
            total_tokens = WINDOW + sum(row["setup_forward_tokens"] for row in arenas)
            if total_calls != 129 or total_tokens != 2624:
                raise RuntimeError("Unexpected hidden startup forward count")
            shared_state_bytes = unique_storage_bytes(
                tensor for runner in self._runners.values() for tensor in state_tensors(runner._arena).values())
            self._engine_metadata = {
                "inference_engine": "window_cuda_graph", "window_tokens": WINDOW, "batch_size": 1,
                "captured_chunk_lengths": list(CAPTURE_LENGTHS),
                "startup_seconds": time.perf_counter() - started,
                "capture_seconds": sum(row["setup_seconds"] for row in arenas),
                "capture_seconds_scope": "sum of per-length warmup, capture and state restoration; seed prefill excluded",
                "startup_forward_calls": total_calls, "startup_forward_tokens": total_tokens,
                "auxiliary_input_tokens": total_tokens,
                "synthetic_seed_tokens": WINDOW,
                "seed_ids_sha256": hashlib.sha256(json.dumps(seed_ids, separators=(",", ":")).encode()).hexdigest(),
                "seed_text_sha256": hashlib.sha256(SEED_TEXT.encode()).hexdigest(),
                "shared_graph_arena_storage_bytes": shared_state_bytes,
                "shared_engine_allocated_delta_bytes": torch.cuda.memory_allocated() - allocated_before,
                "shared_engine_reserved_delta_bytes": torch.cuda.memory_reserved() - reserved_before,
                "shared_memory_scope": "one model-shared engine with 32 private graph arenas/pools; allocation deltas exclude pre-existing model allocations and are not per-session cache size",
                "per_length_arenas": arenas,
                "initialization_excluded_from_request_accounting": True,
                "capture_during_requests": False, "graph_error_fallback": False,
                "eager_scope": "prefix shorter than 512 or input block larger than 32 tokens",
                "maximum_session_tokens": MAX_TOKENS, "generated_tokens": 0,
            }
            self._phase = "ready"
        except Exception:
            self._phase = "failed"
            for runner in self._runners.values():
                try:
                    runner.close()
                except Exception:
                    pass
            self._runners.clear()
            raise

    @property
    def engine_metadata(self):
        with self._lock:
            return copy.deepcopy(self._engine_metadata)

    def stats(self):
        with self._lock:
            return {**self._request_stats, "phase": self._phase}

    def _require_ready(self):
        if self._phase != "ready":
            raise RuntimeError(f"Graph text engine is not ready: {self._phase}")

    def _forward(self, ids, cache):
        with self._lock:
            self._require_ready()
            if not isinstance(ids, (list, tuple)):
                raise ValueError("Text runtime requires a list/tuple of exact token IDs")
            if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self._vocab_size for token in ids):
                raise ValueError("Invalid token ID for the loaded classifier")
            if cache is None:
                prefix = 0
            else:
                from window_attention import WindowCache
                if type(cache) is not WindowCache or cache.window != WINDOW or hasattr(cache, "memories"):
                    raise ValueError("Graph text runtime requires a W512 WindowCache")
                prefix = cache.get_seq_length()
            route = select_execution(prefix, len(ids))
            if route == "graph":
                runner = self._runners.get(len(ids))
                if runner is None:
                    self._phase = "failed"
                    raise RuntimeError("Required exact-length graph was not captured; eager fallback is disabled")
                self._request_stats["graph_attempts"] += 1
                try:
                    result = runner.step(ids, cache)
                except Exception:
                    self._request_stats["graph_failures"] += 1
                    self._phase = "failed"
                    raise
                self._request_stats["graph_calls"] += 1
                self._request_stats["graph_forward_tokens"] += len(ids)
                return result
            try:
                result = super()._forward(ids, cache)
            except Exception:
                self._request_stats["eager_failures"] += 1
                raise
            self._request_stats["eager_calls"] += 1
            self._request_stats["eager_forward_tokens"] += len(ids)
            return result

    def _clone(self, cache):
        with self._lock:
            self._require_ready()
            return super()._clone(cache)

    def new_session(self):
        with self._lock:
            self._require_ready()
            return super().new_session()

    def close(self):
        with self._lock:
            if self._phase == "closed":
                return
            errors = []
            for runner in self._runners.values():
                try:
                    runner.close()
                except Exception as error:
                    errors.append(type(error).__name__)
            self._runners.clear()
            self._phase = "closed"
            if errors:
                raise RuntimeError("Graph engine cleanup failed: " + ", ".join(errors))
