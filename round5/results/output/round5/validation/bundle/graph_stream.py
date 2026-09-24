"""W512/batch-one CUDA Graph classifier runner for one fixed length 1..32.

This is not the text runtime and never changes model weights. Every call copies
an external WindowCache into a private fixed-address arena and returns an
independent cache. Unfilled windows, full/memory models, padding, lengths
outside 1..32 and dynamic RoPE are deliberately unsupported.
"""
from __future__ import annotations

import copy
import math
import threading
import time

WINDOW = 512
CHUNK = 8
MIN_CHUNK = 1
MAX_CHUNK = 32
MAX_TOKENS = 8192
ROLES = ("user", "assistant")
LABELS = ("safe", "unsafe", "controversial")


def validate_parameters(window=WINDOW, batch_size=1, chunk_tokens=CHUNK):
    for name, value, required in (("window", window, WINDOW), ("batch_size", batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value != required:
            raise ValueError(f"Graph prototype requires {name}={required}")
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int) or not MIN_CHUNK <= chunk_tokens <= MAX_CHUNK:
        raise ValueError("Graph chunk_tokens must be an exact integer length in 1..32")


def validate_ids(ids, vocab_size, chunk_tokens=CHUNK):
    validate_parameters(chunk_tokens=chunk_tokens)
    if not isinstance(ids, (list, tuple)) or len(ids) != chunk_tokens:
        raise ValueError(f"Graph runner requires exactly {chunk_tokens} token IDs")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < vocab_size for v in ids):
        raise ValueError("Invalid token ID")


class _Lifecycle:
    def __init__(self):
        self.phase = "initializing"

    def move(self, phase):
        allowed = {"initializing": {"warming", "failed", "closed"},
                   "warming": {"capturing", "failed", "closed"},
                   "capturing": {"ready", "failed", "closed"},
                   "ready": {"failed", "closed"}, "failed": {"closed"}, "closed": set()}
        if phase not in allowed[self.phase]:
            raise RuntimeError(f"Invalid graph lifecycle transition: {self.phase} -> {phase}")
        self.phase = phase

    def require_ready(self):
        if self.phase != "ready":
            raise RuntimeError(f"Graph runner is not ready: {self.phase}")


def require_l20():
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "L20" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("CUDA Graph neural execution requires exactly one visible L20")
    return torch


def state_tensors(cache):
    """Explicit state inventory for this exact HF hybrid cache layout."""
    result = {}
    for index, layer in enumerate(cache.layers):
        if hasattr(layer, "keys"):
            result[(index, "keys", None)] = layer.keys
            result[(index, "values", None)] = layer.values
        if hasattr(layer, "conv_states"):
            for kind in ("conv_states", "recurrent_states"):
                for key, tensor in getattr(layer, kind).items():
                    result[(index, kind, key)] = tensor
    return result


def state_metadata(cache):
    linear = {}
    for index, layer in enumerate(cache.layers):
        if hasattr(layer, "conv_states"):
            linear[index] = {key: copy.deepcopy(getattr(layer, key)) for key in (
                "has_previous_state", "is_conv_states_initialized", "is_recurrent_states_initialized",
                "conv_kernel_size", "record_past")}
    return {"seen": dict(cache.seen), "window": cache.window,
            "first_attention": cache.first_attention, "linear": linear}


def validate_cache(cache, torch, chunk_tokens=CHUNK):
    validate_parameters(chunk_tokens=chunk_tokens)
    from window_attention import WindowCache
    if type(cache) is not WindowCache or hasattr(cache, "memories"):
        raise ValueError("Only the existing WindowCache is supported; full/memory caches are rejected")
    if cache.window != WINDOW or getattr(cache, "offloading", False) or len(cache.layers) != 24:
        raise ValueError("Unsupported cache configuration")
    values = list(cache.seen.values())
    if (not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values)
            or len(set(values)) != 1 or not WINDOW <= values[0] <= MAX_TOKENS - chunk_tokens):
        raise ValueError(f"Cache must contain one fully initialized prefix of 512..{MAX_TOKENS - chunk_tokens} tokens")
    attention = set()
    linear = 0
    for index, layer in enumerate(cache.layers):
        if hasattr(layer, "keys"):
            attention.add(index)
            if not getattr(layer, "is_initialized", False):
                raise ValueError("Attention cache is not initialized")
            for value in (layer.keys, layer.values):
                if not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != 1 or value.shape[-2] != WINDOW - 1:
                    raise ValueError("Attention cache must retain exactly 511 keys and values")
        elif hasattr(layer, "conv_states"):
            linear += 1
            if layer.record_past or set(layer.conv_states) != {0} or set(layer.recurrent_states) != {0}:
                raise ValueError("Unsupported GDN state layout")
            for flags in (layer.has_previous_state, layer.is_conv_states_initialized, layer.is_recurrent_states_initialized):
                if flags != {0: True}:
                    raise ValueError("GDN state must be initialized before capture")
        else:
            raise ValueError("Unknown cache layer type")
    if len(attention) != 6 or linear != 18 or set(cache.seen) != attention:
        raise ValueError("Expected six attention and eighteen GDN layers")
    tensors = state_tensors(cache)
    if len(tensors) != 48 or any(not torch.is_tensor(t) or t.device.type != "cuda" or t.shape[0] != 1 for t in tensors.values()):
        raise ValueError("Incomplete or non-CUDA cache state")
    return values[0]


def storage_pointers(cache):
    return {(str(t.device), t.untyped_storage().data_ptr()) for t in state_tensors(cache).values()
            if t is not None and t.numel()}


def unique_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if tensor is not None and tensor.numel():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def cpu_probabilities(probabilities):
    values = probabilities.float().cpu().tolist()
    if len(values) != 2 or any(len(row) != 3 for row in values):
        raise RuntimeError("Expected both three-class risk heads")
    if any(not all(math.isfinite(v) and 0 <= v <= 1 for v in row) or abs(sum(row) - 1) > 1e-4 for row in values):
        raise RuntimeError("Nonfinite or invalid graph probabilities")
    return {role: dict(zip(LABELS, row)) for role, row in zip(ROLES, values)}


class WindowGraphRunner:
    """One locked graph arena; external caches remain caller-owned snapshots.

    The model must remain frozen and configured as window/W512 for this
    runner's lifetime. close() releases graph state, never model weights.
    """
    def __init__(self, model, seed_cache, warmup_ids, *, window=WINDOW, batch_size=1, chunk_tokens=CHUNK):
        validate_parameters(window, batch_size, chunk_tokens)
        self._torch = require_l20()
        self._lock = threading.RLock()
        self._life = _Lifecycle()
        self._model = model
        self._graph = None
        self._chunk_tokens = chunk_tokens
        self.setup_forward_calls = 0
        self.setup_forward_tokens = 0
        if model.training or any(p.requires_grad or p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("Model must be eval/frozen and entirely on L20")
        backbone = model.backbone
        if (getattr(backbone, "guard_window", None) != WINDOW
                or not hasattr(backbone, "guard_original_forward")
                or hasattr(backbone, "memory_original_forward")):
            raise ValueError("Only the W512 window model is supported")
        if getattr(backbone.rotary_emb, "rope_type", None) != "default":
            raise ValueError("Dynamic/unknown RoPE configuration is outside this prototype")
        self._vocab_size = backbone.config.vocab_size
        validate_ids(warmup_ids, self._vocab_size, self._chunk_tokens)
        validate_cache(seed_cache, self._torch, self._chunk_tokens)
        torch = self._torch
        began = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        try:
            with torch.inference_mode():
                self._arena = copy.deepcopy(seed_cache)
                self._stable = state_tensors(self._arena)
                self._layout = {key: (tuple(t.shape), t.dtype, t.device) for key, t in self._stable.items()}
                self._fixed_metadata = state_metadata(seed_cache)["linear"]
                self._ids = torch.empty((1, self._chunk_tokens), dtype=torch.long, device="cuda")
                self._positions = torch.empty((1, self._chunk_tokens), dtype=torch.long, device="cuda")
                self._offsets = torch.arange(self._chunk_tokens, dtype=torch.long, device="cuda")[None, :]
                self._stream = torch.cuda.Stream()
                self._stream.wait_stream(torch.cuda.current_stream())
                self._life.move("warming")
                with torch.cuda.stream(self._stream):
                    for _ in range(3):
                        self._load(seed_cache, warmup_ids)
                        self._body()
                        self._rebind()
                    self._load(seed_cache, warmup_ids)
                torch.cuda.current_stream().wait_stream(self._stream)
                torch.cuda.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                self._life.move("capturing")
                with torch.cuda.graph(self._graph, stream=self._stream):
                    self._output = self._body()
                self._rebind()
                # Warmup and capture executed real forwards. Restore every
                # tensor plus logical position before accepting any real call.
                self._load(seed_cache, warmup_ids)
                torch.cuda.synchronize()
                self._life.move("ready")
        except Exception:
            self._life.move("failed")
            self._release_resources()
            raise
        self.setup_seconds = time.perf_counter() - began
        self.setup_allocated_delta_bytes = torch.cuda.memory_allocated() - allocated_before
        self.setup_reserved_delta_bytes = torch.cuda.memory_reserved() - reserved_before

    @property
    def chunk_tokens(self):
        return self._chunk_tokens

    def _rebind(self):
        # Rebinding is outside replay. Captured kernels always read the original
        # stable input addresses, and _body copies final K/V back into them.
        for (index, kind, key), tensor in self._stable.items():
            layer = self._arena.layers[index]
            if key is None:
                setattr(layer, kind, tensor)
            else:
                getattr(layer, kind)[key] = tensor

    def _load(self, cache, ids):
        seen = validate_cache(cache, self._torch, self._chunk_tokens)
        incoming = state_tensors(cache)
        if set(incoming) != set(self._stable) or state_metadata(cache)["linear"] != self._fixed_metadata:
            raise ValueError("Cache topology or GDN metadata changed")
        for key, source in incoming.items():
            if (tuple(source.shape), source.dtype, source.device) != self._layout[key]:
                raise ValueError("Cache shape/dtype/device differs from captured arena")
            self._stable[key].copy_(source)
        self._rebind()
        self._arena.seen = dict(cache.seen)
        self._ids.copy_(self._torch.tensor([ids], dtype=self._torch.long, device="cuda"))
        self._positions.copy_(self._offsets + seen)
        return seen

    def _body(self):
        torch = self._torch
        self.setup_forward_calls += 1
        self.setup_forward_tokens += self._chunk_tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._model.backbone(input_ids=self._ids, position_ids=self._positions,
                                          past_key_values=self._arena, use_cache=True)
        if output.past_key_values is not self._arena:
            raise RuntimeError("Backbone replaced the cache object")
        hidden = output.last_hidden_state[:, -1]
        probabilities = torch.stack([self._model.readout(hidden, role)[0].float().softmax(-1)[0] for role in ROLES])
        # Capture the state commits as device operations. Python attribute/dict
        # assignments in WindowCache.update execute only during capture.
        for key, current in state_tensors(self._arena).items():
            target = self._stable[key]
            if current is not target:
                target.copy_(current)
        return probabilities

    def step(self, ids, cache):
        with self._lock:
            self._life.require_ready()
            validate_ids(ids, self._vocab_size, self._chunk_tokens)
            validate_cache(cache, self._torch, self._chunk_tokens)
            torch = self._torch
            try:
                with torch.inference_mode():
                    seen = self._load(cache, ids)
                    self._graph.replay()
                    # Python counters are NOT advanced by CUDA Graph replay.
                    self._arena.seen = {index: seen + self._chunk_tokens for index in self._arena.seen}
                    exported = copy.deepcopy(self._arena)
                    # CPU-visible output is read after state copies on this
                    # stream, including their cost in the public call latency.
                    probabilities = cpu_probabilities(self._output)
                    if storage_pointers(exported) & storage_pointers(self._arena):
                        raise RuntimeError("Returned cache aliases graph arena")
                    if storage_pointers(exported) & storage_pointers(cache):
                        raise RuntimeError("Returned cache aliases the caller's cache")
                    return exported, probabilities
            except Exception:
                self._life.move("failed")
                raise

    def snapshot(self):
        with self._lock:
            self._life.require_ready()
            with self._torch.inference_mode():
                return copy.deepcopy(self._arena)

    def storage_metrics(self):
        with self._lock:
            self._life.require_ready()
            states = list(state_tensors(self._arena).values())
            io = [self._ids, self._positions, self._offsets, self._output]
            return {"chunk_tokens": self._chunk_tokens,
                    "arena_state_storage_bytes": unique_storage_bytes(states),
                    "fixed_io_storage_bytes": unique_storage_bytes(io),
                    "arena_and_io_storage_bytes": unique_storage_bytes([*states, *io]),
                    "setup_seconds": self.setup_seconds,
                    "setup_forward_calls": self.setup_forward_calls,
                    "setup_forward_tokens": self.setup_forward_tokens,
                    "cuda_allocated_delta_bytes": self.setup_allocated_delta_bytes,
                    "cuda_reserved_delta_bytes": self.setup_reserved_delta_bytes}

    def _release_resources(self):
        self._graph = None
        self._output = None
        self._stable = {}
        self._arena = None
        self._ids = None
        self._positions = None
        self._offsets = None
        self._stream = None
        self._model = None

    def close(self):
        with self._lock:
            if self._life.phase == "closed":
                return
            try:
                self._torch.cuda.synchronize()
            finally:
                self._release_resources()
                self._life.move("closed")
