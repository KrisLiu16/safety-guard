"""Append-only text sessions for the Round4 direct risk classifiers.

Every append re-tokenizes the complete training-format conversation. If BPE
changes a previous suffix, restore a full cache snapshot at/before the longest
common token-ID prefix, then replay. Never crop only attention KV: GDN and
associative memory are also recurrent state. Neural execution is L20-only.

TextStreamSession accepts a fake chunk runner for CPU state-machine tests.
TextStreamRuntime is the real model adapter; its sessions share model weights,
but no mutable cache or snapshots. There is no token generation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import math
import threading
import time

MAX_INPUT_TOKENS = 8192
SNAPSHOT_INTERVAL = 64
MAX_SNAPSHOTS = 2
PREFILL_REMAINING_THRESHOLD = 128
ROLES = ("user", "assistant")
RISK_LABELS = ("safe", "unsafe", "controversial")


def serialize(messages):
    return "\n\n".join(message["role"].upper() + ":\n" + message["content"] for message in messages)


def longest_common_prefix(left, right):
    position = 0
    for old, new in zip(left, right):
        if old != new:
            break
        position += 1
    return position


def _encode_kwargs(tokenizer):
    # HF exposes truncation (or **kwargs); the native Tokenizers encode method
    # does not accept that keyword. Inspect once, never retry an encode after a
    # TypeError that could have come from the tokenizer implementation itself.
    kwargs = {"add_special_tokens": False}
    try:
        parameters = inspect.signature(tokenizer.encode).parameters
    except (TypeError, ValueError):
        parameters = {}
    if ("truncation" in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                                          for parameter in parameters.values())):
        kwargs["truncation"] = False
    return kwargs


def _reject_configured_truncation(tokenizer):
    seen = set()
    for backend in (tokenizer, getattr(tokenizer, "backend_tokenizer", None),
                    getattr(tokenizer, "_tokenizer", None)):
        if backend is None or id(backend) in seen:
            continue
        seen.add(id(backend))
        configuration = getattr(backend, "truncation", None)
        if configuration is not None and configuration is not False:
            raise ValueError("Tokenizer truncation is enabled; disable it before using append-only sessions")


def _copy_probabilities(value):
    result = {}
    for role in ROLES:
        probabilities = {label: float(value[role][label]) for label in RISK_LABELS}
        if (not all(math.isfinite(number) and 0 <= number <= 1 for number in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 1e-4):
            raise ValueError("The chunk runner returned invalid risk probabilities")
        result[role] = probabilities
    return result


@dataclass(frozen=True)
class _Snapshot:
    position: int
    cache: object
    probabilities: dict


class TextStreamSession:
    """Transactional state machine, injectable without torch for CPU tests.

    runner(ids, cache) returns (updated_cache, probabilities_by_role). It may
    mutate its input cache. clone_cache must return fully independent mutable
    state: the real adapter uses deepcopy under torch.inference_mode().
    """

    def __init__(self, tokenizer, runner, *, clone_cache=copy.deepcopy, chunk_tokens=32,
                 prefill_chunk_tokens=2048):
        for name, value in (("chunk_tokens", chunk_tokens), ("prefill_chunk_tokens", prefill_chunk_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._tokenizer = tokenizer
        self._encode_kwargs = _encode_kwargs(tokenizer)
        self._runner = runner
        self._clone_cache = clone_cache
        self._chunk_tokens = chunk_tokens
        self._prefill_chunk_tokens = prefill_chunk_tokens
        self._messages = []
        self._ids = []
        self._cache = None
        self._probabilities = None
        self._snapshots = []
        self._lock = threading.RLock()
        self._forward_tokens = 0
        self._replay_tokens = 0
        self._wall_seconds = 0.0

    @property
    def messages(self):
        with self._lock:
            return copy.deepcopy(self._messages)

    @property
    def token_ids(self):
        with self._lock:
            return self._ids.copy()

    @property
    def snapshot_positions(self):
        with self._lock:
            return [snapshot.position for snapshot in self._snapshots]

    def begin_message(self, role, text=""):
        if role not in ROLES or not isinstance(text, str):
            raise ValueError("begin_message requires a user/assistant role and string text")
        started = time.perf_counter()
        with self._lock:
            messages = [*self._messages, {"role": role, "content": text}]
            return self._append_transaction(messages, len(text), started)

    def append_text(self, text):
        if not isinstance(text, str):
            raise ValueError("append_text requires a string")
        started = time.perf_counter()
        with self._lock:
            if not self._messages:
                raise ValueError("Call begin_message before append_text")
            messages = [*self._messages[:-1],
                        {"role": self._messages[-1]["role"], "content": self._messages[-1]["content"] + text}]
            return self._append_transaction(messages, len(text), started)

    def _append_transaction(self, messages, appended_characters, started):
        # All candidate state is private until every forward and validation has
        # succeeded. A runner that mutates its cache and then raises cannot
        # corrupt the committed cache, message text, snapshots or counters.
        tokenizing = time.perf_counter()
        text = serialize(messages)
        _reject_configured_truncation(self._tokenizer)
        encoded = self._tokenizer.encode(text, **self._encode_kwargs)
        if getattr(encoded, "overflowing", None):
            raise ValueError("Tokenizer returned overflowing token segments; truncated input is not permitted")
        ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
        tokenization_seconds = time.perf_counter() - tokenizing
        if not ids or len(ids) > MAX_INPUT_TOKENS:
            raise ValueError(f"Conversation has {len(ids)} tokens; required range is 1..{MAX_INPUT_TOKENS}; no truncation")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids):
            raise ValueError("Tokenizer must return nonnegative integer token IDs")
        old_length = len(self._ids)
        common = longest_common_prefix(self._ids, ids)
        rollback = common < old_length
        changed = ids != self._ids
        final_boundary = (len(ids) // SNAPSHOT_INTERVAL) * SNAPSHOT_INTERVAL
        needed_positions = tuple(position for position in
                                 range(final_boundary - (MAX_SNAPSHOTS - 1) * SNAPSHOT_INTERVAL,
                                       final_boundary + 1, SNAPSHOT_INTERVAL) if position > 0)
        clone_seconds = 0.0

        def clone(cache):
            nonlocal clone_seconds
            if cache is None:
                return None
            before = time.perf_counter()
            result = self._clone_cache(cache)
            clone_seconds += time.perf_counter() - before
            if result is cache:
                raise RuntimeError("Cache clone returned the original mutable object")
            return result

        if not changed:
            offset = old_length
            cache = self._cache
            probabilities = self._probabilities
            snapshots = self._snapshots.copy()
        elif not rollback:
            offset = old_length
            cache = clone(self._cache)
            probabilities = self._probabilities
            snapshots = self._snapshots.copy()
        else:
            snapshots = [snapshot for snapshot in self._snapshots if snapshot.position <= common]
            available_positions = {snapshot.position for snapshot in snapshots}
            missing_positions = [position for position in needed_positions if position not in available_positions]
            # A suffix merge can shrink across a 64-token boundary and make a
            # previously evicted older snapshot one of the final two again.
            # Restoring only at the latest common snapshot would skip that
            # required position forever. Restore before the earliest missing
            # boundary, falling back to zero when no sufficiently old state is
            # still retained. Already-valid later snapshots remain reusable.
            restore_limit = min(common, min(missing_positions, default=common))
            sources = [snapshot for snapshot in snapshots if snapshot.position <= restore_limit]
            snapshot = sources[-1] if sources else None
            offset = snapshot.position if snapshot else 0
            cache = clone(snapshot.cache) if snapshot else None
            probabilities = snapshot.probabilities if snapshot else None
        restored_at = offset
        fallback = rollback and offset == 0
        # Only the final two aligned positions can survive this transaction.
        # Earlier boundaries need neither a forward split nor a temporary
        # snapshot. Pick a rollback source before dropping obsolete snapshots.
        snapshots = [snapshot for snapshot in snapshots if snapshot.position in needed_positions]
        pending_positions = set(needed_positions).difference(snapshot.position for snapshot in snapshots)
        forward_tokens = 0
        forward_calls = 0
        model_seconds = 0.0
        while offset < len(ids):
            remaining = len(ids) - offset
            chunk_limit = (self._prefill_chunk_tokens if remaining > PREFILL_REMAINING_THRESHOLD
                           else self._chunk_tokens)
            boundary = next((position for position in needed_positions
                             if position > offset and position in pending_positions), len(ids))
            end = min(len(ids), offset + chunk_limit, boundary)
            before = time.perf_counter()
            cache, observed = self._runner(ids[offset:end], cache)
            model_seconds += time.perf_counter() - before
            if cache is None:
                raise RuntimeError("Cached chunk runner returned no state")
            probabilities = _copy_probabilities(observed)
            forward_tokens += end - offset
            forward_calls += 1
            offset = end
            if offset in pending_positions:
                snapshot = _Snapshot(offset, clone(cache), _copy_probabilities(probabilities))
                snapshots = sorted([*snapshots, snapshot], key=lambda saved: saved.position)
                pending_positions.remove(offset)
        if tuple(snapshot.position for snapshot in snapshots) != needed_positions:
            raise RuntimeError("Final aligned snapshot positions are incomplete")
        if probabilities is None:
            raise RuntimeError("No classification is available for this token sequence")
        probabilities = _copy_probabilities(probabilities)
        replay_tokens = max(0, min(old_length, len(ids)) - restored_at) if rollback else 0
        replayed_unchanged_tokens = max(0, common - restored_at) if rollback else 0
        target_role = messages[-1]["role"]
        wall_seconds = time.perf_counter() - started
        result = {
            "target_role": target_role,
            "risk_probabilities": probabilities[target_role].copy(),
            "native_tokens": len(ids), "previous_native_tokens": old_length,
            "net_new_tokens": len(ids) - old_length,
            "input_characters": sum(len(message["content"]) for message in messages),
            "serialized_characters": len(text), "appended_characters": appended_characters,
            "forward_tokens": forward_tokens, "forward_calls": forward_calls,
            "replay_tokens": replay_tokens, "replayed_unchanged_prefix_tokens": replayed_unchanged_tokens,
            "token_ids_changed": changed, "common_prefix_tokens": common,
            "rollback": rollback, "restore_position": restored_at,
            "fallback_full_replay": fallback,
            "snapshot_positions": [snapshot.position for snapshot in snapshots],
            "tokenization_seconds": tokenization_seconds, "cache_clone_seconds": clone_seconds,
            "model_call_seconds": model_seconds, "wall_seconds": wall_seconds,
            "cumulative_forward_tokens": self._forward_tokens + forward_tokens,
            "cumulative_replay_tokens": self._replay_tokens + replay_tokens,
            "cumulative_successful_append_wall_seconds": self._wall_seconds + wall_seconds,
            "generated_tokens": 0, "third_risk_class_validated": False,
            "category_output_validated": False, "safe_prefix_release_validated": False,
        }
        # The only commit point. Snapshots are private and never mutated.
        self._messages = messages
        self._ids = ids
        self._cache = cache
        self._probabilities = probabilities
        self._snapshots = snapshots
        self._forward_tokens += forward_tokens
        self._replay_tokens += replay_tokens
        self._wall_seconds += wall_seconds
        return result


def _require_l20():
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "L20" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Neural text-stream execution requires exactly one visible CUDA L20")
    return torch


class TextStreamRuntime:
    """Shared loaded Classifier/tokenizer; call new_session() per conversation.

    Configure full/window/memory and load its exported weights before creating
    this adapter. Architecture changes while sessions exist are unsupported.
    """

    def __init__(self, model, tokenizer, *, chunk_tokens=32, prefill_chunk_tokens=2048):
        self._torch = _require_l20()
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Prepare the classifier with eval().requires_grad_(False)")
        if any(parameter.device.type != "cuda" for parameter in model.parameters()):
            raise ValueError("All classifier parameters must be on the L20")
        if hasattr(model, "lm_head") or hasattr(model.backbone, "lm_head"):
            raise ValueError("A direct classifier must not have a vocabulary output head")
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self._lock = threading.RLock()

    def _forward(self, ids, cache):
        torch = self._torch
        with self._lock, torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
            output = self.model(tokens, past_key_values=cache, use_cache=True)
            hidden = output.last_hidden_state[:, -1]
            probabilities = {}
            for role in ROLES:
                logits, _ = self.model.readout(hidden, role)
                values = logits.float().softmax(-1)[0].cpu().tolist()
                probabilities[role] = dict(zip(RISK_LABELS, values))
            return output.past_key_values, probabilities

    def _clone(self, cache):
        # deepcopy copies all DynamicCache layers, GDN states, window logical
        # counters, memory matrices and FP32 normalizers, including any views.
        with self._lock, self._torch.inference_mode():
            self._torch.cuda.synchronize()
            result = copy.deepcopy(cache)
            self._torch.cuda.synchronize()
            return result

    def new_session(self):
        return TextStreamSession(self.tokenizer, self._forward,
                                 clone_cache=self._clone, chunk_tokens=self.chunk_tokens,
                                 prefill_chunk_tokens=self.prefill_chunk_tokens)


def audit_cache_tensor_isolation(*sessions):
    """L20 test interface: fail on shared tensor storage across session states.

    Includes current caches and both snapshots. Call between append operations;
    this diagnostic is deliberately outside the inference hot path.
    """
    torch = _require_l20()

    def storages(value):
        found, seen = {}, set()

        def visit(item):
            if id(item) in seen:
                return
            seen.add(id(item))
            if isinstance(item, torch.Tensor):
                storage = item.untyped_storage()
                if storage.nbytes():
                    found[(str(item.device), storage.data_ptr())] = storage.nbytes()
            elif isinstance(item, dict):
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)
            elif hasattr(item, "__dict__"):
                visit(vars(item))
        visit(value)
        return found

    records = []
    used = set()
    for index, session in enumerate(sessions):
        with session._lock:
            states = [("current", session._cache)] + [
                (f"snapshot_{snapshot.position}", snapshot.cache) for snapshot in session._snapshots]
            for name, cache in states:
                storage = storages(cache)
                if used.intersection(storage):
                    raise AssertionError(f"Mutable cache tensor storage is shared: session {index}, {name}")
                used.update(storage)
                records.append({"session": index, "state": name,
                                "tensor_storage_bytes": sum(storage.values()), "unique_storages": len(storage)})
    return {"pass": True, "states": records,
            "total_tensor_storage_bytes": sum(record["tensor_storage_bytes"] for record in records)}
