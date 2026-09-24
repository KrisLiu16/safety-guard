"""Canonical fixed-32 execution for whole inputs and transactional text appends.

Only completed 32-token blocks may enter a committed cache. Partial blocks
always have fixed future padding and are replayed from their aligned anchor.
CPU tests inject an engine; actual model execution lives in the L20-only engine.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import threading
import time

from text_stream_runtime import serialize, longest_common_prefix, _encode_kwargs, _reject_configured_truncation

BLOCK = 32
MAX_TOKENS = 8192
ROLES = ('user', 'assistant')
LABELS = ('safe', 'unsafe', 'controversial')


def validate_ids(ids):
    if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= MAX_TOKENS
            or any(type(value) is not int or value < 0 for value in ids)):
        raise ValueError('Canonical input must contain 1..8192 nonnegative integer native IDs')


def encode(tokenizer, messages):
    _reject_configured_truncation(tokenizer)
    value = tokenizer.encode(serialize(messages), **_encode_kwargs(tokenizer))
    if getattr(value, 'overflowing', None):
        raise ValueError('Canonical input refuses implicit tokenizer truncation/overflow')
    ids = list(value.ids if hasattr(value, 'ids') else value)
    validate_ids(ids)
    return ids


def block_probabilities(result, real_count):
    if result.get('valid_tokens') != real_count:
        raise ValueError('Engine returned an inconsistent real token count')
    execution = result['execution']
    if (execution['forward_tokens'] != BLOCK or execution['real_forward_tokens'] != real_count
            or execution['padding_tokens'] != BLOCK - real_count
            or execution['eager_calls'] + execution['graph_calls'] != 1):
        raise ValueError('Canonical engine did not execute exactly one fixed32 block')
    values = result['probabilities_by_role']
    output = {}
    for role in ROLES:
        rows = values[role]
        if len(rows) != BLOCK:
            raise ValueError('Canonical engine must expose all 32 positions for both roles')
        output[role] = []
        for row in rows:
            numbers = [float(value) for value in row]
            if (len(numbers) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in numbers)
                    or abs(sum(numbers) - 1) > 1e-4):
                raise ValueError('Invalid canonical risk probabilities')
            output[role].append(numbers)
    if (real_count < BLOCK and result.get('cache') is not None
            or real_count == BLOCK and result.get('cache') is None):
        raise ValueError('Only full real blocks may return a committed cache')
    return output


def risk_at(values, index):
    return {role: dict(zip(LABELS, values[role][index])) for role in ROLES}


@dataclass(frozen=True)
class Snapshot:
    position: int
    cache: object
    probabilities: dict


class CanonicalTextSession:
    def __init__(self, tokenizer, runner, *, clone_cache=copy.deepcopy, synchronize=lambda: None):
        self._tokenizer, self._runner, self._clone_cache = tokenizer, runner, clone_cache
        self._synchronize = synchronize
        self._lock = threading.RLock()
        self._messages, self._ids, self._snapshots = [], [], []
        self._cache, self._aligned_probabilities, self._probabilities = None, None, None
        self._aligned = 0
        self._forward_tokens = self._real_forward_tokens = self._padding_tokens = self._replay_tokens = 0
        self._wall_seconds = 0.

    @property
    def messages(self):
        with self._lock:
            return copy.deepcopy(self._messages)

    @property
    def token_ids(self):
        with self._lock:
            return list(self._ids)

    @property
    def logical_input_tokens(self):
        return len(self._ids)

    @property
    def aligned_cache_tokens(self):
        return self._aligned

    @property
    def snapshot_positions(self):
        with self._lock:
            return [item.position for item in self._snapshots]

    def begin_message(self, role, text=''):
        if role not in ROLES or not isinstance(text, str):
            raise ValueError('begin_message requires a user/assistant role and string text')
        start = time.perf_counter()
        with self._lock:
            return self._transaction([*self._messages, {'role': role, 'content': text}], len(text), start)

    def append_text(self, text):
        if not isinstance(text, str):
            raise ValueError('append_text requires string text')
        start = time.perf_counter()
        with self._lock:
            if not self._messages:
                raise ValueError('begin_message must precede append_text')
            messages = [*self._messages[:-1], {**self._messages[-1], 'content': self._messages[-1]['content'] + text}]
            return self._transaction(messages, len(text), start)

    def _transaction(self, messages, appended_characters, start):
        token_start = time.perf_counter()
        ids = encode(self._tokenizer, messages)
        token_seconds = time.perf_counter() - token_start
        old_length, length = len(self._ids), len(ids)
        common = longest_common_prefix(self._ids, ids)
        rollback, changed = common < old_length, ids != self._ids
        aligned = length // BLOCK * BLOCK
        needed = tuple(p for p in (aligned - 2 * BLOCK, aligned - BLOCK) if p > 0)
        clone_seconds = 0.

        def clone(cache):
            nonlocal clone_seconds
            if cache is None:
                return None
            began = time.perf_counter()
            result = self._clone_cache(cache)
            clone_seconds += time.perf_counter() - began
            if result is cache:
                raise RuntimeError('Cache clone aliases mutable state')
            return result

        physical = real = padding = calls = model_seconds = 0
        execution = {'eager_calls': 0, 'graph_calls': 0}
        if not changed:
            cache, aligned_probs, probs = self._cache, self._aligned_probabilities, self._probabilities
            snapshots, restored = self._snapshots.copy(), length
        else:
            sources = [item for item in self._snapshots if item.position <= common]
            if self._aligned and self._aligned <= common:
                sources.append(Snapshot(self._aligned, self._cache, self._aligned_probabilities))
            available = {item.position: item for item in sources}
            missing = [p for p in needed if p not in available]
            limit = min(common, aligned, min(missing, default=aligned))
            source = max((item for item in sources if item.position <= limit), key=lambda item: item.position, default=None)
            restored = source.position if source else 0
            offset = restored
            cache = clone(source.cache) if source else None
            aligned_probs = copy.deepcopy(source.probabilities) if source else None
            probs = aligned_probs
            # Retained snapshots are never passed to a mutating runner.
            snapshots = [available[p] for p in needed if p in available]
            if offset in needed and offset not in [item.position for item in snapshots]:
                snapshots.append(Snapshot(offset, clone(cache), copy.deepcopy(aligned_probs)))
            while offset < length:
                real_ids = ids[offset:offset + BLOCK]
                began = time.perf_counter()
                result = self._runner(real_ids, cache, prefix_tokens=offset)
                model_seconds += time.perf_counter() - began
                values = block_probabilities(result, len(real_ids))
                observed = result['execution']
                if (observed['forward_tokens'] != BLOCK or observed['real_forward_tokens'] != len(real_ids)
                        or observed['padding_tokens'] != BLOCK - len(real_ids)
                        or observed['eager_calls'] + observed['graph_calls'] != 1):
                    raise ValueError('Canonical block execution accounting is inconsistent')
                for key in execution:
                    execution[key] += observed[key]
                physical += BLOCK; real += len(real_ids); padding += BLOCK - len(real_ids); calls += 1
                probs = risk_at(values, len(real_ids) - 1)
                if len(real_ids) == BLOCK:
                    cache = result['cache']
                    aligned_probs = copy.deepcopy(probs)
                    offset += BLOCK
                    if offset in needed and offset not in [item.position for item in snapshots]:
                        snapshots.append(Snapshot(offset, clone(cache), copy.deepcopy(aligned_probs)))
                else:
                    # Never advance the aligned cache into dummy future tokens.
                    offset += len(real_ids)
            snapshots.sort(key=lambda item: item.position)
            if tuple(item.position for item in snapshots) != needed:
                raise RuntimeError('Final aligned snapshot set is incomplete')
        if probs is None:
            raise RuntimeError('No classification for this logical input')
        replay = max(0, common - restored) if changed else 0
        target = messages[-1]['role']
        # Snapshot copies may be queued after the last CPU probability read.
        # Wait before both wall-clock sampling and the only commit point; an
        # asynchronous CUDA failure must leave the old session intact.
        self._synchronize()
        wall = time.perf_counter() - start
        result = {'target_role': target, 'risk_probabilities': dict(probs[target]),
                  'risk_probabilities_by_role': copy.deepcopy(probs),
                  'native_tokens': length, 'logical_input_tokens': length, 'aligned_cache_tokens': aligned,
                  'previous_native_tokens': old_length, 'net_new_tokens': length - old_length,
                  'forward_tokens': physical, 'real_forward_tokens': real, 'padding_tokens': padding,
                  'forward_calls': calls, 'replay_tokens': replay, 'replayed_unchanged_prefix_tokens': replay,
                  'retokenized_suffix_tokens': length - common if changed else 0,
                  'input_characters': sum(len(message['content']) for message in messages),
                  'serialized_characters': len(serialize(messages)), 'appended_characters': appended_characters,
                  'common_prefix_tokens': common, 'token_ids_changed': changed,
                  'rollback': rollback, 'restore_position': restored, 'fallback_full_replay': changed and restored == 0 and old_length > 0,
                  'snapshot_positions': [item.position for item in snapshots],
                  'tokenization_seconds': token_seconds, 'cache_clone_seconds': clone_seconds,
                  'model_call_seconds': model_seconds, 'wall_seconds': wall,
                  'cumulative_forward_tokens': self._forward_tokens + physical,
                  'cumulative_real_forward_tokens': self._real_forward_tokens + real,
                  'cumulative_padding_tokens': self._padding_tokens + padding,
                  'cumulative_replay_tokens': self._replay_tokens + replay,
                  'cumulative_successful_append_wall_seconds': self._wall_seconds + wall,
                  'engine_execution': {**execution, 'forward_tokens': physical, 'real_forward_tokens': real,
                                       'padding_tokens': padding, 'initialization_included': False},
                  'generated_tokens': 0, 'third_risk_class_validated': False,
                  'category_output_validated': False, 'safe_prefix_release_validated': False,
                  'execution_definition': 'canonical32-v1'}
        # Single commit point, after all engine calls/probability validation succeed.
        self._messages, self._ids, self._snapshots = messages, ids, snapshots
        self._cache, self._aligned, self._aligned_probabilities, self._probabilities = cache, aligned, aligned_probs, probs
        self._forward_tokens += physical; self._real_forward_tokens += real
        self._padding_tokens += padding; self._replay_tokens += replay; self._wall_seconds += wall
        return result


class CanonicalTextRuntime:
    def __init__(self, model, tokenizer, *, inference_engine='eager', pad_token_id=None, engine=None):
        if engine is None:
            from canonical_block_engine import CanonicalBlockEngine
            engine = CanonicalBlockEngine(model, tokenizer, inference_engine=inference_engine, pad_token_id=pad_token_id)
        self.engine, self.tokenizer = engine, tokenizer
        self._lock = threading.RLock()

    @property
    def engine_metadata(self):
        return copy.deepcopy(self.engine.engine_metadata)

    def stats(self):
        return dict(self.engine.stats())

    def _forward(self, ids, cache, *, prefix_tokens):
        with self._lock:
            return self.engine.evaluate_block(ids, cache, prefix_tokens=prefix_tokens)

    def new_session(self):
        return CanonicalTextSession(self.tokenizer, self._forward, clone_cache=self.engine.clone_cache,
                                    synchronize=self._synchronize)

    def _synchronize(self):
        with self._lock:
            torch = getattr(self.engine, '_torch', None)
            if torch is not None:
                torch.cuda.synchronize()

    def classify_prefixes(self, ids):
        validate_ids(ids)
        started = time.perf_counter()
        values = {role: [] for role in ROLES}
        cache, calls, physical, padding = None, 0, 0, 0
        execution = {'eager_calls': 0, 'graph_calls': 0}
        for offset in range(0, len(ids), BLOCK):
            block = list(ids[offset:offset + BLOCK])
            result = self._forward(block, cache, prefix_tokens=offset)
            observed = block_probabilities(result, len(block))
            for role in ROLES:
                values[role].extend(observed[role][:len(block)])
            for key in execution:
                execution[key] += result['execution'][key]
            cache = result['cache']
            calls += 1; physical += BLOCK; padding += BLOCK - len(block)
        del cache
        self._synchronize()
        return {'probabilities_by_role_all_tokens': values,
                'native_tokens': len(ids), 'logical_input_tokens': len(ids),
                'forward_tokens': physical, 'real_forward_tokens': len(ids), 'padding_tokens': padding,
                'forward_calls': calls, 'generated_tokens': 0, 'wall_seconds': time.perf_counter() - started,
                'execution_contract': self.engine_metadata['execution_contract'],
                'engine_execution': {**execution, 'forward_tokens': physical, 'real_forward_tokens': len(ids),
                                     'padding_tokens': padding, 'initialization_included': False}}

    def close(self):
        with self._lock:
            self.engine.close()
