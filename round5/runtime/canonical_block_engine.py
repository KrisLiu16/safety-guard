"""L20-only canonical 32-token classifier blocks, with one optional CUDA Graph.

Every forward has the same physical shape and reads both heads at all 32
positions. Future padding is ordinary causal input, never generated output or
new user input. A partial block never exports its padding-contaminated cache.
"""
from __future__ import annotations

import copy
import threading
import time
import math

from graph_stream import (
    WindowGraphRunner, require_l20, state_tensors, storage_pointers,
    validate_cache as validate_graph_cache, validate_ids as validate_graph_ids,
)
from text_stream_runtime import _encode_kwargs, _reject_configured_truncation

BLOCK = 32
WINDOW = 512
MAX_TOKENS = 8192
ROLES = ('user', 'assistant')
LABELS = ('safe', 'unsafe', 'controversial')
ENGINES = ('eager', 'window_cuda_graph')


def execution_contract(pad_token_id):
    return {'name': 'canonical32-v1', 'block_tokens': BLOCK, 'pad_token_id': pad_token_id,
            'window': WINDOW, 'backbone_layers': 24, 'partial_cache': 'not_committed',
            'readout': 'all_32_positions_both_roles', 'max_input_tokens': MAX_TOKENS,
            'risk_labels': list(LABELS)}


def prepare_block(token_ids, prefix_tokens, pad_token_id, vocab_size):
    if (isinstance(prefix_tokens, bool) or not isinstance(prefix_tokens, int)
            or prefix_tokens % BLOCK or not 0 <= prefix_tokens <= MAX_TOKENS - BLOCK):
        raise ValueError('Block prefix must be a 32-aligned integer in 0..8160')
    if (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)
            or not 0 <= pad_token_id < vocab_size):
        raise ValueError('A valid in-vocabulary pad token is required')
    if not isinstance(token_ids, (list, tuple)) or not 1 <= len(token_ids) <= BLOCK:
        raise ValueError('A block requires 1..32 real token IDs')
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < vocab_size for v in token_ids):
        raise ValueError('Invalid real token ID')
    return [*token_ids, *([pad_token_id] * (BLOCK - len(token_ids)))], len(token_ids)


def probability_lists(value):
    """Validate/copy a CPU list shaped [two roles, 32 positions, 3 classes]."""
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError('Expected probabilities for both role heads')
    result = {}
    for role, rows in zip(ROLES, value):
        if not isinstance(rows, list) or len(rows) != BLOCK:
            raise RuntimeError('Every role must expose exactly 32 readout positions')
        output = []
        for row in rows:
            if (not isinstance(row, list) or len(row) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 1 for v in row)
                    or abs(sum(row) - 1) > 1e-4):
                raise RuntimeError('Invalid per-position risk probabilities')
            output.append([float(v) for v in row])
        result[role] = output
    return result


def validate_aligned_cache(cache, prefix_tokens, torch):
    if isinstance(prefix_tokens, bool) or not isinstance(prefix_tokens, int) or prefix_tokens % BLOCK:
        raise ValueError('Cache position must be a multiple of 32')
    if prefix_tokens == 0:
        if cache is not None:
            raise ValueError('The empty prefix must use cache=None')
        return
    from window_attention import WindowCache
    if (not BLOCK <= prefix_tokens <= MAX_TOKENS or type(cache) is not WindowCache
            or cache.window != WINDOW or len(cache.layers) != 24 or hasattr(cache, 'memories')
            or getattr(cache, 'offloading', False)):
        raise ValueError('Expected an independent W512 aligned cache')
    attention, linear = set(), 0
    for index, layer in enumerate(cache.layers):
        if hasattr(layer, 'keys'):
            attention.add(index)
            if not getattr(layer, 'is_initialized', False):
                raise ValueError('Uninitialized attention cache')
            for tensor in (layer.keys, layer.values):
                if (not torch.is_tensor(tensor) or tensor.ndim != 4 or tensor.shape[0] != 1
                        or tensor.shape[-2] != min(prefix_tokens, WINDOW - 1)):
                    raise ValueError('Incorrect retained real K/V length')
        elif hasattr(layer, 'conv_states'):
            linear += 1
            if (layer.record_past or set(layer.conv_states) != {0} or set(layer.recurrent_states) != {0}
                    or any(flags != {0: True} for flags in (layer.has_previous_state,
                              layer.is_conv_states_initialized, layer.is_recurrent_states_initialized))):
                raise ValueError('Unsupported GDN state layout')
        else:
            raise ValueError('Unknown hybrid cache layer')
    if (len(attention) != 6 or linear != 18 or set(cache.seen) != attention
            or any(isinstance(value, bool) or not isinstance(value, int) or value != prefix_tokens
                   for value in cache.seen.values())):
        raise ValueError('Cache logical position does not match the aligned prefix')
    tensors = state_tensors(cache)
    if (len(tensors) != 48 or any(not torch.is_tensor(t) or t.device.type != 'cuda'
                                or t.shape[0] != 1 for t in tensors.values())):
        raise ValueError('The complete cache must reside on L20')


def all_position_readout(model, hidden, torch):
    # Preserve [batch=1, physical positions=32, hidden] until the readout.
    return torch.stack([model.readout(hidden, role)[0].float().softmax(-1)[0] for role in ROLES])


class _AllPositionGraphRunner(WindowGraphRunner):
    """Reuse fixed-address state machinery without modifying the old runner."""
    def _body(self):
        torch = self._torch
        self.setup_forward_calls += 1
        self.setup_forward_tokens += BLOCK
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = self._model.backbone(input_ids=self._ids, position_ids=self._positions,
                                          past_key_values=self._arena, use_cache=True)
        if output.past_key_values is not self._arena:
            raise RuntimeError('Backbone replaced the graph cache')
        probabilities = all_position_readout(self._model, output.last_hidden_state, torch)
        for key, current in state_tensors(self._arena).items():
            if current is not self._stable[key]:
                self._stable[key].copy_(current)
        return probabilities

    def step(self, ids, cache, *, export_cache):
        with self._lock:
            self._life.require_ready()
            validate_graph_ids(ids, self._vocab_size, BLOCK)
            validate_graph_cache(cache, self._torch, BLOCK)
            try:
                with self._torch.inference_mode():
                    seen = self._load(cache, ids)
                    self._graph.replay()
                    self._arena.seen = {index: seen + BLOCK for index in self._arena.seen}
                    exported = copy.deepcopy(self._arena) if export_cache else None
                    probabilities = probability_lists(self._output.float().cpu().tolist())
                    if exported is not None and storage_pointers(exported).intersection(
                            storage_pointers(self._arena) | storage_pointers(cache)):
                        raise RuntimeError('Exported graph state aliases caller or arena')
                    return exported, probabilities
            except Exception:
                self._life.move('failed')
                raise


class CanonicalBlockEngine:
    """Frozen-model fixed32 block runner; caller owns committed aligned caches.

    evaluate_block(real_ids, aligned_cache, prefix_tokens=...) returns all 32
    positions, but only len(real_ids) are semantic outputs. The caller must not
    commit a partial block: cache is None unless all 32 IDs are real. Replay,
    net-new user input and accepted-session accounting belong to the caller.
    """
    def __init__(self, model, tokenizer, *, inference_engine='eager', pad_token_id=None):
        if inference_engine not in ENGINES:
            raise ValueError('inference_engine must be eager or window_cuda_graph')
        self._torch = require_l20()
        self._lock = threading.RLock()
        self._phase = 'initializing'
        self._model, self._runner = model, None
        self._engine = inference_engine
        self._vocab_size = model.backbone.config.vocab_size
        self._pad = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
        prepare_block([0], 0, self._pad, self._vocab_size)
        backbone = model.backbone
        if (model.training or any(p.requires_grad or p.device.type != 'cuda' for p in model.parameters())
                or getattr(backbone, 'guard_window', None) != WINDOW
                or not hasattr(backbone, 'guard_original_forward') or hasattr(backbone, 'memory_original_forward')
                or len(backbone.layers) != 24 or getattr(backbone.rotary_emb, 'rope_type', None) != 'default'
                or any('lm_head' in name for name, _ in model.named_modules())):
            raise ValueError('Canonical blocks require the frozen W512 24-layer L20 classifier without lm_head')
        self._counts = {key: 0 for key in ('forward_calls', 'forward_tokens', 'real_forward_tokens',
                        'padding_tokens', 'eager_calls', 'eager_forward_tokens', 'graph_calls',
                        'graph_forward_tokens', 'exported_full_caches', 'partial_calls', 'failed_calls',
                        'attempted_calls', 'attempted_forward_tokens')}
        self._metadata = {'execution_contract': execution_contract(self._pad), 'engine_type': inference_engine,
                          'inference_engine': inference_engine, 'startup_forward_calls': 0,
                          'startup_forward_tokens': 0, 'auxiliary_input_tokens': 0,
                          'capture_seconds': 0., 'startup_seconds': 0., 'captured_chunk_lengths': [],
                          'shared_graph_arena_storage_bytes': 0,
                          'initialization_excluded_from_request_accounting': True,
                          'request_counter_scope': 'Successful forwards plus separate attempted calls/tokens; failed attempts are not claimed as completed input',
                          'capture_during_requests': False, 'graph_error_fallback': False,
                          'generated_tokens': 0}
        started = time.perf_counter()
        try:
            if inference_engine == 'window_cuda_graph':
                _reject_configured_truncation(tokenizer)
                text = 'USER:\n' + ('公开资料用于一般阅读和目录整理。 Public reference text for ordinary reading.\n' * 128)
                encoded = tokenizer.encode(text, **_encode_kwargs(tokenizer))
                if getattr(encoded, 'overflowing', None):
                    raise ValueError('Startup tokenizer returned overflow')
                ids = list(encoded.ids if hasattr(encoded, 'ids') else encoded)
                if len(ids) < WINDOW + BLOCK:
                    raise ValueError('Insufficient canonical startup seed tokens')
                cache = None
                for prefix in range(0, WINDOW, BLOCK):
                    block, _ = prepare_block(ids[prefix:prefix + BLOCK], prefix, self._pad, self._vocab_size)
                    cache, _ = self._eager_block(block, cache, prefix, export_cache=True)
                self._runner = _AllPositionGraphRunner(model, cache, ids[WINDOW:WINDOW + BLOCK], chunk_tokens=BLOCK)
                metrics = self._runner.storage_metrics()
                self._metadata.update(startup_forward_calls=16 + metrics['setup_forward_calls'],
                    startup_forward_tokens=WINDOW + metrics['setup_forward_tokens'],
                    auxiliary_input_tokens=WINDOW + metrics['setup_forward_tokens'],
                    startup_seed_calls=16, startup_seed_tokens=WINDOW,
                    startup_count_scope='Forward invocations: 16 eager32 seed blocks, 3 warmups and 1 capture trace; never user input',
                    capture_seconds=metrics['setup_seconds'], captured_chunk_lengths=[BLOCK],
                    shared_graph_arena_storage_bytes=metrics['arena_state_storage_bytes'], graph_storage=metrics)
                del cache
            self._metadata['startup_seconds'] = time.perf_counter() - started
            self._phase = 'ready'
        except Exception:
            self._phase = 'failed'
            if self._runner is not None:
                self._runner.close()
            raise

    def _require_ready(self):
        if self._phase != 'ready':
            raise RuntimeError('Canonical block engine is not ready: ' + self._phase)

    def _eager_block(self, ids, cache, prefix_tokens, *, export_cache):
        torch = self._torch
        with torch.inference_mode():
            work = copy.deepcopy(cache) if cache is not None else None
            tensor = torch.tensor([ids], dtype=torch.long, device='cuda')
            positions = torch.arange(prefix_tokens, prefix_tokens + BLOCK, dtype=torch.long, device='cuda')[None, :]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = self._model.backbone(input_ids=tensor, position_ids=positions,
                                              past_key_values=work, use_cache=True)
            probabilities = probability_lists(all_position_readout(self._model, output.last_hidden_state, torch).cpu().tolist())
            result = output.past_key_values
            if result is None:
                raise RuntimeError('Canonical forward returned no cache')
            if cache is not None and storage_pointers(result).intersection(storage_pointers(cache)):
                raise RuntimeError('Eager working state aliases the caller cache')
            return (result if export_cache else None), probabilities

    def evaluate_block(self, token_ids, aligned_cache, *, prefix_tokens):
        with self._lock:
            self._require_ready()
            padded, valid = prepare_block(token_ids, prefix_tokens, self._pad, self._vocab_size)
            validate_aligned_cache(aligned_cache, prefix_tokens, self._torch)
            route = 'graph' if self._engine == 'window_cuda_graph' and prefix_tokens >= WINDOW else 'eager'
            self._counts['attempted_calls'] += 1
            self._counts['attempted_forward_tokens'] += BLOCK
            try:
                if route == 'graph':
                    if self._runner is None:
                        raise RuntimeError('Missing fixed32 graph; eager fallback is disabled')
                    cache, probabilities = self._runner.step(padded, aligned_cache, export_cache=valid == BLOCK)
                else:
                    cache, probabilities = self._eager_block(padded, aligned_cache, prefix_tokens,
                                                              export_cache=valid == BLOCK)
                if valid < BLOCK and cache is not None:
                    raise RuntimeError('Partial block must not export padding-contaminated state')
                if valid == BLOCK:
                    validate_aligned_cache(cache, prefix_tokens + BLOCK, self._torch)
                execution = {'eager_calls': int(route == 'eager'), 'graph_calls': int(route == 'graph'),
                             'eager_forward_tokens': BLOCK if route == 'eager' else 0,
                             'graph_forward_tokens': BLOCK if route == 'graph' else 0,
                             'forward_calls': 1, 'forward_tokens': BLOCK,
                             'real_forward_tokens': valid, 'padding_tokens': BLOCK - valid,
                             'exported_full_caches': int(valid == BLOCK), 'partial_calls': int(valid < BLOCK)}
                for key, value in execution.items():
                    self._counts[key] += value
                return {'cache': cache, 'probabilities_by_role': probabilities, 'valid_tokens': valid,
                        'execution': execution, 'generated_tokens': 0}
            except Exception:
                self._counts['failed_calls'] += 1
                self._phase = 'failed'
                raise

    def clone_cache(self, cache):
        with self._lock:
            self._require_ready()
            if cache is None:
                return None
            validate_aligned_cache(cache, cache.get_seq_length(), self._torch)
            try:
                with self._torch.inference_mode():
                    clone = copy.deepcopy(cache)
                    if storage_pointers(clone).intersection(storage_pointers(cache)):
                        raise RuntimeError('Snapshot clone shares mutable cache storage')
                    return clone
            except Exception:
                self._phase = 'failed'
                raise

    @property
    def engine_metadata(self):
        with self._lock:
            return copy.deepcopy(self._metadata)

    def stats(self):
        with self._lock:
            return {**self._counts, 'phase': self._phase}

    def close(self):
        with self._lock:
            if self._phase == 'closed':
                return
            try:
                if self._runner is not None:
                    self._runner.close()
            finally:
                self._runner = None
                self._model = None
                self._phase = 'closed'
