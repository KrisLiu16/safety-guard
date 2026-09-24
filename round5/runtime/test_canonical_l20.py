"""Pre-calibration canonical32 correctness on a fixed window checkpoint.

No bundle or calibrated threshold is needed. All model work runs in a bounded
L20 child; old bulk outputs are diagnostic and cannot pass/fail canonical parity.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

PROBABILITY_ATOL = 1e-6
FP32_STATE_ATOL = 1e-6
FP32_STATE_RTOL = 1e-6
RUNTIME_MODULES = ('text_stream_runtime', 'graph_stream', 'canonical_block_engine', 'canonical_text_runtime')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def runtime_source_receipt(directory):
    directory = Path(directory).resolve()
    hashes, paths = {}, {}
    for name in RUNTIME_MODULES:
        module = sys.modules.get(name)
        expected = directory / (name + '.py')
        if module is None or Path(module.__file__).resolve() != expected:
            raise RuntimeError('Canonical audit imported a runtime module from another directory: ' + name)
        digest = sha(expected)
        if getattr(module, '__canonical_audit_source_sha256__', None) != digest:
            raise RuntimeError('Canonical audit runtime source changed after exact loading: ' + name)
        hashes[expected.name], paths[name] = digest, str(expected)
    return {'runtime_source_sha256': hashes, 'runtime_source_paths': paths}


def load_runtime_modules(directory):
    """Run after the legacy loader edits sys.path; preserve its WindowCache class."""
    directory = Path(directory).resolve()
    sys.path[:] = [str(directory)] + [value for value in sys.path if value != str(directory)]
    for name in RUNTIME_MODULES:
        path = directory / (name + '.py')
        previous = sys.modules.get(name)
        if previous is not None and Path(previous.__file__).resolve() != path:
            raise RuntimeError('Refusing a cached runtime from another directory: ' + name)
        digest = sha(path)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.__canonical_audit_source_sha256__ = digest
    return runtime_source_receipt(directory)


def execution_delta(before, after):
    keys = ('eager_calls', 'graph_calls', 'forward_calls', 'forward_tokens', 'real_forward_tokens', 'padding_tokens')
    delta = {name: after[name] - before[name] for name in keys}
    if any(type(value) is not int or value < 0 for value in delta.values()):
        raise AssertionError('Invalid request engine counter delta')
    return delta


def probability_error(left, right):
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError('Probability role/class keys differ')
        return max(probability_error(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('Probability position count differs')
        return max(probability_error(a, b) for a, b in zip(left, right)) if left else 0.
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise AssertionError('Nonfinite probability')
    return abs(float(left) - float(right))


def assert_probabilities(left, right):
    error = probability_error(left, right)
    if error > PROBABILITY_ATOL:
        raise AssertionError('Canonical same-prefix probability mismatch: ' + str(error))
    return error


def assert_cache(torch, left, right):
    from graph_stream import state_tensors, state_metadata
    if left is None or right is None:
        if left is not right:
            raise AssertionError('One aligned cache is missing')
        return 0.
    if state_metadata(left) != state_metadata(right):
        raise AssertionError('Canonical cache logical metadata differs')
    a, b = state_tensors(left), state_tensors(right)
    if set(a) != set(b):
        raise AssertionError('Incomplete cache tensor inventory')
    maximum = 0.
    for name in a:
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise AssertionError('Cache shape/dtype differs: ' + repr(name))
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise AssertionError('Nonfinite canonical cache state: ' + repr(name))
        error = float((x.float() - y.float()).abs().max().cpu()) if x.numel() else 0.
        maximum = max(maximum, error)
        same = (torch.allclose(x, y, atol=FP32_STATE_ATOL, rtol=FP32_STATE_RTOL)
                if x.dtype == torch.float32 else torch.equal(x, y))
        if not same:
            raise AssertionError('Canonical complete-state mismatch: ' + repr((name, error)))
    return maximum


def verify_fixed_input(args):
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Fixed checkpoint SHA mismatch')
    if args.training_summary:
        summary = json.loads(args.training_summary.read_text())
        if (summary.get('status') != 'completed'
                or summary.get('final_checkpoint_sha256') != args.checkpoint_sha256
                or summary['sft'].get('steps') != 4164
                or summary['classification_rl'].get('updates') != 256):
            raise ValueError('Checkpoint is not the completed, fixed training selection')


def run_child(args):
    report = {'status': 'running', 'pass': False, 'generated_tokens': 0, 'phase': 'hardware',
              'checkpoint_sha256': args.checkpoint_sha256, 'bulk_reference_is_diagnostic_only': True,
              'probability_atol': PROBABILITY_ATOL,
              'state_tolerance': {'bfloat16': 'exact', 'float32_atol': FP32_STATE_ATOL, 'float32_rtol': FP32_STATE_RTOL},
              'script_sha256': sha(__file__)}
    runtimes = []
    try:
        verify_fixed_input(args)
        local = Path(__file__).resolve().parent
        sys.path[:0] = [str(local), str(args.round4_code), str(args.base_root / 'input'), str(args.base_root / 'window')]
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'L20' not in torch.cuda.get_device_name(0):
            raise RuntimeError('All neural checks require exactly one CUDA L20; no CPU/MPS')
        from types import SimpleNamespace
        from infer_classifier import load_classifier
        setup = SimpleNamespace(base_root=args.base_root, base_code_dir=args.base_root / 'input',
                                window_code_dir=args.base_root / 'window', memory_code_dir=args.round4_code,
                                checkpoint=args.checkpoint)
        _, model, tokenizer, _, _, device = load_classifier(setup, 'window')
        sources = load_runtime_modules(local)
        from canonical_text_runtime import CanonicalTextRuntime, encode
        from graph_stream import storage_pointers
        eager = CanonicalTextRuntime(model, tokenizer, inference_engine='eager')
        graph = CanonicalTextRuntime(model, tokenizer, inference_engine='window_cuda_graph')
        runtimes = [eager, graph]
        for runtime in runtimes:
            if any(runtime.stats()[key] != 0 for key in ('eager_calls', 'graph_calls', 'forward_calls',
                                                         'forward_tokens', 'real_forward_tokens', 'padding_tokens')):
                raise AssertionError('Startup capture was mixed into request counters')
        if eager.engine_metadata['execution_contract'] != graph.engine_metadata['execution_contract']:
            raise AssertionError('Eager/graph serving definitions differ')
        report.update(device=device, execution_contract=eager.engine_metadata['execution_contract'],
                      initialization={'eager': eager.engine_metadata, 'graph': graph.engine_metadata},
                      **sources,
                      phase='fixed_block_all_positions')
        write(args.output, report)
        phrase = tokenizer.encode('公开资料用于解释科学证据。 Public reference material discusses evidence and observations.\n', add_special_tokens=False, truncation=False)
        ids = (phrase * (8192 // len(phrase) + 2))[:8192]
        caches = [None, None]
        max_state = max_probability = 0.
        # Same fixed32 execution history, all real positions and two heads.
        for offset in range(0, 608, 32):
            outputs = [runtime.engine.evaluate_block(ids[offset:offset + 32], caches[index], prefix_tokens=offset)
                       for index, runtime in enumerate(runtimes)]
            max_probability = max(max_probability, assert_probabilities(outputs[0]['probabilities_by_role'], outputs[1]['probabilities_by_role']))
            max_state = max(max_state, assert_cache(torch, outputs[0]['cache'], outputs[1]['cache']))
            caches = [output['cache'] for output in outputs]
            if offset == 480:
                seed = eager.engine.clone_cache(caches[0])
        if (graph.stats()['eager_calls'] != 16 or graph.stats()['graph_calls'] != 3
                or eager.stats()['eager_calls'] != 19 or eager.stats()['graph_calls'] != 0):
            raise AssertionError('Graph mode did not actually execute the required filled-window graphs')
        report['fixed_608_execution'] = {'eager': eager.stats(), 'graph': graph.stats()}
        report['continued_semantic_blocks'] = {'real_tokens': 608, 'max_probability_error': max_probability, 'max_state_error': max_state}
        padding_cases = []
        alternate = (eager.engine_metadata['execution_contract']['pad_token_id'] + 1) % model.backbone.config.vocab_size
        for prefix, cache in ((0, None), (512, seed)):
            for count in range(1, 32):
                real_ids = ids[prefix:prefix + count]
                saved = eager.engine.clone_cache(cache)
                outputs = [runtime.engine.evaluate_block(real_ids, cache, prefix_tokens=prefix) for runtime in runtimes]
                error = assert_probabilities(outputs[0]['probabilities_by_role'], outputs[1]['probabilities_by_role'])
                assert_cache(torch, cache, saved)
                if any(output['cache'] is not None for output in outputs):
                    raise AssertionError('A padded partial block exported dummy future state')
                for runtime, output in zip(runtimes, outputs):
                    changed_future = runtime.engine.evaluate_block(real_ids + [alternate] * (32 - count), cache, prefix_tokens=prefix)
                    error = max(error, assert_probabilities(
                        {role: output['probabilities_by_role'][role][:count] for role in ('user', 'assistant')},
                        {role: changed_future['probabilities_by_role'][role][:count] for role in ('user', 'assistant')}))
                padding_cases.append({'prefix': prefix, 'real_count': count, 'max_probability_error': error})
        report['future_padding_cases'] = padding_cases
        report['phase'] = 'whole_allprefix_and_sessions'
        write(args.output, report)
        whole_cases = []
        for length in (1, 31, 32, 33, 511, 512, 513, 8191, 8192):
            outputs = [runtime.classify_prefixes(ids[:length]) for runtime in runtimes]
            error = assert_probabilities(outputs[0]['probabilities_by_role_all_tokens'], outputs[1]['probabilities_by_role_all_tokens'])
            for output in outputs:
                if (any(len(rows) != length for rows in output['probabilities_by_role_all_tokens'].values())
                        or output['real_forward_tokens'] != length
                        or output['forward_tokens'] != math.ceil(length / 32) * 32
                        or output['padding_tokens'] != output['forward_tokens'] - length):
                    raise AssertionError('All-prefix output includes dummy padding or bad token accounting')
            whole_cases.append({'native_tokens': length, 'max_probability_error': error})
        report['whole_all_prefix_cases'] = whole_cases
        session_cases = []
        text = 'x ' * 699 + 'informatio'
        for runtime in runtimes:
            for pieces in ([text], [text[:1], text[1:31], text[31:511], text[511:]], [text[:513], text[513:]]):
                session = runtime.new_session()
                session.begin_message('user', '')
                for piece in [*pieces, 'n', '!']:
                    result = session.append_text(piece)
                    reference = runtime.new_session()
                    observed = reference.begin_message('user', session.messages[-1]['content'])
                    error = assert_probabilities(result['risk_probabilities_by_role'], observed['risk_probabilities_by_role'])
                    state_error = assert_cache(torch, session._cache, reference._cache)
                    session_cases.append({'engine': runtime.engine_metadata.get('engine_type', runtime.engine_metadata.get('inference_engine')),
                                          'native_tokens': result['native_tokens'], 'rollback': result['rollback'],
                                          'net_new_tokens': result['net_new_tokens'], 'max_probability_error': error,
                                          'max_aligned_state_error': state_error})
                other = runtime.new_session()
                other.begin_message('assistant', 'A separate public reading session.')
                unchanged = runtime.engine.clone_cache(session._cache)
                other.append_text(' More neutral context.')
                assert_cache(torch, session._cache, unchanged)
                switched = session.begin_message('assistant', '我可以解释公开资料。')
                expected = runtime.classify_prefixes(encode(tokenizer, session.messages))
                assert_probabilities([list(switched['risk_probabilities_by_role'][role].values()) for role in ('user', 'assistant')],
                                     [expected['probabilities_by_role_all_tokens'][role][-1] for role in ('user', 'assistant')])
                before_ids, before_messages = session.token_ids, session.messages
                before_cache = runtime.engine.clone_cache(session._cache)
                try:
                    session.append_text(' x' * 8300)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Overlong text was not rejected')
                if session.token_ids != before_ids or session.messages != before_messages:
                    raise AssertionError('Rejected text changed the committed session')
                assert_cache(torch, session._cache, before_cache)
                actual_runner = session._runner
                def fail_after_private_work(real_ids, cache, *, prefix_tokens):
                    actual_runner(real_ids, cache, prefix_tokens=prefix_tokens)
                    raise RuntimeError('Injected post-forward transaction failure')
                session._runner = fail_after_private_work
                try:
                    session.append_text(' more')
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Injected transaction failure did not propagate')
                finally:
                    session._runner = actual_runner
                if session.token_ids != before_ids:
                    raise AssertionError('Failed neural transaction committed tokens')
                assert_cache(torch, session._cache, before_cache)
                committed = [session._cache, *[snapshot.cache for snapshot in session._snapshots]]
                seen = set()
                for cache in committed:
                    pointers = storage_pointers(cache)
                    if seen & pointers:
                        raise AssertionError('Current cache and snapshots share mutable storage')
                    seen |= pointers
        if not any(row['rollback'] and row['net_new_tokens'] < 0 for row in session_cases):
            raise AssertionError('The real tokenizer did not exercise the intended BPE contraction')
        crossed = []
        for runtime in runtimes:
            session = runtime.new_session()
            before = session.begin_message('user', 'x ' * 700 + 'informatio')
            old_aligned, old_snapshots = session.aligned_cache_tokens, session.snapshot_positions
            after = session.append_text('n')
            if not (after['rollback'] and after['common_prefix_tokens'] < old_aligned
                    and 0 < after['restore_position'] <= after['common_prefix_tokens']
                    and after['restore_position'] in old_snapshots):
                raise AssertionError('Real BPE case did not restore a prior nonzero aligned snapshot')
            reference = runtime.new_session(); expected = reference.begin_message('user', session.messages[-1]['content'])
            assert_probabilities(after['risk_probabilities_by_role'], expected['risk_probabilities_by_role'])
            assert_cache(torch, session._cache, reference._cache)
            crossed.append({'previous_tokens': before['native_tokens'], 'native_tokens': after['native_tokens'],
                            'old_aligned_cache_tokens': old_aligned, 'old_snapshot_positions': old_snapshots,
                            'common_prefix_tokens': after['common_prefix_tokens'], 'restore_position': after['restore_position']})
        report['crossed_aligned_boundary_bpe'] = crossed
        report['session_cases'] = session_cases
        report['phase'] = '128_append_timing'
        write(args.output, report)
        timings = []
        for runtime in runtimes:
            initial_text = 'x ' * 699 + 'Public material.'
            updates = [' a b c d e f g h' if i % 2 == 0 else ' public facts and neutral reference text.' for i in range(128)]
            warm = runtime.new_session(); warm.begin_message('user', initial_text)
            for update in updates:
                warm.append_text(update)
            measured = runtime.new_session(); measured.begin_message('user', initial_text)
            start_tokens = measured.logical_input_tokens
            counters_before = runtime.stats()
            rows, elapsed = [], []
            torch.cuda.synchronize(); began = time.perf_counter()
            for update in updates:
                started = time.perf_counter(); result = measured.append_text(update); torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - started)
                rows.append({name: result[name] for name in ('net_new_tokens', 'forward_tokens', 'real_forward_tokens', 'padding_tokens', 'replay_tokens', 'forward_calls')})
            seconds = time.perf_counter() - began
            totals = {name: sum(row[name] for row in rows) for name in rows[0]}
            delta = execution_delta(counters_before, runtime.stats())
            if (delta['forward_tokens'] != totals['forward_tokens'] or delta['forward_calls'] != totals['forward_calls']
                    or delta['real_forward_tokens'] != totals['real_forward_tokens']
                    or delta['padding_tokens'] != totals['padding_tokens']
                    or (runtime is graph and delta['graph_calls'] <= 0)):
                raise AssertionError('Timed text engine calls do not match per-append accounting/dispatch')
            if totals['net_new_tokens'] != measured.logical_input_tokens - start_tokens:
                raise AssertionError('ITPS net input denominator cannot be reconstructed')
            ordered = sorted(elapsed)
            timings.append({'inference_engine': runtime.engine_metadata.get('engine_type', runtime.engine_metadata.get('inference_engine')),
                            'measured_calls': 128, 'seconds': seconds, **totals,
                            'start_context_tokens': start_tokens, 'end_context_tokens': measured.logical_input_tokens,
                            'itps': totals['net_new_tokens'] / seconds, 'classifications_per_second': 128 / seconds,
                            'p50_ms': ordered[math.ceil(.5 * len(ordered)) - 1] * 1000,
                            'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1] * 1000,
                            'latencies_seconds': elapsed, 'per_append_counts': rows,
                            'engine_execution_delta': delta,
                            'scope': 'batch1 actual text; includes retokenization, replay, future-padding compute and CPU classification; startup excluded; zero generated tokens'})
        report['actual_text_timings'] = timings
        report['phase'] = 'bulk_diagnostic'
        with torch.inference_mode():
            output = model(torch.tensor([ids[:704]], device='cuda'), use_cache=False)
            bulk = {role: model.readout(output.last_hidden_state[:, -1], role)[0].float().softmax(-1)[0].cpu().tolist()
                    for role in ('user', 'assistant')}
        canonical = eager.classify_prefixes(ids[:704])['probabilities_by_role_all_tokens']
        report['old_bulk_diagnostic'] = {'probabilities': bulk, 'canonical_endpoint': {role: canonical[role][-1] for role in canonical},
                                         'max_probability_difference': probability_error(bulk, {role: canonical[role][-1] for role in canonical}),
                                         'used_for_acceptance': False}
        if runtime_source_receipt(local) != sources:
            raise RuntimeError('Canonical runtime source identities changed during the audit')
        report.update(status='completed', phase='finished', **{'pass': True},
                      graph_same_shape_probability_and_state_pass=True, future_pad_causality_pass=True,
                      session_arrival_invariance_pass=True, no_dummy_cache_commit_pass=True)
        write(args.output, report)
        return 0
    except Exception as error:
        report.update(status='failed', **{'pass': False}, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write(args.output, report)
        return 1
    finally:
        for runtime in runtimes:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--training-summary', type=Path)
    parser.add_argument('--base-root', type=Path, default=Path('/work'))
    parser.add_argument('--round4-code', type=Path, default=Path('/work/round4'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cpu-check-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error('timeout must be 1..3600')
    if args.cpu_check_only:
        if 'torch' in sys.modules:
            raise AssertionError('CPU parser check imported torch')
        assert probability_error({'x': [[.1, .9]]}, {'x': [[.1, .9]]}) == 0
        print(json.dumps({'cpu_checks': 'passed', 'neural_calls': 0}))
        return 0
    if args.checkpoint is None or args.output is None or args.checkpoint_sha256 is None or len(args.checkpoint_sha256) != 64:
        parser.error('--checkpoint --checkpoint-sha256 --output are required for L20 validation')
    if args.child:
        return run_child(args)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a canonical audit result')
    verify_fixed_input(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from process_runner import run_process_group
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    environment.pop('PYTHONPATH', None)
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--child']
    try:
        completed = run_process_group(command, args.output.with_suffix('.log'), args.timeout_seconds, env=environment)
        if completed.returncode != 0 or not args.output.is_file():
            return 1
        return 0 if json.loads(args.output.read_text()).get('pass') is True else 1
    except Exception as error:
        write(args.output, {'status': 'failed', 'pass': False, 'phase': 'supervisor',
                            'error_type': type(error).__name__, 'error': str(error), 'generated_tokens': 0})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
