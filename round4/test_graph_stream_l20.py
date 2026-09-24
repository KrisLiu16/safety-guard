"""Isolated, at-most-15-minute L20 experiment; never promotes an entry point.

Use --cpu-check-only for syntax, parameter and lifecycle checks without torch.
Normal execution supervises a separate worker and writes an explicit failure
report if capture is unsupported, validation fails, or the worker times out.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parent), '/work/validation', '/work/input', '/work/window', '/work/round4']
from graph_stream import (CHUNK, LABELS, ROLES, WindowGraphRunner, _Lifecycle,
                          cpu_probabilities, require_l20, state_metadata, state_tensors,
                          storage_pointers, validate_ids, validate_parameters)

PROBABILITY_ATOL = .001
STATE_TOLERANCES = {'torch.bfloat16': {'atol': .02, 'rtol': .01},
                    'torch.float32': {'atol': .002, 'rtol': .001}}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cpu_checks():
    for path in (Path(__file__), Path(__file__).with_name('graph_stream.py')):
        ast.parse(path.read_text())
    validate_parameters()
    rejected = 0
    for kwargs in ({'window': 1024}, {'batch_size': 2}, {'chunk_tokens': 1}, {'chunk_tokens': True}):
        try:
            validate_parameters(**kwargs)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Unsupported parameter was accepted')
    validate_ids(list(range(8)), 100)
    for ids in ([1] * 7, [1] * 9, [-1] * 8, [True] * 8, [100] * 8, ['1'] * 8):
        try:
            validate_ids(ids, 100)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid input was accepted')
    life = _Lifecycle()
    for phase in ('warming', 'capturing', 'ready'):
        life.move(phase)
    life.require_ready()
    life.move('failed')
    try:
        life.require_ready()
    except RuntimeError:
        rejected += 1
    else:
        raise AssertionError('Failed runner accepted a replay')
    life.move('closed')
    try:
        life.move('warming')
    except RuntimeError:
        rejected += 1
    else:
        raise AssertionError('Closed runner could be reused')
    # The invalid-parameter guard runs before hardware checks or model access.
    try:
        WindowGraphRunner(None, None, None, chunk_tokens=32)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('Unsupported runner construction succeeded')
    probabilities = {role: {'safe': .8, 'unsafe': .1, 'controversial': .1} for role in ROLES}
    assert compare_probabilities(probabilities, probabilities)['pass']
    for invalid in (float('nan'), float('inf'), -float('inf')):
        broken = copy.deepcopy(probabilities)
        broken['user']['unsafe'] = invalid
        result = compare_probabilities(probabilities, broken)
        assert result['pass'] is False and result['max_absolute_error_by_role']['user'] is None
        json.dumps(result, allow_nan=False)
        rejected += 1
    assert 'torch' not in sys.modules, 'CPU-only checks unexpectedly imported torch'
    return {'pass': True, 'invalid_paths_rejected': rejected, 'syntax_files': 2,
            'model_calls': 0, 'torch_imported': False,
            'scope': 'CPU parameter/lifecycle guards only; no CUDA capture claim'}


def cache_fingerprint(cache):
    digest = hashlib.sha256(json.dumps(state_metadata(cache), sort_keys=True).encode())
    for key, tensor in sorted(state_tensors(cache).items()):
        digest.update(str((key, tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(__import__('torch').uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def compare_states(expected, actual, torch):
    left, right = state_tensors(expected), state_tensors(actual)
    topology = set(left) == set(right)
    metadata = state_metadata(expected) == state_metadata(actual)
    families = {}
    passed = topology and metadata
    if topology:
        for key in sorted(left):
            a, b = left[key], right[key]
            if a.shape != b.shape or a.dtype != b.dtype or str(a.dtype) not in STATE_TOLERANCES:
                return {'pass': False, 'metadata_equal': metadata, 'error': f'Unsupported state layout/dtype: {key}'}
            tolerance = STATE_TOLERANCES[str(a.dtype)]
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            if finite:
                delta = (a.float() - b.float()).abs()
                maximum = float(delta.max())
                normalized = float((delta / (tolerance['atol'] + tolerance['rtol'] * a.float().abs())).max())
            else:
                maximum = normalized = None
            valid = finite and normalized <= 1.
            passed = passed and valid
            family = key[1]
            aggregate = families.setdefault(family, {'tensors': 0, 'dtype': str(a.dtype),
                                                      'max_absolute_error': 0., 'max_normalized_error': 0.,
                                                      'finite': True, 'pass': True, 'tolerance': tolerance})
            aggregate['tensors'] += 1
            aggregate['finite'] = aggregate['finite'] and finite
            aggregate['pass'] = aggregate['pass'] and valid
            if finite:
                aggregate['max_absolute_error'] = max(aggregate['max_absolute_error'], maximum)
                aggregate['max_normalized_error'] = max(aggregate['max_normalized_error'], normalized)
    return {'pass': bool(passed), 'topology_equal': topology, 'metadata_equal': metadata, 'families': families}


def compare_probabilities(expected, actual):
    errors = {}
    for role in ROLES:
        pairs = [(expected[role][label], actual[role][label]) for label in LABELS]
        if not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
            errors[role] = None
            continue
        differences = [abs(a - b) for a, b in pairs]
        errors[role] = max(differences) if all(math.isfinite(value) for value in differences) else None
    return {'pass': all(v is not None and v <= PROBABILITY_ATOL for v in errors.values()),
            'max_absolute_error_by_role': errors, 'atol': PROBABILITY_ATOL}


def run_worker(args):
    report = {'status': 'running', 'pass': False, 'variant': 'window', 'window': 512, 'batch_size': 1,
              'chunk_tokens': CHUNK, 'checkpoint': str(args.checkpoint), 'generated_tokens': 0,
              'production_entry_replaced': False, 'training_or_weights_modified': False,
              'fixed_tolerances': {'probability_atol': PROBABILITY_ATOL, 'state': STATE_TOLERANCES},
              'checks': [], 'phase': 'hardware_check'}
    write_report(args.output, report)
    started = time.monotonic()
    graph = None
    torch = None
    try:
        torch = require_l20()
        torch.set_num_threads(4)
        from infer_classifier import load_classifier, sha256_file
        report.update(device=torch.cuda.get_device_name(0), checkpoint_sha256=sha256_file(args.checkpoint), phase='load_window')
        write_report(args.output, report)
        setup = SimpleNamespace(base_root=Path('/work'), base_code_dir=Path('/work/input'),
                                window_code_dir=Path('/work/window'), memory_code_dir=Path('/work/round4'),
                                checkpoint=args.checkpoint)
        _, model, tok, _, _, _ = load_classifier(setup, 'window')

        def token_sequence(prefix):
            text = 'USER:\n' + ''.join(f'{prefix} 第{i}条公开阅读资料，核对章节与日期。 Record {i}: review public material.\n' for i in range(600))
            ids = tok.encode(text, add_special_tokens=False, truncation=False)
            if len(ids) < 4096:
                raise RuntimeError('Insufficient distinct deterministic test tokens')
            return ids[:4096]

        ids_a, ids_b = token_sequence('甲'), token_sequence('乙')
        report['input_token_hashes'] = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in (ids_a, ids_b)]

        def prefill(ids):
            with torch.inference_mode():
                output = model(torch.tensor([ids], device='cuda'), use_cache=True)
                state = output.past_key_values
                torch.cuda.synchronize()
                return state

        def eager(ids, cache):
            with torch.inference_mode():
                work = copy.deepcopy(cache)
                output = model(torch.tensor([ids], device='cuda'), past_key_values=work, use_cache=True)
                hidden = output.last_hidden_state[:, -1]
                probabilities = torch.stack([model.readout(hidden, role)[0].float().softmax(-1)[0] for role in ROLES])
                return output.past_key_values, cpu_probabilities(probabilities)

        def independent(cache):
            with torch.inference_mode():
                return copy.deepcopy(cache)

        def check(name, expected, actual, expected_p=None, actual_p=None):
            with torch.inference_mode():
                state = compare_states(expected, actual, torch)
            probabilities = compare_probabilities(expected_p, actual_p) if expected_p is not None else None
            valid = state['pass'] and (probabilities is None or probabilities['pass'])
            record = {'case': name, 'pass': valid, 'state': state, 'probabilities': probabilities}
            report['checks'].append(record)
            write_report(args.output, report)
            if not valid:
                raise AssertionError(f'Graph correctness check failed: {name}')

        seed_a, seed_b = prefill(ids_a[:1024]), prefill(ids_b[:1536])
        seed_fingerprints = [cache_fingerprint(seed) for seed in (seed_a, seed_b)]
        report['phase'] = 'warm_and_capture'
        write_report(args.output, report)
        graph = WindowGraphRunner(model, seed_a, ids_a[1024:1032])
        report['capture_setup_seconds'] = graph.setup_seconds
        restored_seed = graph.snapshot()
        exact_seed_restore = cache_fingerprint(restored_seed) == seed_fingerprints[0]
        report['warm_capture_seed_restore_bitwise_equal'] = exact_seed_restore
        if not exact_seed_restore:
            raise AssertionError('Warmup/capture failed to restore the exact seed state')
        check('complete_seed_restore_after_warmup_and_capture', seed_a, restored_seed)
        assert seed_fingerprints == [cache_fingerprint(seed) for seed in (seed_a, seed_b)]
        report['phase'] = 'correctness'
        ea, ga = independent(seed_a), independent(seed_a)
        for index in range(8):
            block = ids_a[1024 + index * CHUNK:1024 + (index + 1) * CHUNK]
            ea, ep = eager(block, ea)
            ga, gp = graph.step(block, ga)
            check(f'continuous_{index}', ea, ga, ep, gp)
            if index == 0:
                first_graph_cache, first_graph_probs = independent(ga), copy.deepcopy(gp)
            if index == 3:
                rollback_e, rollback_g = independent(ea), independent(ga)
        rollback_fingerprint = cache_fingerprint(rollback_g)
        branch = ids_b[2000:2008]
        er, ep = eager(branch, rollback_e)
        gr, gp = graph.step(branch, rollback_g)
        check('restore_snapshot_then_different_suffix', er, gr, ep, gp)
        assert cache_fingerprint(rollback_g) == rollback_fingerprint
        # Reusing the same original seed after many replays must reproduce the
        # first replay, proving capture/warm state has not become hidden input.
        again, again_p = graph.step(ids_a[1024:1032], seed_a)
        check('reuse_original_seed_after_many_replays', first_graph_cache, again, first_graph_probs, again_p)

        states_e = [independent(seed_a), independent(seed_b)]
        states_g = [independent(seed_a), independent(seed_b)]
        for turn in range(8):
            index = turn % 2
            other = 1 - index
            untouched = cache_fingerprint(states_g[other])
            tokens, base = (ids_a, 1024) if index == 0 else (ids_b, 1536)
            start = base + (turn // 2) * CHUNK
            states_e[index], ep = eager(tokens[start:start + CHUNK], states_e[index])
            states_g[index], gp = graph.step(tokens[start:start + CHUNK], states_g[index])
            check(f'interleaved_session_{index}_turn_{turn}', states_e[index], states_g[index], ep, gp)
            assert untouched == cache_fingerprint(states_g[other]), 'Other session state changed'
            assert not storage_pointers(states_g[0]) & storage_pointers(states_g[1]), 'Sessions alias each other'
            assert not storage_pointers(states_g[index]) & storage_pointers(graph._arena), 'Session aliases arena'
        assert seed_fingerprints == [cache_fingerprint(seed) for seed in (seed_a, seed_b)]

        report['phase'] = 'warm_full_timing_trajectory'
        write_report(args.output, report)
        blocks = [ids_a[1024 + i * CHUNK:1024 + (i + 1) * CHUNK] for i in range(args.replays)]
        assert all(len(block) == CHUNK for block in blocks)
        for fn in (eager, graph.step):
            state = independent(seed_a)
            for block in blocks:
                state, _ = fn(block, state)
            del state
        torch.cuda.synchronize()

        def benchmark(fn):
            state = independent(seed_a)
            times = []
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for block in blocks:
                tick = time.perf_counter()
                state, probabilities = fn(block, state)
                times.append(time.perf_counter() - tick)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            ordered = sorted(times)
            mid = len(ordered) // 2
            median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            return {'seconds': elapsed, 'new_input_tokens': len(blocks) * CHUNK,
                    'classifications': len(blocks), 'itps': len(blocks) * CHUNK / elapsed,
                    'classifications_per_second': len(blocks) / elapsed,
                    'p50_ms': median * 1000, 'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1] * 1000,
                    'latencies_ms': [value * 1000 for value in times]}, state, probabilities

        report['phase'] = 'timing'
        eager_metrics, eager_final, ep = benchmark(eager)
        graph_metrics, graph_final, gp = benchmark(graph.step)
        check('same_timing_trajectory_final_state', eager_final, graph_final, ep, gp)
        report['performance'] = {
            'initial_context_tokens': 1024, 'chunk_tokens': CHUNK, 'replays': args.replays,
            'eager': eager_metrics, 'graph': graph_metrics,
            'itps_ratio_graph_over_eager': graph_metrics['itps'] / eager_metrics['itps'],
            'p95_ratio_graph_over_eager': graph_metrics['p95_ms'] / eager_metrics['p95_ms'],
            'scope': 'Both APIs return independent external cache and CPU-visible probabilities for both role heads. '
                     'Eager includes input-cache deepcopy; graph includes locked arena copy-in, replay, copy-out and alias check. '
                     'All per-call state allocation/copy, input transfer, position update and CPU output costs are timed. '
                     'No tokenization, BPE rollback selection, HTTP, cross-request batching or input waiting. '
                     'One matching complete trajectory warmed for both paths before measurement.'}
        report.update(status='completed', phase='complete', **{'pass': True},
                      external_seed_unchanged=True, cross_session_isolation_pass=True,
                      rollback_replay_pass=True, graph_arena_return_alias=False,
                      limitation='A fixed-shape window-only research experiment; no full/memory/text-runtime deployment claim.')
    except Exception as error:
        report.update(status='failed', **{'pass': False}, error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        if graph is not None:
            try:
                graph.close()
            except Exception as error:
                report.update(status='failed', **{'pass': False}, cleanup_error=str(error))
        report['elapsed_seconds'] = time.monotonic() - started
        write_report(args.output, report)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/work/output/round4/window/best.safetensors'))
    parser.add_argument('--output', type=Path, default=Path('/work/output/round4_validation/graph_stream_audit.json'))
    parser.add_argument('--replays', type=int, default=24)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--cpu-check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 2 <= args.replays <= 128 or not 1 <= args.timeout_seconds <= 900:
        parser.error('Require 2..128 replays and a wall limit of 1..900 seconds')
    if args.cpu_check_only:
        print(json.dumps(cpu_checks()))
        return
    if args.worker:
        result = run_worker(args)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--checkpoint', str(args.checkpoint),
                   '--output', str(args.output), '--replays', str(args.replays)]
        log_path = args.output.with_suffix('.log')
        try:
            with log_path.open('wb') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
            result = json.loads(args.output.read_text()) if args.output.exists() else {}
            if process.returncode or result.get('status') not in ('completed', 'failed'):
                result.update(status='failed', **{'pass': False}, worker_returncode=process.returncode)
                result.setdefault('error', 'Worker ended without a completed experiment report')
        except subprocess.TimeoutExpired:
            result = json.loads(args.output.read_text()) if args.output.exists() else {}
            result.update(status='failed', **{'pass': False}, error_type='TimeoutExpired',
                          error=f'Independent graph worker exceeded {args.timeout_seconds} seconds and was killed')
        result.update(worker_log=str(log_path), enforced_wall_limit_seconds=args.timeout_seconds,
                      production_entry_replaced=False, training_or_weights_modified=False)
        write_report(args.output, result)
    # Exit zero means the isolated experiment produced a report, not that a
    # graph was correct or faster. Consumers must inspect report.pass.
    print(json.dumps({'graph_stream_report': str(args.output), 'status': result['status'], 'pass': result['pass']}))


if __name__ == '__main__':
    main()
