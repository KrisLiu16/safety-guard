"""Independent <=300s L20 diagnostic; never changes the frozen failed audit.

Compare ordinary eager streaming against one-shot-created state, then feed
identical real development-example continuations through each. Graph/eager
use exactly the same schedule. Original state/probability helpers and their
thresholds are imported unchanged. Completion is not release approval.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PREFIXES = (512, 4096, 7680)
LENGTHS = (1, 8, 32)
FIXED_SHA = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'
ORIGINAL_AUDIT_SHA = '11635ba5abc48a466363aa75ef98b146e65f8dae0cf9fad19d07b98eb833234c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decision_differences(left, right, language, selection):
    values = {}
    for role in ('user', 'assistant'):
        threshold = selection['thresholds'].get(language + '/' + role)
        values[role] = {
            'threshold': threshold,
            'left_unsafe': None if threshold is None else left[role]['unsafe'] >= threshold,
            'right_unsafe': None if threshold is None else right[role]['unsafe'] >= threshold,
            'decision_changed': None if threshold is None else
                (left[role]['unsafe'] >= threshold) != (right[role]['unsafe'] >= threshold),
        }
    return values


def detailed_state_differences(audit, expected, actual):
    torch = audit.torch
    left, right = audit.state.state_tensors(expected), audit.state.state_tensors(actual)
    records = []
    with torch.inference_mode():
        for key in sorted(left):
            a, b = left[key].float(), right[key].float()
            difference = (a - b).abs()
            flat = int(difference.reshape(-1).argmax())
            records.append({'layer': key[0], 'family': key[1], 'dtype': str(left[key].dtype),
                            'shape': list(a.shape), 'max_absolute_error': float(difference.max()),
                            'flat_index_at_max': flat, 'expected_at_max': float(a.reshape(-1)[flat]),
                            'actual_at_max': float(b.reshape(-1)[flat])})
    return records


def select_dev_examples(args, audit, tokenizer, helper):
    data_manifest = json.loads(args.dev.with_name('manifest.json').read_text())
    expected = data_manifest.get('output_hashes', {}).get('dev')
    if expected is None or sha(args.dev) != expected:
        raise RuntimeError('Development data differs from its frozen manifest')
    rows = [json.loads(line) for line in args.dev.read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        ids = row['ids']
        if 8 <= len(ids) <= 192 and row['source_label'] in ('safe', 'unsafe'):
            key = (row['language'], row['target_role'], row['source_label'])
            groups.setdefault(key, []).append(row)
    expected_groups = [(language, role, label) for language, role in
                       (('zh', 'user'), ('zh', 'assistant'), ('en', 'assistant'))
                       for label in ('safe', 'unsafe')]
    selected, scored = [], []
    for key in expected_groups:
        pool = sorted(groups.get(key, []), key=lambda row: hashlib.sha256(row['sample_id'].encode()).hexdigest())[:16]
        if not pool:
            raise RuntimeError('Missing eligible real development diagnostic group: ' + str(key))
        candidates = []
        for row in pool:
            if helper.encode(tokenizer, audit.text.serialize(row['messages'])) != row['ids']:
                raise RuntimeError('Frozen development IDs do not match the bundled tokenizer')
            cache, probabilities = audit.eager_forward(row['ids'])
            del cache
            unsafe = probabilities[row['target_role']]['unsafe']
            observed = {'sample_id': row['sample_id'], 'language': key[0], 'target_role': key[1],
                        'source_label': key[2], 'native_tokens': len(row['ids']),
                        'isolated_probabilities': probabilities, 'non_saturated_target': .05 <= unsafe <= .95}
            scored.append(observed)
            candidates.append((abs(unsafe - .5), row, observed))
        _, row, observed = min(candidates, key=lambda item: (item[0], item[1]['sample_id']))
        selected.append({**observed, 'ids': row['ids'], 'messages': row['messages']})
    return selected, {'dev_sha256': sha(args.dev), 'dev_manifest_sha256': sha(args.dev.with_name('manifest.json')),
                      'selection_scope': 'At most 16 hash-ordered existing dev rows per language/role/label; '
                                         'closest isolated unsafe probability to 0.5 only to stress numerical diagnostics; '
                                         'no weight, model, threshold or quality-metric selection',
                      'scored_candidates': scored, 'selected': selected,
                      'has_non_saturated_selected_target': any(row['non_saturated_target'] for row in selected)}


def run_worker(args):
    report = {'status': 'running', 'diagnostic_completed': False, 'promotion_approved': False,
              'original_audit_pass': False, 'original_gate_changes': False, 'new_thresholds_defined': False,
              'generated_tokens': 0, 'baseline_controls': [], 'continuations': []}
    write(args.output, report)
    runners = {}
    started = time.monotonic()
    try:
        original = json.loads(args.original_audit.read_text())
        if sha(args.original_audit) != ORIGINAL_AUDIT_SHA or original.get('pass') is not False:
            raise RuntimeError('Use the preserved original failed audit snapshot')
        if sha(args.helpers) != original['audit_script']['sha256']:
            raise RuntimeError('The original comparison specification must remain unchanged')
        helper = import_file('_frozen_fast_audit_helpers', args.helpers)
        loader = helper.load_module('standalone_model', args.bundle / 'standalone_model.py')
        torch, model, tokenizer, metadata = loader.load_bundle(args.bundle)
        if metadata['checkpoint_sha256'] != FIXED_SHA:
            raise RuntimeError('Only the unchanged W512 SFT checkpoint is permitted')
        text = helper.load_module('text_stream_runtime', args.bundle / 'text_stream_runtime.py')
        graph = helper.load_module('graph_stream', args.bundle / 'graph_stream.py')
        for name, module in (('standalone_model', loader), ('text_stream_runtime', text), ('graph_stream', graph)):
            if sha(module.__file__) != original['loaded_sources'][name]['sha256']:
                raise RuntimeError('Frozen runtime changed: ' + name)
        eager = text.TextStreamRuntime(model, tokenizer)
        audit = helper.Audit(torch, graph, text, eager, None, report, args.output)
        report.update(device=torch.cuda.get_device_name(0), checkpoint_sha256=FIXED_SHA,
                      original_audit_sha256=ORIGINAL_AUDIT_SHA, helpers_sha256=sha(args.helpers),
                      retained_original_tolerances=original['tolerances'], diagnostic_script_sha256=sha(__file__))
        selected, selection = select_dev_examples(args, audit, tokenizer, helper)
        report['development_continuation_selection'] = selection
        write(args.output, report)
        sequence = helper.encode(tokenizer, 'USER:\n' + ''.join(
            f'第{i}条公开资料，核对目录和日期。 Public reading record {i}: organize ordinary notes.\n'
            for i in range(700)))
        if len(sequence) < 8192:
            raise RuntimeError('Insufficient deterministic prefix source')
        seed512, _ = audit.eager_forward(sequence[:512])
        for length in LENGTHS:
            runners[length] = graph.WindowGraphRunner(model, seed512, sequence[512:512 + length], chunk_tokens=length)
        report['captured_control_lengths'] = list(runners)
        report['capture_cost_scope'] = ('Three initial diagnostic graphs; additional exact remainder lengths are '
                                       'captured once before use, never counted as continuation inputs')
        for prefix_length in PREFIXES:
            seed, _ = audit.eager_forward(sequence[:prefix_length])
            repeated, _ = audit.eager_forward(sequence[:prefix_length])
            report['baseline_controls'].append({'case': f'whole_repeat_{prefix_length}',
                'bitwise_state_equal': audit.fingerprint(seed) == audit.fingerprint(repeated),
                'state': audit.compare_states(seed, repeated)})
            del repeated
            continuation_start = None
            for length in LENGTHS:
                ids = sequence[prefix_length:prefix_length + length]
                prefix = sequence[:prefix_length + length]
                e_cache, e_probs = audit.eager_forward(ids, seed)
                g_cache, g_probs = runners[length].step(ids, seed)
                w_cache, w_probs = audit.eager_forward(prefix)
                control = {'case': f'prefix_{prefix_length}_append_{length}', 'prefix_tokens': len(prefix),
                           'ordinary_eager_vs_whole_state': audit.compare_states(w_cache, e_cache),
                           'ordinary_eager_vs_whole_probabilities': helper.compare_probabilities(w_probs, e_probs, .03),
                           'graph_vs_ordinary_eager_state': audit.compare_states(e_cache, g_cache),
                           'graph_vs_ordinary_eager_probabilities': helper.compare_probabilities(e_probs, g_probs, .001),
                           'ordinary_eager_vs_whole_by_layer': detailed_state_differences(audit, w_cache, e_cache)}
                report['baseline_controls'].append(control)
                if length == 8:
                    continuation_start = (prefix, audit.clone(e_cache), audit.clone(g_cache), audit.clone(w_cache))
                del e_cache, g_cache, w_cache
                write(args.output, report)
            # Start continuations from identical logical text but differently
            # formed states; W and E both use ordinary eager for every future
            # chunk, isolating the consequence of their inherited state delta.
            for sample in selected:
                prefix, e_cache, g_cache, w_cache = continuation_start
                e_cache, g_cache, w_cache = audit.clone(e_cache), audit.clone(g_cache), audit.clone(w_cache)
                future_prefix = list(prefix)
                entry = {'prefix_tokens': len(prefix), 'sample_id': sample['sample_id'],
                         'language': sample['language'], 'target_role': sample['target_role'],
                         'source_label_diagnostic_only': sample['source_label'], 'steps': []}
                for offset in range(0, len(sample['ids']), 32):
                    ids = sample['ids'][offset:offset + 32]
                    # Capture an exact final short remainder once if needed.
                    if len(ids) not in runners:
                        runners[len(ids)] = graph.WindowGraphRunner(model, seed512, sequence[512:512 + len(ids)],
                                                                   chunk_tokens=len(ids))
                    e_cache, e_probs = audit.eager_forward(ids, e_cache)
                    w_cache, w_probs = audit.eager_forward(ids, w_cache)
                    g_cache, g_probs = runners[len(ids)].step(ids, g_cache)
                    future_prefix.extend(ids)
                    entry['steps'].append({'native_prefix_tokens': len(future_prefix), 'new_tokens': len(ids),
                        'eager_stream_history_probabilities': e_probs, 'whole_created_history_probabilities': w_probs,
                        'graph_stream_probabilities': g_probs,
                        'eager_history_vs_whole_history_probabilities': helper.compare_probabilities(w_probs, e_probs, .03),
                        'graph_vs_eager_probabilities': helper.compare_probabilities(e_probs, g_probs, .001),
                        'graph_vs_eager_state': audit.compare_states(e_cache, g_cache),
                        'fixed_threshold_comparison': decision_differences(e_probs, w_probs, sample['language'], metadata['selection']),
                        'non_saturated_target_probability': .05 <= e_probs[sample['target_role']]['unsafe'] <= .95,
                        'non_saturated_probability_present': any(.05 <= values['unsafe'] <= .95
                                                                 for values in e_probs.values())})
                fresh, fresh_probs = audit.eager_forward(future_prefix)
                entry['final_fresh_whole_probabilities'] = fresh_probs
                entry['final_eager_vs_fresh_whole_probabilities'] = helper.compare_probabilities(fresh_probs, e_probs, .03)
                entry['final_eager_vs_fresh_whole_state'] = audit.compare_states(fresh, e_cache)
                entry['final_eager_history_vs_whole_history_state'] = audit.compare_states(w_cache, e_cache)
                report['continuations'].append(entry)
                del fresh, e_cache, g_cache, w_cache
                write(args.output, report)
            del continuation_start, seed
        steps = [step for case in report['continuations'] for step in case['steps']]
        controls = [case for case in report['baseline_controls'] if 'ordinary_eager_vs_whole_state' in case]
        graph_probabilities = [case['graph_vs_ordinary_eager_probabilities'] for case in controls] + [
            step['graph_vs_eager_probabilities'] for step in steps]
        graph_states = [case['graph_vs_ordinary_eager_state'] for case in controls] + [
            step['graph_vs_eager_state'] for step in steps]
        graph_probability_max = max(value for row in graph_probabilities
                                    for value in row['max_absolute_error_by_role'].values())
        graph_state_max = max(family['max_absolute_error'] for row in graph_states
                              for family in row['families'].values())
        report['findings'] = {
            'ordinary_eager_whole_state_failure_reproduced': any(not case['ordinary_eager_vs_whole_state']['pass'] for case in controls),
            'graph_eager_all_original_state_and_probability_gates_pass': all(
                case['graph_vs_ordinary_eager_state']['pass'] and case['graph_vs_ordinary_eager_probabilities']['pass'] for case in controls)
                and all(step['graph_vs_eager_state']['pass'] and step['graph_vs_eager_probabilities']['pass'] for step in steps),
            'graph_eager_max_probability_error': graph_probability_max,
            'graph_eager_max_state_absolute_error': graph_state_max,
            'graph_eager_observed_numerically_exact': graph_probability_max == 0 and graph_state_max == 0,
            'max_state_origin_continuation_probability_error': max(value for step in steps for value in
                step['eager_history_vs_whole_history_probabilities']['max_absolute_error_by_role'].values()),
            'continuation_state_origin_probability_gate_failures': sum(not step['eager_history_vs_whole_history_probabilities']['pass'] for step in steps),
            'continuation_fixed_threshold_changes': sum(row['decision_changed'] is True for step in steps for row in step['fixed_threshold_comparison'].values()),
            'non_saturated_continuation_steps': sum(step['non_saturated_probability_present'] for step in steps),
            'non_saturated_target_continuation_steps': sum(step['non_saturated_target_probability'] for step in steps),
            'non_saturation_coverage_missing': not any(step['non_saturated_target_probability'] for step in steps),
            'scope': 'Observed selected development continuations only; cannot prove every future suffix is insensitive to state differences',
        }
        report['all_captured_control_lengths'] = sorted(runners)
        report['total_capture_setup_seconds'] = sum(runner.setup_seconds for runner in runners.values())
        report.update(status='completed', diagnostic_completed=True, elapsed_seconds=time.monotonic() - started,
                      interpretation='A completed diagnostic does not change the original failed specification or authorize delivery')
        returncode = 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        returncode = 1
    finally:
        for runner in runners.values():
            try:
                runner.close()
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(str(error))
                returncode = 1
        report['elapsed_seconds'] = time.monotonic() - started
        write(args.output, report)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('/work/output/fast_runtime/bundle'))
    parser.add_argument('--original-audit', type=Path, default=Path('/work/output/fast_runtime/fast_text_audit.json'))
    parser.add_argument('--helpers', type=Path, default=Path('/work/round4/fast_runtime/test_fast_text_l20.py'))
    parser.add_argument('--dev', type=Path, default=Path('/work/round4/data/risk_v2/dev.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('/work/output/fast_runtime_diagnostic/prefill_stream_control.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cpu-check-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_check_only:
        ast.parse(Path(__file__).read_text())
        assert max(PREFIXES) + max(LENGTHS) + 192 <= 8192
        assert 'torch' not in sys.modules
        print(json.dumps({'cpu_syntax_checked': True, 'model_calls': 0, 'prefixes': PREFIXES,
                          'lengths': LENGTHS, 'maximum_selected_dev_rows': 6, 'hard_timeout_seconds': 300}))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return run_worker(args)
    if args.output.exists():
        raise FileExistsError('Do not overwrite an earlier diagnostic')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--bundle', str(args.bundle),
               '--original-audit', str(args.original_audit), '--helpers', str(args.helpers),
               '--dev', str(args.dev), '--output', str(args.output)]
    with args.output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            report = json.loads(args.output.read_text()) if args.output.exists() else {}
            report.update(status='timed_out', diagnostic_completed=False, hard_timeout_seconds=300,
                          original_audit_pass=False, promotion_approved=False)
            write(args.output, report)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
