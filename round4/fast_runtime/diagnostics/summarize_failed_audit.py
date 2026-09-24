"""CPU-only aggregation; preserve every original failed gate unchanged."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'fast_text_audit.original.json'
    report = json.loads(source.read_text())
    false_paths, pass_false_paths = [], []
    states, probabilities = defaultdict(list), defaultdict(list)

    def walk(value, path='$', case=None):
        if value is False:
            false_paths.append(path)
            if path.endswith('.pass'):
                pass_false_paths.append(path)
        if isinstance(value, dict):
            case = value.get('case', case)
            if 'families' in value and isinstance(value['families'], dict):
                name = re.sub(r'\[\d+\]', '[]', path.rsplit('.', 1)[-1])
                if name == 'state' and case and case.startswith('whole_8192_'):
                    name = 'whole_state_at_8192'
                elif '.snapshots[' in path:
                    name = 'snapshot_eager_vs_graph_state'
                for family, entry in value['families'].items():
                    states[(name, family, entry.get('dtype'))].append({'path': path, 'case': case, **entry})
            if 'max_absolute_error_by_role' in value:
                name = path.rsplit('.', 1)[-1]
                if name == 'probabilities' and case and case.startswith('whole_8192_'):
                    name = 'whole_probabilities_at_8192'
                elif '.snapshots[' in path:
                    name = 'snapshot_eager_vs_graph_probabilities'
                probabilities[name].append({'path': path, 'case': case, **value})
            for key, nested in value.items():
                walk(nested, f'{path}.{key}', case)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f'{path}[{index}]', case)

    walk(report)
    state_summary = []
    for (comparison, family, dtype), rows in sorted(states.items()):
        abs_rows = [row for row in rows if isinstance(row.get('max_absolute_error'), (int, float))]
        norm_rows = [row for row in rows if isinstance(row.get('max_normalized_error'), (int, float))]
        largest = max(abs_rows, key=lambda row: row['max_absolute_error']) if abs_rows else {}
        normalized = max(norm_rows, key=lambda row: row['max_normalized_error']) if norm_rows else {}
        state_summary.append({'comparison': comparison, 'family': family, 'dtype': dtype,
                              'comparisons': len(rows), 'failed': sum(row.get('pass') is not True for row in rows),
                              'nonfinite': sum(row.get('finite') is not True for row in rows),
                              'max_absolute_error': largest.get('max_absolute_error'), 'absolute_error_case': largest.get('case'),
                              'max_normalized_error': normalized.get('max_normalized_error'), 'normalized_error_case': normalized.get('case'),
                              'tolerances': rows[0].get('tolerance')})
    probability_summary = []
    for comparison, rows in sorted(probabilities.items()):
        result = {'comparison': comparison, 'comparisons': len(rows),
                  'failed': sum(row.get('pass') is not True for row in rows), 'by_role': {}}
        for role in ('user', 'assistant'):
            values = [(row['max_absolute_error_by_role'].get(role), row.get('case')) for row in rows]
            valid = [(value, case) for value, case in values if isinstance(value, (int, float))]
            largest = max(valid, default=(None, None), key=lambda row: row[0])
            result['by_role'][role] = {'max_absolute_error': largest[0], 'case': largest[1],
                                       'missing_or_invalid_count': len(values) - len(valid)}
        probability_summary.append(result)
    failed_checks = [row for row in report['checks'] if row['pass'] is not True]
    families = Counter('exact_length' if row['case'].startswith('exact_length_') else
                       'whole_8192' if row['case'].startswith('whole_8192_') else 'text_pair'
                       for row in failed_checks)
    failure_roots = []
    for row in failed_checks:
        roots = [key for key, item in row.items() if isinstance(item, dict) and item.get('pass') is False]
        failure_roots.append({'case': row['case'], 'failed_comparison_roots': roots})
    startup = {key: value for key, value in report['startup'].items() if key != 'per_length_arenas'}
    timing_keys = ('accepted_appends', 'initial_native_tokens', 'final_native_tokens', 'net_new_native_tokens',
                   'actual_forward_tokens', 'forward_calls', 'replay_tokens', 'rollback_events', 'seconds',
                   'itps_net_native_input', 'p50_ms', 'p95_ms', 'session_storage_before', 'session_storage_after')
    result = {'status': 'cpu_analysis_complete', 'original_status': report['status'], 'original_pass': report['pass'],
              'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'checks': len(report['checks']), 'passed_checks': len(report['checks']) - len(failed_checks),
              'failed_checks': len(failed_checks), 'failed_check_groups': dict(families),
              'failed_comparison_roots': failure_roots, 'all_false_json_paths': false_paths,
              'failed_pass_json_paths': pass_false_paths, 'state_comparisons': state_summary,
              'probability_comparisons': probability_summary,
              'text_performance': {engine: {key: values.get(key) for key in timing_keys}
                                   for engine, values in report['text_performance'].items()},
              'performance_comparison': report['performance_comparison'], 'startup': startup,
              'no_gate_changed': True, 'no_model_calls': True, 'promotion_approved': False}
    target = ROOT / 'failure_summary.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('checks', 'passed_checks', 'failed_checks', 'failed_check_groups',
                                                   'performance_comparison')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
