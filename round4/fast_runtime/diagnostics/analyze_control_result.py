"""CPU-only analysis of the completed frozen prefill/stream control."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / 'prefill_stream_control.original.json'
    result = json.loads(path.read_text())
    entries, probability_failures, argmax_changes, threshold_changes = [], [], [], []
    for case in result['continuations']:
        for index, step in enumerate(case['steps']):
            row = {key: case[key] for key in ('prefix_tokens', 'sample_id', 'language', 'target_role')}
            row.update(step=index, native_prefix_tokens=step['native_prefix_tokens'],
                       probability_comparison=step['eager_history_vs_whole_history_probabilities'])
            entries.append(row)
            if row['probability_comparison']['pass'] is not True:
                probability_failures.append({**row,
                    'eager': step['eager_stream_history_probabilities'],
                    'whole_origin': step['whole_created_history_probabilities'],
                    'threshold_comparison': step['fixed_threshold_comparison']})
            for role in ('user', 'assistant'):
                a = step['eager_stream_history_probabilities'][role]
                b = step['whole_created_history_probabilities'][role]
                a_label, b_label = max(a, key=a.get), max(b, key=b.get)
                if a_label != b_label:
                    argmax_changes.append({**row, 'head': role, 'is_target_head': role == case['target_role'],
                        'eager_argmax': a_label, 'whole_origin_argmax': b_label,
                        'eager_probabilities': a, 'whole_origin_probabilities': b,
                        'fixed_threshold_comparison': step['fixed_threshold_comparison'][role]})
                if step['fixed_threshold_comparison'][role]['decision_changed'] is True:
                    threshold_changes.append({**row, 'head': role,
                                              'comparison': step['fixed_threshold_comparison'][role]})
    baselines = [row for row in result['baseline_controls'] if 'ordinary_eager_vs_whole_state' in row]
    endpoints = [{'sample_id': row['sample_id'], 'prefix_tokens': row['prefix_tokens'],
                  'target_role': row['target_role'],
                  'comparison': row['final_eager_vs_fresh_whole_probabilities']}
                 for row in result['continuations']]
    summary = {
        'status': 'cpu_analysis_complete', 'control_status': result['status'],
        'control_diagnostic_completed': result['diagnostic_completed'],
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'control_script_sha256': result['diagnostic_script_sha256'],
        'control_elapsed_seconds': result['elapsed_seconds'],
        'model_checkpoint_sha256': result['checkpoint_sha256'],
        'original_failed_audit_sha256': result['original_audit_sha256'],
        'original_failed_audit_pass': False, 'promotion_approved': False,
        'baseline_cases': len(baselines),
        'ordinary_eager_vs_whole_state_failures': sum(not row['ordinary_eager_vs_whole_state']['pass'] for row in baselines),
        'repeated_whole_bitwise_equal': {row['case']: row['bitwise_state_equal']
                                        for row in result['baseline_controls'] if 'bitwise_state_equal' in row},
        'continuation_cases': len(result['continuations']), 'continuation_steps': len(entries),
        'prefix_seed_lengths': sorted({row['prefix_tokens'] - 8 for row in entries}),
        'findings_as_recorded': result['findings'],
        'continuation_probability_failures': probability_failures,
        'continuation_argmax_changes': argmax_changes,
        'target_head_argmax_changes': sum(row['is_target_head'] for row in argmax_changes),
        'fixed_calibrated_threshold_changes': threshold_changes,
        'final_eager_vs_fresh_whole_probability_failures': [row for row in endpoints if not row['comparison']['pass']],
        'final_eager_vs_fresh_whole_max_probability_error': max(
            value for row in endpoints for value in row['comparison']['max_absolute_error_by_role'].values()),
        'no_new_model_calls': True, 'no_original_gate_or_threshold_changed': True,
    }
    target = ROOT / 'control_analysis.json'
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: summary[key] for key in (
        'baseline_cases', 'ordinary_eager_vs_whole_state_failures', 'continuation_cases', 'continuation_steps',
        'target_head_argmax_changes', 'final_eager_vs_fresh_whole_max_probability_error')}, indent=2))


if __name__ == '__main__':
    main()
