"""Write a reproducible research inference manifest after collected validation.

No model calls and no test-score-based selection. This is not production approval.
"""
import argparse
import json
from pathlib import Path
from render_results import Artifacts, audit_gate, number


def select(results_root):
    files = Artifacts(results_root)
    final = files.read('final_summary.json')
    audit = files.read('long_stream_audit.json')
    command = files.read('post_training_audit_status.json', outer=True)
    variant = final.get('rl_base_variant')
    if final.get('status') != 'completed' or variant not in ('full', 'window', 'memory'):
        return {'status': 'pending', 'reason': 'Training and fixed architecture selection are incomplete',
                'production_approval': False}
    eligible = []
    failures = {}
    for name in (variant, 'classification_rl'):
        filename = 'summary.json' if name == 'classification_rl' else 'sft_summary.json'
        summary = files.read(f'{name}/{filename}')
        dev = files.read(f'{name}/exported_dev_metrics.json')
        passed, reasons = audit_gate(name, audit, command, summary)
        if passed and summary.get('status') == 'completed' and number(dev.get('selection_score')):
            eligible.append((dev['selection_score'], name, summary, dev))
        else:
            failures[name] = reasons or ['Missing completed training or exported development score']
    if not eligible:
        return {'status': 'no_validated_candidate', 'failures': failures, 'production_approval': False}
    # Stable ordering retains SFT on a tie. Sealed-test metrics are never read.
    score, name, summary, dev = max(eligible, key=lambda value: value[0])
    return {
        'status': 'research_candidate', 'candidate': name, 'variant': variant,
        'checkpoint': f'/work/output/round4/{name}/best.safetensors',
        'checkpoint_sha256': summary['checkpoint_sha256'],
        'pvc': 'safety-guard-base-compare-data', 'base_root': '/work',
        'window': None if variant == 'full' else 512,
        'exported_development_score': score,
        'selection_rule': 'Recorded RL base architecture; higher exported development score between '
                          'its SFT and RL checkpoints that pass the completed 8K audit; SFT on ties',
        'selection_read_sealed_test': False,
        'thresholds': {key: row['threshold_from_calibration'] for key, row in dev['strata'].items()},
        'calibrated_strata': sorted(dev['strata']),
        'risk_class_order': ['safe', 'unsafe', 'controversial'],
        'hard_supervised_classes_this_round': ['safe', 'unsafe'],
        'category_validated': False, 'generated_tokens': 0,
        'streaming_scope': 'Exact token-ID incremental input validated by the 8K audit; '
                           'arbitrary text rollback is implemented and awaits the separate final L20 audit',
        'long_stream_audit': str(files.artifact / 'long_stream_audit.json'),
        'production_approval': False,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, default=root / 'results')
    parser.add_argument('--output', type=Path, default=root / 'MODEL_MANIFEST.json')
    args = parser.parse_args()
    result = select(args.results_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(args.output), 'status': result['status'],
                      'candidate': result.get('candidate'), 'model_calls': 0}))


if __name__ == '__main__':
    main()
