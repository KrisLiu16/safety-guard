"""Prepare a Round5 release identity only after completed training and real SHA verification."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from standalone_model import INITIAL_WINDOW_SHA256, validate_evidence, validate_selection, sha256_file


def prepare(summary_path, canonical_calibration, checkpoint=None):
    paths = {'summary': summary_path, 'sft_summary': summary_path.parent / 'sft/summary.json',
             'rl_summary': summary_path.parent / 'classification_rl/summary.json',
             'canonical_calibration': canonical_calibration}
    documents = {name: json.loads(path.read_text()) for name, path in paths.items()}
    final, rl = documents['summary'], documents['rl_summary']
    if final.get('status') != 'completed' or rl.get('status') != 'completed':
        raise ValueError('Training is incomplete; no provisional release identity may be invented')
    checkpoint = checkpoint or Path(final['final_checkpoint'])
    actual = sha256_file(checkpoint)
    if actual != final['final_checkpoint_sha256'] or actual != rl['checkpoint_sha256']:
        raise ValueError('Actual final weights differ from the completed training selection')
    metrics = rl['best_metrics']
    eligible = metrics['eligible']
    training_promoted = eligible and actual != INITIAL_WINDOW_SHA256
    canonical = documents['canonical_calibration']
    runtime_pass = canonical['canonical_quality_gate_pass']
    promoted = training_promoted and runtime_pass
    candidate = ('initial_window_fallback' if actual == INITIAL_WINDOW_SHA256
                 else 'classification_rl' if rl['selected']['stage'] == 'classification_rl' else 'sft')
    thresholds = canonical['thresholds']
    selection = {
        'format': 'round5-prefix-selection-v1', 'status': ('research_unvalidated_runtime' if not runtime_pass else
                                                         'research_candidate' if promoted else 'fallback_not_promoted'),
        'candidate': candidate, 'variant': 'window', 'window': 512, 'backbone_layers': 24,
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': actual,
        'inference_engine': 'eager', 'graph_validation_passed': False,
        'eligible_candidate_found': eligible, 'promoted_from_initial': promoted,
        'training_promoted_from_initial': training_promoted, 'canonical_quality_gate_pass': runtime_pass,
        'promotion_reason': ('Fixed selected weights passed independent canonical serving gates' if promoted else
                             'Research classifier with diagnostic thresholds; serving gates failed' if not runtime_pass else
                             'No eligible improvement over the initial window checkpoint; retained fallback, not a training promotion'),
        'execution_contract': canonical['execution_contract'],
        'thresholds': thresholds, 'threshold_comparison': '>',
        'calibrated_strata': {mode: sorted(values) for mode, values in thresholds.items()},
        'calibration_scope': {'whole': 'canonical32 complete original target endpoint',
                              'stream': 'canonical32 episode maxima over predeclared native target tokens and native-retokenized text cuts'},
        'arbitrary_bpe_schedule_fpr_guarantee': False, 'episode_stopping_policy': False,
        'selection_read_test': False, 'generated_tokens': 0, 'production_approval': False,
        'risk_class_order': ['safe', 'unsafe', 'controversial'],
        'hard_supervised_classes_this_round': ['safe', 'unsafe'], 'category_validated': False,
        'selection_evidence': {name: {'path': str(path.resolve()), 'sha256': sha256_file(path)} for name, path in paths.items()},
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    validate_selection(selection)
    validate_evidence(selection, documents)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True, help='Completed Round5 output summary.json')
    parser.add_argument('--checkpoint', type=Path, help='Relocated identical weights, still verified by SHA')
    parser.add_argument('--canonical-calibration', type=Path, required=True,
                        help='Post-selection canonical32 calibration+development evidence; old bulk thresholds are forbidden')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a fixed model manifest')
    selection = prepare(args.summary, args.canonical_calibration, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(args.output), 'status': selection['status'],
                      'checkpoint_sha256': selection['checkpoint_sha256'], 'model_calls': 0}))


if __name__ == '__main__':
    main()
