"""Validate graph text serving and repair the frozen primary baseline checks."""
import json
from pathlib import Path
import subprocess
import time

from process_runner import run_process_group

ROOT = Path('/work/round4/fast_runtime')
OUT = Path('/work/output/fast_runtime')
PYTHON = '/work/modern/bin/python'
LEGACY = '/work/legacy/bin/python'
PRIMARY_MANIFEST = Path('/work/validation/MODEL_MANIFEST.json')
WINDOW_SHA256 = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'
PRIMARY_SHA256 = '29fc3e5663eaa333235b1c2ba84d4c0a964f6ece4fd146c9fc6ec01761eaf222'


def validate_fixed_selections(selected, primary):
    if (selected['candidate'] != 'window' or selected['variant'] != 'window'
            or selected.get('inference_engine') != 'window_cuda_graph'
            or selected['checkpoint_sha256'] != WINDOW_SHA256):
        raise RuntimeError('Only the already fixed window checkpoint may enter this validation')
    if (primary.get('status') != 'research_candidate' or primary.get('candidate') != 'classification_rl'
            or primary.get('variant') != 'full' or primary.get('window') is not None
            or primary.get('checkpoint_sha256') != PRIMARY_SHA256
            or primary.get('checkpoint') != '/work/output/round4/classification_rl/best.safetensors'
            or primary.get('production_approval') is not False):
        raise RuntimeError('Primary baseline rechecks must use the fixed full/RL candidate')


def build_stage_commands(selected, primary):
    """Pure command construction: safe to inspect and validate on CPU."""
    validate_fixed_selections(selected, primary)
    manifest = ROOT / 'MODEL_MANIFEST.json'
    compatibility = ROOT / 'keyword_tokenizer_compatibility_v1.json'
    recheck = OUT / 'baseline_recheck'
    commands = [
        ('package', [PYTHON, str(ROOT / 'package_model.py'), '--manifest', str(manifest),
             '--output', str(OUT / 'bundle'), '--model-assets', '/work/models/qwen35',
             '--code-dir', str(ROOT), '--window-code-dir', '/work/window'], 300),
        ('fast_text', [PYTHON, str(ROOT / 'test_fast_text_l20.py'),
             '--bundle', str(OUT / 'bundle'), '--output', str(OUT / 'fast_text_audit.json'),
             '--timeout-seconds', '1740'], 1800),
        ('bundle_audit', [PYTHON, str(ROOT / 'test_bundle_l20.py'),
             '--bundle', str(OUT / 'bundle'), '--manifest', str(manifest),
             '--output', str(OUT / 'bundle_audit.json')], 1200),
        ('official_window', [PYTHON, str(ROOT / 'evaluate_official_adapted.py'), '--kind', 'student',
             '--data-dir', '/work/validation/official_adapted_v1',
             '--source-dir', '/work/validation/qwen3guardtest',
             '--candidate-manifest', str(manifest), '--output', str(OUT / 'official_window')], 2400),
        ('keyword_window', [PYTHON, str(ROOT / 'eval_keyword_probe.py'), '--kind', 'candidate',
             '--data-root', '/work/validation/keyword_probe_v1', '--variant', 'window',
             '--checkpoint', selected['checkpoint'],
             '--calibration-metrics', '/work/output/round4/window/exported_dev_metrics.json',
             '--tokenizer-compatibility', str(compatibility),
             '--output-prefix', str(OUT / 'keyword_window')], 600),
        ('benchmark_candidate_v2', [PYTHON, str(ROOT / 'benchmark_pair.py'), '--kind', 'candidate',
             '--candidate-manifest', str(PRIMARY_MANIFEST),
             '--output', str(recheck / 'speed_candidate_v2.json')], 600),
        ('benchmark_a0_v2', [LEGACY, str(ROOT / 'benchmark_pair.py'), '--kind', 'a0',
             '--candidate-manifest', str(PRIMARY_MANIFEST),
             '--output', str(recheck / 'speed_a0_v2.json')], 600),
        ('keyword_primary', [PYTHON, str(ROOT / 'eval_keyword_probe.py'), '--kind', 'candidate',
             '--data-root', '/work/validation/keyword_probe_v1', '--variant', 'classification_rl',
             '--rl-base-variant', primary['variant'], '--checkpoint', primary['checkpoint'],
             '--calibration-metrics', '/work/output/round4/classification_rl/exported_dev_metrics.json',
             '--tokenizer-compatibility', str(compatibility),
             '--output-prefix', str(recheck / 'keyword_primary')], 600),
    ]
    return commands


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'baseline_recheck').mkdir(exist_ok=True)
    selected = json.loads((ROOT / 'MODEL_MANIFEST.json').read_text())
    primary = json.loads(PRIMARY_MANIFEST.read_text())
    if not (ROOT / 'keyword_tokenizer_compatibility_v1.json').is_file():
        raise FileNotFoundError('Explicit real-modern tokenizer compatibility proof is required')
    commands = build_stage_commands(selected, primary)
    stages = []
    for name, command, timeout in commands:
        start = time.monotonic()
        log_directory = OUT / 'baseline_recheck' if name in ('benchmark_candidate_v2', 'benchmark_a0_v2', 'keyword_primary') else OUT
        log = log_directory / (name + '.log')
        row = {'stage': name, 'command': command, 'required': True,
               'timeout_seconds': timeout, 'log': str(log)}
        print(json.dumps({'stage': name, 'status': 'running'}), flush=True)
        try:
            result = run_process_group(command, log, timeout)
            row.update(returncode=result.returncode, process_completed=result.returncode == 0)
        except (subprocess.TimeoutExpired, OSError) as error:
            row.update(process_completed=False, error=str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                row.update(timed_out=True,
                           process_group_termination=getattr(error, 'process_group_termination', 'unconfirmed'))
        row['seconds'] = time.monotonic() - start
        stages.append(row)
        (OUT / 'stage_status.json').write_text(json.dumps({
            'status': 'completed' if len(stages) == len(commands) else 'running',
            'stages': stages, 'all_required_processes_completed': all(r['process_completed'] for r in stages),
            'candidate': selected['candidate'], 'variant': selected['variant'],
            'inference_engine': selected['inference_engine'], 'checkpoint_sha256': selected['checkpoint_sha256'],
            'primary_quality_candidate': primary['candidate'], 'primary_variant': primary['variant'],
            'primary_checkpoint_sha256': primary['checkpoint_sha256'],
            'primary_manifest': str(PRIMARY_MANIFEST), 'speed_recheck_protocol': 'matched-classifier-token-id-v2',
            'planned_stage_count': len(commands), 'total_stage_timeout_seconds': sum(row[2] for row in commands),
            'old_validation_artifacts_overwritten': False,
            'training_changed': False, 'primary_quality_candidate_changed': False,
            'note': 'Process completion does not prove audit correctness; inspect numerical and behavioral artifacts.',
        }, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
