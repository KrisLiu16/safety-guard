"""Serial, bounded post-training validation in a dedicated single-L20 job.

This coordinator cannot train, select weights, change thresholds on test data,
or claim quality from a zero exit code. Each stage must produce its declared
artifact. A timeout ends the coordinator; the collector then releases the pod.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = Path('/work/output/round5')
PYTHON = '/work/modern/bin/python'
BUDGET_SECONDS = 4200
EVALUATOR_SHA = '4ee145dfaae18d3408cde321d090b156176482551efaf1254a45b3a691d06ded'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources():
    manifest_path = ROOT / 'validation_input_v1.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'round5-validation-input-v1':
        raise ValueError('Wrong frozen validation input format')
    for name, digest in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe relative package path')
        if sha(ROOT / name) != digest:
            raise ValueError('Validation source changed: ' + name)
    for name, digest in manifest['external_files'].items():
        if sha(name) != digest:
            raise ValueError('Frozen external dependency changed: ' + name)
    if sha(ROOT / 'evaluate_final_l20.py') != EVALUATOR_SHA:
        raise ValueError('Training-completion evaluator changed')
    return {'path': str(manifest_path), 'sha256': sha(manifest_path),
            'packaged_files': len(manifest['files']), 'external_files': len(manifest['external_files'])}


def command(script, *arguments):
    return [PYTHON, '-B', str(ROOT / script), *map(str, arguments)]


def plan(checkpoint, digest):
    summary = OUT / 'prefix_v2/summary.json'
    validation = OUT / 'validation'
    manifest = validation / 'MODEL_MANIFEST.json'
    bundle = validation / 'bundle'
    calibration = OUT / 'canonical32_calibration'
    # Timeout allowances are maxima, not reservations. Keep child supervisors
    # shorter than the outer stage so they can terminate their process groups.
    return [
        {'name': 'canonical_core', 'dependencies': [], 'timeout': 720,
         'command': command('runtime/test_canonical_l20.py', '--checkpoint', checkpoint,
            '--checkpoint-sha256', digest, '--training-summary', summary,
            '--output', validation / 'canonical_core_audit.json', '--timeout-seconds', 540),
         'artifact': validation / 'canonical_core_audit.json', 'kind': 'pass'},
        {'name': 'canonical_calibration', 'dependencies': ['canonical_core'], 'timeout': 2000,
         'command': command('calibrate_canonical_l20.py', '--calibrate', '--inference-engine', 'eager',
            '--output', calibration), 'artifact': calibration / 'audit.json', 'kind': 'integrity'},
        {'name': 'prepare_manifest', 'dependencies': ['canonical_calibration'], 'timeout': 90,
         'command': command('runtime/prepare_manifest.py', '--summary', summary,
            '--canonical-calibration', calibration / 'selected/calibration.json', '--output', manifest),
         'artifact': manifest, 'kind': 'selection'},
        {'name': 'package', 'dependencies': ['prepare_manifest'], 'timeout': 180,
         'command': command('runtime/package_model.py', '--manifest', manifest, '--output', bundle),
         'artifact': bundle / 'BUNDLE_MANIFEST.json', 'kind': 'bundle'},
        {'name': 'portable_bundle', 'dependencies': ['package'], 'timeout': 540,
         'command': command('runtime/test_bundle_l20.py', '--manifest', manifest, '--bundle', bundle,
            '--output', validation / 'bundle_audit.json', '--timeout-seconds', 360),
         'artifact': validation / 'bundle_audit.json', 'kind': 'pass'},
        {'name': 'canonical_fresh390', 'dependencies': ['canonical_calibration'], 'timeout': 800,
         'command': command('evaluate_canonical_fresh_l20.py', '--evaluate-fresh',
            '--calibration-output', calibration, '--output', OUT / 'canonical32_fresh390'),
         'artifact': OUT / 'canonical32_fresh390/audit.json', 'kind': 'integrity'},
        {'name': 'canonical_official2441', 'dependencies': ['canonical_calibration'], 'timeout': 1200,
         'command': command('evaluate_canonical_official_l20.py', '--calibration-output', calibration,
            '--output', OUT / 'canonical32_official'),
         'artifact': OUT / 'canonical32_official/metrics.json', 'kind': 'official'},
        {'name': 'bulk_reference_fresh390', 'dependencies': [], 'timeout': 600,
         'command': command('evaluate_final_l20.py', '--output', OUT / 'fresh390_final'),
         'artifact': OUT / 'fresh390_final/audit.json', 'kind': 'integrity'},
    ]


def verify_artifact(stage, checkpoint_sha):
    path, kind = stage['artifact'], stage['kind']
    value = json.loads(path.read_text())
    if kind in ('pass', 'integrity', 'official'):
        marker = 'pass' if kind == 'pass' else 'integrity_pass'
        if value.get('status') != 'completed' or value.get(marker) is not True:
            raise ValueError('Stage artifact did not report completed ' + marker)
        if kind == 'official' and value.get('evaluated_source_rows') != 2441:
            raise ValueError('Official evaluation lost source coverage')
    elif kind == 'selection':
        if (value.get('format') != 'round5-prefix-selection-v1'
                or value.get('status') not in ('research_candidate', 'research_unvalidated_runtime', 'fallback_not_promoted')
                or value.get('checkpoint_sha256') != checkpoint_sha
                or value.get('selection_read_test') is not False):
            raise ValueError('Unexpected fixed selection artifact')
    elif kind == 'bundle':
        if (value.get('format') != 'round5-prefix-classifier-bundle-v1'
                or value.get('checkpoint_sha256') != checkpoint_sha):
            raise ValueError('Packaged candidate identity differs')
    else:
        raise ValueError('Unknown stage artifact kind')
    bound = value.get('checkpoint_sha256', value.get('selected_checkpoint_sha256'))
    if bound is not None and bound != checkpoint_sha:
        raise ValueError('Stage audited a different checkpoint')
    return {'path': str(path), 'sha256': sha(path),
            'canonical_quality_gate_pass': value.get('canonical_quality_gate_pass'),
            'quality_approval_implied': False}


def execute_stages(stages, report, save, runner, deadline, checkpoint_sha):
    outcomes = {}
    aborted = False
    for stage in stages:
        row = {key: stage[key] for key in ('name', 'dependencies', 'command')}
        row['artifact_path'] = str(stage['artifact'])
        remaining = int(deadline - time.monotonic())
        unmet = [name for name in stage['dependencies'] if not outcomes.get(name, False)]
        if aborted or remaining < 30 or unmet:
            row.update(status='skipped', completed=False,
                       reason='previous timeout' if aborted else 'total time budget' if remaining < 30 else 'failed dependency',
                       unmet_dependencies=unmet)
        else:
            timeout = min(stage['timeout'], remaining - 5)
            # Nested supervisors have their own sessions. Never start another
            # GPU stage after an outer timeout; pod release is final cleanup.
            row.update(status='running', timeout_seconds=timeout)
            report['active_stage'] = stage['name']
            save()
            print(json.dumps({'stage': stage['name'], 'status': 'running', 'timeout_seconds': timeout}), flush=True)
            began = time.monotonic()
            try:
                result = runner(stage['command'], OUT / 'validation' / (stage['name'] + '.log'), timeout)
                row['returncode'] = result.returncode
                if result.returncode != 0:
                    raise RuntimeError('Stage process returned ' + str(result.returncode))
                row['artifact_receipt'] = verify_artifact(stage, checkpoint_sha)
                row.update(status='completed', completed=True)
            except subprocess.TimeoutExpired as error:
                aborted = True
                row.update(status='timed_out', completed=False, error=str(error),
                           process_group_termination=getattr(error, 'process_group_termination', 'unknown'),
                           pod_release_required_before_any_further_gpu_work=True)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                row.update(status='failed', completed=False, error=str(error), error_type=type(error).__name__)
            row['seconds'] = time.monotonic() - began
        outcomes[stage['name']] = row['completed']
        report['stages'].append(row)
        save()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report['active_stage'] = None
    return all(outcomes.values())


def main():
    status_path = OUT / 'validation/workflow_status.json'
    if status_path.exists():
        raise FileExistsError('Refusing to overwrite a post-training validation attempt')
    began = time.monotonic()
    report = {'status': 'running', 'pipeline_integrity_pass': False, 'stages': [],
              'started_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
              'budget_seconds': BUDGET_SECONDS, 'generated_tokens': 0,
              'selection_changed': False, 'quality_approval_implied': False}
    save = lambda: write_json(status_path, report)
    save()
    try:
        report['input_receipt'] = verify_sources()
        evaluator = load_module('round5_completion_only', ROOT / 'evaluate_final_l20.py')
        summary, checkpoint, _ = evaluator.completed_training(OUT / 'prefix_v2')
        report.update(checkpoint_sha256=summary['final_checkpoint_sha256'],
                      training_summary_sha256=sha(OUT / 'prefix_v2/summary.json'))
        supervisor = load_module('round5_validation_supervisor', ROOT / 'runtime/process_runner.py')
        stages = plan(checkpoint, summary['final_checkpoint_sha256'])
        passed = execute_stages(stages, report, save, supervisor.run_process_group,
                                began + BUDGET_SECONDS, summary['final_checkpoint_sha256'])
        if verify_sources() != report['input_receipt']:
            raise ValueError('Frozen validation sources changed during execution')
        report.update(status='completed' if passed else 'incomplete', pipeline_integrity_pass=passed)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        save()
    return 0 if report['pipeline_integrity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
