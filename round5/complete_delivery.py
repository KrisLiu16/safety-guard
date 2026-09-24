"""Collect final reports, release the L20, then retrieve a verified CPU-only bundle."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from process_runner import run_process_group
POD = 'safety-guard-prefix-round5-20260924-r1-t9m2k'
JOB = 'safety-guard-prefix-round5-20260924-r1'
VALIDATION_JOB = 'safety-guard-prefix-validation-20260924-r1'
ARTIFACT_JOB = 'safety-guard-prefix-artifacts-20260924-r1'
K = ['kubectl', '--kubeconfig', str(Path.home() / '.kube/cls-og1rjus2-private-latest'), '-n', 'default']


def start_job(yaml, name):
    subprocess.run(K + ['apply', '-f', str(yaml)], check=True, timeout=30)
    # A Job can exist before its pod. Wait for creation before Ready.
    for _ in range(30):
        pods = json.loads(subprocess.check_output(K + ['get', 'pods', '-l', 'job-name=' + name,
                                                     '-o', 'json'], timeout=30))['items']
        if pods:
            break
        time.sleep(1)
    if len(pods) != 1:
        raise RuntimeError('Expected exactly one pod for ' + name)
    pod = pods[0]['metadata']['name']
    subprocess.run(K + ['wait', '--for=condition=Ready', 'pod/' + pod, '--timeout=180s'], check=True, timeout=190)
    current = json.loads(subprocess.check_output(K + ['get', 'pod', pod, '-o', 'json'], timeout=30))
    if current['status']['phase'] != 'Running':
        raise RuntimeError('Expected a running pod for ' + name)
    return pod


def collector_command(pod, job, output, ready, ack):
    return [sys.executable, '-u', '-B', str(ROOT / 'collect_results.py'),
            '--pod', pod, '--job', job, '--output-root', str(output),
            '--ready-marker', ready, '--ack-marker', ack, '--artifact-relative', 'output/round5',
            '--checkpoint-path', '/work/output/round5/prefix_v2/classification_rl/best.safetensors',
            '--checkpoint-path', '/work/output/round5/validation/bundle/classifier.safetensors']


def main():
    status_path = ROOT / 'delivery_status.json'
    if status_path.exists():
        raise FileExistsError('A delivery attempt already exists; inspect it before restarting')
    report = {'status': 'waiting_for_training', 'started_at': datetime.now(timezone.utc).isoformat(),
              'training_pod': POD, 'training_job': JOB, 'local_neural_calls': 0,
              'quality_approval_implied': False}

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(status_path)

    save()
    try:
        collector = collector_command(POD, JOB, ROOT / 'training_results',
                                      'round5_training_ready', 'round5_training_collected')
        with (ROOT / 'training_collector.log').open('xb') as stream:
            result = subprocess.run(collector, stdout=stream, stderr=subprocess.STDOUT, timeout=30000)
        report['training_collector_returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Training collector failed; inspect training_collector.log and preserved PVC')
        gpu_job = json.loads(subprocess.check_output(K + ['get', 'job', JOB, '-o', 'json'], timeout=30))
        report['gpu_job_completed'] = gpu_job.get('status', {}).get('succeeded', 0) == 1
        if not report['gpu_job_completed']:
            raise RuntimeError('Reports returned but GPU job completion is unconfirmed')
        training = json.loads((ROOT / 'training_results/output/round5/prefix_v2/summary.json').read_text())
        if training.get('status') != 'completed':
            raise RuntimeError('Formal training failed; final validation cannot start')
        report.update(status='validating_on_l20', training_final_checkpoint_sha256=training['final_checkpoint_sha256'])
        save()
        # Training and validation never own the physical L20 simultaneously.
        validation_pod = start_job(ROOT / 'validation_worker_v1.yaml', VALIDATION_JOB)
        report.update(validation_pod=validation_pod, validation_job=VALIDATION_JOB)
        save()
        collector = collector_command(validation_pod, VALIDATION_JOB, ROOT / 'results',
                                      'round5_validation_ready', 'round5_validation_collected')
        with (ROOT / 'validation_collector.log').open('xb') as stream:
            result = subprocess.run(collector, stdout=stream, stderr=subprocess.STDOUT, timeout=9000)
        report['validation_collector_returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Validation collector failed; inspect validation_collector.log')
        validation_job = json.loads(subprocess.check_output(K + ['get', 'job', VALIDATION_JOB, '-o', 'json'], timeout=30))
        report['validation_gpu_job_completed'] = validation_job.get('status', {}).get('succeeded', 0) == 1
        if not report['validation_gpu_job_completed']:
            raise RuntimeError('Validation GPU job completion is unconfirmed')
        collected = ROOT / 'results/output/round5'
        workflow = json.loads((collected / 'validation/workflow_status.json').read_text())
        report['validation_pipeline_integrity_pass'] = workflow['pipeline_integrity_pass']
        training_summary_sha = hashlib.sha256((ROOT / 'training_results/output/round5/prefix_v2/summary.json').read_bytes()).hexdigest()
        final_summary_sha = hashlib.sha256((collected / 'prefix_v2/summary.json').read_bytes()).hexdigest()
        if not (workflow.get('training_summary_sha256') == training_summary_sha == final_summary_sha):
            raise RuntimeError('Validation belongs to another training summary')
        audit = json.loads((collected / 'validation/bundle_audit.json').read_text())
        if audit.get('status') != 'completed' or audit.get('pass') is not True:
            raise RuntimeError('Portable bundle L20 audit did not pass; no local release claimed')
        metadata = collected / 'validation/bundle'
        bundle = json.loads((metadata / 'BUNDLE_MANIFEST.json').read_text())
        if not (audit.get('checkpoint_sha256') == bundle.get('checkpoint_sha256')
                == workflow.get('checkpoint_sha256') == training['final_checkpoint_sha256']):
            raise RuntimeError('Training, validation, portable audit and packaged weights disagree')
        report.update(status='downloading_verified_bundle',
                      checkpoint_sha256=bundle['checkpoint_sha256'],
                      canonical_quality_gate_pass=bundle['canonical_quality_gate_pass'],
                      selection_status=bundle['selection_status'])
        save()
        artifact_pod = start_job(ROOT / 'artifact_worker.yaml', ARTIFACT_JOB)
        report['artifact_pod'] = artifact_pod
        save()
        download = [sys.executable, '-u', '-B', str(ROOT / 'runtime/download_release.py'),
                    '--pod', artifact_pod, '--metadata', str(metadata), '--output', str(ROOT / 'release'),
                    '--remote-bundle', '/work/output/round5/validation/bundle']
        attempts = []
        for attempt in range(1, 4):
            try:
                result = run_process_group(download, ROOT / ('release_download_' + str(attempt) + '.log'), 1600)
                attempts.append({'attempt': attempt, 'returncode': result.returncode})
                succeeded = result.returncode == 0
            except subprocess.TimeoutExpired as error:
                attempts.append({'attempt': attempt, 'timed_out': True, 'error': str(error),
                                 'process_group_termination': getattr(error, 'process_group_termination', 'unknown')})
                succeeded = False
            report['download_attempts'] = attempts
            save()
            if succeeded:
                break
            time.sleep(5)
        else:
            raise RuntimeError('Three transfer attempts failed; partial bytes and PVC retained')
        completed = subprocess.run(K + ['wait', '--for=condition=complete', 'job/' + ARTIFACT_JOB,
                                        '--timeout=60s'], timeout=65)
        report.update(status='bundle_delivered' if report['validation_pipeline_integrity_pass'] else 'bundle_delivered_evaluation_incomplete',
                      bundle=str(ROOT / 'release'),
                      artifact_job_completed=completed.returncode == 0,
                      delivery_is_quality_approval=False)
    except Exception as error:
        report.update(status='needs_attention', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
