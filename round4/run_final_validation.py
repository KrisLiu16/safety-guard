"""Serial post-training validation on the same L20; no training or selection."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/work/validation')
OUT = Path('/work/output/round4_validation')
MODERN = '/work/modern/bin/python'
LEGACY = '/work/legacy/bin/python'


def run_process_group(command, log_path, timeout):
    """Bound a whole stage, including its ordinary child/grandchild processes.

    The validation scripts spawn Python workers that inherit their process
    group. A new session keeps their forced timeout cleanup separate from this
    coordinator and from other jobs. SIGKILL is intentional: a child ignoring
    SIGTERM must not continue GPU work or write artifacts after timeout.
    """
    if os.name != 'posix':
        raise OSError('Validation stage process-group supervision requires POSIX')
    with Path(log_path).open('wb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)

        def stop_group():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The complete group has already exited.
            # Reap the direct child after signalling the entire group. Never
            # stop after merely observing that the parent process has exited.
            process.wait()

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop_group()
            error.process_group_id = process.pid
            error.process_group_termination = 'SIGKILL_sent_to_entire_group'
            raise
        except BaseException:
            stop_group()
            raise
    return subprocess.CompletedProcess(command, returncode)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'MODEL_MANIFEST.json'
    selected = json.loads(manifest.read_text())
    if selected['status'] != 'research_candidate':
        raise RuntimeError('Final validation requires a fixed development-selected checkpoint')
    checkpoint_kind = 'classification_rl' if selected['candidate'] == 'classification_rl' else selected['variant']
    calibrated = Path('/work/output/round4') / selected['candidate'] / 'exported_dev_metrics.json'
    commands = [
        ('text_stream', [MODERN, str(ROOT / 'test_text_stream_l20.py')], 1800),
        ('graph_stream', [MODERN, str(ROOT / 'test_graph_stream_l20.py'), '--replays', '128'], 960),
        ('official_student', [MODERN, str(ROOT / 'evaluate_official_adapted.py'), '--kind', 'student',
             '--data-dir', str(ROOT / 'official_adapted_v1'), '--source-dir', str(ROOT / 'qwen3guardtest'),
             '--candidate-manifest', str(manifest), '--output', str(OUT / 'official_student')], 2400),
        ('official_a0', [LEGACY, str(ROOT / 'evaluate_official_adapted.py'), '--kind', 'a0',
             '--data-dir', str(ROOT / 'official_adapted_v1'), '--source-dir', str(ROOT / 'qwen3guardtest'),
             '--candidate-manifest', str(manifest), '--output', str(OUT / 'official_a0')], 2400),
        ('benchmark_candidate', [MODERN, str(ROOT / 'benchmark_pair.py'), '--kind', 'candidate',
             '--candidate-manifest', str(manifest), '--output', str(OUT / 'benchmark_candidate.json')], 600),
        ('benchmark_a0', [LEGACY, str(ROOT / 'benchmark_pair.py'), '--kind', 'a0',
             '--candidate-manifest', str(manifest), '--output', str(OUT / 'benchmark_a0.json')], 600),
        ('keyword_candidate', [MODERN, str(ROOT / 'eval_keyword_probe.py'), '--kind', 'candidate',
             '--data-root', str(ROOT / 'keyword_probe_v1'), '--variant', checkpoint_kind,
             '--checkpoint', selected['checkpoint'], '--calibration-metrics', str(calibrated),
             '--output-prefix', str(OUT / 'keyword_candidate')], 600),
        ('keyword_a0', [LEGACY, str(ROOT / 'eval_keyword_probe.py'), '--kind', 'a0',
             '--data-root', str(ROOT / 'keyword_probe_v1'),
             '--calibration-metrics', '/work/output/round4/a0_reference_metrics.json',
             '--output-prefix', str(OUT / 'keyword_a0')], 600),
        ('package', [MODERN, str(ROOT / 'package_model.py'), '--manifest', str(manifest),
             '--output', str(OUT / 'bundle'), '--model-assets', '/work/models/qwen35',
             '--code-dir', str(ROOT), '--window-code-dir', '/work/window'], 300),
        ('bundle_audit', [MODERN, str(ROOT / 'test_bundle_l20.py'), '--bundle', str(OUT / 'bundle'),
             '--manifest', str(manifest), '--output', str(OUT / 'bundle_audit.json')], 1200),
    ]
    stages = []
    for name, command, timeout in commands:
        started = time.monotonic()
        record = {'stage': name, 'command': command, 'required': name != 'graph_stream'}
        print(json.dumps({'stage': name, 'status': 'running'}), flush=True)
        try:
            result = run_process_group(command, OUT / (name + '.log'), timeout)
            record.update(returncode=result.returncode, process_completed=result.returncode == 0)
        except (subprocess.TimeoutExpired, OSError) as error:
            record.update(process_completed=False, error=str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                record.update(timed_out=True,
                              process_group_id=getattr(error, 'process_group_id', None),
                              process_group_termination=getattr(error, 'process_group_termination', 'unconfirmed'))
        record['seconds'] = time.monotonic() - started
        stages.append(record)
        (OUT / 'stage_status.json').write_text(json.dumps({
            'status': 'completed' if len(stages) == len(commands) else 'running', 'stages': stages,
            'all_processes_completed': all(row['process_completed'] for row in stages),
            'all_required_processes_completed': all(row['process_completed'] for row in stages if row['required']),
            'candidate': selected['candidate'], 'variant': selected['variant'],
            'checkpoint_sha256': selected['checkpoint_sha256'],
            'note': 'Process completion is not an audit pass. Check each numerical and behavioral artifact.',
        }, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
