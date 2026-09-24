"""Bounded L20 diagnosis, one-step smoke, then frozen formal prefix training."""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, '/work/round4/fast_runtime')
from process_runner import run_process_group

ROOT = Path('/work/round5')
OUT = Path('/work/output/round5')
PYTHON = '/work/modern/bin/python'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    common = ['--data-root', str(ROOT / 'data/prefix_v2'),
              '--proof', str(ROOT / 'native_tokenizer_proof.json')]
    commands = [
        ('prefill_state_control', [PYTHON, '-B', str(ROOT / 'test_prefill_stream_control_l20.py'),
            '--bundle', '/work/output/fast_runtime/bundle',
            '--original-audit', '/work/output/fast_runtime/fast_text_audit.json',
            '--helpers', '/work/round4/fast_runtime/test_fast_text_l20.py',
            '--dev', '/work/round4/data/dev.jsonl',
            '--output', str(OUT / 'fast_state_control/prefill_stream_control.json')], 360, False),
        ('training_smoke', [PYTHON, '-B', str(ROOT / 'train_prefix.py'), *common,
            '--smoke-only', '--output', str(OUT / 'smoke_v1')], 600, True),
        ('formal_training', [PYTHON, '-B', str(ROOT / 'train_prefix.py'), *common,
            '--output', str(OUT / 'prefix_v2')], 21600, True),
    ]
    stages = []
    for name, command, timeout, required in commands:
        row = {'stage': name, 'command': command, 'timeout_seconds': timeout,
               'required_for_training': required}
        print(json.dumps({'stage': name, 'status': 'running'}), flush=True)
        start = time.monotonic()
        try:
            result = run_process_group(command, OUT / (name + '.log'), timeout)
            row.update(returncode=result.returncode, process_completed=result.returncode == 0)
        except (subprocess.TimeoutExpired, OSError) as error:
            row.update(process_completed=False, error=str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                row['process_group_termination'] = getattr(error, 'process_group_termination', 'unconfirmed')
        row['seconds'] = time.monotonic() - start
        stages.append(row)
        stop = required and not row['process_completed']
        (OUT / 'stage_status.json').write_text(json.dumps({
            'status': 'failed' if stop else 'completed' if len(stages) == len(commands) else 'running',
            'stages': stages,
            'note': 'Prefill state diagnostic is independent of risk training and cannot rewrite the original failed audit. Smoke failure stops formal training.',
        }, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if stop:
            return


if __name__ == '__main__':
    main()
