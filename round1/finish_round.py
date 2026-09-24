"""Run the remaining acceptance stages after the sequential training pipeline ends."""
import hashlib
import argparse
import json
import shutil
from pathlib import Path
from run_local_round import stage,ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--from-stage');args=parser.parse_args()
    state=json.loads((ROOT/'local_run_status.json').read_text())
    if args.from_stage:
        if state['stage']!=args.from_stage or state['status']!='failed':
            raise SystemExit('Resume must identify the currently failed stage; inspect the actual process first.')
    elif state['stage']!='performance_merged' or state['status']!='succeeded':
        raise SystemExit('Training/performance pipeline has not succeeded; inspect its live process or failure first.')
    checkpoint=json.loads((ROOT/'training/selected_checkpoint.json').read_text())['path']
    steps=[
        ('precision_control',['evaluate.py','--fp32-heads','--mode','both','--scope','all','--out','precision_control']),
        ('stream_baseline',['stream_evaluate.py','--out','baseline']),
        ('stream_post',['stream_evaluate.py','--checkpoint',checkpoint,'--out','post']),
        ('export',['export.py']),
        ('merged_eval',['evaluate.py','--model-path','export/model','--mode','both','--scope','all','--out','merged']),
        ('stream_merged',['stream_evaluate.py','--model-path','export/model','--out','merged']),
        ('refresh_metrics',['refresh_metrics.py']),
        ('demo_verify',['verify_demo.py']),
        ('production_performance',['production_performance.py']),
        ('finalize_package',['finalize_package.py']),
        ('verify_launcher',['verify_launcher.py']),
        ('startup',['measure_startup.py']),
        ('compare',['compare.py']),
        ('audit',['audit.py'])]
    names=[name for name,_ in steps];start=names.index(args.from_stage) if args.from_stage else 0
    for i,(name,command) in enumerate(steps[:7]):
        if i>=start:stage(name,command)
    shutil.copy2(ROOT/'merged/chinese_metrics.json',ROOT/'export/calibration_batch.json')
    shutil.copy2(ROOT/'merged/stream_metrics.json',ROOT/'export/calibration_stream.json')
    manifest=json.loads((ROOT/'export/manifest.json').read_text())
    for name in ['runtime.py','demo.py']:shutil.copy2(ROOT/name,ROOT/'export'/name)
    manifest['calibration_source']='merged checkpoint, independent calibration split'
    manifest['files']={str(p.relative_to(ROOT/'export')):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        for p in (ROOT/'export').rglob('*') if p.is_file() and p.name!='manifest.json'}
    (ROOT/'export/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for i,(name,command) in enumerate(steps[7:],start=7):
        if i>=start:stage(name,command)
