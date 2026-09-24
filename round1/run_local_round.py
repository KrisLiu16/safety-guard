"""Sequential local GPU stages, stopping on any failed verification."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

def stage(name,args):
    status={'stage':name,'status':'running','pid':__import__('os').getpid(),'started_at':time.time()}
    (ROOT/'local_run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (ROOT/f'{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    status.update(status='succeeded' if result.returncode==0 else 'failed',returncode=result.returncode,finished_at=time.time())
    (ROOT/'local_run_status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (ROOT/'local_stages.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    if result.returncode:raise SystemExit(f'Stage failed: {name}; inspect {ROOT/name}.log')
    print(json.dumps(status),flush=True)

if __name__=='__main__':
    stage('smoke',['smoke.py'])
    stage('baseline_official_subset_corrected',['evaluate.py','--mode','official','--scope','subset','--out','baseline'])
    stage('baseline_official_all_corrected',['evaluate.py','--mode','official','--scope','all','--out','baseline'])
    stage('baseline_chinese',['evaluate.py','--mode','chinese','--out','baseline'])
    stage('performance_original',['performance.py','--name','original'])
    stage('train',['train.py'])
    checkpoint=json.loads((ROOT/'training/selected_checkpoint.json').read_text())['path']
    stage('post_eval',['evaluate.py','--checkpoint',checkpoint,'--mode','both','--scope','all','--out','post'])
    stage('performance_rl',['performance.py','--checkpoint',checkpoint,'--name','rl'])
    stage('performance_merged',['performance.py','--checkpoint',checkpoint,'--merge','--name','rl_merged'])
