"""One compact read-only snapshot of the owned Round5 L20 job."""
import json
from pathlib import Path
import subprocess

POD = 'safety-guard-prefix-round5-20260924-r1-t9m2k'
JOB = 'safety-guard-prefix-round5-20260924-r1'
K = ['kubectl', '--kubeconfig', str(Path.home() / '.kube/cls-og1rjus2-private-latest'), '-n', 'default']
REMOTE = r'''
import datetime,json,subprocess
from pathlib import Path
root=Path('/work/output/round5')
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in ('stage_status.json','smoke_v1/smoke.json','prefix_v2/summary.json','validation/workflow_status.json'):
 p=root/name
 if p.is_file():
  d=json.loads(p.read_text())
  if name=='stage_status.json':result['stages']=[{k:r.get(k) for k in ('stage','process_completed','returncode','seconds')} for r in d['stages']]
  elif name.endswith('smoke.json'):result['smoke']={k:d.get(k) for k in ('status','updates','loss','gradient_norm')}
  elif name.endswith('workflow_status.json'):
   result['validation']={k:d.get(k) for k in ('status','active_stage','pipeline_integrity_pass','checkpoint_sha256')}
   result['validation']['stages']=[{k:r.get(k) for k in ('name','status','completed','seconds')} for r in d['stages']]
  else:result['training_final']={k:d.get(k) for k in ('status','final_checkpoint_sha256','generated_tokens')}
for label,relative in [('sft','prefix_v2/sft/losses.jsonl'),('rl','prefix_v2/classification_rl/trajectory_metrics.jsonl')]:
 p=root/relative
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-262144)); lines=f.read().decode(errors='replace').splitlines()
  for line in reversed(lines):
   try:d=json.loads(line)
   except json.JSONDecodeError:continue
   result[label]={k:d.get(k) for k in ('step','steps','epoch','seconds','loss','gradient_norm','input_tokens','weighted_reward') if k in d};break
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
result['gpu']=gpu.stdout.strip() if gpu.returncode==0 else 'unavailable'
print(json.dumps(result))
'''


def main():
    root = Path(__file__).resolve().parent
    delivery_path = root / 'delivery_status.json'
    delivery = json.loads(delivery_path.read_text()) if delivery_path.exists() else {}
    pod_name, job_name = delivery.get('validation_pod', POD), delivery.get('validation_job', JOB)
    pod = subprocess.run(K + ['get', 'pod', pod_name, '-o', 'json'], capture_output=True, text=True, timeout=20)
    if pod.returncode:
        raise RuntimeError(pod.stderr.strip())
    state = json.loads(pod.stdout)
    report = {'pod': pod_name, 'job': job_name, 'phase': state['status']['phase'],
              'delivery_status': delivery.get('status', 'not_started')}
    if report['phase'] == 'Running':
        read = subprocess.run(K + ['exec', '-i', pod_name, '--', 'python', '-'], input=REMOTE,
                              capture_output=True, text=True, timeout=30)
        if read.returncode:
            raise RuntimeError(read.stderr.strip())
        report.update(json.loads(read.stdout))
    destination = root / 'LIVE_STATUS.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
