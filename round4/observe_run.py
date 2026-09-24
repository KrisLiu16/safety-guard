"""Read one compact Round4 snapshot; never launches, stops, or changes workloads."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod', default='safety-guard-risk-round4-r3-20260923-cgttg')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    command = ['kubectl', '--kubeconfig', str(Path.home() / '.kube/cls-og1rjus2-private-latest'),
               '-n', 'default', '--request-timeout=25s']

    def run(arguments):
        result = subprocess.run(command + arguments, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    state = {'observed_at': datetime.now(timezone.utc).isoformat(), 'pod': args.pod}
    try:
        pod = json.loads(run(['get', 'pod', args.pod, '-o', 'json']))
        state['phase'] = pod['status']['phase']
        state['node'] = pod['spec'].get('nodeName')
        state['container_states'] = {r['name']: r['state']
                                     for r in pod['status'].get('containerStatuses', [])}
        logs = run(['logs', args.pod, '--tail=18'])
        events = []
        for line in logs.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        state['recent_events'] = events
        state['last_logged_step'] = next((event for event in reversed(events) if 'step' in event), None)
        if state['phase'] == 'Running':
            script = """from pathlib import Path
import json
root=Path('/work/output/round4')
result={}
for name in ('full','window','memory','classification_rl'):
 p=root/name
 row={'started':p.exists()}
 for filename in ('sft_summary.json','summary.json','exported_dev_metrics.json'):
  if (p/filename).exists(): row[filename]=json.loads((p/filename).read_text())
 if (p/'dev_curve.json').exists():
  curve=json.loads((p/'dev_curve.json').read_text())
  row['dev_curve_scores']=[{'step':x['step'],'selection_score':x['selection_score']} for x in curve]
 row['checkpoint_present']=(p/'best.safetensors').exists()
 result[name]=row
result['training_complete']=(root/'final_summary.json').exists()
result['reports_ready']=(root.parent/'round4_ready').exists()
result['collected_ack']=(root.parent/'round4_collected').exists()
if (root/'long_stream_audit.json').exists():
 a=json.loads((root/'long_stream_audit.json').read_text())
 result['long_stream_audit']={'status':a['status'],'all_candidates_pass':a['all_candidates_pass']}
print(json.dumps(result))
"""
            state['artifacts'] = json.loads(run(['exec', args.pod, '--', 'python', '-c', script]))
            if args.gpu:
                state['gpu'] = run(['exec', args.pod, '--', 'nvidia-smi',
                                    '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader']).strip()
        else:
            state['log_tail'] = logs
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        state['observation_error'] = str(error)
    target = root / 'LIVE_STATUS.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    compact = {k: state[k] for k in ('observed_at', 'phase', 'last_logged_step', 'gpu', 'observation_error') if k in state}
    artifacts = state.get('artifacts', {})
    compact['stages'] = {name: {
        'started': artifacts[name]['started'],
        'completed': 'sft_summary.json' in artifacts[name] or 'summary.json' in artifacts[name],
        'latest_dev': artifacts[name].get('dev_curve_scores', [None])[-1],
    } for name in ('full', 'window', 'memory', 'classification_rl') if name in artifacts}
    compact['training_complete'] = artifacts.get('training_complete', False)
    compact['long_stream_audit'] = artifacts.get('long_stream_audit')
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
