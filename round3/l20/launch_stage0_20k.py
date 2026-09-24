"""Continue C1 Stage 0 distillation on 20k open-corpus examples on L20."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
JOB = 'safety-guard-c1-stage0-20k-20260923'


def run(kubeconfig, *args, check=True, input_file=None):
    command = ['kubectl', f'--kubeconfig={kubeconfig}', '-n', 'default', *args]
    with (open(input_file, 'rb') if input_file else open('/dev/null', 'rb')) as source:
        result = subprocess.run(command, stdin=source, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f'kubectl {args[0]} failed: {result.stderr.strip()}')
    return result


def objects(kubeconfig, kind, *args):
    return json.loads(run(kubeconfig, 'get', kind, *args, '-o', 'json').stdout)


def free_l20(kubeconfig):
    nodes = objects(kubeconfig, 'nodes')['items']
    pods = objects(kubeconfig, 'pods', '-A')['items']
    available = {}
    for node in nodes:
        if node['metadata'].get('labels', {}).get('naive-sandbox/gpu-product') == 'L20':
            available[node['metadata']['name']] = int(node['status']['allocatable'].get('nvidia.com/gpu', '0'))
    for pod in pods:
        node = pod['spec'].get('nodeName')
        if node not in available or pod['status']['phase'] in ('Succeeded', 'Failed'):
            continue
        containers = pod['spec'].get('containers', [])
        init = pod['spec'].get('initContainers', [])
        request = lambda c: int(c.get('resources', {}).get('requests', {}).get('nvidia.com/gpu', '0'))
        available[node] -= max(sum(map(request, containers)), max(map(request, init), default=0))
    return available


def verify_bundle():
    expected = (HERE / 'bundle_stage0_20k.sha256').read_text().split()[0]
    digest = hashlib.sha256()
    with (HERE / 'bundle_stage0_20k.tar').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError('Local bundle checksum mismatch')
    manifest = json.loads((HERE / 'bundle_stage0_20k_manifest.json').read_text())
    if manifest['uses_project_safety_labels'] or manifest['bundle_sha256'] != expected:
        raise RuntimeError('Stage0 package contract mismatch')


def wait_pod(kubeconfig):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        pods = objects(kubeconfig, 'pods', '-l', f'job-name={JOB}')['items']
        for pod in pods:
            if pod['status']['phase'] == 'Running':
                return pod['metadata']['name']
            if pod['status']['phase'] == 'Failed':
                raise RuntimeError(f'Pod failed before upload: {pod["metadata"]["name"]}')
        time.sleep(5)
    raise TimeoutError('L20 Pod did not become Running within 15 minutes')


def wait_ready(kubeconfig, pod):
    deadline = time.monotonic() + 13800
    while time.monotonic() < deadline:
        ready = run(kubeconfig, 'exec', pod, '-c', 'train', '--',
                    'test', '-f', '/work/output/ready', check=False)
        if ready.returncode == 0:
            return
        item = objects(kubeconfig, 'pod', pod)
        if item['status']['phase'] in ('Failed', 'Succeeded'):
            logs = run(kubeconfig, 'logs', pod, '-c', 'train', '--tail=100', check=False)
            raise RuntimeError(f"Stage0 Pod ended early: {item['status']['phase']}\n{logs.stdout[-3000:]}")
        time.sleep(10)
    raise TimeoutError('Stage0 training did not finish within the Job deadline')


def download_file(kubeconfig, pod, remote_name, destination):
    target = destination.with_suffix(destination.suffix + '.part')
    command = ['kubectl', f'--kubeconfig={kubeconfig}', '-n', 'default',
               'exec', pod, '-c', 'train', '--', 'cat', f'/work/output/{remote_name}']
    with target.open('wb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE)
    if result.returncode:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Artifact download failed: {remote_name}: {result.stderr.decode(errors='replace')[:300]}")
    target.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kubeconfig', default='/Users/liuzhihao/.kube/cls-og1rjus2-private-latest')
    parser.add_argument('--out', default=str(HERE / 'stage0_20k_output'))
    args = parser.parse_args()
    kubeconfig = args.kubeconfig
    verify_bundle()
    available = free_l20(kubeconfig)
    print(json.dumps({'free_l20_gpus_by_node': available}), flush=True)
    if max(available.values(), default=0) < 1:
        raise SystemExit('No free L20 GPU; existing Pods were left untouched and no Job was created')
    existing = run(kubeconfig, 'get', 'job', JOB, check=False)
    if existing.returncode == 0:
        raise RuntimeError(f'Job {JOB} already exists; inspect it before re-running')
    run(kubeconfig, 'create', '-f', str(HERE / 'job_stage0_20k.yaml'))
    run(kubeconfig, 'patch', 'job', JOB, '--type=merge', '-p', '{"spec":{"suspend":false}}')
    print(f'Created and resumed {JOB}', flush=True)
    pod = wait_pod(kubeconfig)
    print(f'Uploading bundle to {pod}', flush=True)
    run(kubeconfig, 'exec', '-i', pod, '-c', 'train', '--', 'sh', '-c', 'cat > /work/bundle.tar.part', input_file=HERE / 'bundle_stage0_20k.tar')
    run(kubeconfig, 'exec', '-i', pod, '-c', 'train', '--', 'sh', '-c', 'cat > /work/bundle.sha256', input_file=HERE / 'bundle_stage0_20k.sha256')
    run(kubeconfig, 'exec', pod, '-c', 'train', '--', 'mv', '/work/bundle.tar.part', '/work/bundle.tar')
    print('Bundle uploaded; waiting for L20 distillation', flush=True)
    wait_ready(kubeconfig, pod)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'loss_curve.jsonl', 'stage0_adapter.safetensors'):
        download_file(kubeconfig, pod, name, out / name)
    summary = json.loads((out / 'summary.json').read_text())
    digest = hashlib.sha256((out / 'stage0_adapter.safetensors').read_bytes()).hexdigest()
    if summary['adapter_sha256'] != digest or not summary['not_policy_safety_aligned']:
        raise RuntimeError('Stage0 adapter checksum or status mismatch')
    run(kubeconfig, 'exec', pod, '-c', 'train', '--', 'touch', '/work/output/ack')
    completed = run(kubeconfig, 'wait', f'job/{JOB}', '--for=condition=complete', '--timeout=180s', check=False)
    if completed.returncode:
        logs = run(kubeconfig, 'logs', pod, '-c', 'train', '--tail=100', check=False)
        print(logs.stdout, file=sys.stderr)
        raise RuntimeError(f'Job did not complete: {completed.stderr.strip()}')
    print(json.dumps({'status': 'succeeded', 'output': str(out.resolve()),
                      'optimizer_steps': summary['optimizer_steps'],
                      'dev_before': summary['dev_before']['total'],
                      'dev_after': summary['dev_after']['total'],
                      'adapter_sha256': digest}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
