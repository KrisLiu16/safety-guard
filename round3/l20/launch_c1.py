"""Start the C1 L20 compatibility probe only when one GPU is free."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
JOB = 'safety-guard-c1-l20-probe-20260923'


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
    expected = (HERE / 'bundle_c1.sha256').read_text().split()[0]
    digest = hashlib.sha256()
    with (HERE / 'bundle_c1.tar').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError('Local bundle checksum mismatch')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kubeconfig', default='/Users/liuzhihao/.kube/cls-og1rjus2-private-latest')
    parser.add_argument('--out', default=str(HERE / 'c1_l20_results.json'))
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
    run(kubeconfig, 'create', '-f', str(HERE / 'job_c1.yaml'))
    run(kubeconfig, 'patch', 'job', JOB, '--type=merge', '-p', '{"spec":{"suspend":false}}')
    print(f'Created and resumed {JOB}', flush=True)
    pod = wait_pod(kubeconfig)
    print(f'Uploading bundle to {pod}', flush=True)
    run(kubeconfig, 'exec', '-i', pod, '-c', 'probe', '--', 'sh', '-c', 'cat > /work/bundle.tar.part', input_file=HERE / 'bundle_c1.tar')
    run(kubeconfig, 'exec', '-i', pod, '-c', 'probe', '--', 'sh', '-c', 'cat > /work/bundle.sha256', input_file=HERE / 'bundle_c1.sha256')
    run(kubeconfig, 'exec', pod, '-c', 'probe', '--', 'mv', '/work/bundle.tar.part', '/work/bundle.tar')
    completed = run(kubeconfig, 'wait', f'job/{JOB}', '--for=condition=complete', '--timeout=1800s', check=False)
    if completed.returncode:
        logs = run(kubeconfig, 'logs', pod, '-c', 'probe', '--tail=100', check=False)
        print(logs.stdout, file=sys.stderr)
        raise RuntimeError(f'Job did not complete: {completed.stderr.strip()}')
    # Completed Pods reject kubectl exec; the runner emits the result as its final log line.
    logs = run(kubeconfig, 'logs', pod, '-c', 'probe').stdout
    result = json.loads(logs.strip().splitlines()[-1])
    if result.get('status') != 'passed':
        raise RuntimeError(f'Job completed without a passed result: {result.get("status")}')
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
