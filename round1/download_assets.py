"""Fetch pinned public assets; resumable, with a content-hash manifest."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

def fetch(job):
    url, path = job
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.partial')
        subprocess.run(['curl', '-fLsS', '--retry', '4', '--retry-all-errors',
                        '--connect-timeout', '30', '-C', '-', url, '-o', str(temp)], check=True)
        temp.replace(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    print(json.dumps({'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size}), flush=True)
    return {'path': str(path.relative_to(ROOT)), 'url': url, 'sha256': digest}

if __name__ == '__main__':
    jobs = []
    for manifest_name, repo, directory in [
        ('model_manifest', 'Qwen/Qwen3Guard-Stream-0.6B', 'model'),
        ('benchmark_manifest', 'datasets/Qwen/Qwen3GuardTest', 'benchmark')]:
        manifest = json.loads((ROOT/'source'/f'{manifest_name}.json').read_text())
        for item in manifest['siblings']:
            name = item['rfilename']
            if name.startswith('.'): continue
            jobs.append((f'https://huggingface.co/{repo}/resolve/{manifest["sha"]}/{name}', ROOT/directory/name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, jobs))
    (ROOT/'asset_hashes.json').write_text(json.dumps(results, indent=2)+'\n')
