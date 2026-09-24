"""Fetch locked public checkpoints; never bypass gated repositories."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path('/work/models')
specs = json.loads(Path('/work/input/models_lock.json').read_text())
ROOT.mkdir(exist_ok=True)
results = []
for spec in specs:
    if spec.get('access_blocked'):
        results.append(spec)
        continue
    dest = ROOT / spec['key']
    print(json.dumps({'downloading': spec['repo'], 'revision': spec['revision']}), flush=True)
    snapshot_download(spec['repo'], revision=spec['revision'], local_dir=dest,
                      allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt', '*.model', '*.jinja'],
                      max_workers=3)
    files = {}
    for path in sorted(dest.glob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    result = {**spec, 'path': str(dest), 'files': files}
    results.append(result)
    Path('/work/output/models_downloaded.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'downloaded': spec['repo'], 'bytes': sum(x['bytes'] for x in files.values())}), flush=True)

Path('/work/output/models_downloaded.json').write_text(json.dumps(results, indent=2)+'\n')
