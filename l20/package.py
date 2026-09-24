"""Create a portable, credential-free bundle for the suspended L20 Job."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'round1'
OUTPUT = HERE / 'bundle.tar'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise RuntimeError('bundle.tar already exists; remove it explicitly after inspecting')
    items = [
        (HERE / 'runtime.py', 'runtime.py'),
        (HERE / 'run_cuda.py', 'run_cuda.py'),
        (HERE / 'reference.json', 'reference.json'),
        (SOURCE / 'data/dev.jsonl', 'dev.jsonl'),
        (SOURCE / 'data/train.jsonl', 'train.jsonl'),
    ]
    items += [(path, f'model/{path.name}') for path in sorted((SOURCE / 'model').iterdir()) if path.is_file()]
    items += [(path, f'checkpoint/{path.name}') for path in sorted((SOURCE / 'training/step-0150').iterdir()) if path.is_file()]
    manifest = {}
    for path, name in items:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest[name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    with tarfile.open(OUTPUT, 'w') as tar:
        for path, name in items:
            tar.add(path, arcname=f'bundle/{name}', recursive=False)
    digest = sha256(OUTPUT)
    (HERE / 'bundle.sha256').write_text(f'{digest}  /work/bundle.tar\n')
    (HERE / 'bundle_manifest.json').write_text(json.dumps({
        'model_revision': '419364a715de9840d47b1457982f64ff37f90ed4',
        'checkpoint': 'round1/training/step-0150',
        'files': manifest, 'tar_sha256': digest, 'tar_bytes': OUTPUT.stat().st_size,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tar_bytes': OUTPUT.stat().st_size, 'tar_sha256': digest, 'files': len(items)}))


if __name__ == '__main__':
    main()
