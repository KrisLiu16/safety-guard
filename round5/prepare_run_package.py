"""Freeze reviewed prefix inputs without touching earlier training artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    project = root.parent
    output = root / 'training_input_v1.tar.gz'
    if output.exists():
        raise FileExistsError('Frozen archive already exists')
    files = ['train_prefix.py', 'stream_metrics.py', 'prepare_prefix_data.py',
             'prepare_fresh_holdout.py', 'PREFIX_TRAINING_PROTOCOL.md', 'TRAIN_PREFIX_USAGE.md',
             'test_prefix_protocol.py', 'test_train_prefix_cpu.py']
    files += ['data/prefix_v2/' + name for name in ('train.jsonl', 'calibration.jsonl', 'dev.jsonl', 'manifest.json')]
    files += ['data/fresh_holdout_v1/' + name for name in ('episodes.jsonl', 'manifest.json')]
    prefix = json.loads((root / 'data/prefix_v2/manifest.json').read_text())
    for split, digest in prefix['output_hashes'].items():
        assert sha(root / f'data/prefix_v2/{split}.jsonl') == digest
    holdout = json.loads((root / 'data/fresh_holdout_v1/manifest.json').read_text())
    assert sha(root / 'data/fresh_holdout_v1/episodes.jsonl') == holdout['episode_sha256']
    assert prefix['builder_sha256'] == sha(root / 'prepare_prefix_data.py')
    assert holdout['builder_sha256'] == sha(root / 'prepare_fresh_holdout.py')
    external_paths = {
        '/work/round4/train_risk.py': 'round4/train_risk.py',
        '/work/round4/risk_metrics.py': 'round4/risk_metrics.py',
        '/work/round4/memory_attention.py': 'round4/memory_attention.py',
        '/work/input/train_base.py': 'round3/l20/base_compare/train_base.py',
        '/work/input/experiment_common.py': 'round3/l20/base_compare/experiment_common.py',
        '/work/input/speed_probe.py': 'round3/l20/base_compare/speed_probe.py',
        '/work/window/window_attention.py': 'round3/l20/window_probe/window_attention.py',
        '/work/window/run_probe.py': 'round3/l20/window_probe/run_probe.py',
    }
    external = {remote: sha(project / local) for remote, local in external_paths.items()}
    external['/work/output/round4/window/best.safetensors'] = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'
    hashes = {name: sha(root / name) for name in files}
    with tempfile.TemporaryDirectory(prefix='guard-prefix-inputs-') as directory:
        stage = Path(directory)
        for name in files:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        external_file = stage / 'external_files.sha256'
        external_file.write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(external.items())))
        hashes['external_files.sha256'] = sha(external_file)
        (stage / 'input_files.sha256').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())))
        with tarfile.open(output, 'w:gz') as tar:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    manifest = {'archive_sha256': sha(output), 'files': hashes, 'external_files': external,
                'raw_generation_format_changed': False, 'frozen_holdout_selection_use': False,
                'neural_model_calls': 0}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archive': str(output), 'archive_sha256': manifest['archive_sha256'],
                      'bytes': output.stat().st_size, 'input_files': len(hashes), 'external_files': len(external)}))


if __name__ == '__main__':
    main()
