"""Freeze final validation code; never replace the live training inputs."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    project = root.parent
    archive = root / 'validation_input_v1.tar.gz'
    receipt = root / 'validation_input_v1.manifest.json'
    if archive.exists() or receipt.exists():
        raise FileExistsError('Validation v1 has already been frozen')
    names = ['post_training_validate.py', 'evaluate_final_l20.py',
             'calibrate_canonical_l20.py', 'evaluate_canonical_fresh_l20.py',
             'evaluate_canonical_official_l20.py', 'CANONICAL_CALIBRATION_USAGE.md',
             'CANONICAL_OFFICIAL_USAGE.md', 'FINAL_EVALUATION_USAGE.md']
    names += [str(path.relative_to(root)) for path in sorted((root / 'runtime').glob('*.py'))]
    names += ['runtime/BUNDLE_USAGE.md']
    files = {name: sha(root / name) for name in names}
    # Training code/data are already frozen on the PVC. Verify but do not
    # archive them, so extraction cannot change an in-flight training input.
    existing = ['train_prefix.py', 'stream_metrics.py', 'prepare_prefix_data.py',
                'prepare_fresh_holdout.py', 'PREFIX_TRAINING_PROTOCOL.md',
                'native_tokenizer_proof.json']
    existing += ['data/prefix_v2/' + name for name in
                 ('train.jsonl', 'calibration.jsonl', 'dev.jsonl', 'manifest.json')]
    existing += ['data/fresh_holdout_v1/' + name for name in ('episodes.jsonl', 'manifest.json')]
    external = {'/work/round5/' + name: sha(root / name) for name in existing}
    helpers = {
        '/work/round4/infer_classifier.py': 'round4/infer_classifier.py',
        '/work/round4/evaluate_official_adapted.py': 'round4/evaluate_official_adapted.py',
        '/work/round4/train_risk.py': 'round4/train_risk.py',
        '/work/round4/risk_metrics.py': 'round4/risk_metrics.py',
        '/work/round4/memory_attention.py': 'round4/memory_attention.py',
        '/work/input/train_base.py': 'round3/l20/base_compare/train_base.py',
        '/work/input/experiment_common.py': 'round3/l20/base_compare/experiment_common.py',
        '/work/input/speed_probe.py': 'round3/l20/base_compare/speed_probe.py',
        '/work/window/window_attention.py': 'round3/l20/window_probe/window_attention.py',
        '/work/window/run_probe.py': 'round3/l20/window_probe/run_probe.py',
    }
    external.update({remote: sha(project / local) for remote, local in helpers.items()})
    manifest = {'format': 'round5-validation-input-v1', 'files': files,
                'external_files': external, 'training_inputs_in_archive': False,
                'training_and_checkpoint_selection_changed': False, 'neural_calls': 0}
    with tempfile.TemporaryDirectory(prefix='round5-validation-freeze-') as temporary:
        manifest_file = Path(temporary) / receipt.name
        manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
        with tarfile.open(archive, 'w:gz') as destination:
            for name in sorted(names):
                destination.add(root / name, arcname=name, recursive=False)
            destination.add(manifest_file, arcname=receipt.name, recursive=False)
        receipt.write_bytes(manifest_file.read_bytes())
    summary = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'manifest_sha256': sha(receipt), 'files': len(files), 'external_files': len(external),
               'post_training_script_sha256': files['post_training_validate.py']}
    (root / 'validation_input_v1.archive.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
