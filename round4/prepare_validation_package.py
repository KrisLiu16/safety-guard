"""Freeze the final L20 validation inputs after development-only selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from standalone_model import validate_selection


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=root / 'MODEL_MANIFEST.json')
    parser.add_argument('--output', type=Path, default=root / 'validation_package_v1.tar.gz')
    args = parser.parse_args()
    selection = validate_selection(json.loads(args.manifest.read_text()))
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite a frozen validation package: {args.output}')
    with tempfile.TemporaryDirectory(prefix='guard-validation-') as temp:
        stage = Path(temp)
        scripts = ['infer_classifier.py', 'memory_attention.py', 'text_stream_runtime.py',
                   'test_text_stream_l20.py', 'evaluate_official_adapted.py', 'run_final_validation.py',
                   'standalone_model.py', 'package_model.py', 'run_guard.py', 'test_bundle_l20.py', 'BUNDLE_USAGE.md',
                   'graph_stream.py', 'test_graph_stream_l20.py', 'GRAPH_STREAM_NOTES.md', 'eval_keyword_probe.py',
                   'benchmark_pair.py', 'BENCHMARK_PAIR_USAGE.md']
        for name in scripts:
            shutil.copyfile(root / name, stage / name)
        shutil.copyfile(root / 'validation_start.sh', stage / 'start_v1.sh')
        shutil.copyfile(args.manifest, stage / 'MODEL_MANIFEST.json')
        shutil.copytree(root / 'data/official_adapted_v1', stage / 'official_adapted_v1')
        shutil.copytree(root / 'data/keyword_probe_v1', stage / 'keyword_probe_v1')
        (stage / 'qwen3guardtest').mkdir()
        for name in ('thinking', 'thinking_loc', 'response_loc'):
            shutil.copyfile(root.parent / 'round1/benchmark' / (name + '.jsonl'),
                            stage / 'qwen3guardtest' / (name + '.jsonl'))
        files = {str(path.relative_to(stage)): digest(path) for path in sorted(stage.rglob('*')) if path.is_file()}
        (stage / 'input_files.sha256').write_text(''.join(f'{checksum}  {name}\n' for name, checksum in files.items()))
        with tarfile.open(args.output, 'w:gz') as archive:
            for path in sorted(stage.iterdir()):
                archive.add(path, arcname=path.name)
    spec = {'archive': str(args.output), 'archive_sha256': digest(args.output),
            'checkpoint_sha256': selection['checkpoint_sha256'], 'candidate': selection['candidate'],
            'variant': selection['variant'], 'files': files, 'training_or_selection_changed': False}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps({key: value for key, value in spec.items() if key != 'files'}))


if __name__ == '__main__':
    main()
