"""Freeze independent graph-text validation code and its existing data inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from standalone_model import validate_selection
from run_fast_validation import build_stage_commands


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(piece)
    return value.hexdigest()


def validate_proof(original):
    """Check collected real-modern CPU evidence; do not run tokenizers or models."""
    evaluator = original / 'eval_keyword_probe.py'
    spec = importlib.util.spec_from_file_location('_frozen_probe_checker', evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_root = original / 'data/keyword_probe_v1'
    manifest, rows = module.frozen_probe(data_root)
    proof_path = original / 'keyword_tokenizer_compatibility_v1.json'
    proof = json.loads(proof_path.read_text())
    expected = {
        'version': 'keyword-tokenizer-row-compatibility-v1', 'status': 'passed',
        'scope': module.COMPATIBILITY_SCOPE, 'global_tokenizer_equivalence': False,
        'neural_model_loaded': False, 'neural_forward_calls': 0,
        'calibration_or_predictions_read': False, 'library_versions': module.COMPATIBILITY_VERSIONS,
        'reference_tokenizer_raw_sha256': manifest['tokenizer_sha256'],
        'probe_manifest_sha256': sha(data_root / 'manifest.json'),
        'probe_jsonl_sha256': sha(data_root / 'probe.jsonl'),
        'verified_rows': len(rows), 'mismatched_rows': 0,
        'frozen_row_ids_sha256': module.row_ids_sha(rows),
        'actual_row_ids_sha256': module.row_ids_sha(rows),
        'row_checks': module.row_tokenizer_checks(rows), 'checker_source_sha256': sha(evaluator),
    }
    for key, value in expected.items():
        if proof.get(key) != value:
            raise ValueError('Tokenizer compatibility evidence mismatch: ' + key)
    if len(rows) != 1024:
        raise ValueError('This validation is fixed to the original 1024-row keyword probe')
    assets = original / 'validation_results/output/round4_validation/bundle/assets'
    asset_hashes = module.tokenizer_asset_hashes(assets)
    if (proof['runtime_tokenizer_assets_sha256'] != asset_hashes
            or proof['runtime_tokenizer_raw_sha256'] != asset_hashes['tokenizer.json']
            or proof['runtime_tokenizer_canonical_json_sha256'] != module.canonical_json_sha(
                json.loads((assets / 'tokenizer.json').read_text()))):
        raise ValueError('Collected runtime tokenizer assets differ from the real-modern proof')
    return proof_path, proof


def prepare_inputs(root):
    original = root.parent
    selection = validate_selection(json.loads((root / 'MODEL_MANIFEST.json').read_text()))
    primary_manifest = original / 'MODEL_MANIFEST.json'
    primary_selection = json.loads(primary_manifest.read_text())
    commands = build_stage_commands(selection, primary_selection)
    frozen = json.loads((original / 'validation_package_v1.tar.manifest.json').read_text())
    if sha(primary_manifest) != frozen['files']['MODEL_MANIFEST.json']:
        raise ValueError('Primary manifest must remain byte-identical to the original validation package')
    proof_path, proof = validate_proof(original)
    external = {'/work/validation/' + name: value for name, value in frozen['files'].items()
                if name.startswith(('official_adapted_v1/', 'keyword_probe_v1/', 'qwen3guardtest/'))}
    external['/work/validation/MODEL_MANIFEST.json'] = sha(primary_manifest)
    for variant, summary_name, manifest in (
            ('window', 'sft_summary.json', selection),
            ('classification_rl', 'summary.json', primary_selection)):
        folder = original / 'results/output/round4' / variant
        summary = json.loads((folder / summary_name).read_text())
        metrics = json.loads((folder / 'exported_dev_metrics.json').read_text())
        if (summary.get('status') != 'completed' or summary.get('variant') != manifest['variant']
                or summary.get('checkpoint_sha256') != manifest['checkpoint_sha256']):
            raise ValueError('Checkpoint summary conflicts with the fixed selection: ' + variant)
        thresholds = {key: row['threshold_from_calibration'] for key, row in metrics['strata'].items()}
        if thresholds != manifest['thresholds']:
            raise ValueError('Calibration thresholds conflict with the fixed selection: ' + variant)
        for filename in ('exported_dev_metrics.json', summary_name):
            external['/work/output/round4/' + variant + '/' + filename] = sha(folder / filename)
    scripts = ('standalone_model.py', 'run_guard.py', 'package_model.py', 'text_stream_runtime.py',
               'graph_stream.py', 'graph_text_runtime.py', 'test_fast_text_l20.py', 'test_bundle_l20.py',
               'BUNDLE_USAGE.md', 'MODEL_MANIFEST.json', 'run_fast_validation.py', 'process_runner.py',
               'start_v1.sh', 'README.md')
    sources = {name: root / name for name in scripts}
    for name in ('infer_classifier.py', 'evaluate_official_adapted.py', 'eval_keyword_probe.py',
                 'benchmark_pair.py', 'BENCHMARK_PAIR_USAGE.md', proof_path.name):
        sources[name] = original / name
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    metadata = {
        'checkpoint_sha256': selection['checkpoint_sha256'], 'candidate': selection['candidate'],
        'variant': selection['variant'], 'inference_engine': selection['inference_engine'],
        'primary_candidate': primary_selection['candidate'], 'primary_variant': primary_selection['variant'],
        'primary_checkpoint_sha256': primary_selection['checkpoint_sha256'],
        'primary_manifest_sha256': sha(primary_manifest),
        'tokenizer_compatibility_sha256': sha(proof_path), 'keyword_probe_verified_rows': proof['verified_rows'],
        'planned_stage_count': len(commands), 'total_stage_timeout_seconds': sum(row[2] for row in commands),
        'training_changed': False, 'primary_quality_candidate_changed': False,
    }
    return sources, external, metadata


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root / 'fast_validation_package_v1.tar.gz')
    parser.add_argument('--check-only', action='store_true', help='Validate all inputs without creating an archive or manifest')
    args = parser.parse_args()
    sources, external, metadata = prepare_inputs(root)
    if args.check_only:
        print(json.dumps(dict(metadata, status='inputs_verified_not_frozen', source_file_count=len(sources),
                              external_file_count=len(external), archive_created=False)))
        return
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Refusing to overwrite a frozen fast validation package')
    with tempfile.TemporaryDirectory(prefix='guard-fast-validation-') as directory:
        stage = Path(directory)
        for name, source in sources.items():
            shutil.copyfile(source, stage / name)
        (stage / 'external_files.sha256').write_text(''.join(f'{value}  {name}\n' for name, value in sorted(external.items())))
        files = {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob('*')) if path.is_file()}
        (stage / 'input_files.sha256').write_text(''.join(f'{value}  {name}\n' for name, value in files.items()))
        with tarfile.open(args.output, 'w:gz') as archive:
            for path in sorted(stage.iterdir()):
                archive.add(path, arcname=path.name)
    record = dict(metadata, archive=str(args.output), archive_sha256=sha(args.output),
                  files=files, external_files=external)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key not in ('files', 'external_files')}))


if __name__ == '__main__':
    main()
