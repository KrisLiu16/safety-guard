"""Frozen Qwen3GuardTest adaptation with canonical32 all-position classification.

The fixed checkpoint and runtime are bound to completed training and locked
canonical calibration. Official labels never select weights or thresholds.
This script only evaluates the student; historical A0 is verified and reused.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

CALIBRATOR_SHA = '835c335182e84698c3c6a7b90670a1c57ef74fabc86be3d5f95296f778193666'
LOCKED_EVALUATOR_SHA = 'ace7d10c026cbf69f5fb98ee50ac7bba24596a4d7503d2aa552c57790adfc5c0'
OFFICIAL_EVALUATOR_SHA = 'fa057016e844baf262e0a8ebf153345d23071f3d9b40300ccc6d3125f437a593'
OFFICIAL_MANIFEST_SHA = '161972a7e4df0e10821d6b0776a7b33d3b778a8a898573a52c5e8f6bd6be6202'
A0_METRICS_SHA = '9ff0b35279092bd89337e0c4920ffad027e62deaeb0a8f56b37c2ff760e06b65'
BULK_METRICS_SHA = '22a0a8f310cb34ca1d78fcd056f95da1b89f294172aa2d36b5288b6e5aeee52f'
INITIAL_WINDOW_SHA = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'
COUNTS = {'thinking': 1059, 'thinking_loc': 569, 'response_loc': 813}
SPLITS = tuple(COUNTS)
EXPECTED_UNIQUE = 1872
EXPECTED_CACHE_HITS = 569
VERSIONS = {'transformers': '5.17.0', 'tokenizers': '0.23.2'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_exact(name, path, digest):
    path = Path(path).resolve()
    if sha(path) != digest:
        raise ValueError('Frozen source SHA changed: ' + name)
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError('Refusing a helper from another directory: ' + name)
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def sequence_key(row):
    # Same two-component cache rule as the frozen official evaluator.
    return ids_digest(row['ids']), row['eval_start_index']


def validate_prepared_rows(split, rows, sources):
    if len(rows) != COUNTS[split] or len(sources) != COUNTS[split]:
        raise ValueError('Official split coverage changed: ' + split)
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source-record identity within split')
    for index, row in enumerate(rows):
        ids = row['ids']
        if (row['split'] != split or row['row_index'] != index or row['target_role'] != 'assistant'
                or row['status'] != 'ready' or row['label'] not in ('Safe', 'Unsafe')
                or not isinstance(ids, list) or not 1 <= len(ids) <= 8192
                or any(type(token) is not int or token < 0 for token in ids)
                or row['token_count'] != len(ids) or row['endpoint_index'] != len(ids) - 1
                or row['eval_end_index_exclusive'] != len(ids)
                or type(row['eval_start_index']) is not int or not 0 <= row['eval_start_index'] < len(ids)
                or row['native_ids_sha256'] != ids_digest(ids)):
            raise ValueError('Frozen official native protocol changed: ' + split + '/' + str(index))
        raw = sources[index]
        if raw['unique_id'] != row['unique_id'] or raw['label'] != row['label']:
            raise ValueError('Adaptation no longer matches the frozen source row')
    if split != 'thinking' and any(row['label'] != 'Unsafe' for row in rows):
        raise ValueError('Location split label composition changed')


def coverage_receipt(data):
    keys, hits, per_split = set(), 0, {}
    real_tokens = physical_blocks = 0
    for split in SPLITS:
        new, reused = 0, 0
        for row in data[split]:
            key = sequence_key(row)
            if key in keys:
                reused += 1
            else:
                keys.add(key); new += 1
                real_tokens += len(row['ids'])
                physical_blocks += math.ceil(len(row['ids']) / 32)
        hits += reused
        per_split[split] = {'requested': len(data[split]), 'unique_forward_sequences': new,
                            'exact_prediction_cache_hits': reused}
    if (sum(len(rows) for rows in data.values()) != 2441 or len(keys) != EXPECTED_UNIQUE
            or hits != EXPECTED_CACHE_HITS):
        raise ValueError('Expected exactly 2441 rows, 1872 unique native sequences/start pairs and 569 duplicate cache hits')
    return {'rows': 2441, 'unique_forward_sequences': len(keys), 'exact_prediction_cache_hits': hits,
            'expected_unique_real_tokens': real_tokens, 'expected_physical_block_calls': physical_blocks,
            'expected_physical_tokens': physical_blocks * 32, 'expected_padding_tokens': physical_blocks * 32 - real_tokens,
            'splits': per_split}


def load_official_inputs(data_dir, source_dir):
    manifest_path = data_dir / 'manifest.json'
    if sha(manifest_path) != OFFICIAL_MANIFEST_SHA:
        raise ValueError('Frozen official preparation manifest changed')
    manifest = json.loads(manifest_path.read_text())
    data, receipts = {}, {}
    for split in SPLITS:
        prepared_path, source_path = data_dir / (split + '.jsonl'), source_dir / (split + '.jsonl')
        if (sha(prepared_path) != manifest['output_hashes'][split + '.jsonl']
                or sha(source_path) != manifest['splits'][split]['source_sha256']):
            raise ValueError('Frozen official input/source SHA changed: ' + split)
        rows, sources = read_rows(prepared_path), read_rows(source_path)
        validate_prepared_rows(split, rows, sources)
        data[split] = rows
        receipts[split] = {'adapted_sha256': sha(prepared_path), 'source_sha256': sha(source_path),
                           'requested': len(rows), 'source_dataset_revision': manifest['source_dataset_revision']}
    return data, {'manifest_sha256': OFFICIAL_MANIFEST_SHA, 'files': receipts, **coverage_receipt(data)}


def verify_native_ids(tokenizer, data, serialize):
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError('No implicit tokenizer truncation is allowed')
    checked, token_count = 0, 0
    digest = hashlib.sha256()
    for split in SPLITS:
        for row in data[split]:
            text = serialize(row['messages'])
            if hashlib.sha256(text.encode()).hexdigest() != row['canonical_text_sha256']:
                raise ValueError('Native official serialization changed')
            actual = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if actual != row['ids']:
                raise ValueError('Actual modern native IDs differ; replacing IDs is forbidden: ' + row['sample_id'])
            digest.update(json.dumps([row['sample_id'], actual, row['eval_start_index']], separators=(',', ':')).encode() + b'\n')
            checked += 1; token_count += len(actual)
    if checked != 2441:
        raise ValueError('Incomplete native tokenizer verification')
    return {'status': 'passed', 'verified_rows': checked, 'verified_source_tokens': token_count,
            'all_native_ids_and_start_sha256': digest.hexdigest(), 'mismatched_rows': 0,
            'replacement_ids_written': False, 'neural_forward_calls': 0}


def reference_metrics(path, expected_sha, kind, checkpoint_sha):
    if sha(path) != expected_sha:
        raise ValueError('Historical reference metrics changed: ' + str(path))
    report = json.loads(path.read_text())
    if (report.get('status') != 'completed' or report.get('kind') != kind
            or report.get('checkpoint_sha256') != checkpoint_sha
            or report.get('selection_uses_this_benchmark') is not False
            or report.get('generated_tokens') != 0 or report.get('neural_forward_calls') != EXPECTED_UNIQUE):
        raise ValueError('Historical reference does not have the original identity/protocol')
    for split, count in COUNTS.items():
        if report['splits'][split]['coverage'] != {'requested': count, 'evaluated': count, 'excluded': {}}:
            raise ValueError('Historical reference lacks complete original-source coverage')
    return {'path': str(path.resolve()), 'sha256': expected_sha, 'original_kind': kind,
            'original_checkpoint_sha256': checkpoint_sha, 'reused_without_new_model_calls': True,
            'same_frozen_source_manifest_sha256': OFFICIAL_MANIFEST_SHA, 'metrics': report}


def classify_record(row, all_probabilities, official):
    probabilities = all_probabilities['assistant']
    if len(probabilities) != len(row['ids']):
        raise ValueError('Official protocol requires exactly one assistant risk output per real native token')
    for values in probabilities:
        if (len(values) != 3 or not all(isinstance(p, (int, float)) and not isinstance(p, bool)
                                       and math.isfinite(p) and 0 <= p <= 1 for p in values)
                or abs(sum(values) - 1) > 1e-4):
            raise ValueError('Invalid official native risk distribution')
    # Python max retains the first class on ties, matching torch.argmax.
    predictions = [max(range(3), key=values.__getitem__) for values in probabilities]
    start = row['eval_start_index']
    action, position = official.decision(predictions[start:])
    return {'status': 'ok', 'input_tokens': len(row['ids']), 'eval_start_index': start,
            'decision': action, 'decision_index_in_evaluated_native_tokens': position,
            'endpoint_decision': predictions[-1], 'endpoint_probabilities': probabilities[-1]}


def run(args, cal, locked, audit, official, report):
    summary, checkpoint, metrics, trainer, _, training_data = cal.common_inputs(args, audit)
    # Pick the already locked implementation if the caller did not explicitly name one.
    if args.inference_engine is None:
        saved = audit.read_json(args.calibration_output / 'selected/calibration.json')
        args.inference_engine = saved['engine_metadata']['inference_engine']
    if args.inference_engine not in ('eager', 'window_cuda_graph'):
        raise ValueError('Unsupported locked canonical implementation')
    documents, calibration_audit_sha = locked.locked_calibrations(args, audit, cal, summary, training_data, metrics)
    selected = documents['selected']['artifact']
    if args.pad_token_id is None:
        args.pad_token_id = selected['execution_contract']['pad_token_id']
    elif args.pad_token_id != selected['execution_contract']['pad_token_id']:
        raise ValueError('Official padding token differs from locked canonical calibration')
    del training_data
    data, input_receipt = load_official_inputs(args.data_dir, args.source_dir)
    historical = {
        'a0_original_qwen3guard_stream': reference_metrics(args.a0_metrics, A0_METRICS_SHA, 'a0', None),
        'round4_window_sft_old_bulk_diagnostic': reference_metrics(args.bulk_reference_metrics, BULK_METRICS_SHA, 'student', INITIAL_WINDOW_SHA),
    }
    report.update(phase='cpu_native_id_verification', official_input_receipt=input_receipt,
                  training_summary_sha256=sha(args.training_output / 'summary.json'),
                  checkpoint_sha256=summary['final_checkpoint_sha256'],
                  canonical_calibration_artifact_sha256=documents['selected']['artifact_sha256'],
                  canonical_calibration_audit_sha256=calibration_audit_sha,
                  execution_contract=selected['execution_contract'],
                  canonical_quality_gate_pass=selected['canonical_quality_gate_pass'],
                  original_training_eligible=summary['classification_rl']['best_metrics']['eligible'],
                  fixed_weights_retained_initial=summary['final_checkpoint_sha256'] == INITIAL_WINDOW_SHA,
                  historical_references=historical)
    audit.write_json(args.output / 'metrics.json', report)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError('Official native ID proof requires the same modern tokenizer versions')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True, trust_remote_code=False)
    native_proof = verify_native_ids(tokenizer, data, trainer.serialize)
    native_proof.update(library_versions=versions, tokenizer_class=type(tokenizer).__name__,
                        tokenizer_assets_sha256=trainer.asset_hashes(args.tokenizer_root))
    audit.write_json(args.output / 'official_native_tokenizer_proof.json', native_proof)
    report['official_native_tokenizer_proof_sha256'] = sha(args.output / 'official_native_tokenizer_proof.json')
    del tokenizer
    loader = cal.prepare_modules(args)
    torch = model = runtime = None
    began = time.monotonic()
    cache, totals = {}, {'unique_forward_sequences': 0, 'exact_prediction_cache_hits': 0,
                          'real_forward_tokens': 0, 'physical_forward_tokens': 0,
                          'physical_block_forward_calls': 0, 'padding_tokens': 0,
                          'eager_block_calls': 0, 'graph_block_calls': 0}
    try:
        torch, model, tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        if device['runtime_versions'] != audit.read_json(args.calibration_output / 'audit.json')['device']['runtime_versions']:
            raise ValueError('Official evaluation runtime versions differ from locked canonical calibration')
        metadata, sources = runtime.engine_metadata, cal.source_receipt(args)
        if metadata['execution_contract'] != selected['execution_contract']:
            raise ValueError('Official observations would use another execution contract')
        locked.matching_sources(sources, selected)
        # Recheck the actually loaded tokenizer; startup uses only a separate synthetic seed.
        native_loaded = verify_native_ids(tokenizer, data, trainer.serialize)
        if native_loaded['all_native_ids_and_start_sha256'] != native_proof['all_native_ids_and_start_sha256']:
            raise ValueError('Loaded model tokenizer differs from CPU pre-neural proof')
        report.update(phase='canonical_official_observations', device=device, engine_metadata=metadata, **sources)
        audit.write_json(args.output / 'metrics.json', report)
        for split in SPLITS:
            results, split_totals = [], {name: 0 for name in totals}
            destination = args.output / (split + '_predictions.jsonl')
            with destination.open('x') as stream:
                for index, row in enumerate(data[split]):
                    record = {key: row[key] for key in ('sample_id', 'split', 'row_index', 'unique_id', 'label')}
                    key = sequence_key(row)
                    if key in cache:
                        prediction = cache[key]
                        accounting = {'physical_forward_tokens': 0, 'physical_block_forward_calls': 0,
                                      'real_forward_tokens': 0, 'padding_tokens': 0, 'eager_block_calls': 0, 'graph_block_calls': 0}
                        split_totals['exact_prediction_cache_hits'] += 1
                        cache_hit = True
                    else:
                        values, observed = cal.validated_prefixes(runtime, row['ids'], selected['execution_contract'])
                        prediction = classify_record(row, values, official)
                        cache[key] = prediction
                        accounting = {'physical_forward_tokens': observed['forward_tokens'],
                                      'physical_block_forward_calls': observed['forward_calls'],
                                      'real_forward_tokens': observed['native_tokens'], 'padding_tokens': observed['padding_tokens'],
                                      'eager_block_calls': observed['eager_calls'], 'graph_block_calls': observed['graph_calls']}
                        split_totals['unique_forward_sequences'] += 1
                        cache_hit = False
                        del values
                    for name, value in accounting.items():
                        split_totals[name] += value
                    record.update(prediction, inference_cache_hit=cache_hit,
                                  execution_definition='canonical32-v1', execution=accounting, generated_tokens=0)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    results.append(record)
                    if (index + 1) % 100 == 0:
                        stream.flush()
                        print(json.dumps({'stage': 'canonical_official', 'split': split,
                                          'completed': index + 1, 'requested': len(data[split])}), flush=True)
            if len(results) != COUNTS[split] or any(row['status'] != 'ok' for row in results):
                raise ValueError('Official coverage is incomplete; no silent exclusions')
            summarized = official.summarize(results)
            summarized.update(execution=split_totals, predictions_sha256=sha(destination),
                              source_input_tokens_including_repeated_rows=sum(row['input_tokens'] for row in results))
            report['splits'][split] = summarized
            for name, value in split_totals.items():
                totals[name] += value
            report.update(execution_totals=dict(totals), seconds=time.monotonic() - began)
            audit.write_json(args.output / 'metrics.json', report)
        if (totals['unique_forward_sequences'] != EXPECTED_UNIQUE
                or totals['exact_prediction_cache_hits'] != EXPECTED_CACHE_HITS
                or totals['physical_forward_tokens'] != 32 * totals['physical_block_forward_calls']
                or totals['physical_forward_tokens'] != totals['real_forward_tokens'] + totals['padding_tokens']):
            raise ValueError('Official unique-sequence, fixed-block or padding accounting failed')
        final_stats = runtime.stats()
        checks = {'forward_calls': totals['physical_block_forward_calls'],
                  'forward_tokens': totals['physical_forward_tokens'],
                  'real_forward_tokens': totals['real_forward_tokens'], 'padding_tokens': totals['padding_tokens']}
        if (any(final_stats[name] != value for name, value in checks.items())
                or totals['real_forward_tokens'] != input_receipt['expected_unique_real_tokens']
                or totals['physical_block_forward_calls'] != input_receipt['expected_physical_block_calls']):
            raise ValueError('Actual engine work differs from complete unique-sequence coverage')
        after = cal.source_receipt(args)
        if not cal.sources_unchanged(sources, after) or sha(checkpoint) != summary['final_checkpoint_sha256']:
            raise ValueError('Runtime implementation or fixed weights changed during official evaluation')
        locked.matching_sources(after, selected)
        report.update(**after, final_engine_stats=final_stats)
    finally:
        if runtime is not None:
            runtime.close()
        del runtime, model
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()
    if (sha(args.calibration_output / 'audit.json') != calibration_audit_sha
            or sha(Path(documents['selected']['artifact_path'])) != documents['selected']['artifact_sha256']):
        raise ValueError('Locked canonical calibration changed during official evaluation')
    report.update(status='completed', integrity_pass=True, phase='finished', execution_totals=totals,
                  evaluated_source_rows=2441, quality_acceptance='Descriptive official adapted benchmark; never approves, promotes, or changes the fixed candidate.')


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, default=Path('/work/output/round5/prefix_v2'))
    parser.add_argument('--data-root', type=Path, default=Path('/work/round5/data/prefix_v2'))
    parser.add_argument('--calibration-output', type=Path, default=Path('/work/output/round5/canonical32_calibration'))
    parser.add_argument('--data-dir', type=Path, default=Path('/work/validation/official_adapted_v1'))
    parser.add_argument('--source-dir', type=Path, default=Path('/work/validation/qwen3guardtest'))
    parser.add_argument('--a0-metrics', type=Path, default=Path('/work/output/round4_validation/official_a0/metrics.json'))
    parser.add_argument('--bulk-reference-metrics', type=Path, default=Path('/work/output/fast_runtime/official_window/metrics.json'))
    parser.add_argument('--code-root', type=Path, default=root)
    parser.add_argument('--runtime-code', type=Path, default=root / 'runtime')
    parser.add_argument('--tokenizer-root', type=Path, default=Path('/work/models/qwen35'))
    parser.add_argument('--training-tokenizer-proof', type=Path, default=Path('/work/round5/native_tokenizer_proof.json'))
    parser.add_argument('--round4-code', type=Path, default=Path('/work/round4'))
    parser.add_argument('--base-root', type=Path, default=Path('/work'))
    parser.add_argument('--base-code-dir', type=Path, default=Path('/work/input'))
    parser.add_argument('--window-code-dir', type=Path, default=Path('/work/window'))
    parser.add_argument('--inference-engine', choices=('eager', 'window_cuda_graph'),
                        help='Default uses the already locked canonical calibration implementation')
    parser.add_argument('--pad-token-id', type=int)
    parser.add_argument('--output', type=Path, default=Path('/work/output/round5/canonical32_official'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cal = load_exact('canonical_calibration', args.code_root / 'calibrate_canonical_l20.py', CALIBRATOR_SHA)
    locked = load_exact('locked_canonical_evaluation', args.code_root / 'evaluate_canonical_fresh_l20.py', LOCKED_EVALUATOR_SHA)
    audit = cal.load_local('evaluate_final_l20', args.code_root / 'evaluate_final_l20.py', cal.EVALUATOR_SHA)
    official = load_exact('frozen_official_rules', args.round4_code / 'evaluate_official_adapted.py', OFFICIAL_EVALUATOR_SHA)
    if args.output.exists():
        raise FileExistsError('Official output exists; refusing silent overwrite or repeated test evaluation')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'integrity_pass': False, 'kind': 'round5_canonical_student', 'splits': {},
              'phase': 'fixed_training_and_canonical_calibration_gate', 'evaluator_sha256': sha(__file__),
              'official_rule_source_sha256': OFFICIAL_EVALUATOR_SHA, 'calibrator_sha256': CALIBRATOR_SHA,
              'selection_uses_this_benchmark': False, 'selection_changed': False,
              'thresholds_tuned_on_official': False, 'official_decision_uses_calibrated_thresholds': False,
              'production_approval': False, 'third_risk_class_validated': False, 'category_validated': False,
              'generated_tokens': 0, 'a0_model_calls_this_run': 0,
              'protocol': 'Frozen adapted native IDs/eval_start_index and original unsafe-priority two-consecutive argmax rule; canonical32 assistant probabilities at every real position.',
              'reuse_rule': 'Exact full native token IDs and evaluation start, within this fixed checkpoint only; 1872 forward sequences for 2441 source rows.',
              'overlap_note': 'thinking/thinking_loc overlap; report separately, never pool scores. Location splits are Unsafe-only: FPR is undefined.',
              'location_metrics': None,
              'location_note': 'Original Qwen3 token localization indices are not applied to native Qwen3.5 tokens.',
              'reference_note': 'A0 retains its original identity; prior window/SFT bulk is a historical diagnostic with its own checkpoint SHA, not a new student result.'}
    began = time.monotonic()
    audit.write_json(args.output / 'metrics.json', report)
    try:
        run(args, cal, locked, audit, official, report)
    except Exception as error:
        report.update(status='failed', integrity_pass=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        audit.write_json(args.output / 'metrics.json', report)
    return 0 if report['integrity_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
