"""L20 evaluation of complete Qwen3GuardTest splits, with explicit protocol scope.

The student uses its trained serialization and native token boundaries. This is
not a reproduction of the reference tokenizer's location/delay measurements.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def decision(values):
    for risk in (1, 2):
        for index in range(len(values) - 1):
            if values[index] == values[index + 1] == risk:
                return risk, index + 1
    return 0, -1


def binary_metrics(rows, decision_key='decision', loose=False):
    tp = fp = tn = fn = 0
    for row in rows:
        truth = row['label'] == 'Unsafe'
        pred = row[decision_key] == 1 or (loose and row[decision_key] == 2)
        tp += int(truth and pred)
        fp += int(not truth and pred)
        tn += int(not truth and not pred)
        fn += int(truth and not pred)
    return {'n': len(rows), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'recall': tp / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def summarize(rows):
    valid = [row for row in rows if row['status'] == 'ok']
    excluded = {}
    for row in rows:
        if row['status'] != 'ok':
            reason = row.get('reason', row['status'])
            excluded[reason] = excluded.get(reason, 0) + 1
    return {'coverage': {'requested': len(rows), 'evaluated': len(valid), 'excluded': excluded},
            'native_two_consecutive_strict': binary_metrics(valid),
            'native_two_consecutive_loose': binary_metrics(valid, loose=True),
            'endpoint_argmax_unsafe': binary_metrics(valid, 'endpoint_decision'),
            'location_metrics': None,
            'location_note': 'Original Qwen3 annotation token indices are not applied to native student tokens.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('student', 'a0'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    assert torch.cuda.device_count() == 1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = ['/work/input', '/work/window', '/work/round4', '/work/validation']
    selected = json.loads(args.candidate_manifest.read_text())
    if selected['status'] != 'research_candidate':
        raise RuntimeError('No completed and validated development-selected candidate')
    if args.kind == 'student':
        from infer_classifier import load_classifier, sha256_file, serialize
        checkpoint = Path(selected['checkpoint'])
        if sha256_file(checkpoint) != selected['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint checksum changed')
        setup = SimpleNamespace(base_root=Path('/work'), base_code_dir=Path('/work/input'),
                                window_code_dir=Path('/work/window'), memory_code_dir=Path('/work/round4'),
                                checkpoint=checkpoint)
        _, model, tok, _, _, _ = load_classifier(setup, selected['variant'])
    else:
        from runtime import load, logits
        model, tok = load(dtype=torch.bfloat16, device='cuda', model_path=Path('/work/models/guard'))
        model.eval().requires_grad_(False)
    summaries = {}
    prediction_cache = {}
    forward_calls = 0
    began = time.monotonic()
    with torch.inference_mode():
        for split in ('thinking', 'thinking_loc', 'response_loc'):
            prepared = read(args.data_dir / (split + '.jsonl'))
            sources = read(args.source_dir / (split + '.jsonl'))
            results = []
            destination = args.output / (split + '_predictions.jsonl')
            with destination.open('w') as handle:
                for index, row in enumerate(prepared):
                    record = {key: row[key] for key in ('sample_id', 'split', 'row_index', 'unique_id', 'label')}
                    if args.kind == 'student':
                        ids = row['ids']
                        if tok.encode(serialize(row['messages']), add_special_tokens=False) != ids:
                            raise RuntimeError('Prepared native IDs differ from the runtime tokenizer')
                        start = row['eval_start_index']
                        if row['status'] != 'ready':
                            record.update(status='excluded', reason=row['exclude_reason'])
                    else:
                        raw = sources[row['row_index']]
                        if raw['unique_id'] != row['unique_id'] or raw['label'] != row['label']:
                            raise RuntimeError('Reference source rows do not match the frozen adaptation')
                        # Keep the template's final newline: the published annotation
                        # IDs include it. The project's ordinary runtime.encode strips it.
                        text = tok.apply_chat_template(raw['message'], tokenize=False,
                                                       add_generation_prompt=False, enable_thinking=False)
                        ids = tok.encode(text, add_special_tokens=False)
                        tokens = [tok.decode([token]) for token in ids]
                        if split.startswith('thinking'):
                            position = next(i for i in range(1, len(tokens))
                                            if tokens[i - 1] == '<|im_start|>' and tokens[i] == 'assistant')
                            start = position + 4
                        else:
                            position = next(i for i in range(1, len(tokens))
                                            if tokens[i - 1] == '</think>' and tokens[i] == '\n\n')
                            start = position + 1
                        if raw['input_ids'] is not None and ids != raw['input_ids']:
                            record.update(status='excluded', reason='reference_annotation_token_mismatch')
                        if len(ids) > 8192:
                            record.update(status='excluded', reason='reference_overlength')
                    if 'status' not in record:
                        cache_key = (hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(), start)
                        if cache_key in prediction_cache:
                            record.update(prediction_cache[cache_key], inference_cache_hit=True)
                            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                            results.append(record)
                            continue
                        tensor = torch.tensor([ids], device='cuda')
                        if args.kind == 'student':
                            hidden = model(tensor).last_hidden_state[0]
                            risk, _ = model.readout(hidden, 'assistant')
                            del hidden
                        else:
                            output = model(input_ids=tensor, use_cache=False, logits_to_keep=0)
                            risk = logits(output, 'assistant')[0].float()
                            del output
                        if tuple(risk.shape) != (len(ids), 3):
                            raise RuntimeError('Benchmark requires one three-class readout per input position')
                        if not torch.isfinite(risk).all():
                            raise RuntimeError(f'Nonfinite official benchmark risk logits: {split}/{index}')
                        predictions = risk.argmax(-1).cpu().tolist()
                        endpoint = risk[-1].float().softmax(-1).cpu().tolist()
                        action, position = decision(predictions[start:])
                        record.update(status='ok', input_tokens=len(ids), eval_start_index=start,
                                      decision=action, decision_index_in_evaluated_native_tokens=position,
                                      endpoint_decision=predictions[-1], endpoint_probabilities=endpoint)
                        prediction_cache[cache_key] = {key: record[key] for key in (
                            'status', 'input_tokens', 'eval_start_index', 'decision',
                            'decision_index_in_evaluated_native_tokens', 'endpoint_decision', 'endpoint_probabilities')}
                        record['inference_cache_hit'] = False
                        forward_calls += 1
                        del risk, tensor
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    results.append(record)
                    if (index + 1) % 100 == 0:
                        handle.flush()
                        print(json.dumps({'benchmark': args.kind, 'split': split, 'completed': index + 1,
                                          'requested': len(prepared), 'seconds': time.monotonic() - began}), flush=True)
            summaries[split] = summarize(results)
            (args.output / 'metrics.json').write_text(json.dumps({
                'status': 'completed' if len(summaries) == 3 else 'running', 'kind': args.kind,
                'checkpoint_sha256': selected['checkpoint_sha256'] if args.kind == 'student' else None,
                'splits': summaries, 'selection_uses_this_benchmark': False,
                'protocol': 'Student: canonical training serialization and native two-consecutive decisions. '
                            'A0: original chat template/tokenizer and published two-consecutive decisions. '
                            'Template/token granularity differ, so this is an adapted quality comparison.',
                'overlap_note': 'thinking and thinking_loc overlap; no pooled total score. '
                                'Location splits contain no safe examples, so FPR is undefined.',
                'neural_forward_calls': forward_calls,
                'reuse_rule': 'Exact full input token IDs and evaluation start, within the same checkpoint only',
                'generated_tokens': 0, 'seconds': time.monotonic() - began}, ensure_ascii=False, indent=2) + '\n')
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
