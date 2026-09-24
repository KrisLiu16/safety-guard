"""Bounded L20 inference, cache, and one-update training portability test."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from runtime import load, restore_adapter, encode, forward_ids, Stream, sync

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def percentile(values, percentage):
    ordered = sorted(values)
    return ordered[math.ceil(percentage * len(ordered)) - 1]


def timed(call):
    sync()
    start = time.perf_counter()
    result = call()
    sync()
    return (time.perf_counter() - start) * 1000, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    gpu = torch.cuda.get_device_name(0)
    if 'L20' not in gpu.upper():
        raise RuntimeError(f'Expected an L20, found {gpu}')
    torch.manual_seed(20260922)
    model, tokenizer = load(dtype=torch.bfloat16, device='cuda', model_path=HERE / 'model')
    restore_adapter(model, HERE / 'checkpoint')
    model.eval()
    dev = {row['sample_id']: row for row in read_jsonl(HERE / 'dev.jsonl')}
    references = json.loads((HERE / 'reference.json').read_text())
    errors, decisions = [], []
    with torch.no_grad():
        for ref in references:
            row = dev[ref['sample_id']]
            ids = encode(tokenizer, row['messages'])
            actual = forward_ids(model, ids, ref['role'])[-1].softmax(-1).cpu().tolist()
            errors.append(max(abs(a-b) for a,b in zip(actual, ref['probs'])))
            decisions.append(actual.index(max(actual)) == ref['probs'].index(max(ref['probs'])))

        row = dev[references[0]['sample_id']]
        ids = encode(tokenizer, row['messages'])
        stream = Stream(model, tokenizer)
        stream_prob = stream.update_ids(ids[:-1], row['target_role'])
        stream_prob = stream.update_ids(ids, row['target_role']).cpu()
        full_prob = forward_ids(model, ids, row['target_role'])[-1].softmax(-1).cpu()
        stream_error = (stream_prob - full_prob).abs().max().item()

        # Compare one new token with recomputing its 1K-token prefix; both include synchronization.
        long_ids = encode(tokenizer, [{'role': 'user', 'content': '请解释公开政策文本的历史背景。' * 220}])[:1024]
        if len(long_ids) < 1024:
            raise RuntimeError('Benchmark prompt did not reach 1024 tokens')
        for _ in range(3):
            state = Stream(model, tokenizer)
            state.update_ids(long_ids[:-1], 'user')
            state.update_ids(long_ids, 'user')
        cached_ms, full_ms = [], []
        for _ in range(20):
            state = Stream(model, tokenizer)
            state.update_ids(long_ids[:-1], 'user')
            ms, _ = timed(lambda: state.update_ids(long_ids, 'user'))
            cached_ms.append(ms)
        for _ in range(5):
            ms, _ = timed(lambda: forward_ids(model, long_ids, 'user'))
            full_ms.append(ms)

    # Transient update only: no checkpoint is saved and no held-out labels enter training.
    train = read_jsonl(HERE / 'train.jsonl')
    chosen = []
    for role in ('user', 'assistant'):
        for label in ('safe', 'unsafe'):
            chosen.append(next(row for row in train if row['target_role'] == role and row['label'] == label))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    rewards = []
    for row in chosen:
        logits = forward_ids(model, encode(tokenizer, row['messages']), row['target_role'])[-1]
        logp = logits.log_softmax(-1)
        p = logp.exp()
        expected = 1 if row['label'] == 'unsafe' else 0
        reward = torch.tensor([1. if expected == 0 else -2., 1. if expected == 1 else -2., -.25], device=model.device)
        actions = torch.multinomial(p.detach(), 4, replacement=True)
        advantage = reward[actions] - (p.detach() * reward).sum()
        loss = -(advantage * logp[actions]).mean() / len(chosen)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite CUDA policy loss')
        loss.backward()
        rewards.append((p.detach() * reward).sum().item())
    gradients = [p.grad for p in params if p.grad is not None]
    grad_ok = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True).item()
    optimizer.step()
    sync()

    result = {
        'gpu': gpu, 'gpu_memory_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'reference_cases': len(references), 'reference_max_probability_error': max(errors),
        'reference_decision_agreement': sum(decisions), 'stream_probability_error': stream_error,
        'policy_update_examples': len(chosen), 'finite_gradients': grad_ok, 'gradient_norm': grad_norm,
        'cached_1k_one_token_p50_ms': statistics.median(cached_ms),
        'cached_1k_one_token_p95_ms': percentile(cached_ms, .95),
        'full_1k_one_token_p50_ms': statistics.median(full_ms),
        'full_1k_one_token_p95_ms': percentile(full_ms, .95),
        'cuda_peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    result['status'] = ('passed' if grad_ok and len(references) == 16 and all(decisions)
                        and max(errors) <= .05 and stream_error <= .01 else 'failed')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
