"""Audit exported Round4 weights beyond the local window, on L20 only.

This is an additional validation artifact. It does not change training, labels,
checkpoint selection, or the sealed test. Numerical failures block promotion.
"""
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path[:0] = ['/work/input', '/work/window', '/work/round4']
import torch
from train_risk import make_model, configure, OUT, DATA
from experiment_common import read, write, sha
from memory_attention import memory_state_bytes
from window_attention import cache_storage
from run_probe import short_recurrent

PREFIXES = (513, 1025, 4097, 8192)
SCHEDULES = ([257, 1, 7, 127, 31, 512], [1024, 32, 8, 1])
TOLERANCE = .03


def input_tokens(rows, separator, reverse=False):
    ordered = sorted(rows, key=lambda row: row['sample_id'], reverse=reverse)
    tokens = []
    for row in ordered:
        tokens.extend(row['ids'])
        tokens.extend(separator)
        if len(tokens) >= PREFIXES[-1]:
            return tokens[:PREFIXES[-1]]
    raise ValueError('Insufficient development tokens for the long-stream audit')


@torch.inference_mode()
def audit_checkpoint(name, variant, checkpoint, rows):
    model, tok = make_model(variant, checkpoint)
    model.eval().requires_grad_(False)
    configure(model, variant, 512)
    # load() resets kernel wrappers, so apply the short-token optimization after it.
    short_recurrent(True)
    gates = [float(p.detach().abs().max()) for n, p in model.named_parameters()
             if 'memory_gate' in n]
    records = []
    input_hashes = []
    separator = tok.encode('\n\n--- NEXT MESSAGE ---\n\n', add_special_tokens=False)
    for reverse in (False, True):
        tokens = input_tokens(rows, separator, reverse)
        input_hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        ids = torch.tensor([tokens], device='cuda')
        expected = {}
        for length in PREFIXES:
            output = model(ids[:, :length])
            expected[length] = {
                role: model.readout(output.last_hidden_state[:, -1], role)[0].softmax(-1).cpu()
                for role in ('user', 'assistant')
            }
            del output
        for schedule in SCHEDULES:
            cache = None
            offset = 0
            iteration = 0
            for length in PREFIXES:
                while offset < length:
                    end = min(length, offset + schedule[iteration % len(schedule)])
                    output = model(ids[:, offset:end], past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    # Classify each new chunk using both role heads, without token generation.
                    observed = {
                        role: model.readout(output.last_hidden_state[:, -1], role)[0].softmax(-1).cpu()
                        for role in ('user', 'assistant')
                    }
                    del output
                    offset = end
                    iteration += 1
                errors = {role: float((observed[role] - expected[length][role]).abs().max())
                          for role in observed}
                extra = memory_state_bytes(cache) if hasattr(cache, 'memories') else 0
                records.append({'input_order': 'reverse' if reverse else 'forward',
                                'schedule': schedule, 'prefix_tokens': length,
                                'risk_probability_errors': errors,
                                'state_bytes': cache_storage(cache)['unique_storage_bytes'] + extra,
                                'memory_bytes': extra})
            del cache
        del ids, expected
    all_errors = [error for row in records for error in row['risk_probability_errors'].values()]
    finite = all(math.isfinite(error) for error in all_errors)
    max_error = max(all_errors) if finite else None
    bounded = True
    if variant != 'full':
        for reverse in (False, True):
            for schedule in SCHEDULES:
                sizes = [r['state_bytes'] for r in records
                         if r['input_order'] == ('reverse' if reverse else 'forward')
                         and r['schedule'] == schedule]
                bounded = bounded and len(set(sizes)) == 1
    result = {'name': name, 'variant': variant, 'checkpoint_sha256': sha(checkpoint),
              'max_probability_error': max_error, 'probability_tolerance': TOLERANCE,
              'long_stream_parity_pass': finite and max_error < TOLERANCE,
              'bounded_state_pass': bounded if variant != 'full' else None,
              'pass': finite and max_error < TOLERANCE and bounded,
              'absolute_memory_gates': gates, 'input_token_hashes': input_hashes,
              'records': records}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.device_count() == 1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    selected = json.loads((OUT / 'final_summary.json').read_text())
    rows = read(DATA / 'dev.jsonl')
    candidates = [(name, name, OUT / name / 'best.safetensors') for name in ('full', 'window', 'memory')]
    candidates.append(('classification_rl', selected['rl_base_variant'],
                       OUT / 'classification_rl/best.safetensors'))
    results = {}
    for name, variant, checkpoint in candidates:
        try:
            results[name] = audit_checkpoint(name, variant, checkpoint, rows)
        except Exception as error:
            results[name] = {'name': name, 'variant': variant, 'pass': False,
                             'error_type': type(error).__name__, 'error': str(error)}
            gc.collect()
            torch.cuda.empty_cache()
        write(OUT / 'long_stream_audit.json', {
            'status': 'completed' if len(results) == len(candidates) else 'running',
            'all_candidates_pass': all(r['pass'] for r in results.values()),
            'device': torch.cuda.get_device_name(0), 'window': 512,
            'prefix_lengths': PREFIXES, 'generated_tokens': 0, 'results': results,
            'scope': 'Exported weights; exact token-ID numerical parity and cache bounds. '
                     'Development-message concatenations have no long-context semantic gold labels. '
                     'No arbitrary-text rollback, HTTP, or production approval.'})
        print(json.dumps({'long_stream_audit': name, 'pass': results[name]['pass']}), flush=True)


if __name__ == '__main__':
    main()
