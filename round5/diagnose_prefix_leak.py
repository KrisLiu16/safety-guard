"""CPU-only post-mortem of Round5 saved predictions: prompt-risk leak and gate noise.

Reads only files already collected under results/output/round5 and the frozen
risk_v2 / raw public data. No tokenizer, model, network or cluster calls.
Official rows are used descriptively; nothing here selects weights or thresholds.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / 'results/output/round5'
STRATA = (('zh', 'user'), ('zh', 'assistant'), ('en', 'assistant'))
CHECKPOINTS = {'initial': 'prefix_v2/sft/evaluation_0', 'sft_epoch1': 'prefix_v2/sft/evaluation_1',
               'sft_epoch2': 'prefix_v2/sft/evaluation_2', 'rl_64': 'prefix_v2/classification_rl/evaluation_64',
               'rl_256': 'prefix_v2/classification_rl/evaluation_256'}


def jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def bucket(index):
    for limit, name in ((1, '<=1'), (5, '2-5'), (20, '6-20'), (100, '21-100'), (500, '101-500')):
        if index <= limit:
            return name
    return '>500'


def official_fire_positions():
    """Where the two-consecutive-argmax rule first fires on official thinking."""
    rows = jsonl(OUT / 'canonical32_official/thinking_predictions.jsonl')
    result = {}
    for label in ('Safe', 'Unsafe'):
        chosen = [r for r in rows if r['label'] == label]
        fired = [r['decision_index_in_evaluated_native_tokens'] for r in chosen if r['decision'] == 1]
        result[label] = {'n': len(chosen), 'fired': len(fired),
                         'fire_index_buckets': dict(collections.Counter(map(bucket, fired)))}
    return result


def first_cut(row):
    observations = sorted(row['text_cut_observations'], key=lambda o: o['cut_chars'])
    return observations[0]['p_unsafe']


def quantiles(values):
    values = sorted(values)
    return {f'p{int(q * 100)}': round(values[int(q * (len(values) - 1))], 4) for q in (.5, .75, .9, .95)}


def dev_argmax_leak():
    """Canonical32 dev: risk at the first visible target character vs whole stream, at argmax (0.5)."""
    rows = jsonl(OUT / 'canonical32_calibration/initial_window/dev_predictions.jsonl')
    result = {}
    for language, role in STRATA:
        for label in ('safe', 'unsafe'):
            chosen = [r for r in rows if (r['language'], r['target_role'], r['source_label']) == (language, role, label)]
            first = [first_cut(r) for r in chosen]
            result[f'{language}/{role}/{label}'] = {
                'n': len(chosen), 'first_cut_p_unsafe': quantiles(first),
                'first_cut_above_half': sum(p > .5 for p in first) / len(chosen),
                'stream_max_above_half': sum(r['stream_max_p_unsafe'] > .5 for r in chosen) / len(chosen),
                'endpoint_above_half': sum(r['endpoint_p_unsafe'] > .5 for r in chosen) / len(chosen)}
    return result


def auc(positive, negative):
    negative = sorted(negative)
    wins = sum(bisect.bisect_left(negative, v) + .5 * (bisect.bisect_right(negative, v) - bisect.bisect_left(negative, v))
               for v in positive)
    return wins / (len(positive) * len(negative))


def recall_at_fpr(positive, negative, fpr=.05):
    """Diagnostic operating point read off dev itself; never used as a deployment threshold."""
    threshold = sorted(negative, reverse=True)[int(fpr * len(negative))]
    return sum(v > threshold for v in positive) / len(positive)


def macro(rows, ids, key, metric):
    total = 0.
    for stratum in STRATA:
        chosen = [rows[i] for i in ids if (rows[i]['language'], rows[i]['target_role']) == stratum]
        positive = [r[key] for r in chosen if r['source_label'] == 'unsafe']
        negative = [r[key] for r in chosen if r['source_label'] == 'safe']
        total += metric(positive, negative)
    return total / len(STRATA)


def checkpoint_comparison(samples, seed):
    loaded = {name: {r['sample_id']: r for r in jsonl(OUT / path / 'dev_predictions.jsonl')}
              for name, path in CHECKPOINTS.items()}
    base = loaded['initial']
    ids = sorted(base)
    families = collections.defaultdict(list)
    for sample_id in ids:
        families[base[sample_id]['family']].append(sample_id)
    names = sorted(families)
    rng = random.Random(seed)
    draws = [[i for f in rng.choices(names, k=len(names)) for i in families[f]] for _ in range(samples)]
    result = {}
    for name, rows in loaded.items():
        entry = {}
        for key, mode in (('endpoint_p_unsafe', 'whole'), ('stream_max_p_unsafe', 'stream')):
            for metric_name, metric in (('auc', auc), ('recall_at_5pct_fpr', recall_at_fpr)):
                value = macro(rows, ids, key, metric)
                entry[f'{mode}_{metric_name}'] = value
                if name != 'initial':
                    diffs = sorted(macro(rows, d, key, metric) - macro(base, d, key, metric) for d in draws)
                    entry[f'{mode}_{metric_name}_minus_initial'] = {
                        'observed': value - macro(base, ids, key, metric),
                        'ci95': [diffs[int(.025 * samples)], diffs[int(.975 * samples) - 1]]}
        result[name] = entry
    return {'family_bootstrap_samples': samples, 'seed': seed, 'families': len(names), 'checkpoints': result}


def training_label_coupling():
    """How much the prompt alone predicts the assistant label in frozen risk_v2 public data."""
    raw = {}
    for split in ('train', 'valid'):
        for row in jsonl(PROJECT / f'round3/data/raw/nemotron_safety_guard_v3_zh/{split}.jsonl'):
            raw[row['id']] = row
    candidates = {r['candidate_id']: r['source_id']
                  for r in jsonl(PROJECT / 'round3/data/candidates/nemotron_assistant_candidates.jsonl')}
    nemotron = {}
    for split in ('train', 'calibration', 'dev'):
        counts, unmapped = collections.Counter(), 0
        for row in jsonl(PROJECT / f'round4/data/risk_v2/{split}.jsonl'):
            if row['target_role'] != 'assistant' or not row['source'].startswith('nvidia/'):
                continue
            source = raw.get(candidates.get(row['sample_id'], row['sample_id']))
            if source is None:
                unmapped += 1
                continue
            counts[f'prompt_{source["prompt_label"]}->response_{row["source_label"]}'] += 1
        nemotron[split] = {'counts': dict(counts), 'unmapped': unmapped}
    prompts = collections.defaultdict(set)
    with gzip.open(PROJECT / 'round3/data/raw/beavertails_round0_30k/train.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            prompts[row['prompt']].add(bool(row['is_safe']))
    beaver = {'prompts': len(prompts), 'mixed_label_prompts': sum(len(v) == 2 for v in prompts.values()),
              'all_unsafe_prompts': sum(v == {False} for v in prompts.values()),
              'all_safe_prompts': sum(v == {True} for v in prompts.values())}
    return {'nemotron_zh_assistant': nemotron, 'beavertails_round0_30k': beaver}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnosis/prefix_leak_diagnosis.json')
    args = parser.parse_args()
    report = {'version': 'round5-prefix-leak-diagnosis-v1', 'model_calls': 0, 'network_calls': 0,
              'official_thinking_fire_positions': official_fire_positions(),
              'dev_canonical32_initial_argmax_leak': dev_argmax_leak(),
              'dev_checkpoint_threshold_free_comparison': checkpoint_comparison(args.bootstrap, args.seed),
              'training_prompt_response_label_coupling': training_label_coupling()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
