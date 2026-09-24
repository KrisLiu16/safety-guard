"""Freeze a new public-source holdout, never used for Round5 selection."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from prepare_prefix_data import build_record, encode, serialize, sha, stable_number


def text_hash(value):
    canonical = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip().casefold()
    return hashlib.sha256(canonical.encode()).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def main():
    root = Path(__file__).resolve().parent
    project = root.parent
    out = root / 'data/fresh_holdout_v1'
    if out.exists():
        raise FileExistsError('Refusing to overwrite the fresh holdout')
    from tokenizers import Tokenizer
    tokenizer_path = project / 'round3/l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    exclusions = list((project / 'round4/data/risk_v2').glob('*.jsonl'))
    for directory in (project / 'round3/data').iterdir():
        if directory.is_dir():
            exclusions += [directory / (name + '.jsonl') for name in ('train', 'dev', 'test', 'official_test')
                           if (directory / (name + '.jsonl')).is_file()]
    exclusions += [project / 'round1/data' / (name + '.jsonl')
                   for name in ('train', 'dev', 'calibration', 'test')]
    exclusions += list((project / 'round1/benchmark').glob('*.jsonl'))
    families, seen_text, excluded_ids = set(), set(), set()
    input_hashes = {}
    for path in sorted(set(exclusions)):
        input_hashes[str(path.relative_to(project))] = sha(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('family'):
                    families.add(row['family'])
                if row.get('sample_id'):
                    excluded_ids.add(row['sample_id'])
                for value in strings(row):
                    if len(value) >= 20:
                        seen_text.add(text_hash(value))
    source = project / 'round4/data/public_pools.jsonl'
    pools = defaultdict(list)
    rejected = Counter()
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['split_pool'] != 'test':
                continue
            if row['family'] in families or row['sample_id'] in excluded_ids:
                rejected['previous_family_or_id'] += 1
                continue
            if any(text_hash(m['content']) in seen_text for m in row['messages'] if len(m['content']) >= 20):
                rejected['previous_message_text'] += 1
                continue
            ids = encode(tokenizer, serialize(row['messages'])).ids
            if len(ids) > 4096:
                rejected['token_limit'] += 1
                continue
            row['ids'] = ids
            pools[row['language'] + '/' + row['target_role'] + '/' + row['source_label']].append(row)
    keys = sorted(pools, key=lambda k: (len({r['family'] for r in pools[k]}), k))
    if len(keys) != 6:
        raise ValueError('Expected exactly six existing public strata/label cells')
    # Hall's condition over the six strata gives the largest equal quota with
    # globally unique families, independent of model scores or test predictions.
    family_rows = {}
    for key in keys:
        by_family = {}
        for row in sorted(pools[key], key=lambda r: stable_number('fresh-round5:' + r['sample_id'])):
            by_family.setdefault(row['family'], row)
        family_rows[key] = by_family
    bounds = []
    for mask in range(1, 1 << len(keys)):
        subset = [keys[i] for i in range(len(keys)) if mask & (1 << i)]
        union = set().union(*(set(family_rows[k]) for k in subset))
        bounds.append((len(union) // len(subset), subset, len(union)))
    quota = min(100, min(b[0] for b in bounds))
    if quota < 1:
        raise RuntimeError('No balanced unseen-family holdout exists')
    neighbors = {k: sorted(family_rows[k], key=lambda f: stable_number('fresh-family:' + f)) for k in keys}
    family_slot, slot_family = {}, {}
    def assign(slot, visited):
        for family in neighbors[slot[0]]:
            if family in visited:
                continue
            visited.add(family)
            if family not in family_slot or assign(family_slot[family], visited):
                family_slot[family] = slot
                slot_family[slot] = family
                return True
        return False
    for key in keys:
        for index in range(quota):
            if not assign((key, index), set()):
                raise RuntimeError('Family matching failed despite feasibility bound')
    selected = [build_record(family_rows[key][family], tokenizer, 'fresh_holdout')
                for (key, index), family in slot_family.items()]
    selected.sort(key=lambda r: stable_number('fresh-output:' + r['sample_id']))
    assert len(selected) == len({r['family'] for r in selected}) == 6 * quota
    out.mkdir(parents=True)
    path = out / 'episodes.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in selected))
    manifest = {
        'version': 'round5-fresh-public-holdout-v1', 'episodes': len(selected), 'unique_families': len(selected),
        'quota_per_stratum_label': quota,
        'quota_rule': 'Largest equal quota <=100 per cell compatible with globally unique unseen families; Hall subset bound and deterministic augmenting-path assignment, before any model evaluation.',
        'limiting_family_capacity': min(bounds, key=lambda b: b[0]),
        'preparation_history': '100-per-cell availability and 64-per-cell greedy allocation attempts failed before producing outputs; exact family matching now determines the frozen quota from input availability only.',
        'strata': dict(Counter(r['language'] + '/' + r['target_role'] + '/' + r['source_label'] for r in selected)),
        'episode_sha256': sha(path), 'source_public_pool_sha256': sha(source),
        'excluded_input_hashes': input_hashes, 'tokenizer_sha256': sha(tokenizer_path),
        'prefix_builder_sha256': sha(root / 'prepare_prefix_data.py'), 'builder_sha256': sha(Path(__file__)),
        'rejections': dict(rejected), 'source_split': 'preexisting public_pools test partition only',
        'training_use': False, 'calibration_use': False, 'selection_use': False,
        'previous_model_evaluation_in_this_project': False,
        'neural_calls_during_preparation': 0,
        'limits': 'Excluded recorded families, IDs and normalized exact message text from listed inputs; semantic near-duplicates, unrecorded historical processing and foundation pretraining exposure are not exhaustively known. No independent English-user stratum. Source policies may differ.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'excluded_input_hashes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
