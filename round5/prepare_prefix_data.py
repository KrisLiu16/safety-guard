"""CPU-only derived prefix supervision; raw v12 and frozen Round4 stay immutable."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SUFFIXES = ("\n\n[END]\n", "\n\n---\n", "\n\n[End of entry]\n", "\n\n[记录结束]\n")
MAX_ANCHORS = 32


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def serialize(messages):
    return '\n\n'.join(m['role'].upper() + ':\n' + m['content'] for m in messages)


def stable_number(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def choose_positions(values, maximum=MAX_ANCHORS):
    values = sorted(set(values))
    if len(values) <= maximum:
        return values
    # Always retain the earliest observed content and late positions.
    selected = set(values[:4] + values[-2:])
    for i in range(maximum - 6):
        selected.add(values[round(i * (len(values) - 1) / (maximum - 7))])
    for value in values:
        if len(selected) >= maximum:
            break
        selected.add(value)
    return sorted(selected)


def text_cuts(length):
    if length <= 0:
        return []
    return sorted({max(1, min(length, n)) for n in
                   (1, 2, length // 4, length // 2, 3 * length // 4,
                    length - 2, length - 1, length)})


def encode(tokenizer, text):
    result = tokenizer.encode(text, add_special_tokens=False)
    if result.overflowing or not result.ids or len(result.ids) > 8192:
        raise ValueError('No empty input, implicit truncation or over-limit view allowed')
    return result


def build_record(row, tokenizer, split):
    messages = row['messages']
    if messages[-1]['role'] != row['target_role']:
        raise ValueError('Last message is not the annotated target')
    if row['source_label'] not in ('safe', 'unsafe'):
        raise ValueError('Unknown source label')
    text = serialize(messages)
    content = messages[-1]['content']
    content_start = len(text) - len(content)
    original = encode(tokenizer, text)
    if original.ids != row['ids']:
        raise ValueError('Original frozen IDs differ: ' + row['sample_id'])
    target_positions = [i for i, (_, end) in enumerate(original.offsets)
                        if end > content_start]
    result = dict(row)
    result.update(base_split=split, original_ids_sha256=ids_sha(row['ids']),
                  target_content_start_char=content_start,
                  target_token_positions=target_positions,
                  prefix_training_version='prefix-supervision-v2')
    if split == 'train':
        # Whole Safe is NOT a reliable hard label for every prefix. All inherited
        # interior labels, including rubric templates, remain explicitly weak.
        weak_positions = choose_positions(i for i in target_positions if i < len(row['ids']) - 1)
        result['anchors'] = ([{'token_end_exclusive': i + 1, 'label': 0,
                               'confidence': 0.1, 'label_origin': 'whole_safe_inherited_weak'}
                              for i in weak_positions] if row['source_label'] == 'safe' else [])
        suffix = SUFFIXES[stable_number('suffix:' + row['sample_id']) % len(SUFFIXES)]
        augmented_messages = [dict(m) for m in messages]
        augmented_messages[-1]['content'] += suffix
        augmented = encode(tokenizer, serialize(augmented_messages))
        # Prefix contains the complete original target before receiving a hard
        # source-label inheritance. This does not certify that source labels are gold.
        # Byte-level tokenizers can give several partial UTF-8 tokens the same
        # character end offset. Offsets alone do not prove complete visibility.
        after_complete = choose_positions(i for i, (_, end) in enumerate(augmented.offsets)
                                          if end >= len(text) and end > content_start
                                          and tokenizer.decode(augmented.ids[:i + 1],
                                                               skip_special_tokens=False).startswith(text))
        result['augmentation'] = {
            'kind': 'neutral_suffix_after_complete_target', 'suffix': suffix,
            'ids': augmented.ids, 'ids_sha256': ids_sha(augmented.ids),
            'epoch_phase': stable_number('augmentation-phase:' + row['sample_id']) % 2,
            'anchors': [{'token_end_exclusive': i + 1,
                         'label': int(row['source_label'] == 'unsafe'), 'confidence': 1.0,
                         'label_origin': 'complete_target_plus_neutral_suffix'} for i in after_complete],
            'original_endpoint_reused': False,
            'complete_target_verified_by': 'exact decoded causal prefix starts with original serialized text',
        }
        if not after_complete:
            raise ValueError('Neutral suffix produced no complete-target anchors')
    else:
        result['text_cut_views'] = []
        for cut in text_cuts(len(content)):
            cut_messages = [dict(m) for m in messages]
            cut_messages[-1]['content'] = content[:cut]
            ids = encode(tokenizer, serialize(cut_messages)).ids
            result['text_cut_views'].append({'cut_chars': cut, 'ids': ids, 'ids_sha256': ids_sha(ids)})
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=root.parent / 'round4/data/risk_v2')
    parser.add_argument('--tokenizer-json', type=Path,
                        default=root.parent / 'round3/l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json')
    parser.add_argument('--output', type=Path, default=root / 'data/prefix_v2')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite derived prefix inputs')
    from tokenizers import Tokenizer
    import tokenizers
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    if tokenizer.truncation is not None:
        raise ValueError('Tokenizer truncation must be disabled')
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    if sha(args.tokenizer_json) != source_manifest['tokenizer_sha256']:
        raise ValueError('Reference tokenizer identity mismatch')
    args.output.mkdir(parents=True)
    counts, hashes, tokens, statistics, families = {}, {}, {}, {}, {}
    for split in ('train', 'calibration', 'dev'):
        source_path = args.source / (split + '.jsonl')
        if sha(source_path) != source_manifest['output_hashes'][split]:
            raise ValueError('Frozen source hash mismatch: ' + split)
        count, token_count, stats = 0, 0, Counter()
        families[split] = set()
        destination = args.output / (split + '.jsonl')
        with source_path.open() as incoming, destination.open('w') as outgoing:
            for line in incoming:
                row = json.loads(line)
                prepared = build_record(row, tokenizer, split)
                outgoing.write(json.dumps(prepared, ensure_ascii=False, separators=(',', ':')) + '\n')
                count += 1
                token_count += len(row['ids'])
                families[split].add(row['family'])
                stats['target_empty'] += not bool(prepared['target_token_positions'])
                if split == 'train':
                    stats['weak_safe_anchors'] += len(prepared['anchors'])
                    stats['suffix_anchors_' + row['source_label']] += len(prepared['augmentation']['anchors'])
                    stats['phase_' + str(prepared['augmentation']['epoch_phase']) + '_' + row['source_label']] += 1
                else:
                    stats['text_cut_views'] += len(prepared['text_cut_views'])
        counts[split], tokens[split], hashes[split] = count, token_count, sha(destination)
        statistics[split] = dict(stats)
    for a in families:
        for b in families:
            if a < b and families[a] & families[b]:
                raise ValueError('Family split leakage')
    manifest = {
        'version': 'prefix-supervision-v2', 'counts': counts, 'original_tokens': tokens,
        'output_hashes': hashes, 'statistics': statistics,
        'source_manifest_sha256': sha(args.source / 'manifest.json'),
        'source_split_sha256': {k: source_manifest['output_hashes'][k] for k in counts},
        'tokenizer_sha256': sha(args.tokenizer_json), 'builder_tokenizers_version': tokenizers.__version__,
        'builder_sha256': sha(Path(__file__)), 'suffixes': SUFFIXES,
        'training_original_endpoint_separate_forward': True,
        'augmentation_schedule': 'epoch % 2 == epoch_phase; exactly once per record across two epochs',
        'source_labels_and_weights_unchanged': True, 'original_v12_changed': False,
        'unknown_unsafe_interior_anchors': 0, 'teacher_model_calls': 0,
        'test_or_official_predictions_read': False, 'family_disjoint': True,
        'runtime_requirement': 'Before L20 neural work, verify every original/augmented/text-cut ID using actual modern AutoTokenizer; never replace saved IDs on mismatch.',
        'limits': 'Whole-safe interior labels are weak (0.1); full-target labels retain original source uncertainty and weights. No natural unsafe onset annotation. Calibration measures episodes on fixed native and up-to-eight text-cut trajectories only.',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
