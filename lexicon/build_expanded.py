"""Merge licensed candidate lists while keeping source and role boundaries."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'derived/d967c30b'
OUT = ROOT / 'derived/expanded-20260923'
HOUBB = ROOT / 'sources/houbb-sensitive-word-data/src/main/resources'
FWWDN = ROOT / 'sources/fwwdn-sensitive-stop-words'
SPACE = ROOT / 'sources/spacegather-worldwide-sensitive'
COLD = ROOT / 'corpora/COLDataset/COLDataset'


def norm(value):
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def check_locks():
    base_lock = json.loads((ROOT / 'source_lock.json').read_text())
    locks = json.loads((ROOT / 'additional_source_lock.json').read_text())
    for lock in [base_lock] + locks:
        source = ROOT / lock.get('local_path', 'sources/sensitive-lexicon')
        actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
        expected = set(lock['files'])
        if actual != expected:
            raise ValueError(f'Source file set changed: {lock["source_id"]}')
        for name, info in lock['files'].items():
            if sha(source / name) != info['sha256']:
                raise ValueError(f'Source hash changed: {lock["source_id"]}/{name}')
    return locks


def mirror_terms():
    result = []
    for path in sorted(FWWDN.glob('*.txt')):
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            parts = line.split(',') if path.name in ('政治类.txt', '色情类.txt') else [line]
            result.extend(norm(piece) for piece in parts if piece.strip())
    return result


def main():
    locks = check_locks()
    if OUT.exists():
        raise FileExistsError(f'Immutable build already exists: {OUT}')
    records = {}
    for base in jsonl(BASE / 'master.jsonl'):
        key = base['canonical']
        records[key] = {
            'id': base['id'], 'canonical': key, 'display': base['display'],
            'variants': set(base['variants']), 'kinds': set(base['kinds']),
            'refs': {f'sensitive-lexicon/{r["path"]}': list(r['positions']) for r in base['source_refs']},
            'occurrences': base['occurrences'], 'houbb_tags': set(), 'english_aux': False,
        }
    base_keys = set(records)
    fww = mirror_terms()
    fww_unique = set(fww)
    if fww_unique - base_keys:
        raise ValueError('fwwdn mirror unexpectedly added terms; inspect before changing merge policy')

    source_count = 0
    for name, english in [('sensitive_word_dict.txt', False), ('sensitive_word_dict_en.txt', True)]:
        path = HOUBB / name
        source_path = f'houbb-sensitive-word-data/src/main/resources/{name}'
        for position, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            raw = line.strip()
            if not raw:
                continue
            key = norm(raw)
            if key not in records:
                records[key] = {'id': hashlib.sha256(key.encode()).hexdigest()[:20],
                                'canonical': key, 'display': raw, 'variants': set(),
                                'kinds': {'text'}, 'refs': {}, 'occurrences': 0,
                                'houbb_tags': set(), 'english_aux': False}
            row = records[key]
            row['variants'].add(raw)
            row['kinds'].add('text')
            row['refs'].setdefault(source_path, []).append(position)
            row['occurrences'] += 1
            row['english_aux'] |= english
            source_count += 1

    before_space = set(records)
    space_count = 0
    for language in ('zh-CN', 'zh-TW'):
        for path in sorted((SPACE / language).glob('*.csv')):
            source_path = f'spacegather-worldwide-sensitive/{language}/{path.name}'
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for position, values in enumerate(csv.reader(stream), 1):
                    for value in values:
                        raw = value.strip()
                        if not raw:
                            continue
                        key = norm(raw)
                        if key not in records:
                            records[key] = {'id': hashlib.sha256(key.encode()).hexdigest()[:20],
                                            'canonical': key, 'display': raw, 'variants': set(),
                                            'kinds': {'text'}, 'refs': {}, 'occurrences': 0,
                                            'houbb_tags': set(), 'english_aux': False}
                        row = records[key]
                        row['variants'].add(raw)
                        row['kinds'].add('text')
                        row['refs'].setdefault(source_path, []).append(position)
                        row['occurrences'] += 1
                        space_count += 1

    tags = {}
    for line in (HOUBB / 'sensitive_word_tags.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        word, codes = line.rsplit(' ', 1)
        tags.setdefault(norm(word), set()).update(codes.split(','))
    tag_only = len(set(tags) - set(records))
    for key, codes in tags.items():
        if key in records:
            records[key]['houbb_tags'].update(codes)
    allow = {norm(line) for line in (HOUBB / 'sensitive_word_allow.txt').read_text(encoding='utf-8-sig').splitlines()
             if line.strip()}
    allow_overlap = allow & set(records)

    OUT.mkdir(parents=True)
    master = []
    groups = {'text': [], 'url_or_domain': [], 'combination_candidate': []}
    tasks = []
    for key in sorted(records):
        item = records[key]
        kinds = item['kinds']
        role = ('combination_candidate' if 'combination_candidate' in kinds else
                'url_or_domain' if 'url_or_domain' in kinds else 'text')
        row = {'id': item['id'], 'canonical': key, 'display': item['display'],
               'variants': sorted(item['variants']), 'kinds': sorted(kinds), 'role': role,
               'source_refs': [{'path': path, 'positions': sorted(positions)}
                               for path, positions in sorted(item['refs'].items())],
               'occurrences': item['occurrences'], 'houbb_tag_codes': sorted(item['houbb_tags']),
               'houbb_allowlist_overlap': key in allow_overlap,
               'english_aux_source': item['english_aux'],
               'review_state': 'unreviewed_candidate', 'auto_block': False}
        master.append(row)
        groups[role].append(key)
        if role == 'text':
            task_key = 'lex-' + row['id']
            tasks.append({'task_key': task_key, 'origin_group_id': task_key,
                          'word': row['display'], 'canonical': key,
                          'source_refs': row['source_refs'], 'review_state': row['review_state'],
                          'generation_state': 'not_started', 'auto_block': False})
    write_jsonl(OUT / 'master.jsonl', master)
    write_jsonl(OUT / 'task_seeds.jsonl', tasks)
    for role, values in groups.items():
        (OUT / f'{role}.txt').write_text('\n'.join(values) + '\n', encoding='utf-8')
    write_jsonl(OUT / 'allowlist_controls.jsonl',
                [{'canonical': key, 'in_candidate_master': key in records,
                  'source': 'houbb-sensitive-word-data/src/main/resources/sensitive_word_allow.txt'}
                 for key in sorted(allow)])

    cold = {}
    for path in sorted(COLD.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            cold[path.name] = sum(1 for _ in csv.DictReader(stream))
    summary = {
        'sources': [{'id': 'konsheng/Sensitive-lexicon', 'revision': 'd967c30b053fa40b06c5a0dddf0be493f2dfae46',
                     'declared_license': 'MIT', 'merge_role': 'candidate'},
                    *[{'id': x['source_id'], 'revision': x['revision'],
                       'declared_license': x['declared_license'], 'merge_role': x['role']} for x in locks]],
        'base_candidates': len(base_keys), 'houbb_main_entries': source_count - 12,
        'houbb_aux_english_entries': 12,
        'new_candidates_vs_base': len(records) - len(base_keys),
        'spacegather_chinese_entries': space_count,
        'spacegather_new_vs_previous': len(set(records) - before_space),
        'total_candidates': len(records),
        'role_counts': {name: len(values) for name, values in groups.items()},
        'generation_task_seeds': len(tasks), 'model_requests_submitted': 0,
        'fwwdn_rows': len(fww), 'fwwdn_unique': len(fww_unique), 'fwwdn_new_vs_base': 0,
        'houbb_tag_keys': len(tags), 'houbb_tag_only_keys_not_merged': tag_only,
        'allowlist_control_count': len(allow), 'allowlist_overlap_with_master': len(allow_overlap),
        'cold_comment_rows': cold, 'cold_role': 'independent_context_benchmark_not_training',
        'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'meaning': 'Unreviewed source candidates, not legal findings or auto-block rules',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('base_candidates', 'new_candidates_vs_base',
                                             'total_candidates', 'role_counts',
                                             'generation_task_seeds', 'fwwdn_new_vs_base')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
