"""Build an auditable candidate lexicon from frozen upstream files.

The output is a source index for data generation and review, not an auto-block list.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'sources/sensitive-lexicon'
LOCK = ROOT / 'source_lock.json'
OUTPUT = ROOT / 'derived/d967c30b'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(value):
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def kind(path, raw):
    if path == 'Vocabulary/非法网址.txt':
        return 'url_or_domain'
    if '+' in raw:
        return 'combination_candidate'
    return 'text'


def emit(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    lock = json.loads(LOCK.read_text())
    actual = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
    expected = set(lock['files'])
    if actual != expected:
        raise ValueError(f'Source file set changed: added={sorted(actual-expected)}, missing={sorted(expected-actual)}')
    for name, info in lock['files'].items():
        if digest(SOURCE / name) != info['sha256']:
            raise ValueError(f'Source hash changed: {name}')
    if OUTPUT.exists():
        raise FileExistsError(f'Immutable build already exists: {OUTPUT}')

    entries = []
    per_file = {}
    for file in sorted((SOURCE / 'Vocabulary').glob('*.txt')):
        name = str(file.relative_to(SOURCE))
        lines = file.read_text(encoding='utf-8-sig').splitlines()
        used = 0
        for number, line in enumerate(lines, 1):
            raw = line.strip()
            if not raw:
                continue
            key = normalize(raw)
            if not key:
                raise ValueError(f'Empty normalized entry: {name}:{number}')
            entries.append({'source': name, 'position': number, 'raw': raw, 'key': key,
                            'kind': kind(name, raw)})
            used += 1
        per_file[name] = {'lines': len(lines), 'nonempty_entries': used,
                          'sha256': lock['files'][name]['sha256']}

    trchat_path = 'ThirdPartyCompatibleFormats/TrChat/SensitiveLexicon.json'
    trchat = json.loads((SOURCE / trchat_path).read_text(encoding='utf-8'))
    if not isinstance(trchat.get('words'), list):
        raise ValueError('Unexpected TrChat JSON shape')
    trchat_count = 0
    for number, value in enumerate(trchat['words'], 1):
        if not isinstance(value, str):
            raise TypeError(f'Non-string TrChat word at index {number}')
        raw = value.strip()
        if not raw:
            continue
        key = normalize(raw)
        if not key:
            raise ValueError(f'Empty normalized TrChat word at index {number}')
        entries.append({'source': trchat_path, 'position': number, 'raw': raw, 'key': key,
                        'kind': kind(trchat_path, raw)})
        trchat_count += 1
    per_file[trchat_path] = {'lines': len(trchat['words']), 'nonempty_entries': trchat_count,
                             'sha256': lock['files'][trchat_path]['sha256'],
                             'lastUpdateDate': trchat.get('lastUpdateDate')}

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry['key']].append(entry)
    master = []
    for key in sorted(grouped):
        rows = grouped[key]
        references = defaultdict(list)
        for row in rows:
            references[row['source']].append(row['position'])
        master.append({
            'id': hashlib.sha256(key.encode('utf-8')).hexdigest()[:20],
            'canonical': key,
            'display': rows[0]['raw'],
            'variants': sorted({row['raw'] for row in rows}),
            'kinds': sorted({row['kind'] for row in rows}),
            'source_refs': [{'path': path, 'positions': positions}
                            for path, positions in sorted(references.items())],
            'occurrences': len(rows),
            'review_state': 'unreviewed_candidate',
            'auto_block': False,
        })

    OUTPUT.mkdir(parents=True)
    emit(OUTPUT / 'source_rows.jsonl', entries)
    emit(OUTPUT / 'master.jsonl', master)
    # A URL listed again in another source remains a URL indicator, not a text task.
    text_terms = [row['canonical'] for row in master if row['kinds'] == ['text']]
    urls = [row['canonical'] for row in master if 'url_or_domain' in row['kinds']]
    combinations = [row for row in master if 'combination_candidate' in row['kinds']]
    (OUTPUT / 'text_candidates.txt').write_text('\n'.join(text_terms) + '\n', encoding='utf-8')
    (OUTPUT / 'url_indicators.txt').write_text('\n'.join(urls) + '\n', encoding='utf-8')
    emit(OUTPUT / 'combination_candidates.jsonl', combinations)

    seed_file = ROOT.parent / 'pilot/seeds.jsonl'
    seeds = [json.loads(line) for line in seed_file.read_text().splitlines() if line]
    seed_keys = {normalize(row['word']) for row in seeds}
    exact_unique = len({row['raw'] for row in entries})
    summary = {
        'source_id': lock['source_id'], 'revision': lock['revision'],
        'declared_license': lock['declared_license'],
        'raw_files': len(lock['files']), 'vocabulary_files': len(per_file) - 1,
        'vocabulary_nonempty_entries': sum(v['nonempty_entries'] for k, v in per_file.items()
                                              if k.startswith('Vocabulary/')),
        'trchat_nonempty_entries': trchat_count,
        'all_source_entries': len(entries), 'exact_unique_entries': exact_unique,
        'normalized_unique_entries': len(master),
        'text_candidates': len(text_terms), 'url_indicators': len(urls),
        'combination_candidates': len(combinations),
        'selected_seed_count': len(seed_keys), 'selected_seed_coverage': len(seed_keys & set(grouped)),
        'known_examples': {term: term in grouped for term in ('维尼', '圣上')},
        'per_file': per_file,
        'outputs': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                    for p in sorted(OUTPUT.iterdir()) if p.is_file()},
        'meaning': 'Unreviewed candidate terms with provenance; not policy labels or an auto-block list',
    }
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('vocabulary_nonempty_entries', 'trchat_nonempty_entries',
                                              'all_source_entries', 'exact_unique_entries',
                                              'normalized_unique_entries', 'text_candidates',
                                              'url_indicators', 'combination_candidates',
                                              'selected_seed_coverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
