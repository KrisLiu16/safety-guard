"""Independently check that the frozen raw sources survive consolidation."""
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'sources/sensitive-lexicon'
DERIVED = ROOT / 'derived/d967c30b'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def normalize(text):
    return unicodedata.normalize('NFKC', text.strip()).casefold()


def main():
    independent = []
    for path in sorted((SOURCE / 'Vocabulary').glob('*.txt')):
        independent += [normalize(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    trchat = json.loads((SOURCE / 'ThirdPartyCompatibleFormats/TrChat/SensitiveLexicon.json').read_text())
    independent += [normalize(word) for word in trchat['words'] if word.strip()]
    source_rows = list(rows(DERIVED / 'source_rows.jsonl'))
    master = list(rows(DERIVED / 'master.jsonl'))
    assert len(independent) == len(source_rows) == 89452
    assert len(set(independent)) == len(master) == 51085
    assert set(independent) == {row['canonical'] for row in master}
    assert sum(row['occurrences'] for row in master) == len(source_rows)
    assert all(not row['auto_block'] and row['review_state'] == 'unreviewed_candidate' for row in master)
    assert sum(len(ref['positions']) for row in master for ref in row['source_refs']) == len(source_rows)
    text = set((DERIVED / 'text_candidates.txt').read_text().splitlines())
    urls = set((DERIVED / 'url_indicators.txt').read_text().splitlines())
    combos = {row['canonical'] for row in rows(DERIVED / 'combination_candidates.jsonl')}
    assert not (text & urls or text & combos or urls & combos)
    assert text | urls | combos == {row['canonical'] for row in master}
    by_key = {row['canonical']: row for row in master}
    for term in ('维尼', '圣上', '刘广智', '廖伯年'):
        assert term in by_key
    for term in ('刘广智', '廖伯年'):
        assert any(ref['path'].startswith('ThirdPartyCompatibleFormats/') for ref in by_key[term]['source_refs'])
    summary = json.loads((DERIVED / 'summary.json').read_text())
    for name, expected in summary['outputs'].items():
        actual = hashlib.sha256((DERIVED / name).read_bytes()).hexdigest()
        assert actual == expected['sha256'], name
    print(json.dumps({'status': 'passed', 'source_rows': len(source_rows),
                      'normalized_candidates': len(master), 'source_files': summary['raw_files']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
