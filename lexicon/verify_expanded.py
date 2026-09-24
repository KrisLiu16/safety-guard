"""Check expanded candidate coverage against the frozen upstream sources."""
import hashlib
import csv
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'derived/expanded-20260923'
HOUBB = ROOT / 'sources/houbb-sensitive-word-data/src/main/resources'
SPACE = ROOT / 'sources/spacegather-worldwide-sensitive'


def norm(s):
    return unicodedata.normalize('NFKC', s.strip()).casefold()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    base = {r['canonical'] for r in rows(ROOT / 'derived/d967c30b/master.jsonl')}
    added = set()
    for name in ('sensitive_word_dict.txt', 'sensitive_word_dict_en.txt'):
        added |= {norm(line) for line in (HOUBB / name).read_text(encoding='utf-8-sig').splitlines() if line.strip()}
    for language in ('zh-CN', 'zh-TW'):
        for path in (SPACE / language).glob('*.csv'):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                added |= {norm(value) for row in csv.reader(stream) for value in row if value.strip()}
    master = list(rows(OUT / 'master.jsonl'))
    keys = {r['canonical'] for r in master}
    assert keys == base | added
    assert len(keys) == len(master) == 107282
    assert all(r['review_state'] == 'unreviewed_candidate' and not r['auto_block'] for r in master)
    role_sets = {role: set((OUT / f'{role}.txt').read_text().splitlines())
                 for role in ('text', 'url_or_domain', 'combination_candidate')}
    assert set.union(*role_sets.values()) == keys
    assert all(not (role_sets[a] & role_sets[b]) for a in role_sets for b in role_sets if a < b)
    tasks = list(rows(OUT / 'task_seeds.jsonl'))
    assert len(tasks) == len(role_sets['text']) == 92492
    assert {t['canonical'] for t in tasks} == role_sets['text']
    assert len({t['task_key'] for t in tasks}) == len(tasks)
    by_key = {r['canonical']: r for r in master}
    assert by_key['维尼']['auto_block'] is False
    assert by_key['圣上']['auto_block'] is False
    assert by_key['恶搞']['houbb_allowlist_overlap']
    assert by_key['游戏机']['houbb_allowlist_overlap']
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['fwwdn_new_vs_base'] == 0
    assert summary['spacegather_new_vs_previous'] == 8
    assert sum(summary['cold_comment_rows'].values()) == 37480
    for filename, expected in summary['output_hashes'].items():
        assert hashlib.sha256((OUT / filename).read_bytes()).hexdigest() == expected, filename
    print(json.dumps({'status': 'passed', 'candidates': len(master), 'task_seeds': len(tasks),
                      'additional_vs_base': len(keys-base), 'cold_comments': 37480}, ensure_ascii=False))


if __name__ == '__main__':
    main()
