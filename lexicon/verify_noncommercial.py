"""Check the Citizen Lab research layer and its boundary from permissive sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_nc/citizenlab-1eeb5e6'
ARCHIVE = ROOT / 'research_sources/citizenlab/chat-censorship-1eeb5e6.tar.gz'
BASE = ROOT / 'derived/expanded-20260923/master.jsonl'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    meta = json.loads((ARCHIVE.parent / 'source_manifest.json').read_text())
    assert digest(ARCHIVE) == meta['sha256'] == summary['archive_sha256']
    for filename, expected in summary['output_hashes'].items():
        assert digest(OUT / filename) == expected, filename
    base = {r['canonical'] for r in rows(BASE)}
    research = set((OUT / 'candidate_keys.txt').read_text().splitlines())
    combined = set((OUT / 'combined_candidate_keys.txt').read_text().splitlines())
    assert len(base) == 107282
    assert len(research) == summary['research_source_candidates'] == 477165
    assert combined == base | research
    assert len(combined) == summary['combined_noncommercial_candidates'] == 492640
    assert len(combined-base) == summary['research_new_vs_permissive_base'] == 385358
    catalog = {r['path'] for r in rows(OUT / 'source_file_catalog.jsonl')}
    assert len(catalog) == summary['selected_source_files'] == 1968
    seen = set()
    new_text = set()
    count = 0
    for row in rows(OUT / 'candidate_terms.jsonl'):
        key = row['canonical']
        assert key in research and key not in seen
        assert row['candidate_status'] == 'source_recorded_unlabeled'
        assert row['source_license'] == 'CC BY-NC-SA 4.0' and not row['auto_block']
        assert row['generation_eligible'] == (row['role'] == 'text')
        assert row['source_examples'] and all(x['path'] in catalog for x in row['source_examples'])
        if row['new_vs_permissive_base'] and row['role'] == 'text':
            new_text.add(key)
        seen.add(key)
        count += 1
    assert count == len(research)
    task_keys = set()
    for task in rows(OUT / 'new_task_seeds.jsonl'):
        assert task['canonical'] in new_text
        assert task['generation_state'] == 'not_started' and not task['auto_block']
        task_keys.add(task['canonical'])
    assert task_keys == new_text
    assert len(task_keys) == summary['new_text_task_seeds'] == 357083
    task_manifest = json.loads((OUT / 'combined_task_seeds_manifest.json').read_text())
    assert digest(OUT / 'combined_task_seeds.jsonl') == task_manifest['sha256']
    assert task_manifest['tasks'] == 449575
    assert task_manifest['by_layer'] == {'permissive': 92492, 'noncommercial': 357083}
    assert not task_manifest['manual_review_required_for_candidate_inclusion']
    assert '维尼' in combined and '圣上' in combined
    assert summary['model_requests_submitted'] == 0
    print(json.dumps({'status': 'passed', 'research_candidates': count,
                      'combined_candidates': len(combined), 'new_task_seeds': len(task_keys),
                      'source_files': len(catalog)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
