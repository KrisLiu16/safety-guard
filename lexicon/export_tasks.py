"""Turn text candidates into stable future generation task seeds.

This does not call a model, schedule cloud work, or assign safety labels.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'derived/d967c30b/master.jsonl'
TARGET = ROOT / 'derived/d967c30b/task_seeds.jsonl'
MANIFEST = ROOT / 'derived/d967c30b/task_seeds_manifest.json'


def main():
    if TARGET.exists() or MANIFEST.exists():
        raise FileExistsError('Task export already exists; inspect before replacing')
    count = 0
    with SOURCE.open(encoding='utf-8') as source, TARGET.open('w', encoding='utf-8') as target:
        for line in source:
            row = json.loads(line)
            if row['kinds'] != ['text']:
                continue
            task_key = 'lex-' + row['id']
            task = {
                'task_key': task_key, 'origin_group_id': task_key,
                'word': row['display'], 'canonical': row['canonical'],
                'aliases': row['variants'], 'source_refs': row['source_refs'],
                'source_revision': 'd967c30b053fa40b06c5a0dddf0be493f2dfae46',
                'review_state': 'unreviewed_candidate', 'generation_state': 'not_started',
                'license_notice': 'sources/sensitive-lexicon/LICENSE',
            }
            target.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    MANIFEST.write_text(json.dumps({'count': count, 'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                                    'model_requests_submitted': 0, 'safety_labels_assigned': 0},
                                   ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'task_seeds': count, 'model_requests_submitted': 0}))


if __name__ == '__main__':
    main()
