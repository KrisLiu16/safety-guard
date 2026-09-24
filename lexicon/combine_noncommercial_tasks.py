"""Export one stable, unlabeled task seed per text candidate across both layers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'derived/expanded-20260923/task_seeds.jsonl'
NEW = ROOT / 'research_nc/citizenlab-1eeb5e6/new_task_seeds.jsonl'
OUT = ROOT / 'research_nc/citizenlab-1eeb5e6/combined_task_seeds.jsonl'
MANIFEST = ROOT / 'research_nc/citizenlab-1eeb5e6/combined_task_seeds_manifest.json'


def source_licenses(refs):
    result = set()
    for ref in refs:
        path = ref['path']
        if path.startswith(('sensitive-lexicon/', 'spacegather-worldwide-sensitive/')):
            result.add('MIT')
        elif path.startswith('houbb-sensitive-word-data/'):
            result.add('Apache-2.0')
        else:
            raise ValueError(f'Unknown source path: {path}')
    return sorted(result)


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError('Combined task export already exists')
    seen = set()
    by_layer = {'permissive': 0, 'noncommercial': 0}
    with OUT.open('w', encoding='utf-8') as output:
        for path, layer in ((BASE, 'permissive'), (NEW, 'noncommercial')):
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    task = json.loads(line)
                    key = task['canonical']
                    if key in seen:
                        raise ValueError(f'Duplicate task canonical: {key}')
                    seen.add(key)
                    task.pop('review_state', None)
                    task['candidate_status'] = 'included_unlabeled'
                    task['manual_review_required_for_inclusion'] = False
                    task['data_layer'] = layer
                    task['source_licenses'] = (source_licenses(task['source_refs']) if layer == 'permissive'
                                               else ['CC BY-NC-SA 4.0'])
                    task['safety_label_status'] = 'not_assigned'
                    task['auto_block'] = False
                    output.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + '\n')
                    by_layer[layer] += 1
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps({'tasks': sum(by_layer.values()), 'by_layer': by_layer,
                                    'sha256': digest, 'model_requests_submitted': 0,
                                    'manual_review_required_for_candidate_inclusion': False},
                                   ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tasks': sum(by_layer.values()), 'by_layer': by_layer}, ensure_ascii=False))


if __name__ == '__main__':
    main()
