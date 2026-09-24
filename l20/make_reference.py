"""Freeze a small M5 reference set for the L20 portability check."""
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'round1'
sys.path.insert(0, str(SOURCE))
from runtime import load, restore_adapter, batch_probs  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    rows = read_jsonl(SOURCE / 'data/dev.jsonl')
    model, tokenizer = load(dtype=torch.bfloat16, device='mps')
    restore_adapter(model, SOURCE / 'training/step-0150')
    choices = {}
    with torch.no_grad():
        for row in rows:
            group = (row['target_role'], row['label'])
            if len(choices.get(group, [])) >= 4:
                continue
            probs = batch_probs(model, tokenizer, [row['messages']])[0]
            if max(probs) >= 0.95:
                choices.setdefault(group, []).append({
                    'sample_id': row['sample_id'], 'role': row['target_role'],
                    'label': row['label'], 'probs': probs,
                })
            if sum(map(len, choices.values())) == 16:
                break
    expected = {(role, label) for role in ('user', 'assistant') for label in ('safe', 'unsafe')}
    if set(choices) != expected or any(len(v) != 4 for v in choices.values()):
        raise RuntimeError(f'Insufficient strong reference cases: { {str(k): len(v) for k,v in choices.items()} }')
    result = [item for group in sorted(choices) for item in choices[group]]
    (HERE / 'reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cases': len(result), 'groups': {str(k): len(v) for k,v in choices.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
