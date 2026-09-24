"""Freeze a shared source-label classification experiment, not project alignment."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/base_comparison_v1'


def fingerprint(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().encode()).hexdigest()


def walk_strings(value):
    if isinstance(value, str):
        if len(value) >= 20:
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


def write(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Frozen dataset exists; inspect before replacing')
    exclude = set()
    for path in list((ROOT.parent/'round1/benchmark').glob('*.jsonl')) + [ROOT/'data/raw/nemotron_safety_guard_v3_zh/valid.jsonl']:
        for line in path.open():
            for text in walk_strings(json.loads(line)):
                exclude.add(fingerprint(text))
    pools = defaultdict(list)
    rejected = Counter()
    seen = set()
    sources = {}
    paths = [ROOT/'data/candidates/nemotron_user_candidates.jsonl',
             ROOT/'data/candidates/nemotron_assistant_candidates.jsonl']
    for path in paths:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            row = json.loads(line)
            if row['source_label'] not in ('safe', 'unsafe'):
                continue
            messages = row['messages']
            family = fingerprint('nemotron:'+messages[0]['content'])
            sample = {'sample_id': row['candidate_id'], 'family': family,
                      'messages': messages, 'target_role': row['target_role'],
                      'source_label': row['source_label'], 'source_label_origin': row['source_label_origin'],
                      'source': row['source'], 'source_revision': row['source_revision'],
                      'source_license': row['source_license'], 'source_split': 'train',
                      'language': 'zh', 'project_policy_label': None,
                      'objective': 'source_policy_classification_probe'}
            pool = 'zh_'+row['target_role']
            add(sample, pool, pools, seen, exclude, rejected)
    path = ROOT/'data/raw/beavertails_round0_30k/train.jsonl.gz'
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path, 'rt') as stream:
        for index,line in enumerate(stream):
            row = json.loads(line)
            if not isinstance(row.get('is_safe'), bool):
                continue
            messages = [{'role':'user','content':row.get('prompt','')},
                        {'role':'assistant','content':row.get('response','')}]
            sample = {'sample_id':f'beaver-{index}', 'family':fingerprint('beaver:'+row.get('prompt','')),
                      'messages':messages, 'target_role':'assistant',
                      'source_label':'safe' if row['is_safe'] else 'unsafe',
                      'source_label_origin':'original_dataset_label',
                      'source':'PKU-Alignment/BeaverTails/round0/30k',
                      'source_revision':'8401fe609d288129cc684a9b3be6a93e41cfe678',
                      'source_license':'CC BY-NC 4.0', 'source_split':'train',
                      'language':'en', 'project_policy_label':None,
                      'objective':'source_policy_classification_probe'}
            add(sample, 'en_assistant', pools, seen, exclude, rejected)
    quotas={'train':{'zh_user':512,'zh_assistant':512,'en_assistant':1024},
            'dev':{'zh_user':64,'zh_assistant':64,'en_assistant':128},
            'test_en':{'en_assistant':256}}
    selected={name:[] for name in quotas}
    for split,groups in quotas.items():
        for pool,quota in groups.items():
            for label in ('safe','unsafe'):
                rows=sorted(pools[(split,pool,label)],key=lambda r:fingerprint('rank-v1:'+r['sample_id']))
                if len(rows)<quota:
                    raise RuntimeError(f'Insufficient {split}/{pool}/{label}: {len(rows)} < {quota}')
                selected[split].extend(rows[:quota])
    for split,rows in selected.items():
        random.Random(20260923).shuffle(rows)
    families={s:{r['family'] for r in rows} for s,rows in selected.items()}
    assert not families['train'] & (families['dev']|families['test_en'])
    assert not families['dev'] & families['test_en']
    zh_test=[]
    for line in (ROOT/'data/source_valid_probe.jsonl').open():
        row=json.loads(line)
        row.update(sample_id='nemotron-valid-'+str(row['source_id'])+'-'+row['target_role'],language='zh')
        zh_test.append(row)
    selected['test']=zh_test+selected.pop('test_en')
    OUT.mkdir()
    hashes={split:write(OUT/(split+'.jsonl'),rows) for split,rows in selected.items()}
    manifest={'version':'base-comparison-source-v1','counts':{s:len(r) for s,r in selected.items()},
              'strata':{s:dict(Counter(f"{r['language']}/{r['target_role']}/{r['source_label']}" for r in rows)) for s,rows in selected.items()},
              'source_hashes':sources,'output_hashes':hashes,'rejected':dict(rejected),
              'source_labels_are_project_gold':False,'final_safety_alignment':False,
              'split_rule':'normalized prompt family hash modulo100: train<70, dev70-84, heldout>=85; official Nemotron valid separate',
              'decontamination':'exact normalized message text exclusion against local Qwen3GuardTest and Nemotron valid; not a near-duplicate guarantee',
              'training_policy':'public source labels for exploratory model comparison; original candidate eligibility and synthetic schemas unchanged',
              'max_training_characters':1000}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


def add(sample,pool,pools,seen,exclude,rejected):
    texts=[m['content'] for m in sample['messages']]
    if any(not isinstance(t,str) or not t.strip() or t.strip()=='None' for t in texts):
        rejected['empty']+=1;return
    if sum(map(len,texts))>1000:
        rejected['long']+=1;return
    if any(fingerprint(t) in exclude for t in texts):
        rejected['eval_exact_overlap']+=1;return
    content=fingerprint(json.dumps(sample['messages'],ensure_ascii=False,sort_keys=True))
    if content in seen:
        rejected['duplicate_conversation']+=1;return
    seen.add(content)
    bucket=int(sample['family'][:12],16)%100
    split='train' if bucket<70 else 'dev' if bucket<85 else 'test_en'
    pools[(split,pool,sample['source_label'])].append(sample)


if __name__=='__main__':main()
