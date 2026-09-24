"""Freeze risk-label training/calibration/dev/test data; CPU tokenization only."""
from collections import defaultdict,Counter
import hashlib,json,random
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'data/risk_v1'
def read(p):return [json.loads(l) for l in p.open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(s):return hashlib.sha256(s.encode()).hexdigest()
def serialize(ms):return '\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in ms)


def main():
    if OUT.exists():raise FileExistsError('Frozen risk dataset exists')
    tok_path=PROJECT/'round3/l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json'
    tok=Tokenizer.from_file(str(tok_path));pools=defaultdict(list);rejected=Counter()
    old_test={r['sample_id'] for r in read(PROJECT/'round3/data/base_comparison_v1/test.jsonl')}
    for row in read(ROOT/'data/public_pools.jsonl'):
        split=row.pop('split_pool');ids=tok.encode(serialize(row['messages']),add_special_tokens=False).ids
        if not ids or len(ids)>4096:rejected['token_limit']+=1;continue
        if split=='test' and row['sample_id'] in old_test:rejected['previous_test_seen']+=1;continue
        row['ids']=ids;pools[split,row['language']+'/'+row['target_role'],row['source_label']].append(row)
    quotas={'train':{'zh/user':3500,'zh/assistant':2500,'en/assistant':4000},
            'calibration':{'zh/user':150,'zh/assistant':150,'en/assistant':150},
            'dev':{'zh/user':200,'zh/assistant':200,'en/assistant':200},
            'sealed_test':{'zh/user':200,'zh/assistant':200,'en/assistant':200}}
    chosen={}
    for split,groups in quotas.items():
        rows=[]
        for group,n in groups.items():
            for label in ('safe','unsafe'):
                pool=sorted(pools['test' if split=='sealed_test' else split,group,label],key=lambda r:rank('risk-v1:'+r['sample_id']))
                assert len(pool)>=n,(split,group,label,len(pool),n)
                rows.extend(pool[:n])
        chosen[split]=rows
    synthetic=PROJECT/'round2/oneword2_v12/partial_snapshot_50'
    raw=read(synthetic/'examples.jsonl');families=defaultdict(list)
    for ref in read(synthetic/'weak_supervision_index.jsonl'):
        if ref['split']=='train':families[ref['family']].append(raw[ref['raw_line']-1])
    selected_families=sorted(families,key=lambda x:rank('risk-weak:'+x))[:2048]
    for family in selected_families:
        for r in families[family]:
            ids=tok.encode(serialize(r['messages']),add_special_tokens=False).ids
            assert len(ids)<=4096
            chosen['train'].append({'sample_id':r['sample_id'],'family':'synthetic-'+family,'messages':r['messages'],
                                   'ids':ids,'target_role':r['target_role'],'source_label':r['label'],
                                   'language':r['language'],'source':'aster-dev-272/v12',
                                   'source_group':r['source_group'],'source_license':r['source_licenses'],
                                   'source_label_origin':'luna_synthetic_unverified','weight':.2,
                                   'label_tier':'synthetic_weak','raw_content_sha256':r['content_sha256']})
    # Explicitly separate harmless background, keeping the original target intact.
    # This is a low-weight synthetic context stressor, not natural long-risk gold.
    public=[r for r in chosen['train'] if r['label_tier']=='public_source_reference']
    random.Random(20260924).shuffle(public)
    for i,base in enumerate(public[:1024]):
        row=json.loads(json.dumps(base));lang=row['language']
        if lang=='zh':
            background='以下是与新请求无关的已结束背景记录。\n'+''.join(f'记录 {j}：按顺序整理一般阅读资料，核对章节和页码。\n' for j in range(55))
            acknowledgment='背景记录已结束。下面开始独立的新请求。'
        else:
            background='Unrelated completed background log follows.\n'+''.join(f'Record {j}: Organize general reading material and check chapter and page numbers.\n' for j in range(70))
            acknowledgment='The background log is finished. A separate new request follows.'
        row['messages']=[{'role':'user','content':background},{'role':'assistant','content':acknowledgment}]+row['messages']
        row['ids']=tok.encode(serialize(row['messages']),add_special_tokens=False).ids
        if len(row['ids'])>4096:rejected['augmented_token_limit']+=1;continue
        row.update(sample_id=row['sample_id']+'-background',weight=.4,label_tier='synthetic_context_weak')
        chosen['train'].append(row)
    family_sets={s:{r['family'] for r in rows} for s,rows in chosen.items()}
    for s,a in family_sets.items():
        for t,b in family_sets.items():
            if s<t:assert not a&b,(s,t)
    OUT.mkdir();hashes={}
    for split,rows in chosen.items():
        random.Random(20260924).shuffle(rows)
        path=OUT/(split+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));hashes[split]=sha(path)
    manifest={'version':'risk-supervised-v1','teacher_alignment_loss':False,
              'counts':{s:len(r) for s,r in chosen.items()},'tokens':{s:sum(len(x['ids']) for x in r) for s,r in chosen.items()},
              'strata':{s:dict(Counter(x['language']+'/'+x['target_role']+'/'+x['source_label'] for x in r)) for s,r in chosen.items()},
              'training_tiers':dict(Counter(r['label_tier'] for r in chosen['train'])),
              'output_hashes':hashes,'tokenizer_sha256':sha(tok_path),'rejected':dict(rejected),
              'family_disjoint':True,'sealed_test_previous_neural_use':False,
              'limits':'Public source policies differ; synthetic labels unverified; no public English-user evaluation stratum; exact decontamination only.',
              'input_schema':'derived training view; original v12 generation and classification service schemas unchanged'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
