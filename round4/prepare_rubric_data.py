"""Add same-word weak pairs and explicit benign rubric cases; keep raw v12 intact."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,heapq,re,unicodedata,random
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent;P=ROOT.parent;OUT=ROOT/'data/risk_v2'
def read(path):
    with path.open() as f:
        for line in f:yield json.loads(line)
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serialize(ms):return '\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in ms)
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,list):
        for v in x:yield from strings(v)
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)


def main():
    if OUT.exists():raise FileExistsError('Frozen rubric dataset already exists')
    source=P/'round2/oneword2_v12/completed_snapshot_v2'
    if not (source/'summary.json').exists():raise RuntimeError('Completed archive extraction is required')
    tokenizer=P/'round3/l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer));chosen={s:list(read(ROOT/f'data/risk_v1/{s}.jsonl')) for s in ['calibration','dev','sealed_test']}
    chosen['train']=[r for r in read(ROOT/'data/risk_v1/train.jsonl') if r['label_tier']!='synthetic_weak']
    excluded=set()
    for s in ['calibration','dev','sealed_test']:
        for r in chosen[s]:excluded.update(digest(norm(m['content'])) for m in r['messages'])
    for path in list((P/'round1/benchmark').glob('*.jsonl'))+[P/'round3/data/raw/nemotron_safety_guard_v3_zh/valid.jsonl']:
        for r in read(path):excluded.update(digest(norm(t)) for t in strings(r) if len(t)>=20)
    seen=set();candidates=[]
    for r in read(source/'term_status.jsonl'):
        if not r['complete']:continue
        family=digest(norm(r['word']))
        if int(family[:12],16)%100>=80 or family in seen:continue
        seen.add(family)
        candidates.append((digest('round4-pair:'+family),r['task_key'],family,r['word']))
    candidates=heapq.nsmallest(6144,candidates);by_key={x[1]:x for x in candidates};pairs=defaultdict(list)
    for r in read(source/'examples.jsonl'):
        if r['task_key'] in by_key:pairs[r['task_key']].append(r)
    accepted=[];rejections=Counter()
    for _,key,family,word in candidates:
        pair=pairs[key]
        if len(pair)!=2 or {r['language'] for r in pair}!={'zh','en'} or {r['label'] for r in pair}!={'safe','unsafe'}:
            rejections['pair_invalid']+=1;continue
        if any(digest(norm(m['content'])) in excluded for r in pair for m in r['messages']):
            rejections['evaluation_exact_overlap']+=1;continue
        derived=[]
        for r in pair:
            ids=tok.encode(serialize(r['messages']),add_special_tokens=False).ids
            if len(ids)>4096:break
            derived.append({'sample_id':r['sample_id'],'family':'synthetic-'+family,'messages':r['messages'],'ids':ids,
                            'target_role':r['target_role'],'language':r['language'],'source_label':r['label'],
                            'source':'aster-dev-272/v12','source_group':r['source_group'],'source_license':r['source_licenses'],
                            'source_label_origin':'luna_synthetic_unverified','weight':.2,'label_tier':'synthetic_weak',
                            'pair_id':'synthetic-'+family,'raw_content_sha256':r['content_sha256']})
        if len(derived)!=2:rejections['token_limit']+=1;continue
        chosen['train'].extend(derived);accepted.append((family,word,pair))
        if len(accepted)==4096:break
    assert len(accepted)==4096,len(accepted)
    rubric_words=0
    for family,word,pair in accepted:
        # Keep the quoted text bounded and unable to break out of the template.
        if not re.fullmatch(r'[\w\u3400-\u9fff .+·-]{1,40}',word) or re.search(r'\d{6,}',word):continue
        unsafe=next(r for r in pair if r['label']=='unsafe')
        zh_question=f'请提供关于“{word}”的中立背景介绍，并区分事实与观点。'
        en_question=f'For a text-processing exercise, count the Unicode code points in the quoted term “{word}”.'
        cases=[('zh','user',[{'role':'user','content':zh_question}],'neutral_background'),
               ('en','user',[{'role':'user','content':en_question}],'quoted_text_processing'),
               ('zh','assistant',[*unsafe['messages'],{'role':'assistant','content':f'关于“{word}”，我可以提供中立资料或安全防护信息，无法协助威胁、诈骗或侵犯隐私。'}],'safe_refusal'),
               ('en','assistant',[{'role':'user','content':en_question},{'role':'assistant','content':f'The quoted term “{word}” contains {len(word)} Unicode code points.'}],'benign_text_answer')]
        for i,(language,role,messages,rule) in enumerate(cases):
            assert not any(digest(norm(m['content'])) in excluded for m in messages)
            ids=tok.encode(serialize(messages),add_special_tokens=False).ids
            chosen['train'].append({'sample_id':f'rubric-{family[:24]}-{i}','family':'synthetic-'+family,'pair_id':'synthetic-'+family,
                                    'messages':messages,'ids':ids,'target_role':role,'language':language,'source_label':'safe',
                                    'source':'benign-keyword-rubric-v1','source_label_origin':'explicit_template_rubric',
                                    'weight':.8,'label_tier':'rubric_benign','rubric_rule':rule,'word':word})
        rubric_words+=1
        if rubric_words==1024:break
    assert rubric_words==1024,rubric_words
    families={s:{r['family'] for r in rows} for s,rows in chosen.items()}
    for a in families:
        for b in families:
            if a<b:assert not families[a]&families[b]
    OUT.mkdir();hashes={}
    for split,rows in chosen.items():
        random.Random(20260925).shuffle(rows)
        path=OUT/(split+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));hashes[split]=sha(path)
    manifest={'version':'risk-supervised-rubric-v2','teacher_alignment_loss':False,
              'counts':{s:len(r) for s,r in chosen.items()},'tokens':{s:sum(len(x['ids']) for x in r) for s,r in chosen.items()},
              'training_tiers':dict(Counter(r['label_tier'] for r in chosen['train'])),
              'strata':{s:dict(Counter(r['language']+'/'+r['target_role']+'/'+r['source_label'] for r in rows)) for s,rows in chosen.items()},
              'output_hashes':hashes,'tokenizer_sha256':sha(tokenizer),'archive_snapshot_sha256':sha(source/'selected_attempts.json'),
              'rejections':dict(rejections),'family_disjoint':True,'sealed_test_previous_project_eval':None,'sealed_test_scope':'Excluded from this H24 lineage training and previous test IDs; historical corpus processing not exhaustively known',
              'limits':'Public labels follow source policies, weak labels unverified; template rubric is limited to these constructed contexts; exact decontamination is not semantic decontamination; no public English-user test stratum.',
              'generation_schema_changed':False,'classification_contract_changed':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
