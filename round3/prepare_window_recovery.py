"""CPU-only preparation of a small, frozen long-document recovery experiment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
import unicodedata
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/window_recovery_v1'


def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().lower()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,list):
        for v in x:yield from strings(v)
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
def grams(text):
    return {digest(text[i:i+64]) for i in range(0,len(text)-63,32)}


def main():
    if OUT.exists():raise FileExistsError('Frozen recovery dataset already exists')
    tokenizer_path=ROOT/'l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json'
    tok=Tokenizer.from_file(str(tokenizer_path))
    excluded=set()
    paths=list((ROOT.parent/'round1/benchmark').glob('*.jsonl'))+[ROOT/'data/base_comparison_v1/dev.jsonl',ROOT/'data/base_comparison_v1/test.jsonl']
    for path in paths:
        for line in path.open():
            for text in strings(json.loads(line)):
                if len(text)>=64:excluded.update(grams(norm(text)))
    pools={('train','zh'):[],('train','en'):[],('dev','zh'):[],('dev','en'):[]}
    seen=set();reject=Counter();sources={}
    for lang in ('zh','en'):
        path=ROOT/f'data/pretrain_sources/raw_samples/{lang}-10000.parquet'
        sources[lang]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=pq.read_table(path).to_pylist()
        rows.sort(key=lambda r:digest('window-recovery:'+str(r['id'])))
        for row in rows:
            text=row['text'];clean=norm(text);family=urlsplit(row['url']).hostname or 'unknown'
            split='dev' if int(digest(family)[:8],16)%5==0 else 'train'
            quota=16 if split=='dev' else 64
            if len(pools[(split,lang)])>=quota:continue
            h=digest(clean)
            if h in seen:reject['duplicate']+=1;continue
            if any(s in clean for s in ('@','password=','api_key=','secret_key=')):
                reject['conservative_pii_secret_filter']+=1;continue
            # These heuristics reduce obvious noise; not a claim of complete PII removal.
            if len(clean)<1200 or len(set(clean))/len(clean)<.002:
                reject['short_or_repetitive']+=1;continue
            if grams(clean)&excluded:reject['benchmark_shingle_overlap']+=1;continue
            ids=tok.encode('USER:\n'+text,add_special_tokens=False).ids
            if len(ids)<1536:reject['fewer_than_1536_tokens']+=1;continue
            ids=ids[:3072]
            sample={'sample_id':f'{lang}-'+h[:24],'language':lang,'ids':ids,
                    'target_role':'user','source_label':None,'family':family,
                    'source':'fineweb-2/cmn_Hani' if lang=='zh' else 'fineweb-edu',
                    'source_id':row['id'],'source_url':row['url'],'source_license':'ODC-BY',
                    'original_text_sha256':digest(text),'derived_prefix':True,
                    'purpose':'same-visible-prefix representation recovery; no safety gold label'}
            pools[(split,lang)].append(sample);seen.add(h)
    for (split,lang),rows in pools.items():assert len(rows)==(16 if split=='dev' else 64),(split,lang,len(rows))
    train=pools['train','zh']+pools['train','en'];dev=pools['dev','zh']+pools['dev','en']
    assert not {r['family'] for r in train}&{r['family'] for r in dev}
    OUT.mkdir()
    hashes={}
    for split,rows in [('train',train),('dev',dev)]:
        rows.sort(key=lambda r:digest('order:'+r['sample_id']))
        p=OUT/(split+'.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'version':'window-recovery-v1','counts':{'train':len(train),'dev':len(dev)},
              'tokens':{'train':sum(len(r['ids']) for r in train),'dev':sum(len(r['ids']) for r in dev)},
              'token_range':[1536,3072],'source_hashes':sources,'output_hashes':hashes,
              'tokenizer_sha256':hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
              'rejections':dict(reject),'domain_disjoint':True,'safety_labels_available':False,
              'decontamination':'stride-32 normalized 64-character shingle exclusion against local official benchmark and frozen source dev/test; incomplete near-duplicate protection',
              'scope':'small natural-text representation-recovery pilot; not a multi-turn safety gold set'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
