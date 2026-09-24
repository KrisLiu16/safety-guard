"""Freeze seed families before modeling; filter synthetic data without model predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
FAMILIES = [
    ['中南海', '大陆当局', '中国当局', '北京当局', '政府', '中央领导', '大陆官方'],
    ['共产党', '中共'],
    ['腐败', '贪污', '受贿罪', '收受股票', '以权谋私', '经济犯罪', '徇私舞弊'],
    ['玩忽职守', '滥用职权', '官僚主义'],
    ['严重违纪', '违纪', '开除党籍', '撤销职务'],
    ['有期徒刑', '无期徒刑'],
    ['巨额财产', '来源不明罪', '没收个人财产'],
]

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

def freeze():
    path = ROOT/'family_splits.json'
    if path.exists(): return json.loads(path.read_text())
    seeds = [json.loads(l) for l in (ROOT.parent/'pilot/seeds.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for seed in seeds:
        family = next((f'family-{i:02d}' for i,g in enumerate(FAMILIES) if seed['word'] in g), seed['seed_id'])
        groups[family].append(seed)
    ratios = {'train':.6, 'dev':.1, 'calibration':.1, 'test':.2}
    counts = Counter(); result = {}
    rng = random.Random(20260922)
    families = list(groups); rng.shuffle(families)
    families.sort(key=lambda f: -len(groups[f]))
    for family in families:
        split = max(ratios, key=lambda s: 50*ratios[s]-counts[s])
        counts[split] += len(groups[family])
        for seed in groups[family]:
            result[seed['seed_id']] = {'word':seed['word'], 'family':family, 'split':split}
    dump(path, {'seed':20260922, 'rule':'manual conservative topic/alias families, greedy balanced by word count',
                'word_counts':dict(counts), 'items':result})
    return json.loads(path.read_text())

def prepare(source):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from runtime import encode
    from transformers import AutoTokenizer
    mapping = freeze()['items']
    source = Path(source)
    review_ids = {json.loads(l)['sample_id'] for l in (source/'review_required.jsonl').read_text().splitlines()}
    candidates = [json.loads(l) for l in (source/'examples.jsonl').read_text().splitlines()]
    out = ROOT/'data'; out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'model', trust_remote_code=True, local_files_only=True)
    accepted, rejected = [], []
    words = sorted((v['word'] for v in mapping.values()), key=len, reverse=True)
    def normalized(row):
        text = '\n'.join(m['content'] for m in row['messages'])
        for word in words: text = text.replace(word, '主题')
        return re.sub(r'\W+', '', text).lower()
    for row in sorted(candidates, key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest()):
        row.update(mapping[row['seed_id']])
        reason = None
        if row['sample_id'] in review_ids: reason = 'generator_quality_flags'
        # Catch other named seed topics crossing family partitions.
        present = [v for v in mapping.values() if v['word'] in '\n'.join(m['content'] for m in row['messages'])]
        if any(v['split'] != row['split'] for v in present): reason = 'cross_split_seed_reference'
        ids = encode(tokenizer, row['messages'])
        row['token_count'] = len(ids)
        if len(ids)>1024: reason = 'training_context_overflow'
        if reason: rejected.append({'sample_id':row['sample_id'],'reason':reason}); continue
        row['token_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        accepted.append(row)
    # Topic-substituted character n-grams catch close templated paraphrases across all splits.
    texts = [normalized(row) for row in accepted]
    X = TfidfVectorizer(analyzer='char', ngram_range=(3,5), dtype=np.float32).fit_transform(texts)
    neighbors = NearestNeighbors(metric='cosine', algorithm='brute', radius=.08).fit(X)
    removed = set(); pairs = []
    for start in range(0,len(accepted),128):
        distances, indices = neighbors.radius_neighbors(X[start:start+128], return_distance=True)
        for offset, (ds, js) in enumerate(zip(distances, indices)):
            i=start+offset
            if i in removed: continue
            for distance,j in zip(ds,js):
                j=int(j)
                if j<=i or j in removed: continue
                removed.add(j)
                pairs.append({'kept':accepted[i]['sample_id'],'removed':accepted[j]['sample_id'],
                              'cosine_similarity':float(1-distance), 'cross_split':accepted[i]['split']!=accepted[j]['split']})
    clean = [r for i,r in enumerate(accepted) if i not in removed]
    for split in ('train','dev','calibration','test'):
        rows=[r for r in clean if r['split']==split]
        jsonl(out/f'{split}.jsonl',rows)
    jsonl(out/'quarantine.jsonl', rejected)
    jsonl(out/'near_duplicates.jsonl', pairs)
    # Generated prefixes remain unverified; extract an explicit review queue, never auto-promote.
    jsonl(out/'prefix_review_queue.jsonl', [r for r in clean if r['split']=='train' and r['decidable_at_char']])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}
    summary={'source':str(source),'raw_structurally_valid':len(candidates),'clean':len(clean),
        'quarantined':len(rejected),'near_duplicates':len(removed),
        'counts':dict(Counter((r['split']+'/'+r['target_role']+'/'+r['label']) for r in clean)),
        'label_provenance':'Luna synthetic, automatic screening only; not independent gold',
        'family_manifest_sha256':hashlib.sha256((ROOT/'family_splits.json').read_bytes()).hexdigest(),'hashes':hashes}
    dump(out/'manifest.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source'); args=parser.parse_args()
    if args.source: prepare(args.source)
    else: print(json.dumps(freeze(), ensure_ascii=False,indent=2))
