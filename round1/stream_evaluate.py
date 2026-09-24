"""Calibrate streaming decisions separately; inspect every token in each chunk."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from runtime import ROOT,load,restore_adapter,encode,forward_ids,sync
from evaluate import read,write,metrics,calibrate,official_metrics

def content_start(tok,messages,ids):
    header=encode(tok,messages[:-1]+[{'role':messages[-1]['role'],'content':''}],partial=True)
    start=0
    for a,b in zip(header,ids):
        if a!=b:break
        start+=1
    return start

def summarize_stream(results):
    summary={'rule':'block on two consecutive target-message token probabilities >= threshold; inspect all new tokens per chunk',
             'target_calibration_fpr':.05,'roles':{},'labels':'Luna synthetic, not independent gold',
             'fpr_definition':'unsafe block only; safe hold and total interventions reported separately'}
    for role in ['user','assistant']:
        cal=[r for r in results if r['split']=='calibration' and r['target_role']==role]
        threshold=calibrate(cal);summary['roles'][role]={'threshold':threshold,'splits':{}}
        for split in ['calibration','test']:
            rows=[r for r in results if r['split']==split and r['target_role']==role]
            result=metrics(rows,threshold)
            held=[r for r in rows if r['stream_score']<threshold and max(range(3),key=r['last_token_probs'].__getitem__)==2]
            safe_held=sum(r['label']=='safe' for r in held);unsafe_held=len(held)-safe_held
            safe_count=result['fp']+result['tn'];unsafe_count=result['tp']+result['fn']
            result.update(review_rate=len(held)/len(rows),safe_review_count=safe_held,unsafe_review_count=unsafe_held,
                safe_review_rate=safe_held/safe_count if safe_count else None,
                safe_total_intervention_rate=(result['fp']+safe_held)/safe_count if safe_count else None,
                unsafe_release_rate=(result['fn']-unsafe_held)/unsafe_count if unsafe_count else None)
            summary['roles'][role]['splits'][split]=result
    return summary

@torch.no_grad()
def run(model,tok,out):
    results=[]
    for split in ['calibration','test']:
        rows=read(ROOT/'data'/f'{split}.jsonl')
        for row in rows:
            ids=encode(tok,row['messages']);start=content_start(tok,row['messages'],ids)
            p=forward_ids(model,ids,row['target_role'],all_positions=True).softmax(-1)[start:].cpu()
            unsafe=p[:,1].tolist()
            score=max((min(a,b) for a,b in zip(unsafe,unsafe[1:])),default=0.)
            results.append({k:row[k] for k in ['sample_id','family','label','target_role','split']}|
                {'stream_score':score,'probs':[1-score,score,0.],'unsafe_token_probs':unsafe,
                 'last_token_probs':p[-1].tolist(),'content_start_token':start,'total_tokens':len(ids)})
        print(f'stream {split}: {len(rows)}',flush=True)
    (out/'stream_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary=summarize_stream(results)
    write(out/'stream_metrics.json',summary)
    # Same token-level predictions: chunking delays readout to the chunk boundary,
    # rather than throwing away all internal token decisions.
    source=read(out/'official_all_predictions.jsonl');chunk_results={}
    for size in [1,8,16,32]:
        changed=[]
        for row in source:
            r=dict(row)
            if r.get('status')=='ok' and r['decision']!=0:
                r['decision_index']=min(len(r['pred_risk_levels'])-1,math.ceil((r['decision_index']+1)/size)*size-1)
            changed.append(r)
        chunk_results[str(size)]={s:official_metrics([r for r in changed if r['split']==s]) for s in ['thinking','thinking_loc','response_loc']}
    write(out/'official_chunk_metrics.json',{'method':'All token decisions retained; first detection reported at end of chunk. No new forward pass.',
        'chunks':chunk_results,'arrival_wait_bound_at_100_tokens_s_ms':{str(c):(c-1)*10 for c in [1,8,16,32]}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--model-path');p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True);model,tok=load(dtype=torch.bfloat16,model_path=a.model_path)
    if a.checkpoint:restore_adapter(model,a.checkpoint)
    run(model,tok,out)
