"""One bounded finite-action policy-gradient run; never prompts the model with labels."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
import psutil
import torch
from runtime import ROOT, load, encode, forward_ids, attach_lora, save_adapter, sync
from evaluate import read, write, metrics, batch_probs

def token_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

def instances(tok):
    rows=read(ROOT/'data/train.jsonl')
    # Prefix records enter only through an explicit reviewed file, with causal labels.
    reviewed=ROOT/'data/reviewed_prefixes.jsonl'
    if reviewed.exists():rows += read(reviewed)
    result=[]
    for r in rows:
        ids=encode(tok,r['messages'],partial=r.get('is_prefix',False))
        if len(ids)>1024:raise ValueError('Unquarantined overlength training example')
        result.append({'id':r['sample_id'],'ids':ids,'role':r['target_role'],'label':r['label'],
            'token_hash':token_hash(ids),'family':r['family']})
    return result

@torch.no_grad()
def reference(model,rows):
    path=ROOT/'reference_probabilities.jsonl'
    cached={r['id']:r for r in read(path)} if path.exists() else {}
    with path.open('a') as f:
        for i,row in enumerate(rows):
            old=cached.get(row['id'])
            if old:
                assert old['token_hash']==row['token_hash'] and old['role']==row['role'];continue
            p=forward_ids(model,row['ids'],row['role'])[-1].softmax(-1).cpu().tolist()
            obj={k:row[k] for k in ['id','token_hash','role']}|{'probs':p}
            f.write(json.dumps(obj)+'\n');f.flush();cached[row['id']]=obj
            if (i+1)%100==0:print(f'reference {i+1}/{len(rows)}',flush=True)
    return cached

@torch.no_grad()
def dev_score(model,tok):
    model.eval();rows=read(ROOT/'data/dev.jsonl');pred=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4]
        probs=batch_probs(model,tok,[r['messages'] for r in batch])
        pred += [{'label':r['label'],'probs':p} for r,p in zip(batch,probs)]
    m=metrics(pred);model.train()
    # Balanced accuracy with an explicit review cost, fixed before checkpoint selection.
    m['selection_score']=.5*(m['recall']+1-m['false_positive_rate'])-.1*m['review_rate']
    return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-only',action='store_true');args=parser.parse_args()
    torch.manual_seed(20260922);random.seed(20260922)
    smoke=json.loads((ROOT/'smoke_results.json').read_text());assert smoke['status']=='passed'
    config=json.loads((ROOT/'config.json').read_text());cfg=config['training']
    model,tok=load(dtype=torch.bfloat16)
    rows=instances(tok);ref=reference(model,rows)
    if args.reference_only:return
    destination=ROOT/'training';destination.mkdir(exist_ok=True)
    if (destination/'training_metrics.jsonl').exists():raise RuntimeError('Existing training run: inspect checkpoint before any restart')
    params=attach_lora(model);model.train()
    adapters=[p for n,p in params if n.endswith(('.A','.B'))];heads=[p for n,p in params if not n.endswith(('.A','.B'))]
    optimizer=torch.optim.AdamW([{'params':adapters,'lr':cfg['adapter_lr']},{'params':heads,'lr':cfg['head_lr']}],weight_decay=0.)
    groups=Counter((r['role'],r['label']) for r in rows)
    weights={key:len(rows)/(len(groups)*n) for key,n in groups.items()}
    random.shuffle(rows)
    budget=min(cfg['max_optimizer_steps'],math.ceil(len(rows)/cfg['gradient_accumulation']))
    start=time.monotonic();training_seconds=0.;peak=0;best=-float('inf');selected=None;seen=0;tokens=0
    baseline_swap=psutil.swap_memory().used
    with (destination/'training_metrics.jsonl').open('w') as log:
        for step in range(1,budget+1):
            batch=rows[(step-1)*16:step*16]
            if not batch:break
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True);losses=[];rewards=[];kls=[]
            for row in batch:
                logp=forward_ids(model,row['ids'],row['role'])[-1].log_softmax(-1);p=logp.exp()
                expected=1 if row['label']=='unsafe' else 0
                reward=torch.tensor([1. if expected==0 else -2.,1. if expected==1 else -2.,-.25],device='mps')
                actions=torch.multinomial(p.detach(),4,replacement=True)
                # Detached exact baseline reduces variance without changing the PG objective.
                advantage=reward[actions]-(p.detach()*reward).sum()
                policy_loss=-(advantage*logp[actions]).mean()
                refp=torch.tensor(ref[row['id']]['probs'],device='mps').clamp(min=1e-8)
                kl=(p*(logp-refp.log())).sum()
                loss=(weights[(row['role'],row['label'])]*policy_loss+cfg['kl_beta']*kl)/len(batch)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite policy loss')
                loss.backward();losses.append(loss.item());rewards.append((p.detach()*reward).sum().item());kls.append(kl.item())
                tokens+=len(row['ids']);seen+=1
            norm=torch.nn.utils.clip_grad_norm_([p for n,p in params],1.,error_if_nonfinite=True)
            optimizer.step();sync();elapsed=time.monotonic()-begin;training_seconds+=elapsed
            used=torch.mps.driver_allocated_memory();peak=max(peak,used)
            row={'step':step,'examples':seen,'tokens':tokens,'loss':sum(losses),'expected_reward':sum(rewards)/len(rewards),
                 'kl':sum(kls)/len(kls),'grad_norm':norm.item(),'seconds':elapsed,'active_seconds':training_seconds,
                 'mps_driver_bytes':used,'rss_bytes':psutil.Process().memory_info().rss,'swap_used':psutil.swap_memory().used}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step==20 and training_seconds/20*budget>7200:
                budget=min(budget,100)
                print('Measured pace exceeds 2h; limiting to at most 100 updates within wall cap.',flush=True)
            thermal_limited=False
            if step%20==0:
                thermal=subprocess.check_output(['pmset','-g','therm'],text=True)
                (destination/f'thermal-{step:04d}.txt').write_text(thermal)
                limits=re.findall(r'CPU_Speed_Limit\s*=\s*(\d+)',thermal)
                thermal_limited=any(int(v)<80 for v in limits)
            reasons=[]
            if step>=budget:reasons.append('one_pass_or_step_budget')
            if training_seconds>=7200:reasons.append('active_time_cap')
            if used>24*1024**3:reasons.append('memory_cap')
            if psutil.swap_memory().used-baseline_swap>2*1024**3:reasons.append('swap_growth')
            if psutil.virtual_memory().available<2*1024**3:reasons.append('low_system_memory')
            if thermal_limited:reasons.append('thermal_limit')
            stop=bool(reasons)
            if step%50==0 or stop:
                score=dev_score(model,tok);path=destination/f'step-{step:04d}'
                meta={'step':step,'examples':seen,'tokens':tokens,'active_seconds':training_seconds,'dev':score,
                    'base_revision':config['model']['revision'],'algorithm':cfg['algorithm'],'trainable_dtype':'float32',
                    'backbone_dtype':'bfloat16','reference_kl_beta':cfg['kl_beta'],'reward_safe':[1,-2,-.25],
                    'reward_unsafe':[-2,1,-.25],'group_loss_weights':{str(k):v for k,v in weights.items()},
                    'data_manifest_sha256':hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest()}
                save_adapter(model,path,meta)
                if score['selection_score']>best:best=score['selection_score'];selected=str(path)
                write(destination/'selected_checkpoint.json',{'path':selected,'dev_selection_score':best})
            if stop:break
    write(destination/'summary.json',{'checkpoint':selected,'steps':step,'examples':seen,'tokens':tokens,
        'active_seconds':training_seconds,'wall_seconds':time.monotonic()-start,'peak_mps_driver_bytes':peak,
        'trainable_parameters':sum(p.numel() for n,p in params),'seed':20260922,'one_pass':seen==len(rows),
        'stop_reasons':reasons})

if __name__=='__main__':main()
