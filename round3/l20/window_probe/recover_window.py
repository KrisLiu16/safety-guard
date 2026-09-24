"""Matched L20 full-history/window recovery pilot; no generated output tokens."""
import gc
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path[:0]=['/work/input','/work/window']
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from train_base import load,Classifier,encode_rows,evaluate,loss_for,batch
from experiment_common import read,write,write_rows,summarize,sha
from window_attention import set_window

ROOT=Path('/work');OUT=ROOT/'output/window_recovery_v1'


def classifier():
    backbone,tok,_=load('qwen35');model=Classifier(backbone).to('cuda')
    path=ROOT/'output/qwen35_full/classifier.safetensors'
    spec=json.loads(path.with_name('summary.json').read_text());assert sha(path)==spec['checkpoint_sha256']
    state=load_file(str(path));model.load_state_dict(state,strict=True);del state
    return model,tok,spec['checkpoint_sha256']


def targets(teacher,row):
    ids=torch.tensor([row['ids']],device='cuda')
    indices=sorted(set([1023,len(row['ids'])-1]))
    with torch.no_grad():
        h=teacher(ids).last_hidden_state[0,indices].float()
        risk,category=teacher.readout(h,'user')
    return ids,indices,h,risk.softmax(-1),category.softmax(-1)


def recovery_loss(model,bundle):
    ids,indices,hidden,risk,category=bundle
    actual=model(ids).last_hidden_state[0,indices].float()
    r,c=model.readout(actual,'user')
    cosine=(1-F.cosine_similarity(actual,hidden,dim=-1)).mean()
    risk_kl=F.kl_div(r.log_softmax(-1),risk,reduction='batchmean')
    category_kl=F.kl_div(c.log_softmax(-1),category,reduction='batchmean')
    return cosine+.3*risk_kl+.1*category_kl,cosine,risk_kl


def short_preservation(model,teacher,rows,pad):
    ids,mask=batch(rows,pad)
    with torch.no_grad():th=teacher(ids,mask).last_hidden_state
    sh=model(ids,mask).last_hidden_state;losses=[]
    for i,row in enumerate(rows):
        end=len(row['ids'])-1;s=sh[i,end].float();t=th[i,end].float()
        sr,sc=model.readout(s,row['target_role'])
        with torch.no_grad():tr,tc=teacher.readout(t,row['target_role'])
        losses.append(1-F.cosine_similarity(s,t,dim=-1)+.3*F.kl_div(sr.log_softmax(-1),tr.softmax(-1),reduction='sum')
                      +.1*F.kl_div(sc.log_softmax(-1),tc.softmax(-1),reduction='sum'))
    return torch.stack(losses).mean()


def export_state(model):
    return {k:v.detach().to(device='cpu',dtype=torch.bfloat16 if k.startswith('backbone.') else v.dtype).contiguous()
            for k,v in model.state_dict().items()}


@torch.no_grad()
def evaluate_long(model,teacher,rows):
    model.eval();results=[]
    for row in rows:
        bundle=targets(teacher,row)
        loss,cosine,kl=recovery_loss(model,bundle)
        results.append({'sample_id':row['sample_id'],'tokens':len(row['ids']),
                        'loss':float(loss),'cosine_distance':float(cosine),'risk_kl':float(kl)})
    return {'rows':results,'mean_cosine_distance':sum(r['cosine_distance'] for r in results)/len(results),
            'mean_risk_kl':sum(r['risk_kl'] for r in results)/len(results),
            'scope':'held-out document domains; teacher agreement, not safety ground truth'}


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['joint','structure'],default='joint')
    args=parser.parse_args();structure=args.mode=='structure'
    if structure:OUT=ROOT/'output/window_recovery_v2'
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);torch.manual_seed(20260923)
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'recovery/data/manifest.json').read_text())
    for split in ['train','dev']:assert sha(ROOT/f'recovery/data/{split}.jsonl')==manifest['output_hashes'][split]
    assert sha(ROOT/'output/qwen35_full/tokenizer/tokenizer.json')==manifest['tokenizer_sha256']
    long_train=read(ROOT/'recovery/data/train.jsonl');long_dev=read(ROOT/'recovery/data/dev.jsonl')
    teacher,tok,source_sha=classifier();teacher.eval().requires_grad_(False)
    data={s:encode_rows(tok,read(ROOT/f'output/teacher/{s}.jsonl')) for s in ('train','dev','test','official_test')}
    short=[r for r in data['train'] if r.get('source_label') in ('safe','unsafe')]
    random.Random(20260923).shuffle(short);short=short[:len(long_train)*4]
    assert len(short)==len(long_train)*4
    write(OUT/'run_spec.json',{'source_checkpoint_sha256':source_sha,'long_manifest':manifest,
                              'short_sample_ids':[r['sample_id'] for r in short],
                              'steps_per_branch':len(long_train),'branches':['full','w512'],
                              'loss':'long same-visible-prefix cosine + risk/category KL; short H24 preservation' if structure else 'source/teacher classification + long cosine + 0.3 risk KL + 0.1 category KL',
                              'all_backbone_parameters_updated':True,'classification_heads_frozen':structure,
                              'source_hard_labels_used':not structure,'rl':False,'mode':args.mode})
    for name,window in [('full',None),('w512',512)]:
        out=OUT/name;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():raise FileExistsError('Finished branch exists; do not retrain silently')
        model,_,_=classifier();set_window(model.backbone,window);model.eval().requires_grad_(False)
        before=evaluate_long(model,teacher,long_dev);write(out/'long_dev_before.json',before)
        best_state=export_state(model) if structure else None
        best_score=before['mean_cosine_distance'];best_step=0;dev_history=[]
        model.backbone.float();model.train().requires_grad_(True)
        if structure:
            model.heads.requires_grad_(False)
            # Keep the teacher/student residual-stream compute dtype identical
            # while retaining FP32 master weights and optimizer state.
            model.backbone.embed_tokens.register_forward_hook(lambda module,inputs,output:output.to(torch.bfloat16))
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':3e-6},
                                     {'params':model.heads.parameters(),'lr':1e-5}],weight_decay=.01)
        start=time.monotonic();log=[]
        with (out/'losses.jsonl').open('w') as f:
            for step,row in enumerate(long_train):
                bundle=targets(teacher,row)
                optimizer.zero_grad(set_to_none=True)
                # Separate backward passes bound activation memory without altering the summed objective.
                long_loss,cosine,kl=recovery_loss(model,bundle);long_loss.backward()
                if structure:
                    short_loss=.1*short_preservation(model,teacher,short[step*4:step*4+4],tok.pad_token_id)
                else:short_loss=loss_for(model,short[step*4:step*4+4],tok.pad_token_id)
                short_loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad) or not torch.isfinite(long_loss+short_loss):raise RuntimeError('Nonfinite recovery gradient/loss')
                optimizer.step()
                entry={'step':step+1,'long_loss':float(long_loss),'source_loss':float(short_loss),
                       'long_tokens':len(row['ids']),'seconds':time.monotonic()-start}
                f.write(json.dumps(entry)+'\n');log.append(entry)
                if (step+1)%16==0:
                    f.flush();print(json.dumps({'branch':name,**entry}),flush=True)
                del bundle,long_loss,short_loss
                if structure and (step+1)%32==0:
                    measured=evaluate_long(model,teacher,long_dev)
                    score=measured['mean_cosine_distance']
                    dev_history.append({'step':step+1,'cosine_distance':score,'risk_kl':measured['mean_risk_kl']})
                    if score<best_score:
                        best_score=score;best_step=step+1;best_state=export_state(model)
                    model.train()
        seconds=time.monotonic()-start
        del optimizer;model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
        if structure:
            model.load_state_dict(best_state,strict=True);del best_state
            write(out/'dev_selection.json',{'criterion':'minimum held-out long-document cosine distance; includes step 0',
                                            'selected_step':best_step,'history':dev_history})
        model.backbone.to(torch.bfloat16)
        state=export_state(model)
        checkpoint=out/'classifier.safetensors';save_file(state,str(checkpoint));del state
        write(out/'architecture.json',{'window':window,'query_block':128,'layers_removed':0,
                                      'loader':'Classifier + set_window(backbone, window); no LM head'})
        after=evaluate_long(model,teacher,long_dev);write(out/'long_dev_after.json',after)
        predictions={s:evaluate(model,tok,data[s]) for s in ('dev','test','official_test')}
        for split,rows in predictions.items():write_rows(out/f'{split}_predictions.jsonl',rows)
        write(out/'source_metrics.json',summarize(predictions['dev'],predictions['test']))
        write(out/'official_complete_metrics.json',summarize(predictions['dev'],predictions['official_test']))
        with torch.inference_mode():
            probe=torch.tensor([long_dev[0]['ids']],device='cuda');cut=1301
            full,_,_=model.step(probe,None,False)
            _,_,cache=model.step(probe[:,:cut],None,True)
            chunk,_,_=model.step(probe[:,cut:],cache,True)
            error=float((full.softmax(-1)-chunk.softmax(-1)).abs().max());assert error<.03,error
        write(out/'summary.json',{'status':'completed','branch':name,'steps':len(long_train),
                                 'long_train_tokens':sum(len(r['ids']) for r in long_train),
                                 'short_examples':len(short),'training_seconds':seconds,
                                 'checkpoint_sha256':sha(checkpoint),'has_lm_head':False,'training':'full_parameters',
                                 'classification_heads_frozen':structure,'source_hard_labels_used':not structure,
                                 'selected_step':best_step if structure else len(long_train),'mode':args.mode,
                                 'long_dev_cosine_before':before['mean_cosine_distance'],
                                 'long_dev_cosine_after':after['mean_cosine_distance'],
                                 'full_stream_probability_error':error,
                                 'not_claimed_policy_aligned_or_converged':True})
        print(json.dumps({'branch':name,'completed':True}),flush=True)
        del model,cache,full,chunk,probe;gc.collect();torch.cuda.empty_cache()
    write(OUT/'summary.json',{'status':'completed','branches':['full','w512'],'safety_gold':False,
                             'same_initial_weights_and_training_examples':True})


if __name__=='__main__':main()
