"""L20 risk-supervised comparison and discrete-classification RL.

No teacher representation/logit matching. Selection uses held-out risk labels.
No token generation: RL samples classification actions from one forward pass.
"""
import gc,json,math,random,sys,time
from pathlib import Path
sys.path[:0]=['/work/input','/work/window','/work/round4']
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from risk_metrics import metrics
from train_base import load,Classifier,batch
from experiment_common import read,write,write_rows,sha
from window_attention import set_window
from memory_attention import attach_memory
from memory_attention import memory_state_bytes
from window_attention import cache_storage
from experiment_common import quantiles

ROOT=Path('/work');OUT=ROOT/'output/round4';DATA=ROOT/'round4/data'
SEED=20260925;EPOCHS=2;EFFECTIVE_BATCH=16;MICROBATCH=4


def configure(model,variant,window=512):
    if variant=='full':return
    if variant=='window':set_window(model.backbone,window)
    else:attach_memory(model.backbone,window)


def make_model(variant,checkpoint=None):
    backbone,tok,_=load('qwen35');model=Classifier(backbone).to('cuda')
    initial=ROOT/'output/qwen35_full/classifier.safetensors'
    assert sha(initial)==json.loads(initial.with_name('summary.json').read_text())['checkpoint_sha256']
    if checkpoint is None:
        state=load_file(str(initial));model.load_state_dict(state,strict=True);del state
        configure(model,variant)
    else:
        configure(model,variant)
        state=load_file(str(checkpoint));model.load_state_dict(state,strict=True);del state
    return model,tok


def training_mode(model):
    model.backbone.float();model.train().requires_grad_(True)
    model.backbone.embed_tokens.register_forward_hook(lambda module,inputs,output:output.to(torch.bfloat16))
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # Existing category fields are preserved in the checkpoint, but this round
    # trains and validates risk only; no category accuracy is claimed.
    for role in model.heads:model.heads[role]['category'].requires_grad_(False)


def risk_logits(model,rows,pad):
    ids,mask=batch(rows,pad);hidden=model(ids,mask).last_hidden_state
    last=hidden[torch.arange(len(rows),device='cuda'),mask.sum(1)-1]
    slots=[None]*len(rows)
    for role in ('user','assistant'):
        indices=[i for i,r in enumerate(rows) if r['target_role']==role]
        if indices:
            logits,_=model.readout(last[indices],role)
            for j,i in enumerate(indices):slots[i]=logits[j]
    return torch.stack(slots)


@torch.inference_mode()
def predict(model,rows,pad):
    model.eval();result=[]
    ordered=sorted(rows,key=lambda r:len(r['ids']))
    for i in range(0,len(ordered),4):
        group=ordered[i:i+4];ps=risk_logits(model,group,pad).softmax(-1).cpu().tolist()
        result.extend({'sample_id':r['sample_id'],'family':r['family'],'language':r['language'],
                       'target_role':r['target_role'],'source_label':r['source_label'],'probs':p} for r,p in zip(group,ps))
    return result


def export(model,path):
    state={n:p.detach().to(device='cpu',dtype=torch.bfloat16 if n.startswith('backbone.') else p.dtype).contiguous()
           for n,p in model.state_dict().items()}
    temporary=path.with_name(path.name+'.tmp')
    save_file(state,str(temporary));temporary.replace(path);del state


def groups_for_epoch(rows,epoch):
    rng=random.Random(SEED+epoch);ordered=sorted(rows,key=lambda r:len(r['ids']));groups=[]
    for start in range(0,len(ordered),256):
        block=ordered[start:start+256];rng.shuffle(block)
        groups.extend(block[i:i+EFFECTIVE_BATCH] for i in range(0,len(block),EFFECTIVE_BATCH))
    rng.shuffle(groups);return groups


def validate(model,tok,variant,data):
    configure(model,variant,512)
    cal=predict(model,data['calibration'],tok.pad_token_id)
    dev=predict(model,data['dev'],tok.pad_token_id)
    return metrics(cal,dev)


@torch.inference_mode()
def runtime_metrics(model,tok,variant,dev):
    from run_probe import short_recurrent
    short_recurrent(True);model.eval();configure(model,variant,512)
    maximum=0.
    # Actual risk examples, both target roles, with exact token-ID chunking.
    cases=sorted(dev,key=lambda r:r['sample_id'])[:16]
    for row in cases:
        ids=torch.tensor([row['ids']],device='cuda')
        full=model(ids).last_hidden_state[:,-1];fr,_=model.readout(full,row['target_role'])
        state=None;offset=0;i=0
        while offset<ids.shape[1]:
            end=min(ids.shape[1],offset+[1,8,32,127][i%4])
            o=model(ids[:,offset:end],past_key_values=state,use_cache=True);state=o.past_key_values
            r,_=model.readout(o.last_hidden_state[:,-1],row['target_role']);offset=end;i+=1
        maximum=max(maximum,float((fr.softmax(-1)-r.softmax(-1)).abs().max()))
    phrase=tok.encode('公开资料与 public information。\n',add_special_tokens=False)
    whole=torch.tensor([(phrase*2000)[:9216]],device='cuda');records=[]
    for context in (512,8192):
        for chunk in (1,8,32):
            _,_,state=model.step(whole[:,:context],None,True)
            for _ in range(4):_,_,state=model.step(whole[:,context:context+chunk],state,True)
            del state;gc.collect();torch.cuda.empty_cache()
            _,_,state=model.step(whole[:,:context],None,True)
            stored=cache_storage(state)['unique_storage_bytes']
            if hasattr(state,'memories'):stored+=memory_state_bytes(state)
            times=[];torch.cuda.synchronize();started=time.perf_counter()
            for i in range(24):
                t=time.perf_counter();r,_,state=model.step(whole[:,context+i*chunk:context+(i+1)*chunk],state,True)
                r.softmax(-1).cpu().tolist();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            elapsed=time.perf_counter()-started
            records.append({'context_tokens':context,'chunk_tokens':chunk,'new_input_tokens':24*chunk,
                            'itps':24*chunk/elapsed,'classifications_per_second':24/elapsed,
                            'state_bytes_at_prefill':stored,**quantiles(times)})
            del state;gc.collect();torch.cuda.empty_cache()
    return {'risk_example_stream_probability_error':maximum,'stream_parity_pass':maximum<.03,
            'risk_examples_checked':len(cases),'generated_tokens':0,'records':records,
            'scope':'L20 warmed single-session token-ID model calls; CPU-visible risk result; no text tokenization, HTTP or batching'}


def sft(variant,data):
    out=OUT/variant;out.mkdir(exist_ok=True)
    if (out/'sft_summary.json').exists():return json.loads((out/'sft_summary.json').read_text())
    torch.manual_seed(SEED);model,tok=make_model(variant)
    initial=validate(model,tok,variant,data);write(out/'dev_initial.json',initial)
    best=initial['selection_score'];best_step=0;checkpoint=out/'best.safetensors';export(model,checkpoint)
    training_mode(model)
    memory=[p for n,p in model.named_parameters() if 'memory_' in n and p.requires_grad]
    heads=[p for n,p in model.named_parameters() if n.startswith('heads.') and p.requires_grad]
    backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and 'memory_' not in n and p.requires_grad]
    spec=[{'params':backbone,'lr':8e-6,'base_lr':8e-6},{'params':heads,'lr':5e-5,'base_lr':5e-5}]
    if memory:spec.append({'params':memory,'lr':3e-4,'base_lr':3e-4})
    optimizer=torch.optim.AdamW(spec,weight_decay=.01,eps=1e-6)
    all_groups=[(epoch,g) for epoch in range(EPOCHS) for g in groups_for_epoch(data['train'],epoch)]
    total=len(all_groups);start=time.monotonic();tokens=0;curve=[]
    write(out/'setup.json',{'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),
                          'total_steps':total,'epochs':EPOCHS,'effective_batch':EFFECTIVE_BATCH,'microbatch':MICROBATCH,
                          'loss':'weighted risk cross entropy only','teacher_alignment':False,'category_supervision':False,
                          'window_curriculum':[2048,1024,512] if variant!='full' else None,'selection_window':512 if variant!='full' else None})
    with (out/'losses.jsonl').open('w') as log:
        for step,(epoch,group) in enumerate(all_groups,1):
            fraction=step/total;window=2048 if fraction<=.25 else 1024 if fraction<=.5 else 512
            configure(model,variant,window);model.train();optimizer.zero_grad(set_to_none=True)
            lr_scale=min(1.,step/64)*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for param_group in optimizer.param_groups:param_group['lr']=param_group['base_lr']*lr_scale
            value=0.
            for begin in range(0,len(group),MICROBATCH):
                rows=group[begin:begin+MICROBATCH];logits=risk_logits(model,rows,tok.pad_token_id)
                targets=torch.tensor([int(r['source_label']=='unsafe') for r in rows],device='cuda')
                weights=torch.tensor([r.get('weight',1.) for r in rows],device='cuda')
                loss=(F.cross_entropy(logits,targets,reduction='none')*weights).sum()/len(group)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised risk loss')
                loss.backward();value+=float(loss);tokens+=sum(len(r['ids']) for r in rows)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            record={'step':step,'steps':total,'epoch':epoch,'loss':value,'input_tokens':tokens,
                    'training_window':None if variant=='full' else window,'seconds':time.monotonic()-start}
            log.write(json.dumps(record)+'\n')
            if step%64==0:log.flush();print(json.dumps({'variant':variant,**record}),flush=True)
            if step%512==0 or step==total:
                measured=validate(model,tok,variant,data);curve.append({'step':step,**measured})
                write(out/'dev_curve.json',curve)
                if measured['selection_score']>best:
                    best=measured['selection_score'];best_step=step;export(model,checkpoint)
                # One resumable latest state for the active branch, atomically replaced.
                temporary=OUT/'latest_training_state.tmp'
                torch.save({'variant':variant,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},temporary)
                temporary.replace(OUT/'latest_training_state.pt')
                print(json.dumps({'variant':variant,'validation_step':step,'score':measured['selection_score'],'best_step':best_step}),flush=True)
    del optimizer
    model.zero_grad(set_to_none=True);model.eval().requires_grad_(False);model.backbone.to(torch.bfloat16)
    selected=load_file(str(checkpoint));model.load_state_dict(selected,strict=True);del selected
    artifact_dev=validate(model,tok,variant,data);write(out/'exported_dev_metrics.json',artifact_dev)
    runtime=runtime_metrics(model,tok,variant,data['dev']);write(out/'runtime.json',runtime)
    result={'variant':variant,'status':'completed','steps':total,'selected_step':best_step,
            'dev_score':artifact_dev['selection_score'],'best_training_dev_score':best,
            'input_tokens':tokens,'seconds':time.monotonic()-start,'checkpoint_sha256':sha(checkpoint),'has_lm_head':False}
    result['stream_parity_pass']=runtime['stream_parity_pass']
    write(out/'sft_summary.json',result)
    del model,backbone,heads,memory;gc.collect();torch.cuda.empty_cache()
    return result


def classification_rl(variant,data):
    out=OUT/'classification_rl';out.mkdir(exist_ok=True)
    reference,tok=make_model(variant,OUT/variant/'best.safetensors');reference.eval().requires_grad_(False)
    model,_,=make_model(variant,OUT/variant/'best.safetensors');training_mode(model)
    configure(model,variant,512);configure(reference,variant,512)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-7,weight_decay=0.,eps=1e-6)
    pool=[r for r in data['train'] if r['label_tier'] in ('public_source_reference','rubric_benign','synthetic_weak')]
    rng=random.Random(SEED+91);rng.shuffle(pool)
    before=validate(model,tok,variant,data);best=before['selection_score'];best_step=0
    checkpoint=out/'best.safetensors';export(model,checkpoint);curve=[]
    torch.manual_seed(SEED+91);start=time.monotonic()
    with (out/'trajectory_metrics.jsonl').open('w') as log:
        for step in range(1,257):
            rows=[pool[((step-1)*8+i)%len(pool)] for i in range(8)]
            model.train();optimizer.zero_grad(set_to_none=True)
            logits=risk_logits(model,rows,tok.pad_token_id);pu=logits.softmax(-1)[:,1].clamp(1e-5,1-1e-5)
            with torch.no_grad():pref=risk_logits(reference,rows,tok.pad_token_id).softmax(-1)[:,1].clamp(1e-5,1-1e-5)
            y=torch.tensor([int(r['source_label']=='unsafe') for r in rows],device='cuda')
            weights=torch.tensor([r.get('weight',1.) for r in rows],device='cuda')
            actions=torch.bernoulli(pu.detach()[:,None].expand(-1,4))
            # Reward rubric: correct +1, false positive -3, false negative -4.
            r0=torch.where(y==0,1.,-4.);r1=torch.where(y==1,1.,-3.)
            rewards=torch.where(actions.bool(),r1[:,None],r0[:,None])
            baseline=(1-pu.detach())*r0+pu.detach()*r1
            logp=actions*pu[:,None].log()+(1-actions)*(1-pu[:,None]).log()
            policy_loss=-((rewards-baseline[:,None])*logp).mean(1)
            kl=pu*(pu.log()-pref.log())+(1-pu)*((1-pu).log()-(1-pref).log())
            ce=F.cross_entropy(logits,y,reduction='none')
            loss=((policy_loss+.1*kl+.2*ce)*weights).mean();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite classification RL update')
            optimizer.step()
            log.write(json.dumps({'step':step,'weighted_reward':float((rewards.mean(1)*weights).mean()),
                                  'policy_loss':float(policy_loss.mean()),'reference_policy_kl':float(kl.mean()),
                                  'mean_false_positive_actions':float((actions[y==0]==1).float().mean()) if (y==0).any() else None,
                                  'mean_false_negative_actions':float((actions[y==1]==0).float().mean()) if (y==1).any() else None})+'\n')
            if step%64==0:
                log.flush();m=validate(model,tok,variant,data);curve.append({'step':step,**m});write(out/'dev_curve.json',curve)
                if m['selection_score']>best:best=m['selection_score'];best_step=step;export(model,checkpoint)
                print(json.dumps({'stage':'classification_rl','step':step,'dev_score':m['selection_score'],'best_step':best_step}),flush=True)
    del optimizer
    model.zero_grad(set_to_none=True);model.eval().requires_grad_(False);model.backbone.to(torch.bfloat16)
    selected=load_file(str(checkpoint));model.load_state_dict(selected,strict=True);del selected
    artifact_dev=validate(model,tok,variant,data);write(out/'exported_dev_metrics.json',artifact_dev)
    write(out/'summary.json',{'status':'completed','variant':variant,'updates':256,'selected_step':best_step,
                             'dev_before':before,'best_dev_score':best,'checkpoint_sha256':sha(checkpoint),
                             'exported_dev_score':artifact_dev['selection_score'],
                             'algorithm':'REINFORCE with exact per-input reward baseline, reference-policy KL and supervised anchor',
                             'actions_per_input':4,'generated_tokens':0,'reward':{'correct':1,'false_positive':-3,'false_negative':-4},
                             'weak_labels_downweighted':True,'seconds':time.monotonic()-start,
                             'scope':'classification actions; not sequential allow/hold/block control'} )
    del model,reference;gc.collect();torch.cuda.empty_cache()


def main():
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    rubric=json.loads((ROOT/'round4/rubric.json').read_text())
    assert rubric['reward']=={'correct':1,'false_positive':-3,'false_negative':-4}
    for split,h in manifest['output_hashes'].items():assert sha(DATA/(split+'.jsonl'))==h
    data={s:read(DATA/(s+'.jsonl')) for s in ('train','calibration','dev')}
    assert all(r['weight']==rubric['data_weights'][r['label_tier']] for r in data['train'])
    write(OUT/'risk_run_spec.json',{'data_manifest':manifest,'variants':['full','window','memory'],
                                  'epochs':EPOCHS,'semantic_teacher_loss':False,'rubric':rubric,
                                  'objective':'risk labels, false positives and false negatives',
                                  'category_supervision':False,'third_risk_class_has_no_new_hard_supervision':True,
                                  'selection_uses_sealed_test':False})
    summaries=[sft(v,data) for v in ('full','window','memory')]
    eligible=[r for r in summaries if r['stream_parity_pass']]
    if not eligible:raise RuntimeError('No candidate passed streaming correctness')
    best=max(eligible,key=lambda x:x['dev_score'])['variant']
    classification_rl(best,data)
    # Read the new test set only after all model/checkpoint choices are fixed.
    sealed=read(DATA/'sealed_test.jsonl')
    final={}
    for name,variant,path in [(v,v,OUT/v/'best.safetensors') for v in ('full','window','memory')]+[('classification_rl',best,OUT/'classification_rl/best.safetensors')]:
        model,tok=make_model(variant,path);model.eval().requires_grad_(False)
        cal=predict(model,data['calibration'],tok.pad_token_id);test=predict(model,sealed,tok.pad_token_id)
        write_rows(OUT/name/'sealed_test_predictions.jsonl',test);final[name]=metrics(cal,test)
        write(OUT/name/'sealed_test_metrics.json',final[name])
        if name=='classification_rl':write(OUT/name/'runtime.json',runtime_metrics(model,tok,variant,data['dev']))
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/'final_summary.json',{'status':'completed','sft':summaries,'rl_base_variant':best,'sealed_test':final,
                                  'has_lm_head':False,'production_approval':False})
    print(json.dumps({'round4':'completed','rl_base_variant':best}),flush=True)


if __name__=='__main__':main()
