"""Train retained backbones and direct classification heads on L20.

Default: head warmup followed by full-parameter post-training. LoRA is an
explicit optional control, not the architecture-improvement experiment.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from experiment_common import read, write, write_rows, serialize, summarize, sha
from speed_probe import benchmark

ROOT=Path('/work')


class LoRA(nn.Module):
    def __init__(self,base,rank=8):
        super().__init__();self.base=base;self.scale=2.
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,x):
        return self.base(x)+((x.float()@self.A.T@self.B.T)*self.scale).to(x.dtype)


def attach(backbone):
    for p in backbone.parameters():p.requires_grad_(False)
    count=0
    for name,module in list(backbone.named_modules()):
        if isinstance(module,nn.Linear):
            parent,_,attr=name.rpartition('.')
            setattr(backbone.get_submodule(parent),attr,LoRA(module));count+=1
    return count


@torch.no_grad()
def merge(backbone):
    for name,module in list(backbone.named_modules()):
        if isinstance(module,LoRA):
            module.base.weight.add_((module.B@module.A*module.scale).to(module.base.weight.dtype))
            parent,_,attr=name.rpartition('.')
            setattr(backbone.get_submodule(parent),attr,module.base)


class Classifier(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;d=backbone.config.hidden_size
        self.heads=nn.ModuleDict({role:nn.ModuleDict({
            'projection':nn.Sequential(nn.Linear(d,512),nn.LayerNorm(512),nn.SiLU()),
            'risk':nn.Linear(512,3),'category':nn.Linear(512,categories)})
            for role,categories in [('user',9),('assistant',8)]})
    def readout(self,hidden,role):
        h=self.heads[role]['projection'](hidden.float())
        return self.heads[role]['risk'](h),self.heads[role]['category'](h)
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.backbone(input_ids=input_ids,attention_mask=attention_mask,
                                 past_key_values=past_key_values,use_cache=use_cache)
    def step(self,ids,cache,cached):
        output=self(ids,past_key_values=cache,use_cache=cached)
        risk,category=self.readout(output.last_hidden_state[:,-1],'user')
        return risk,category,output.past_key_values if cached else None


def enable_fla():
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    def wrap(fn):
        def call(q,k,v,g,beta,initial_state=None,output_final_state=False,
                 use_qk_l2norm_in_kernel=True,cu_seqlens=None,**kwargs):
            return fn(q,k,v,g=g,beta=beta,initial_state=initial_state,
                      output_final_state=output_final_state,
                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,cu_seqlens=cu_seqlens)
        return call
    module.torch_chunk_gated_delta_rule=wrap(chunk_gated_delta_rule)
    module.torch_recurrent_gated_delta_rule=wrap(fused_recurrent_gated_delta_rule)
    return {'gdn_chunk':'fla-0.5.2','gdn_recurrent':'fla-0.5.2','short_convolution':'Transformers torch implementation'}


def load(key):
    path=ROOT/'models'/key
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    kernels={'attention':'torch sdpa'}
    if key=='qwen35':
        kernels.update(enable_fla())
        from transformers import Qwen3_5ForConditionalGeneration
        full,info=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,
                   dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        backbone=full.model.language_model
        del full;gc.collect()
    else:
        backbone,info=AutoModel.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
                      attn_implementation='sdpa',output_loading_info=True)
    missing=info.get('missing_keys',[])
    if missing:raise RuntimeError('Pretrained weights missing: '+str(missing[:8]))
    if hasattr(backbone,'lm_head'):raise RuntimeError('Classifier backbone must not have a vocabulary output head')
    return backbone.to('cuda'),tok,kernels


def encode_rows(tok,rows,max_len=8192):
    output=[]
    for row in rows:
        ids=tok.encode(serialize(row['messages']),add_special_tokens=False)
        if not ids or len(ids)>max_len:
            raise RuntimeError(f"No silent truncation: {row['sample_id']} has {len(ids)} tokens")
        output.append({**row,'ids':ids})
    return output


def batch(rows,pad):
    length=max(len(row['ids']) for row in rows)
    ids=torch.full((len(rows),length),pad,device='cuda',dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        ids[i,:len(row['ids'])]=torch.tensor(row['ids'],device='cuda');mask[i,:len(row['ids'])]=1
    return ids,mask


@torch.inference_mode()
def evaluate(model,tok,rows):
    model.eval();predictions=[]
    ordered=sorted(rows,key=lambda row:len(row['ids']))
    for start in range(0,len(ordered),4):
        chunk=ordered[start:start+4];ids,mask=batch(chunk,tok.pad_token_id or tok.eos_token_id)
        output=model(ids,mask)
        for index,row in enumerate(chunk):
            risk,category=model.readout(output.last_hidden_state[index,len(row['ids'])-1],row['target_role'])
            predictions.append({k:row.get(k) for k in ['sample_id','language','target_role','source_label','source','source_split']}|
                               {'probs':risk.softmax(-1).cpu().tolist(),'category_probs':category.softmax(-1).cpu().tolist(),
                                'input_tokens':len(row['ids'])})
    return predictions


def loss_for(model,rows,pad):
    ids,mask=batch(rows,pad);output=model(ids,mask)
    losses=[]
    for index,row in enumerate(rows):
        risk,category=model.readout(output.last_hidden_state[index,len(row['ids'])-1],row['target_role'])
        teacher=torch.tensor(row['teacher_risk'],device='cuda')
        loss=.3*F.kl_div(risk.log_softmax(-1),teacher,reduction='sum')
        if row.get('source_label') in ('safe','unsafe'):
            target=torch.tensor(0 if row['source_label']=='safe' else 1,device='cuda')
            loss=loss+F.cross_entropy(risk,target)
        if row.get('source_label')=='unsafe' or row['teacher_risk'][1]+row['teacher_risk'][2]>.5:
            target=torch.tensor(row['teacher_category'],device='cuda')
            loss=loss+.2*F.kl_div(category.log_softmax(-1),target,reduction='sum')
        losses.append(loss)
    return torch.stack(losses).mean()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen3','qwen35','llama'],required=True)
    parser.add_argument('--max-steps',type=int,default=1280)
    parser.add_argument('--training-mode',choices=['full','lora'],default='full')
    parser.add_argument('--head-warmup-steps',type=int,default=64)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if torch.cuda.device_count()!=1 or 'L20' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This experiment requires one physical L20')
    torch.set_num_threads(4);torch.manual_seed(20260923);random.seed(20260923)
    run_name=args.model+'_'+args.training_mode
    out=ROOT/'output'/(run_name+'_smoke' if args.smoke_only else run_name);out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed model already exists')
    backbone,tok,kernels=load(args.model)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base_count=sum(p.numel() for p in backbone.parameters())
    if args.training_mode=='lora':
        adapted=attach(backbone)
    else:
        adapted=0
        # FP32 trainable weights and Adam state; BF16 forward under autocast.
        backbone.float().requires_grad_(True)
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=Classifier(backbone).to('cuda')
    manifest=json.loads((ROOT/'output/teacher/manifest.json').read_text())
    data={}
    for split in ('train','dev','test','official_test'):
        path=ROOT/f'output/teacher/{split}.jsonl'
        if sha(path)!=manifest['hashes'][split]:raise RuntimeError('Teacher data hash mismatch')
        data[split]=encode_rows(tok,read(path))
    max_training_length=max(len(row['ids']) for row in data['train'])
    backbone_trainable=[p for p in backbone.parameters() if p.requires_grad]
    heads=list(model.heads.parameters())
    base_lr=1e-5 if args.training_mode=='full' else 1e-4
    head_lr=1e-4
    optimizer=torch.optim.AdamW([{'params':backbone_trainable,'lr':base_lr},{'params':heads,'lr':head_lr}],weight_decay=.01)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write(out/'setup.json',{'base_parameters':base_count,'trainable_parameters':trainable,
                           'adapted_linear_modules':adapted,'kernels':kernels,'max_training_tokens':max_training_length,
                           'data_counts':{s:len(r) for s,r in data.items()},'has_lm_head':False,
                           'training_mode':args.training_mode,'head_warmup_steps':args.head_warmup_steps,
                           'architecture_change':'direct classification heads only; original token-mixing layers retained',
                           'training_status':'source_label_and_teacher_probe_not_project_aligned'})
    print(json.dumps({'model':args.model,'stage':'loaded','trainable':trainable,'kernels':kernels}),flush=True)
    if args.smoke_only:
        model.train();loss=loss_for(model,data['train'][:4],tok.pad_token_id);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(backbone_trainable+heads,1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Smoke non-finite loss/gradient')
        if not any(p.grad is not None and bool(p.grad.abs().max()>0) for p in backbone_trainable):
            raise RuntimeError('Smoke: no backbone gradients')
        model.eval()
        with torch.inference_mode():
            probe=torch.tensor([data['dev'][0]['ids']],device='cuda')
            full,_,_=model.step(probe,None,False)
            cut=max(1,probe.shape[1]//2)
            _,_,cache=model.step(probe[:,:cut],None,True)
            chunk,_,_=model.step(probe[:,cut:],cache,True)
            difference=float((full.softmax(-1)-chunk.softmax(-1)).abs().max())
            if difference>.03:raise RuntimeError(f'Smoke stream parity failed: {difference}')
        write(out/'summary.json',{'model':args.model,'backward_finite':True,'loss':float(loss),
                                  'gradient_norm':float(grad),'full_stream_probability_error':difference,
                                  'training_mode':args.training_mode,'backbone_gradients_verified':True,
                                  'kernels':kernels,'physical_gpu':torch.cuda.get_device_name(0)})
        print(json.dumps({'model':args.model,'smoke':'passed','parity_error':difference}),flush=True)
        return
    initial=evaluate(model,tok,data['dev']);write_rows(out/'initial_dev_predictions.jsonl',initial)
    # One deterministic pass; length buckets reduce padding without truncating labels.
    ordered=sorted(data['train'],key=lambda r:len(r['ids']))
    batches=[ordered[i:i+4] for i in range(0,len(ordered),4)]
    random.Random(20260923).shuffle(batches)
    batches=batches[:args.max_steps]
    if not batches:raise ValueError('At least one training batch required')
    warmup_steps=min(args.head_warmup_steps,max(0,len(batches)-1))
    if args.training_mode=='full' and warmup_steps:
        backbone.requires_grad_(False)
    model.train();torch.cuda.reset_peak_memory_stats();start=time.monotonic();actual_tokens=0
    losses=[]
    with (out/'losses.jsonl').open('w') as log:
        for step,rows in enumerate(batches,1):
            if args.training_mode=='full' and step==warmup_steps+1:
                backbone.requires_grad_(True)
                print(json.dumps({'model':args.model,'stage':'full_backbone_unfrozen','step':step}),flush=True)
            progress=step/len(batches);scale=min(1.,step/max(1,len(batches)*.05))*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            optimizer.param_groups[0]['lr']=base_lr*scale;optimizer.param_groups[1]['lr']=head_lr*scale
            optimizer.zero_grad(set_to_none=True);loss=loss_for(model,rows,tok.pad_token_id)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(backbone_trainable+heads,1.)
            if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
            optimizer.step();actual_tokens+=sum(len(r['ids']) for r in rows)
            value=float(loss.detach());losses.append(value)
            log.write(json.dumps({'step':step,'loss':value,'tokens':actual_tokens,
                                 'stage':'heads' if args.training_mode=='full' and step<=warmup_steps else args.training_mode})+'\n')
            if step%50==0:
                log.flush();print(json.dumps({'model':args.model,'step':step,'steps':len(batches),'loss':sum(losses[-50:])/min(50,len(losses)),'seconds':round(time.monotonic()-start,1)}),flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    model.eval();dev=evaluate(model,tok,data['dev'])
    if args.training_mode=='lora':
        checkpoint=out/'adapter_and_heads.safetensors'
        state={n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
        save_file(state,str(checkpoint));del state
        merge(backbone)
    else:
        checkpoint=out/'classifier.safetensors'
        backbone.to(torch.bfloat16)
        state={n:p.detach().cpu().contiguous() for n,p in model.state_dict().items()}
        save_file(state,str(checkpoint));del state
        backbone.config.to_json_file(out/'backbone_config.json')
        tok.save_pretrained(out/'tokenizer')
    merged_dev=evaluate(model,tok,data['dev'])
    by_id={r['sample_id']:r for r in dev}
    error=max(abs(a-b) for row in merged_dev for a,b in zip(row['probs'],by_id[row['sample_id']]['probs']))
    if error>.03:raise RuntimeError(f'Exported classifier risk error {error}')
    del optimizer,backbone_trainable,heads;gc.collect();torch.cuda.empty_cache()
    predictions={'dev':merged_dev,'test':evaluate(model,tok,data['test']),
                 'official_test':evaluate(model,tok,data['official_test'])}
    for split,rows in predictions.items():write_rows(out/f'{split}_predictions.jsonl',rows)
    write(out/'source_metrics.json',summarize(predictions['dev'],predictions['test']))
    write(out/'official_complete_metrics.json',summarize(predictions['dev'],predictions['official_test']))
    reference=AutoTokenizer.from_pretrained(ROOT/'models/qwen3',local_files_only=True)
    measured=benchmark(model.step,tok,reference);write(out/'speed.json',measured)
    write(out/'summary.json',{'model':args.model,'status':'completed','steps':len(batches),
                             'trained_examples':sum(map(len,batches)),'native_training_tokens':actual_tokens,
                             'training_seconds':seconds,'training_tokens_per_second':actual_tokens/seconds,
                             'base_parameters':base_count,'trainable_parameters':trainable,
                             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                             'training_mode':args.training_mode,'head_only_steps':warmup_steps if args.training_mode=='full' else 0,
                             'export_max_risk_error':error,'checkpoint_sha256':sha(checkpoint),
                             'has_lm_head':False,'source_policy_probe_only':True,'kernels':kernels,
                             'not_claimed_converged':True})
    print(json.dumps({'model':args.model,'status':'completed'}),flush=True)


if __name__=='__main__':main()
