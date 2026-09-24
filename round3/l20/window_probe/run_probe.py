"""L20-only H24 window correctness, quality and warmed performance experiment."""
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/work/input')
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from train_base import load,Classifier,encode_rows,evaluate,loss_for,enable_fla
from experiment_common import read,write,write_rows,summarize,sha,quantiles
from window_attention import set_window,tiled_window_attention,cache_storage

OUT=Path('/work/output/window_probe_v1')


def oracle_test():
    torch.manual_seed(3407)
    q=torch.randn(2,4,97,16,device='cuda',dtype=torch.float64,requires_grad=True)
    k=torch.randn(2,2,116,16,device='cuda',dtype=torch.float64,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.ones(2,116,device='cuda',dtype=torch.bool);mask[1,-13:]=False
    actual=tiled_window_attention(q,k,v,33,16,mask)
    qi=torch.arange(19,116,device='cuda')[:,None];ki=torch.arange(116,device='cuda')[None,:]
    permitted=((ki<=qi)&(ki>qi-33))[None,None]&mask[:,None,None,:]
    expected=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=permitted)
    delta=(actual-expected).abs().max().item()
    ag=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
    eg=torch.autograd.grad(expected.square().sum(),(q,k,v))
    grad_delta=max((a-b).abs().max().item() for a,b in zip(ag,eg))
    assert delta<1e-8 and grad_delta<1e-8,(delta,grad_delta)
    return {'independent_dense_mask_output_error':delta,'gradient_error':grad_delta,
            'includes_gqa_padding_nonzero_history':True}


def short_recurrent(enabled):
    enable_fla()
    if not enabled:return
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    original=module.torch_chunk_gated_delta_rule
    def dispatch(q,k,v,g,beta,initial_state=None,output_final_state=False,
                 use_qk_l2norm_in_kernel=True,cu_seqlens=None,**kwargs):
        fn=fused_recurrent_gated_delta_rule if (not torch.is_grad_enabled() and initial_state is not None and q.shape[1]<=32) else original
        return fn(q,k,v,g=g,beta=beta,initial_state=initial_state,output_final_state=output_final_state,
                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,cu_seqlens=cu_seqlens,**kwargs)
    module.torch_chunk_gated_delta_rule=dispatch


@torch.inference_mode()
def probs(model,ids,cache=None,cached=False):
    r,_,state=model.step(ids,cache,cached)
    return r.float().softmax(-1),state


@torch.inference_mode()
def correctness(model,tok):
    ids=torch.tensor([tok.encode('USER:\n'+('公开资料用于阅读理解。 Public information is for reading.\n'*400),add_special_tokens=False)[:2305]],device='cuda')
    set_window(model.backbone,None);full,_=probs(model,ids[:,:511])
    set_window(model.backbone,512);same,_=probs(model,ids[:,:511])
    unchanged=float((full-same).abs().max());assert unchanged<.03,unchanged
    results=[]
    for window in (512,1024):
        set_window(model.backbone,window);expected,_=probs(model,ids)
        for schedule in ([257,1,7,31,128],[512,512,512,512,257]):
            position=0;state=None;index=0
            while position<ids.shape[1]:
                n=schedule[index%len(schedule)];end=min(position+n,ids.shape[1])
                actual,state=probs(model,ids[:,position:end],state,True)
                position=end;index+=1
            delta=float((expected-actual).abs().max());storage=cache_storage(state)
            assert delta<.03,(window,schedule,delta)
            assert storage['logical_tokens_seen']==ids.shape[1]
            assert len(storage['attention_layers'])==6
            assert all(x['retained_tokens']==window-1 for x in storage['attention_layers'])
            before=storage['unique_storage_bytes']
            _,state=probs(model,ids[:,:512],state,True)
            after=cache_storage(state)['unique_storage_bytes']
            assert before==after,(before,after)
            results.append({'window':window,'chunk_schedule':schedule,'probability_error':delta,
                            'cache':storage,'storage_after_512_more_tokens':after})
    # A later suffix cannot affect any earlier hidden state.
    set_window(model.backbone,512)
    a=model(ids[:,:700]).last_hidden_state[:,:500].clone()
    edited=ids[:,:700].clone();edited[:,500:]=(edited[:,500:]+1)%model.backbone.config.vocab_size
    b=model(edited).last_hidden_state[:,:500]
    causal_error=float((a-b).abs().max());assert causal_error<.001,causal_error
    return {'unchanged_inside_window_probability_error':unchanged,'cases':results,
            'future_suffix_earlier_hidden_error':causal_error}


@torch.inference_mode()
def speed(model,tok,window,fast):
    set_window(model.backbone,window);short_recurrent(fast)
    phrase=tok.encode('公开资料与 public information。\n',add_special_tokens=False)
    whole=torch.tensor([(phrase*1800)[:9216]],device='cuda')
    records=[]
    for context in (512,2048,8192):
        for chunk in (1,8,32):
            # Warm every exact cached-call shape before measuring. First-run
            # compilation costs are reported separately, never mixed with ITPS.
            torch.cuda.synchronize();warm_start=time.perf_counter()
            _,warm=probs(model,whole[:,:context],cached=True)
            for i in range(4):_,warm=probs(model,whole[:,context:context+chunk],warm,True)
            torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm_start
            del warm;gc.collect();torch.cuda.empty_cache()
            _,state=probs(model,whole[:,:context],cached=True)
            storage_before=cache_storage(state);times=[]
            torch.cuda.synchronize();started=time.perf_counter()
            for i in range(24):
                t=time.perf_counter()
                p,state=probs(model,whole[:,context+i*chunk:context+(i+1)*chunk],state,True)
                p.cpu().tolist();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            elapsed=time.perf_counter()-started
            # Same mode, exact final token sequence: no architecture-parity claim.
            full,_=probs(model,whole[:,:context+24*chunk]);delta=float((full-p).abs().max())
            assert delta<.03,(window,fast,context,chunk,delta)
            records.append({'window':window,'short_recurrent':fast,'initial_context_tokens':context,
                            'chunk_tokens':chunk,'classified_new_tokens':24*chunk,'results':24,
                            'native_itps':24*chunk/elapsed,'classifications_per_second':24/elapsed,
                            **quantiles(times),'warmup_seconds_excluded':warm_seconds,
                            'state_before':storage_before,'state_after':cache_storage(state),
                            'full_stream_probability_error':delta})
            print(json.dumps({'speed_case':records[-1]}),flush=True)
            del state,p,full;gc.collect();torch.cuda.empty_cache()
    return records


def main():
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);torch.manual_seed(3407);OUT.mkdir(exist_ok=True)
    write(OUT/'attention_oracle.json',oracle_test())
    backbone,tok,kernels=load('qwen35');model=Classifier(backbone).to('cuda')
    source=Path('/work/output/qwen35_full/classifier.safetensors')
    expected=json.loads(source.with_name('summary.json').read_text())['checkpoint_sha256']
    assert sha(source)==expected
    state=load_file(str(source));model.load_state_dict(state,strict=True);del state
    model.eval().requires_grad_(False)
    write(OUT/'correctness.json',correctness(model,tok))
    print(json.dumps({'window_correctness':'passed'}),flush=True)
    data={s:encode_rows(tok,read(f'/work/output/teacher/{s}.jsonl')) for s in ('train','dev','test','official_test')}
    # Test the actual full-network backward path before proposing recovery training.
    set_window(backbone,32);model.train().requires_grad_(True);backbone.float()
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    loss=loss_for(model,data['train'][:4],tok.pad_token_id)
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(loss) and torch.isfinite(grad)
    attention_gradient=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if '.self_attn.' in n and p.grad is not None)
    assert attention_gradient>0
    write(OUT/'backward.json',{'loss':float(loss),'gradient_norm':float(grad),'attention_gradient_l1':attention_gradient,
                             'optimizer_steps':0,'window':32})
    model.zero_grad(set_to_none=True);backbone.to(torch.bfloat16);model.eval().requires_grad_(False)
    for window in (512,1024):
        set_window(backbone,window)
        predictions={s:evaluate(model,tok,data[s]) for s in ('dev','test','official_test')}
        for s,rows in predictions.items():write_rows(OUT/f'w{window}_{s}_predictions.jsonl',rows)
        write(OUT/f'w{window}_source_metrics.json',summarize(predictions['dev'],predictions['test']))
        write(OUT/f'w{window}_official_complete_metrics.json',summarize(predictions['dev'],predictions['official_test']))
        write(OUT/f'w{window}_length_coverage.json',{s:{'total':len(data[s]),'beyond_window':sum(len(x['ids'])>window for x in data[s])} for s in data})
        print(json.dumps({'quality_complete':window}),flush=True)
    records=[]
    # Isolate recurrent-kernel changes from the architecture change.
    for window,fast in ((None,False),(None,True),(512,True),(1024,True)):
        records+=speed(model,tok,window,fast)
        write(OUT/'speed.json',{'scope':'warmed L20 token-ID model calls, CPU-visible risk probabilities; 24 appends/case; no HTTP',
                              'records':records})
    write(OUT/'summary.json',{'status':'completed','source_checkpoint_sha256':expected,
                             'parameters':sum(p.numel() for p in model.parameters()),'layers_removed':0,
                             'attention_layers_changed':6,'recovery_training_steps':0,'has_lm_head':False,
                             'scope':'architecture and token-ID runtime probe, not arbitrary text streaming or deployment approval',
                             'kernel_sources':['https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html',
                                               'https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py']})
    print(json.dumps({'window_probe':'completed'}),flush=True)


if __name__=='__main__':main()
