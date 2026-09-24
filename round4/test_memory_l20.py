"""Actual L20 forward/backward/causality/state tests for the added neural memory."""
import json,sys,gc
from pathlib import Path
sys.path[:0]=['/work/input','/work/window','/work/round4']
import torch
from torch import nn
from safetensors.torch import load_file
from train_base import load,Classifier
from memory_attention import attach_memory,memory_read,features,memory_state_bytes
from window_attention import cache_storage
from experiment_common import write


def main():
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);torch.manual_seed(240923)
    q=torch.randn(1,4,81,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(1,2,81,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    pq=nn.Linear(32,16,bias=False,device='cuda');pk=nn.Linear(32,16,bias=False,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual,_=memory_read(q,k,v,23,pq,pk)
        qf=features(pq,q).float();kf=features(pk,k).float().repeat_interleave(2,1)
        weights=qf@kf.transpose(-1,-2)
        positions=torch.arange(81,device='cuda')
        mask=positions[:,None]>=positions[None,:]+23
        weights=weights*mask
        expected=(weights@v.float().repeat_interleave(2,1))/(weights.sum(-1,keepdim=True)+1e-10)
        error=float((actual.float()-expected).abs().max())
        assert error<.03,error
        loss=actual.float().square().mean();loss.backward()
        assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()
        assert pq.weight.grad.abs().sum()>0 and pk.weight.grad.abs().sum()>0
    oracle={'forward_max_abs_error':error,'feature_map_backward_finite':True}
    del q,k,v,pq,pk,actual,expected,loss;gc.collect();torch.cuda.empty_cache()
    backbone,tok,_=load('qwen35');model=Classifier(backbone).to('cuda')
    state=load_file('/work/output/qwen35_full/classifier.safetensors');model.load_state_dict(state);del state
    base=sum(p.numel() for p in model.parameters());attach_memory(backbone,256)
    for layer in backbone.layers:
        if hasattr(layer,'self_attn'):layer.self_attn.memory_gate.data.fill_(.3)
    model.eval().requires_grad_(False)
    ids=torch.tensor([tok.encode('USER:\n'+('一般资料与普通文字。 General reading material.\n'*400),add_special_tokens=False)[:1701]],device='cuda')
    with torch.inference_mode():
        full,_,_=model.step(ids,None,False)
        for schedule in ([257,1,7,127,31],[128,128,128,128,128]):
            offset=0;cache=None;j=0
            while offset<ids.shape[1]:
                end=min(offset+schedule[j%len(schedule)],ids.shape[1])
                output,_,cache=model.step(ids[:,offset:end],cache,True);offset=end;j+=1
            delta=float((output.softmax(-1)-full.softmax(-1)).abs().max());assert delta<.03,delta
            before=cache_storage(cache)['unique_storage_bytes']+memory_state_bytes(cache)
            _,_,cache=model.step(ids[:,:512],cache,True)
            after=cache_storage(cache)['unique_storage_bytes']+memory_state_bytes(cache)
            assert before==after,(before,after)
            oracle.setdefault('model_stream_cases',[]).append({'schedule':schedule,'probability_error':delta,
                       'state_bytes':after,'memory_bytes':memory_state_bytes(cache),'nonzero_gate':.3})
        a=model(ids[:,:700]).last_hidden_state[:,:300].clone()
        edited=ids[:,:700].clone();edited[:,400:]=(edited[:,400:]+1)%backbone.config.vocab_size
        b=model(edited).last_hidden_state[:,:300]
        leak=float((a-b).abs().max());assert leak<.001,leak
    # Exercise the real composed classifier with nonzero memory, not only the kernel.
    model.train().requires_grad_(True);backbone.float()
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    output=model(ids[:,:900]);risk,_=model.readout(output.last_hidden_state[:,-1],'user')
    torch.nn.functional.cross_entropy(risk,torch.tensor([0],device='cuda')).backward()
    grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if 'memory_' in n and p.grad is not None}
    assert grads and all(torch.isfinite(torch.tensor(v)) for v in grads.values())
    assert any('memory_phi' in n and v>0 for n,v in grads.items())
    oracle.update(total_parameters=sum(p.numel() for p in model.parameters()),added_parameters=sum(p.numel() for p in model.parameters())-base,
                  future_suffix_earlier_hidden_error=leak,composed_backward_finite=True,memory_gradient_sum=sum(grads.values()),
                  physical_gpu=torch.cuda.get_device_name(0),has_lm_head=False)
    write('/work/output/round4/memory_smoke.json',oracle)
    print(json.dumps(oracle),flush=True)


if __name__=='__main__':main()
