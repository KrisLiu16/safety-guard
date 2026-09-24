"""Real CPU/MPS parity, cached prefix, padded batch, and backward validation."""
import gc
import json
import platform
import time
import torch
from runtime import ROOT, load, encode, forward_ids, batch_probs, Stream, attach_lora, sync

def main():
    started=time.time(); torch.manual_seed(20260922)
    messages=[]
    for text in ['请介绍公开的政府预算查询渠道。', '我不同意这项政策，请分析它的利弊。',
                 '请帮我伪造证据陷害甲某。', '我准备威胁邻居，让他不敢再投诉。',
                 '引用威胁言论进行反驳时应该注意什么？', '请解释如何保护个人隐私。',
                 'Please summarize this public report.', 'I will hurt my neighbor tonight.']:
        messages.append([{'role':'user','content':text}])
        messages.append([{'role':'user','content':text},{'role':'assistant','content':'我可以协助讨论事实和合法途径，不会帮助伤害他人或伪造证据。'}])
    model,tok=load(device='cpu')
    with torch.no_grad():
        cpu=[forward_ids(model,encode(tok,m),m[-1]['role'])[-1].softmax(-1) for m in messages]
    model.to('mps'); sync()
    results={'torch':torch.__version__,'python':platform.python_version(),'device':str(model.device),
             'mps_available':torch.backends.mps.is_available(),'cases':len(messages)}
    with torch.no_grad():
        mps=[forward_ids(model,encode(tok,m),m[-1]['role'])[-1].softmax(-1).cpu() for m in messages]
    delta=max((a-b).abs().max().item() for a,b in zip(cpu,mps)); results['cpu_mps_probability_max_error']=delta
    assert delta<.002, delta
    cache_errors=[]; processed=[]
    for m in messages[:6]:
        ids=encode(tok,m); stream=Stream(model,tok)
        for end in sorted(set([3, min(13,len(ids)),len(ids)])):
            p=stream.update_ids(ids[:end],m[-1]['role']).cpu()
            with torch.no_grad(): ref=forward_ids(model,ids[:end],m[-1]['role'])[-1].softmax(-1).cpu()
            cache_errors.append((p-ref).abs().max().item())
        assert stream.processed_tokens==len(ids)
        processed.append(stream.processed_tokens)
    # A user message, text append with BPE changes, and then a response on the same cache.
    stream=Stream(model,tok)
    for m,partial in [([{'role':'user','content':'hello wor'}],True),
                      ([{'role':'user','content':'hello world'}],True),
                      ([{'role':'user','content':'hello world'}],False),
                      ([{'role':'user','content':'hello world'},{'role':'assistant','content':'你好。'}],True),
                      ([{'role':'user','content':'hello world'},{'role':'assistant','content':'你好。'}],False)]:
        p=stream.update(m,partial=partial).cpu()
        with torch.no_grad(): ref=forward_ids(model,encode(tok,m,partial=partial),m[-1]['role'])[-1].softmax(-1).cpu()
        cache_errors.append((p-ref).abs().max().item())
    results['cache_probability_max_error']=max(cache_errors)
    assert max(cache_errors)<.002
    padded=batch_probs(model,tok,messages[:4])
    results['batch_probability_max_error']=max((torch.tensor(a)-b).abs().max().item() for a,b in zip(padded,mps[:4]))
    assert results['batch_probability_max_error']<.002
    del model; gc.collect(); torch.mps.empty_cache()
    gradients={}
    for dtype in [torch.float32,torch.bfloat16]:
        model,tok=load(dtype=dtype); params=attach_lora(model); model.train()
        before={n:p.detach().clone() for n,p in params}
        opt=torch.optim.AdamW([p for n,p in params],lr=1e-4)
        ids=encode(tok,messages[4])
        begin=time.monotonic()
        p=forward_ids(model,ids,'user')[-1].softmax(-1)
        reference=mps[4].to('mps')
        parity=(p.detach()-reference).abs().max().item()
        loss=-(p*torch.tensor([-1.,1.,-.25],device='mps')).sum()
        loss.backward()
        assistant_probs=forward_ids(model,encode(tok,messages[5]),'assistant')[-1].softmax(-1)
        assistant_loss=-(assistant_probs*torch.tensor([1.,-1.,-.25],device='mps')).sum()
        assistant_loss.backward()
        grad={n:float(p.grad.float().norm().item()) for n,p in params if p.grad is not None}
        assert grad and all(torch.isfinite(p.grad).all().item() for n,p in params if p.grad is not None)
        assert any(v>0 for n,v in grad.items() if n.endswith('.B'))
        assert grad['query_risk_level_head.weight']>0
        assert grad['risk_level_head.weight']>0
        opt.step();sync()
        changed=[n for n,p in params if not torch.equal(p.detach(),before[n])]
        assert 'query_risk_level_head.weight' in changed
        assert 'risk_level_head.weight' in changed
        assert any(n.endswith('.B') for n in changed)
        assert all(('category_head' not in n) for n in changed)
        gradients[str(dtype)]={'loss':loss.item(),'changed':changed,'finite_gradients':True,
            'trainable_parameters':sum(p.numel() for n,p in params),'seconds':time.monotonic()-begin,
            'fp32_probability_error':parity,'mps_allocated_bytes':torch.mps.current_allocated_memory()}
        if dtype==torch.bfloat16: assert parity<.035,parity
        del model,opt,params,before,p,loss;gc.collect();torch.mps.empty_cache()
    results['gradients']=gradients;results['seconds']=time.time()-started
    results['status']='passed';results['selected_precision']='bfloat16'
    (ROOT/'smoke_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
