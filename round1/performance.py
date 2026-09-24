"""Synchronized on-device measurements with identical token streams and explicit batching."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from runtime import ROOT,load,Stream,forward_ids,restore_adapter,merge_lora,sync,logits

def timed(fn):
    sync();start=time.perf_counter();value=fn();sync()
    return value,time.perf_counter()-start

def stats(values):
    return {'n':len(values),'p50_ms':float(np.median(values)*1000),'p95_ms':float(np.percentile(values,95)*1000),'mean_ms':float(np.mean(values)*1000)}

@torch.no_grad()
def run(model,tok,name,out):
    started=time.monotonic();records=[]
    tokens=tok.encode('请根据公开资料客观分析这项政策的背景、依据与影响，并区分事实和观点。',add_special_tokens=False)
    allids=(tokens*((8192//len(tokens))+1))[:8192]
    for length in [256,1024,4096,8192]:
        ids=allids[:length];prefix=ids[:-32];continuation=ids[-32:]
        ref,_=timed(lambda:forward_ids(model,ids,'assistant')[-1].softmax(-1))
        modes=([('published',1)] if name=='original' else [])+[('incremental',c) for c in [1,8,16,32]]
        for mode,chunk in modes:
            latencies=[];prefills=[];totals=[];errors=[]
            for repeat in range(4):
                if mode=='published':
                    state=model.stream_generate(torch.tensor(prefix,device='mps'))
                    _,prefill=timed(lambda:next(state))
                    per=[]
                    for token in continuation:
                        output,seconds=timed(lambda:state.send(token));per.append(seconds)
                    prob=output[0][0,-1].float().softmax(-1)
                    state.close();del state
                else:
                    stream=Stream(model,tok);_,prefill=timed(lambda:stream.update_ids(prefix,'assistant'))
                    per=[]
                    for end in range(len(prefix)+chunk,len(ids)+1,chunk):
                        prob,seconds=timed(lambda:stream.update_ids(ids[:end],'assistant'));per.append(seconds)
                    assert stream.processed_tokens==length
                    del stream
                errors.append((prob-ref).abs().max().item())
                if repeat:prefills.append(prefill);latencies.extend(per);totals.append(prefill+sum(per))
            if max(errors)>.04:raise AssertionError(f'Cache/published parity failure: {max(errors)}')
            r={'model':name,'mode':mode,'total_tokens':length,'prefill_tokens':length-32,'new_tokens':32,
               'chunk':chunk,'concurrency':1,'chunk_latency':stats(latencies),'prefill_latency':stats(prefills),
               'total_latency':stats(totals),'continuation_tokens_per_second':32*3/sum(latencies),
               'probability_max_error':max(errors),'mps_driver_bytes':torch.mps.driver_allocated_memory(),
               'rss_bytes':psutil.Process().memory_info().rss,
               'buffer_wait_max_ms_at_100_tokens_s':(chunk-1)*10,'buffer_wait_is_modeled_not_measured':True}
            records.append(r);print(json.dumps(r),flush=True)
            if time.monotonic()-started>1800:raise TimeoutError('Performance budget exhausted')
        torch.mps.empty_cache()
    for length in [256,1024]:
        for batch in [1,2,4]:
            x=torch.tensor([allids[:length]]*batch,device='mps');times=[]
            for repeat in range(11):
                _,seconds=timed(lambda:model(input_ids=x,use_cache=False,logits_to_keep=1))
                if repeat:times.append(seconds)
            records.append({'model':name,'mode':'equal_length_batch','tokens_per_request':length,
                'batch':batch,'concurrent_ready_requests':batch,'request_latency':stats(times),
                'requests_per_second':batch*len(times)/sum(times),'tokens_per_second':batch*length*len(times)/sum(times),
                'queue_wait':'zero: all requests ready together; no arrival-process SLO claim',
                'mps_driver_bytes':torch.mps.driver_allocated_memory()})
    result={'model':name,'measurements':records,'wall_seconds':time.monotonic()-started,
        'warmup':'first repetition discarded','timing':'MPS synchronize before and after each forward',
        'power':__import__('subprocess').check_output(['pmset','-g','batt'],text=True).strip()}
    (out/f'{name}_performance.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');parser.add_argument('--merge',action='store_true')
    parser.add_argument('--name',default='original');parser.add_argument('--out',default=str(ROOT/'performance'));args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();model,tok=load(dtype=torch.bfloat16);sync();cold=time.monotonic()-start
    if args.checkpoint:restore_adapter(model,args.checkpoint)
    if args.merge:
        ids=tok.encode('请介绍政府公开信息查询渠道。',add_special_tokens=False)
        with torch.no_grad():before=forward_ids(model,ids,'user')[-1].softmax(-1)
        merge_lora(model)
        with torch.no_grad():after=forward_ids(model,ids,'user')[-1].softmax(-1)
        error=(before-after).abs().max().item();assert error<.035
        (out/'merge_parity.json').write_text(json.dumps({'probability_error':error})+'\n')
    result=run(model,tok,args.name,out);result['cold_load_seconds']=cold
    (out/f'{args.name}_performance.json').write_text(json.dumps(result,indent=2)+'\n')
