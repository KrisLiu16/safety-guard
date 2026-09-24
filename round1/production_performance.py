"""All-token head timing and ready-client queue latency for independent KV sessions."""
import json
import time
import numpy as np
import torch
from runtime import ROOT,load,Stream,logits,sync
from evaluate import write

def summarize(xs):return {'n':len(xs),'p50_ms':float(np.median(xs)*1000),'p95_ms':float(np.percentile(xs,95)*1000)}

if __name__=='__main__':
    model,tok=load(dtype=torch.bfloat16,model_path=ROOT/'export/model')
    tokens=tok.encode('请根据公开资料客观分析，区分事实与观点，并保护个人隐私。',add_special_tokens=False)
    ids=(tokens*(1024//len(tokens)+1))[:1024];results=[]
    with torch.no_grad():
        for concurrent in [1,2,4]:
            for chunk in [1,16]:
                service=[];observed=[];prefills=[];wall=[]
                for repeat in range(4):
                    sessions=[Stream(model,tok) for _ in range(concurrent)]
                    sync();start=time.perf_counter()
                    for s in sessions:s.update_ids(ids[:-32],'assistant',all_positions=True)
                    sync();prefill=time.perf_counter()-start
                    start=time.perf_counter()
                    for end in range(992+chunk,1025,chunk):
                        sync();ready=time.perf_counter()
                        for s in sessions:
                            sync();begin=time.perf_counter()
                            s.update_ids(ids[:end],'assistant',all_positions=True)
                            # Includes classification readout transfer, not only GPU enqueue.
                            logits(s.last,'assistant')[0].float().softmax(-1).cpu().tolist();sync()
                            finished=time.perf_counter()
                            if repeat:service.append(finished-begin);observed.append(finished-ready)
                    if repeat:wall.append(time.perf_counter()-start);prefills.append(prefill)
                    assert all(s.processed_tokens==1024 for s in sessions)
                results.append({'active_sessions':concurrent,'chunk':chunk,'total_tokens_per_session':1024,
                    'new_tokens_per_session':32,'all_new_token_heads':True,'service_latency':summarize(service),
                    'ready_client_latency_including_queue':summarize(observed),'group_prefill_latency':summarize(prefills),
                    'aggregate_new_tokens_per_second':concurrent*32*3/sum(wall),
                    'mps_driver_bytes':torch.mps.driver_allocated_memory(),
                    'schedule':'Independent KV sessions, single MPS device, round-robin; every session ready at each tick.'})
                print(json.dumps(results[-1]),flush=True)
    write(ROOT/'production_performance.json',{'results':results,'warmup':'first of four repetitions excluded',
        'limitation':'Closed-loop ready-client workload, not open-loop arrivals or a production SLO guarantee.'})
