"""Exercise exported weights, JSONL CLI, all-token chunks, and cache reuse."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from runtime import ROOT,load,encode,Stream,logits,forward_ids,bucketed_batch_probs,sync
from stream_evaluate import content_start
from evaluate import read,write
from demo import Session

def controller_contract():
    class FakeStream:
        def __init__(self,*args):
            self.ids=[1,2,3];self.tokenizer=None;self.last_start=0;self.last_changed=True;self.processed_tokens=3
            p=torch.tensor([[[.005,.99,.005],[.005,.99,.005],[.99,.005,.005]]]).log()
            self.last=SimpleNamespace(risk_level_logits=p,query_risk_level_logits=p)
        def update(self,*args,**kwargs):return torch.tensor([.99,.005,.005])
    with patch('demo.Stream',FakeStream),patch('demo.encode',return_value=[]):
        session=Session(None,None,{'user':.5,'assistant':.5})
        result=session.push([{'role':'assistant','content':'需要被拦截的块内内容。'}],final=True)
        assert result['action']=='block' and result['released_text']=='',result
    return 'unsafe interior of chunk blocks even when final token is safe'

def cli(mode):
    source=ROOT/f'demo_{mode}.jsonl'
    result=subprocess.run([sys.executable,str(ROOT/'export/demo.py'),'--mode',mode,'--model-path',str(ROOT/'export/model'),
        '--calibration',str(ROOT/'export'/f'calibration_{mode}.json')],input=source.read_text(),text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    (ROOT/f'demo_{mode}_output.jsonl').write_text(result.stdout)
    return [json.loads(line) for line in result.stdout.splitlines() if line]

if __name__=='__main__':
    contract=controller_contract()
    batch=cli('batch')[0];assert len(batch)==2 and all('probs' in r for r in batch)
    assert batch[0]['action']=='release' and batch[1]['action']=='block',batch
    streamed=cli('stream');assert len(streamed)==4
    assert streamed[0]['action']=='block',streamed
    assert streamed[-1]['action']=='release',streamed
    expected=json.loads((ROOT/'demo_stream.jsonl').read_text().splitlines()[-1])['messages'][-1]['content']
    assert ''.join(r.get('released_text','') for r in streamed[1:])==expected
    assert len({r['seq_id'] for r in streamed})==1
    model,tok=load(dtype=torch.bfloat16,model_path=ROOT/'export/model')
    thresholds=json.loads((ROOT/'export/calibration_stream.json').read_text())['roles']
    # Select fixed development cases before checking agreement; no test tuning.
    candidates=read(ROOT/'data/dev.jsonl');chosen=[]
    for role in ['user','assistant']:
        for label in ['safe','unsafe']:
            chosen += [r for r in candidates if r['target_role']==role and r['label']==label][:2]
    evidence=[];latencies={c:[] for c in [1,8,16,32]}
    with torch.no_grad():
        for row in chosen:
            ids=encode(tok,row['messages']);role=row['target_role'];start=content_start(tok,row['messages'],ids)
            full=forward_ids(model,ids,role,all_positions=True).softmax(-1).cpu()
            threshold=thresholds[role]['threshold'];risk=full[start:,1]
            full_block=any(min(float(a),float(b))>=threshold for a,b in zip(risk,risk[1:]))
            for chunk in [1,8,16,32]:
                stream=Stream(model,tok);stream.update_ids(ids[:start],role,all_positions=True)
                seen=[]
                for end in range(start+chunk,len(ids)+chunk,chunk):
                    end=min(end,len(ids));sync();begin=time.perf_counter()
                    stream.update_ids(ids[:end],role,all_positions=True);sync()
                    latencies[chunk].append(time.perf_counter()-begin)
                    seen.extend(logits(stream.last,role)[0].float().softmax(-1).cpu().tolist())
                actual=torch.tensor(seen);ref=full[start:];assert actual.shape==ref.shape
                error=(actual-ref).abs().max().item();assert error<.04,error
                block=any(min(float(a),float(b))>=threshold for a,b in zip(actual[:,1],actual[1:,1]))
                assert full_block==block,{'id':row['sample_id'],'chunk':chunk,'max_error':error}
                assert stream.processed_tokens==len(ids)
                evidence.append({'sample_id':row['sample_id'],'chunk':chunk,'probability_max_error':error,
                    'decision_matches_full':block==full_block,'block':block,'processed_tokens':stream.processed_tokens})
        conversations=[r['messages'] for r in chosen]
        one=bucketed_batch_probs(model,tok,conversations,max_batch_size=1)
        four=bucketed_batch_probs(model,tok,conversations,max_batch_size=4)
        batch_error=max(abs(a-b) for p,q in zip(one,four) for a,b in zip(p,q));assert batch_error<.04
    summary={'status':'passed','cli_batch_cases':len(batch),'cli_stream_updates':len(streamed),
        'controller_contract':contract,
        'all_token_cache_cases':evidence,'bucketed_batch_max_probability_error':batch_error,
        'production_all_token_chunk_latency':{str(c):{'n':len(t),'p50_ms':float(np.median(t)*1000),'p95_ms':float(np.percentile(t,95)*1000)} for c,t in latencies.items()},
        'timing_note':'Exported model, every new token head evaluated, GPU synchronized; development-case lengths, no arrival waiting included.'}
    write(ROOT/'demo_verification.json',summary);print(json.dumps(summary,indent=2))
