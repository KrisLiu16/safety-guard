"""Local JSONL batch and streamed-conversation demo. No network service or cloud calls."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
import torch
from runtime import ROOT,LABELS,load,bucketed_batch_probs,Stream,restore_adapter,merge_lora,encode,logits

class Session:
    def __init__(self,model,tokenizer,thresholds,buffer_chars=32,deadline_ms=5000):
        self.stream=Stream(model,tokenizer);self.thresholds=thresholds
        self.buffer_chars=buffer_chars;self.seq_id=uuid.uuid4().hex
        self.released=0;self.blocked=False;self.last_role=None;self.previous=''
        self.turn=None;self.token_scores={}
        self.deadline_ms=deadline_ms

    def push(self,messages,final=False):
        begin=time.monotonic()
        role=messages[-1]['role'];content=messages[-1]['content']
        if role!=self.last_role or len(messages)!=self.turn:
            self.released=0;self.blocked=False;self.previous='';self.last_role=role
            self.turn=len(messages);self.token_scores={}
        # Already emitted bytes cannot be changed. Text edits require a fresh session.
        if not content.startswith(self.previous[:self.released]):
            return {'seq_id':self.seq_id,'action':'hold','error':'revision_of_released_text'}
        try:
            probs=self.stream.update(messages,partial=not final,all_positions=True).cpu().tolist()
        except (ValueError,RuntimeError) as exc:
            return {'seq_id':self.seq_id,'action':'hold','error':type(exc).__name__,'released_char':self.released}
        label=LABELS[max(range(3),key=probs.__getitem__)]
        # Inspect every newly processed token, including risks inside a chunk.
        # Roll back readouts too when BPE changes the previously cached suffix.
        empty=messages[:-1]+[{'role':role,'content':''}]
        header=encode(self.stream.tokenizer,empty,partial=True)
        content_start=0
        for a,b in zip(header,self.stream.ids):
            if a!=b:break
            content_start+=1
        if self.stream.last_changed:
            self.token_scores={i:p for i,p in self.token_scores.items() if i<self.stream.last_start}
            new_probs=logits(self.stream.last,role)[0].float().softmax(-1).cpu().tolist()
            for offset,p in enumerate(new_probs):
                index=self.stream.last_start+offset
                if index>=content_start:self.token_scores[index]=p
        ordered=sorted(self.token_scores)
        first_risk=None
        for i,j in zip(ordered,ordered[1:]):
            if j==i+1 and self.token_scores[i][1]>=self.thresholds[role] and self.token_scores[j][1]>=self.thresholds[role]:
                first_risk=j;break
        if first_risk is not None:self.blocked=True
        elapsed_ms=(time.monotonic()-begin)*1000
        deadline_exceeded=elapsed_ms>self.deadline_ms
        old=self.released
        if self.blocked:action='block'
        elif deadline_exceeded:action='hold'
        elif label=='controversial':action='hold'
        else:
            self.released=max(self.released,len(content) if final else max(0,len(content)-self.buffer_chars))
            action='release' if self.released>old else 'hold'
        self.previous=content
        return {'seq_id':self.seq_id,'role':role,'label':label,'probs':dict(zip(LABELS,probs)),
            'action':action,'reviewed_char':len(content),'released_char':self.released,
            'released_text':content[old:self.released],'buffer_chars':len(content)-self.released,
            'processed_tokens':self.stream.processed_tokens,'first_risk_token':first_risk,
            'decision_rule':'two consecutive token scores over threshold, all tokens within chunks inspected',
            'elapsed_ms':elapsed_ms,'deadline_exceeded':deadline_exceeded,
            'final':final}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['batch','stream'],required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--model-path');parser.add_argument('--calibration');parser.add_argument('--merge',action='store_true')
    parser.add_argument('--buffer-chars',type=int,default=32);parser.add_argument('--deadline-ms',type=float,default=5000)
    args=parser.parse_args()
    if args.buffer_chars<0 or args.deadline_ms<=0:parser.error('buffer must be >=0 and deadline must be >0')
    model,tok=load(dtype=torch.bfloat16,model_path=args.model_path)
    if args.checkpoint:restore_adapter(model,args.checkpoint)
    if args.merge:merge_lora(model)
    thresholds={'user':.5,'assistant':.5}
    if args.calibration:
        obj=json.loads(Path(args.calibration).read_text());thresholds={k:v['threshold'] for k,v in obj['roles'].items()}
    sessions={}
    for line in sys.stdin:
        try:
            obj=json.loads(line)
            if args.mode=='batch':
                conversations=obj['conversations'];probs=bucketed_batch_probs(model,tok,conversations)
                result=[{'role':m[-1]['role'],'probs':dict(zip(LABELS,p)),
                    'label':LABELS[max(range(3),key=p.__getitem__)],
                    'action':'block' if p[1]>=thresholds[m[-1]['role']] else ('hold' if p[2]>max(p[:2]) else 'release')}
                    for m,p in zip(conversations,probs)]
            else:
                sid=obj.get('session','default')
                if obj.get('close'):
                    sessions.pop(sid,None)
                    print(json.dumps({'session':sid,'action':'closed'}),flush=True)
                    continue
                messages=obj.get('messages')
                if not isinstance(messages,list) or not messages or any(not isinstance(m,dict) or not isinstance(m.get('content'),str) or m.get('role') not in ['system','user','assistant'] for m in messages):
                    raise ValueError('Expected a nonempty list of text messages')
                if messages[-1]['role'] not in ['user','assistant']:raise ValueError('Invalid target role')
                if sid not in sessions and len(sessions)>=4:
                    print(json.dumps({'session':sid,'action':'hold','error':'four_session_capacity_reached'}),flush=True)
                    continue
                if obj.get('reset') or sid not in sessions:sessions[sid]=Session(model,tok,thresholds,args.buffer_chars,args.deadline_ms)
                result=sessions[sid].push(obj['messages'],obj.get('final',False))
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except (ValueError,KeyError,TypeError,IndexError,RuntimeError) as exc:
            print(json.dumps({'action':'hold','error':type(exc).__name__}),flush=True)

if __name__=='__main__':main()
