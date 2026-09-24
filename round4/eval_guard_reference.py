"""Evaluate the original safety guard on the new held-out risk test after selection."""
from pathlib import Path
import sys,json
sys.path[:0]=['/work/input','/work/round4']
import torch
from runtime import load,encode,logits
from experiment_common import read,write,write_rows
from risk_metrics import metrics


@torch.inference_mode()
def main():
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    model,tok=load(dtype=torch.bfloat16,device='cuda',model_path=Path('/work/models/guard'))
    outputs={}
    for split in ['calibration','sealed_test']:
        rows=read('/work/round4/data/'+split+'.jsonl')
        prepared=sorted([(r,encode(tok,r['messages'])) for r in rows],key=lambda x:len(x[1]))
        predictions=[]
        for start in range(0,len(prepared),8):
            group=prepared[start:start+8];length=max(len(ids) for _,ids in group)
            ids=torch.full((len(group),length),tok.pad_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
            for i,(_,tokens) in enumerate(group):ids[i,:len(tokens)]=torch.tensor(tokens,device='cuda');mask[i,:len(tokens)]=1
            result=model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False)
            for i,(row,tokens) in enumerate(group):
                p=logits(result,row['target_role'])[i,len(tokens)-1].float().softmax(-1).cpu().tolist()
                predictions.append({k:row[k] for k in ['sample_id','family','language','target_role','source_label']}|{'probs':p})
        outputs[split]=predictions
        write_rows('/work/output/round4/a0_'+split+'_predictions.jsonl',predictions)
    scores=metrics(outputs['calibration'],outputs['sealed_test'])
    write('/work/output/round4/a0_reference_metrics.json',scores)
    print(json.dumps({'reference':'original Qwen3Guard-Stream-0.6B','sealed_test':scores}),flush=True)


if __name__=='__main__':main()
