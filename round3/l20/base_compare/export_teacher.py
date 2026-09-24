"""Freeze A0 soft targets and matched full-input baseline on one L20."""
import copy
import json
from pathlib import Path
import time

import torch
from transformers import DynamicCache
from runtime import load, encode, logits
from experiment_common import read, write, write_rows, summarize, sha
from speed_probe import benchmark

ROOT=Path('/work/input')
OUT=Path('/work/output/teacher')


@torch.inference_mode()
def predict(model,tok,rows):
    result=[]
    ordered=sorted([(row,encode(tok,row['messages'],partial=row.get('partial',False))) for row in rows],key=lambda x:len(x[1]))
    for start in range(0,len(ordered),8):
        batch=ordered[start:start+8];length=max(len(ids) for _,ids in batch)
        x=torch.full((len(batch),length),tok.pad_token_id,device='cuda',dtype=torch.long)
        mask=torch.zeros_like(x)
        for index,(_,ids) in enumerate(batch):
            x[index,:len(ids)]=torch.tensor(ids,device='cuda');mask[index,:len(ids)]=1
        output=model(input_ids=x,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False)
        for index,(row,ids) in enumerate(batch):
            risk=logits(output,row['target_role'])[index,len(ids)-1].float().softmax(-1).cpu().tolist()
            category=(output.query_category_logits if row['target_role']=='user' else output.category_logits)
            category=category[index,len(ids)-1].float().softmax(-1).cpu().tolist()
            result.append({**row,'teacher_risk':risk,'teacher_category':category,'probs':risk,'teacher_input_tokens':len(ids)})
        if start%512==0:print(json.dumps({'teacher_processed':min(start+8,len(rows)),'total':len(rows)}),flush=True)
    by_id={r['sample_id']:r for r in result}
    return [by_id[r['sample_id']] for r in rows]


def main():
    assert torch.cuda.device_count()==1 and 'L20' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for split in ('train','dev','test'):
        assert sha(ROOT/f'data/{split}.jsonl')==manifest['output_hashes'][split]
    model,tok=load(dtype=torch.bfloat16,device='cuda',model_path=Path('/work/models/guard'))
    assert not hasattr(model,'lm_head')
    outputs={}
    for split in ('train','dev','test','official_test'):
        rows=read(ROOT/f'data/{split}.jsonl')
        if split=='train':
            partial=[]
            for index,row in enumerate(rows):
                if index%4:continue
                text=row['messages'][-1]['content']; cut=max(1,len(text)*(1+index%3)//4)
                candidate=copy.deepcopy(row)
                candidate['messages'][-1]['content']=text[:cut]
                candidate.update(sample_id=row['sample_id']+'-prefix',source_label=None,partial=True)
                partial.append(candidate)
            rows+=partial
        outputs[split]=predict(model,tok,rows)
        write_rows(OUT/f'{split}.jsonl',outputs[split])
    write(OUT/'source_metrics.json',summarize(outputs['dev'],outputs['test']))
    write(OUT/'official_complete_metrics.json',summarize(outputs['dev'],outputs['official_test']))
    def step(ids,cache,cached):
        # This all-full-attention model needs no sliding-window cache config.
        # Transformers 4.55.0 passes max_cache_len to DynamicLayer when config
        # is supplied; explicitly constructing its ordinary cache avoids that
        # incompatible constructor path, without changing the attention model.
        if cached and cache is None:cache=DynamicCache()
        output=model(input_ids=ids,past_key_values=cache,use_cache=cached,logits_to_keep=1)
        return output.query_risk_level_logits[:,-1],output.query_category_logits[:,-1],output.past_key_values if cached else None
    write(OUT/'speed.json',benchmark(step,tok,tok))
    write(OUT/'manifest.json',{'teacher':'Qwen3Guard-Stream-0.6B','has_lm_head':False,
                              'same_visible_prefix_targets':True,'counts':{s:len(r) for s,r in outputs.items()},
                              'hashes':{s:sha(OUT/f'{s}.jsonl') for s in outputs},
                              'categories':'source Qwen single-choice head, not project harm ontology',
                              'official_scope':'same frozen 250 examples, complete-input readout; not published debounced streaming protocol'})
    print(json.dumps({'teacher_export_complete':True}),flush=True)


if __name__=='__main__':main()
