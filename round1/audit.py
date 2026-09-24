"""Evidence checks for the explicit round-one deliverables, not a substitute for review."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from runtime import ROOT,encode
from transformers import AutoTokenizer
from evaluate import read,write

def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

if __name__=='__main__':
    evidence={}
    env=json.loads((ROOT/'environment.json').read_text());assert env['chip']=='Apple M5' and env['memory_bytes']==32*1024**3
    smoke=json.loads((ROOT/'smoke_results.json').read_text());assert smoke['status']=='passed' and smoke['device']=='mps:0'
    assert all('risk_level_head.weight' in x['changed'] and 'query_risk_level_head.weight' in x['changed'] for x in smoke['gradients'].values())
    evidence['environment']={'status':'passed','source':['environment.json','smoke_results.json','requirements.lock.txt']}
    assets=json.loads((ROOT/'asset_hashes.json').read_text())
    for item in assets:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    submit=json.loads((ROOT/'generation_submit.json').read_text())['data']['run']
    assert submit['concurrency']==50 and submit['repetition_count']==1 and submit['max_attempts']==1
    gen=json.loads((ROOT/'generated/generation_audit.json').read_text());assert gen['responses']==495 and gen['raw_examples']==4950
    # Shortfall is retained as an explicit failed attempt, never labeled 5,000 produced.
    summary=json.loads((ROOT/'generated/summary.json').read_text());assert summary['succeeded_tasks']==49 and summary['failed_tasks']==1
    evidence['generation']={'status':'completed_with_documented_shortfall','raw':4950,'target':5000,'retries':0,'failed_tasks':1}
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for name,digest in manifest['hashes'].items():assert sha(ROOT/'data'/name)==digest,name
    datasets={s:read(ROOT/'data'/f'{s}.jsonl') for s in ['train','dev','calibration','test']}
    ids=[r['sample_id'] for rows in datasets.values() for r in rows];assert len(ids)==len(set(ids))==4095
    for a,rows in datasets.items():
        for b,others in datasets.items():
            if a!=b:assert not {r['family'] for r in rows}&{r['family'] for r in others}
    evidence['data']={'status':'passed','counts':{s:len(r) for s,r in datasets.items()},'prefixes':len(read(ROOT/'data/reviewed_prefixes.jsonl')),
        'label_limit':'Synthetic labels plus limited Codex spot review; not human gold or a guarantee against all semantic paraphrase leakage.'}
    train=json.loads((ROOT/'training/summary.json').read_text());assert train['steps']==159 and train['examples']==2541 and train['one_pass']
    assert train['trainable_parameters']==330752 and train['active_seconds']<7200
    logs=read(ROOT/'training/training_metrics.jsonl');assert len(logs)==159 and all(torch.isfinite(torch.tensor(r['grad_norm'])) for r in logs)
    reference={r['id']:r for r in read(ROOT/'reference_probabilities.jsonl')};tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True)
    for row in datasets['train']+read(ROOT/'data/reviewed_prefixes.jsonl'):
        tokens=encode(tok,row['messages'],partial=row.get('is_prefix',False))
        assert reference[row['sample_id']]['token_hash']==hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        assert reference[row['sample_id']]['role']==row['target_role']
    selected=json.loads((ROOT/'training/selected_checkpoint.json').read_text());checkpoint=Path(selected['path'])
    adapters=load_file(str(checkpoint/'adapter.safetensors'));heads=load_file(str(checkpoint/'risk_heads.safetensors'))
    assert len(adapters)==32 and len(heads)==2
    assert {int(k.split('.')[2]) for k in adapters}==set(range(20,28))
    assert all(float(v.norm())>0 for k,v in adapters.items() if k.endswith('.B'))
    with safe_open(str(ROOT/'model/model.safetensors'),framework='pt') as original:
        assert all(not torch.equal(v,original.get_tensor(k).float()) for k,v in heads.items())
    scores=[json.loads(p.read_text())['dev']['selection_score'] for p in (ROOT/'training').glob('step-*/metadata.json')]
    assert selected['dev_selection_score']==max(scores)
    evidence['training']={'status':'passed','selected':checkpoint.name,'steps':159,'reference_inputs_verified':len(reference),
        'updated_lora_layers':list(range(20,28)),'risk_heads_updated':list(heads),'trainable_parameters':330752}
    counts={'thinking':1059,'thinking_loc':569,'response_loc':813};keys=None
    for directory in ['baseline','post','merged','precision_control']:
        rows=read(ROOT/directory/'official_all_predictions.jsonl');current={(r['split'],r['row_index']) for r in rows}
        assert len(rows)==len(current)==sum(counts.values()) and all(r['status']=='ok' for r in rows)
        assert Counter(r['split'] for r in rows)==Counter(counts)
        if keys is None:keys=current
        else:assert current==keys
        chinese=read(ROOT/directory/'chinese_predictions.jsonl')
        assert {r['sample_id'] for r in chinese if r['split']=='test'}=={r['sample_id'] for r in datasets['test']}
    evidence['evaluation']={'status':'passed','official_same_full_ids':counts,'chinese_test':len(datasets['test']),
        'same_scope_runs':['baseline','post','merged','precision_control'],'metric_parity':'official_metrics_parity.json'}
    for name in ['original','rl','rl_merged']:
        perf=json.loads((ROOT/'performance'/f'{name}_performance.json').read_text())
        cells={(r.get('total_tokens'),r.get('chunk')) for r in perf['measurements'] if r['mode']=='incremental'}
        assert cells=={(n,c) for n in [256,1024,4096,8192] for c in [1,8,16,32]}
        assert {r['batch'] for r in perf['measurements'] if r['mode']=='equal_length_batch'}=={1,2,4}
    production=json.loads((ROOT/'production_performance.json').read_text());assert {r['active_sessions'] for r in production['results']}=={1,2,4}
    demo=json.loads((ROOT/'demo_verification.json').read_text());assert demo['status']=='passed'
    assert len(demo['all_token_cache_cases'])==32 and all(r['decision_matches_full'] for r in demo['all_token_cache_cases'])
    evidence['inference']={'status':'passed','cache_smoke':'smoke_results.json','all_token_exported_cache':'demo_verification.json',
        'independent_live_sessions':'production_performance.json','prefix_reuse':'demo_stream_output.jsonl'}
    export=json.loads((ROOT/'export/manifest.json').read_text())
    for path,digest in export['files'].items():assert sha(ROOT/'export'/path)==digest,path
    assert export['reload_probability_max_error']<1e-6
    assert sha(ROOT/'export/base_model/model.safetensors')==sha(ROOT/'model/model.safetensors')
    assert json.loads((ROOT/'launcher_verification.json').read_text())['status']=='passed'
    changed=[]
    with safe_open(str(ROOT/'model/model.safetensors'),framework='pt') as base, safe_open(str(ROOT/'export/model/model.safetensors'),framework='pt') as merged:
        assert set(base.keys())==set(merged.keys())
        for key in base.keys():
            if not torch.equal(base.get_tensor(key).float(),merged.get_tensor(key).float()):changed.append(key)
    allowed=set(heads)|{f'model.layers.{i}.self_attn.{name}.weight' for i in range(20,28) for name in ['q_proj','v_proj']}
    assert set(changed)<=allowed and set(heads)<=set(changed)
    evidence['export_and_frozen_weights']={'status':'passed','changed_parameters':changed,'all_other_base_tensors_unchanged':True}
    for name in ['ROUND1_REPORT.md','baseline_report.md','evaluation_before_after.json','performance_before_after.json','README.md']:
        assert (ROOT/name).stat().st_size>0,name
    evidence['export_and_report']={'status':'passed','manifest':'export/manifest.json','report':'ROUND1_REPORT.md'}
    write(ROOT/'completion_audit.json',{'status':'checks_passed','evidence':evidence,
        'scope_limit':'One bounded experiment; neither guaranteed quality improvement nor production approval.'})
    print(json.dumps(evidence,indent=2))
