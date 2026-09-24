"""Export the selected adapter, original heads, and a merged standalone model."""
import gc
import hashlib
import json
import shutil
from pathlib import Path
import torch
from runtime import ROOT,load,restore_adapter,merge_lora,batch_probs
from evaluate import read,write

if __name__=='__main__':
    selected=json.loads((ROOT/'training/selected_checkpoint.json').read_text())
    checkpoint=Path(selected['path']);out=ROOT/'export';out.mkdir(exist_ok=True)
    model,tok=load(dtype=torch.bfloat16);restore_adapter(model,checkpoint)
    rows=read(ROOT/'data/dev.jsonl')[:32];conversations=[r['messages'] for r in rows]
    with torch.no_grad():before=sum([batch_probs(model,tok,conversations[i:i+4]) for i in range(0,len(rows),4)],[])
    merge_lora(model)
    with torch.no_grad():after=sum([batch_probs(model,tok,conversations[i:i+4]) for i in range(0,len(rows),4)],[])
    error=max(abs(a-b) for p,q in zip(before,after) for a,b in zip(p,q));assert error<.035,error
    modeldir=out/'model';model.save_pretrained(modeldir,safe_serialization=True);tok.save_pretrained(modeldir)
    # The upstream custom class is not registered for automatic source copying.
    for source in (ROOT/'model').glob('*.py'):shutil.copy2(source,modeldir/source.name)
    shutil.copy2(ROOT/'model/LICENSE',modeldir/'LICENSE')
    shutil.copy2(checkpoint/'risk_heads.safetensors',modeldir/'risk_heads_fp32.safetensors')
    for name in ['adapter.safetensors','risk_heads.safetensors','metadata.json']:shutil.copy2(checkpoint/name,out/name)
    for name in ['config.json','requirements.lock.txt','family_splits.json']:shutil.copy2(ROOT/name,out/name)
    for name in ['runtime.py','demo.py','demo_batch.jsonl','demo_stream.jsonl']:shutil.copy2(ROOT/name,out/name)
    (out/'README.md').write_text('''# M5 Guard 首轮模型导出

`model/` 是合并权重；`adapter.safetensors` 与 `risk_heads.safetensors` 保留可复核的未合并更新。
使用随包提供的 `runtime.py` / `demo.py`，会明确恢复 FP32 风险头；直接用全局 BF16 加载并不保证相同数值结果。
依赖见 `requirements.lock.txt`。本包只在 Apple M5、32 GB、MPS 上实测，不是生产服务。

在本目录运行（将 Python 替换为已安装依赖的解释器）：

```sh
python demo.py --mode batch --calibration calibration_batch.json < demo_batch.jsonl
python demo.py --mode stream --calibration calibration_stream.json < demo_stream.jsonl
```

batch 与 stream 各用独立阈值。流式模式每个 session 独立 KV，逐个检查新 token，最多同时保留 4 个 session；用 `{"session":"名称","close":true}` 释放状态。已释放文本不能撤回。完整实验结论见上级 `ROUND1_REPORT.md`。
''')
    shutil.copy2(ROOT/'data/manifest.json',out/'data_manifest.json')
    shutil.copy2(ROOT/'post/chinese_metrics.json',out/'calibration_batch.json')
    shutil.copy2(ROOT/'post/stream_metrics.json',out/'calibration_stream.json')
    del model;gc.collect();torch.mps.empty_cache()
    restored,tok=load(dtype=torch.bfloat16,model_path=modeldir)
    with torch.no_grad():roundtrip=sum([batch_probs(restored,tok,conversations[i:i+4]) for i in range(0,len(rows),4)],[])
    reload_error=max(abs(a-b) for p,q in zip(after,roundtrip) for a,b in zip(p,q));assert reload_error<1e-6,reload_error
    manifest={'selected_checkpoint':str(checkpoint),'parity_cases':len(rows),'merge_probability_max_error':error,
        'reload_probability_max_error':reload_error,'merge_argmax_changes':sum(max(range(3),key=p.__getitem__)!=max(range(3),key=q.__getitem__) for p,q in zip(before,after)),
        'loader':'round1/runtime.py load(model_path=export/model) restores FP32 risk heads explicitly',
        'files':{str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}}
    write(out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
