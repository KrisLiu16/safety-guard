"""Keep the exact trained adapter path as default; expose merged weights as an option."""
import hashlib
import json
from pathlib import Path
import shutil
from runtime import ROOT

if __name__=='__main__':
    out=ROOT/'export';base=out/'base_model';base.mkdir(exist_ok=True)
    for file in (ROOT/'model').iterdir():
        if file.is_file() and not (base/file.name).exists():shutil.copy2(file,base/file.name)
    merged_config=json.loads((out/'model/config.json').read_text());merged_config['round1_adapter_merged']=True
    (out/'model/config.json').write_text(json.dumps(merged_config,ensure_ascii=False,indent=2)+'\n')
    for name in ['runtime.py','demo.py']:shutil.copy2(ROOT/name,out/name)
    for mode,source in [('batch','chinese_metrics.json'),('stream','stream_metrics.json')]:
        shutil.copy2(ROOT/'post'/source,out/f'calibration_{mode}_unmerged.json')
    (out/'run_guard.py').write_text('''"""Recommended unmerged checkpoint; --merged selects the measured merge variant."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--mode',required=True,choices=['batch','stream'])
parser.add_argument('--merged',action='store_true')
parser.add_argument('--buffer-chars',type=int,default=32)
parser.add_argument('--deadline-ms',type=float,default=5000)
options=parser.parse_args();mode=options.mode;merged=options.merged
args=['--mode',mode,'--buffer-chars',str(options.buffer_chars),'--deadline-ms',str(options.deadline_ms),
      '--model-path',str(ROOT/('model' if merged else 'base_model'))]
if not merged:args+=['--checkpoint',str(ROOT)]
args+=['--calibration',str(ROOT/f"calibration_{mode}{'' if merged else '_unmerged'}.json")]
sys.argv=[sys.argv[0],*args]
from demo import main
main()
''')
    (out/'README.md').write_text('''# M5 Guard 第一轮导出

默认采用基座 + FP32 LoRA/风险头，保留开发集选出的原始训练数值路径。`--merged` 启用已测量的合并权重；合并更快一点，但 BF16 舍入会造成少量分类/暂挂变化，详见上级实验报告。

在本目录使用已安装 `requirements.lock.txt` 中依赖的 Python：

```sh
python run_guard.py --mode batch < demo_batch.jsonl
python run_guard.py --mode stream < demo_stream.jsonl
python run_guard.py --mode batch --merged < demo_batch.jsonl
```

`base_model/` 为固定 revision 原版；`adapter.safetensors` 和 `risk_heads.safetensors` 为训练更新；`model/` 为可选合并权重。两个变体、两种输入方式分别使用自己的校准文件。不要在合并权重上再次叠加 adapter。

流式模式检查块内每个新 token，保留 32 字符缓冲，最多保留 4 个会话；用 `{"session":"名称","close":true}` 释放会话。默认 5 秒软时限：不能抢占已提交 GPU 的算子，返回时若超时则 hold，不释放新文本。`label` 是模型 argmax，`action` 是阈值/缓冲策略的动作，两者可能不同。已释放文字不可撤回。

这是本机 M5/MPS 验证的研究原型；中文测试标签来自合成数据，不能把 calibration 上的 5% 误拦目标当作线上保证。完整数据、质量和吞吐对比见上级 `ROUND1_REPORT.md`。
''')
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['recommended_variant']='base_model + adapter/risk_heads (unmerged)'
    manifest['optional_variant']='model/ (merged)'
    manifest['recommended_reason']='Preserve the exact selected training path; merged BF16 rounding changes some held-out review decisions.'
    manifest['files']={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        for p in out.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Finalized both variants; default launcher preserves unmerged training precision.')
