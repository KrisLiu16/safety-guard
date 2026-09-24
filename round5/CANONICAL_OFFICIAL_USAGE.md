# Canonical32 官方适配评估

`evaluate_canonical_official_l20.py` 只评估固定 Round5 checkpoint，不训练、选模或调阈值。它要求训练完整结束、已有锁定 canonical calibration，并验证原 cal/dev 预测及源码/版本/权重绑定。即使服务门槛未通过或保留初始 fallback，仍允许研究评估，结果显式保留状态，不构成模型批准。

```sh
/work/modern/bin/python /work/round5/evaluate_canonical_official_l20.py \
  --training-output /work/output/round5/prefix_v2 \
  --calibration-output /work/output/round5/canonical32_calibration \
  --data-dir /work/validation/official_adapted_v1 \
  --source-dir /work/validation/qwen3guardtest \
  --output /work/output/round5/canonical32_official
```

默认自动使用已校准的 eager/graph 实现与 pad ID；显式提供不同引擎或 pad 会被拒绝。所有神经调用仍只允许单张 CUDA L20。输出目录存在时拒绝重复覆盖。

数据保持原三个片：thinking 1059、thinking_loc 569、response_loc 813，共 2441 行。模型用原训练序列化和每行冻结 native IDs、eval_start_index，在 canonical32 中读取全部真实位置的 assistant 三类概率，再执行冻结脚本的原判决：先在整个评估区间找连续两个 Unsafe；没有时才找连续两个 Controversial；否则 Safe。Unsafe-only 与包含 Controversial 两套指标及句末 argmax 均保留，不把争议档当新增已验收能力，不套原 Qwen3 标注位置计算定位分数。

恰好 569 个重复的完整 IDs+start 结果复用，因此是 **1872 个唯一推理序列**，不能写成 2441 个神经 forward。每个序列又有 ceil(n/32) 次物理块调用；报告单列 physical_block_forward_calls、physical_forward_tokens、真实输入与 padding，生成 tokens=0。两重叠片不汇总成独立总分；两个纯 Unsafe 片的 FPR 无定义。

脚本逐行验证实际 modern tokenizer ID，原 saved IDs 永不替换。A0 只读取既有完整同源结果，固定其历史 SHA 与 `kind=a0`；不会加载 A0 模型或把 A0 标成新学生。旧 round4 window/SFT 的 bulk 结果另作历史诊断，保留原 checkpoint SHA，与新学生结果分别保存。

主要输出为 `metrics.json`、三个 `*_predictions.jsonl` 及 `official_native_tokenizer_proof.json`。完整覆盖是成功条件，不能通过丢掉失败行改善指标。`integrity_pass=true` 仅表示评估完整可信，不代表服务质量过关。CPU 测试：`python -m unittest test_canonical_official_cpu.py`；这些测试不执行神经模型。
