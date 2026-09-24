# 固定 W512 前缀后训练入口

`train_prefix.py` 只写新的 `/work/output/round5/`，起点固定为 round4 window/SFT 的完整 24 层 checkpoint（SHA `bb16a3…30d2`）。category 参数冻结；风险投影、风险头和 backbone 全参数训练。输出是分类概率，零生成 token。

先在实际 modern 环境完成 CPU 分词证明；不会加载神经模型：

```sh
/work/modern/bin/python /work/round5/train_prefix.py --verify-data-only \
  --data-root /work/round5/data/prefix_v2 \
  --proof /work/round5/native_tokenizer_proof.json
```

证明检查 85,399 个原始/增强/文本切分视图的原 token IDs，并对 207,810 个 suffix 锚点用实际 native decoder 确认原始完整文本已经出现。保存数据、脚本、tokenizer 资产和库版本哈希；不重新生成或替换 ID。源码冻结后才能生成最终证明，正式训练将逐项验证关联。

只在单张 CUDA L20 上运行。先用独立输出目录做一个真实更新的 smoke，再使用正式输出：

```sh
/work/modern/bin/python /work/round5/train_prefix.py --smoke-only \
  --output /work/output/round5/prefix_v2_smoke
/work/modern/bin/python /work/round5/train_prefix.py \
  --output /work/output/round5/prefix_v2
```

SFT 为 2 个 epoch，每个 33,311 条原记录、2,082 次更新，effective batch 16 / microbatch 4；backbone/head LR 为 8e-6 / 5e-5，沿用原 warmup/cosine 调度。每条原句末 CE 始终在原 IDs 上计算；原 safe interior 锚 confidence=.1，仅其指定 epoch 激活 suffix。前缀项 `.5 * weight * sum(confidence * CE) / 有效锚点数`；原始与增强分开 backward。只在 0、epoch 1、epoch 2 评估。

RL 从选定 SFT 出发，以自身冻结 SFT 为参考，8 条记录 / microbatch 4、LR 5e-7、256 次更新，每 64 次评估。仅原完整句末和允许来源的 complete suffix；每锚 4 个二元分类动作，先在原记录内平均，再施加来源权重。沿用 +1/-3/-4、精确动作期望 baseline、.1 KL、每可信锚点 .2 CE；不把自然 safe interior 当 hard reward，也不是 episode stopping 策略。`trajectory_metrics.jsonl` 记录 source ID、来源权重、位置、动作与参考概率、4 个动作、reward、baseline、CE/KL/policy loss 及最终归一化系数，支持逐动作复算；常规进度输出不打印这些明细。

所有评估使用导出的 BF16 backbone checkpoint，保存逐例原 token 风险、文本切分末尾风险和指标。original 与独立 cut 均按长度排序 batch 4；相同 IDs+role 的 cut 合并，完整 cut 直接复用同次原 endpoint。calibration 只定阈值，dev 只作固定门槛与选模；完整句 recall 基线始终为初始 checkpoint 的值。没有 eligible 候选则保留起点并明确不晋升，不读 test/official/fresh holdout。`observations()` 与 `evaluate_checkpoint()` 可供后续独立验收复用。

中断后可在同输出目录加 `--resume`：SFT 每 256 步及 epoch 末、RL 每 64 步原子保存模型、optimizer、RNG 和选择状态。恢复截掉未提交日志尾部，并校验/恢复与原子状态一致的 best checkpoint。旧 round4 状态和权重始终保留。CPU 检查为 `python -m unittest test_train_prefix_cpu.py`；这些检查不替代 L20 smoke、数值一致性或正式效果验收。
