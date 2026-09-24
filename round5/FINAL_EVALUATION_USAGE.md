# 固定新保留集的最终 L20 评估

这一步只在 Round5 主训练完成并且最终 checkpoint、selected、best_metrics、run binding 全部对应后执行。它不训练、不选择模型，也不在新保留集拟合阈值。

默认命令：

    /work/modern/bin/python /work/round5/evaluate_final_l20.py \
      --training-output /work/output/round5/prefix_v2 \
      --data-root /work/round5/data/prefix_v2 \
      --fresh-root /work/round5/data/fresh_holdout_v1 \
      --training-tokenizer-proof /work/round5/native_tokenizer_proof.json \
      --output /work/output/round5/fresh390_final

脚本需要同目录的已冻结 train_prefix.py 和 stream_metrics.py。它验证源文件 SHA、已完成训练的 modern tokenizer proof 绑定，再在实际 Transformers 5.17.0 / Tokenizers 0.23.2 环境重新核对 fresh 390 条原始输入和全部字符切点的 native IDs、角色内容边界；不替换任何保存 IDs。模型调用只能使用单张可见 L20。

输出包括：

- audit.json：完整门槛、checkpoint/calibration 身份、覆盖、family 隔离、固定阈值结果和初始 window 对照差值。
- fresh_tokenizer_proof.json：实际 modern 原始/cut 输入逐 ID 验证、资产/版本与计数。
- initial_window/predictions.jsonl、metrics.json：固定初始 window，使用原 sft/evaluation_0 的校准阈值。
- selected/predictions.jsonl、metrics.json：已冻结最终模型，使用它实际选中评估点保存的校准阈值。

保存每条 endpoint/native/text-cut/stream 分数、whole/stream 固定阈值与严格“大于”判决。完整覆盖为 390 条、390 个唯一 family、三个语言/角色分层各 65 safe + 65 unsafe。如果最终 SHA 与初始 window 相同，神经观测复用一次并注明，两个结果仍各自使用其已保存校准回执。

阈值只来自原训练 calibration_predictions 和 metrics；脚本先用旧 cal/dev 逐例复算 metrics 以验证来源，对 fresh 只应用这些数值。它不会调用新保留集的选模函数。integrity_pass 表示身份、覆盖和计算过程通过，不表示新模型在质量上超过初始模型，也不是生产准入。

运行时复查 fresh 与冻结 train/cal/dev 的 family/ID 无交集。21 个更早历史文件的排除依据是已独立 CPU 审过并固定 SHA 的 fresh manifest，不假定 GPU 容器中仍挂载所有历史文件。历史文本排除仅针对长度至少 20 字符；短文本有明确豁免，不能声称任意长度都零重复或语义无污染。

默认拒绝覆盖已有输出目录；失败会保留 audit.json 和错误阶段。不得把失败后的重复执行当成新的模型/阈值搜索。新数据仅在候选冻结后作这一次最终观测；旧官方与旧 sealed 应在另一路明确标为回归集。

本地 CPU 检查：

    python3 round5/test_evaluate_final_cpu.py -v

这些 fixtures 不导入 torch 或 transformers，不运行模型，不代表 L20 评估已经成功。

