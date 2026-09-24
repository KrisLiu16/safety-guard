# C1 开放语料 Stage0 续训报告

2026-09-23，腾讯云 `cls-og1rjus2` 的单张 NVIDIA L20 上完成 Job `safety-guard-c1-stage0-20k-20260923`。从[1k pilot adapter](stage0_output/summary.json)接续，对[固定的 20k train / 2k dev 清单](../data/stage0_distillation_20k/manifest.json)作同一可见前缀的 A0→C1 软分布与表征蒸馏；**未使用来源安全标签或项目政策标签**。权重、代码和语料打包哈希见 `bundle_stage0_20k_manifest.json`，远端校验后训练，下载时再次核对 adapter SHA。

| 指标 | 续训前 | 续训后 |
|---|---:|---:|
| dev total teacher-matching loss | 1.3504 | 1.0523 |
| dev risk KL | 0.7071 | 0.5552 |
| dev category KL | 1.5902 | 1.1978 |
| dev representation cosine loss | 0.8314 | 0.6887 |

20,000 条训练输入中 19,776 条进入训练、224 条超长跳过；开发集 2,000 条中 1,973 条进入复评。共 2,472 optimizer steps、330,752 个可训练 LoRA/head 参数，峰值 CUDA 已分配约 2.09 GiB，训练与复评墙钟约 1,146 秒。输出[summary](stage0_20k_output/summary.json)、[adapter](stage0_20k_output/stage0_adapter.safetensors)和[损失曲线](stage0_20k_output/loss_curve.jsonl)。Adapter SHA-256：`a0150b08ca515a8449e32ee29c3598e815d07fd900b5d8182bca2facde50f682`。

这个结果只证明 C1 在保留来源族隔离的开放语料开发集上更接近冻结教师。它不证明政治公共事务误拦、风险漏判、流式证据起点或服务延迟已达目标。下一步用独立自然语境和 Qwen3GuardTest 等基准比较 A0、pilot C1、续训 C1，再准备人工金标的最终安全对齐；正式速度仍须同卡同负载对照。
