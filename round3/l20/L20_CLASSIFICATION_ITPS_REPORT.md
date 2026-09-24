# 直接分类模型 A0/C1：L20 质量与输入速率首轮对照

2026-09-23，Job `safety-guard-classification-itps-20260923` 在一张 NVIDIA L20 上完成。哈希锁定的[评测包清单](bundle_classification_eval_manifest.json)包含原版 A0、14 层 C1、20k 开放语料 Stage0 adapter 和冻结的 Qwen3GuardTest 子集。输出为[完整汇总](classification_eval_output/summary.json)、[A0 明细](classification_eval_output/a0_original/classification_benchmark.json)与[C1 明细](classification_eval_output/c1_stage0_20k/classification_benchmark.json)。两个模型都直接输出风险/类别 logits，无 `lm_head`、无生成 token；吞吐单位为输入 token/s（ITPS）。

| 同卡指标 | A0 原版 0.6B | C1 Stage0 14 层 |
|---|---:|---:|
| 参数量 | 597,111,296 | 376,878,080 |
| 2,048 输入 token 整句中位 ITPS | 57,774 | 112,215 |
| 256 历史 + 逐 1 token、单会话 ITPS | 59.0 | 115.8 |
| 1,024 历史 + 32 token/chunk、32 会话 ITPS | 1,592.9 | 2,978.0 |
| 256 历史 + 逐 1 token 的分类 P95 | 19.25 ms | 8.89 ms |
| 官方 `thinking` 100 条 strict F1 | 0.819 | **0.603** |
| `thinking` 安全样本误报率 | 0.191 | **0.596** |
| `response_loc` 100 条 unsafe 漏判率 | 0.15 | **0.38** |

C1 的首轮本机 HTTP 完整分类（24 次、每次约 130 输入 token）测得 **9,146 ITPS**、70.4 次分类/s、P95 14.57 ms；这是单机 localhost、串行 GPU 工作线程、热身后的短文本回放，未包含真实网络/队列，也没有 A0 HTTP 对照。第一次流式追加确实返回三档风险和类别；整句与 1/8/16 token 切块末态的最大风险概率差约 0.0042、类别差约 0.0073。LoRA 合并后风险/类别概率最大差约 0.0029。

**结论：C1 当前不能进入安全服务。** 教师匹配开发损失下降不等于独立风险能力恢复；官方子集误报和漏判明显恶化，约 2 倍的输入速率收益不能抵消这一点。暂保留 A0 为质量基线，C1 仅作架构实验。`thinking_loc` 和 `response_loc` 定位分片都是 unsafe，不能从它们推断误报率；`thinking` 100 条也只是回归子集，需更大独立集与区间估计。当前分数未作项目金标校准。

下一步先验证质量恢复：比较未训练 C1、Stage0 C1 与更浅裁剪幅度；再用来源族隔离的开放数据做结构蒸馏/辅助训练，冻结 Qwen3GuardTest 只作测试。仅在 C1 接近 A0 的固定误报率下召回和流式定位时，进一步测试优化后的 token-ID 快路、文本 HTTP 并发和 H1/H2 混合架构。首轮包在新增 `append_token_ids()` 之前冻结，首轮流式数字来自累计 ID 接口；快路须单独 L20 实测，不能沿用这些数字。
