# Round4 训练结果

生成时间（UTC）：2026-09-23T18:41:13+00:00。本报告只读取 `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4` 及收集器状态，不运行模型。

训练主流程记录为 **completed**；这仅表示训练和评估流程结束，不能据此判断长流审计通过或允许部署。

## 完成状态

| 模型 | 训练状态 | 导出权重开发评估 | 保留测试 | 速度 | 长流门槛 |
| --- | --- | --- | --- | --- | --- |
| full / SFT | completed | 已记录 | 已记录 | 已记录 | 通过且证据完整 |
| window / SFT | completed | 已记录 | 已记录 | 已记录 | 通过且证据完整 |
| memory / SFT | completed | 已记录 | 已记录 | 已记录 | 未通过或证据未齐 |
| 分类 RL | completed | 已记录 | 已记录 | 已记录 | 通过且证据完整 |

## 开发集选模

阈值从独立校准集的安全样本选取；开发评分为宏平均风险召回减去 `5 × 平均超额误报率`，误报预算为 5%。保留测试只报告结果，不用于重新挑选架构或检查点。

| 模型 | 完成更新数 | 选中步骤 | 导出权重开发分数 | 宏召回 | 宏 FPR | 宏 PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| full / SFT | 4,164 | 2,048 | 0.6117 | 69.50% | 6.50% | 0.9166 |
| window / SFT | 4,164 | 4,096 | 0.6100 | 70.17% | 6.83% | 0.9102 |
| memory / SFT | 4,164 | 2,560 | 0.6017 | 62.67% | 4.83% | 0.9061 |
| 分类 RL | 256 | 192 | 0.6283 | 70.33% | 6.33% | 0.9163 |

已记录的 RL 起点架构：**full**。该选择来自训练流程，报告不会按测试集结果改选其他架构。

在已固定架构的 SFT 与 RL 检查点之间，仅按导出权重开发分数和长流门槛，**分类 RL** 可作为下一步验收候选。分数相同保留 SFT；这不构成生产部署批准。

## 保留测试的风险能力

所有分层的阈值只来自各自校准集。宏平均对各语言/角色分层等权；它不是按样本量加权的总体指标。原版 A0 仅作同一新测试集上的质量参考。

| 模型 | 样本数 | 宏召回 | 宏 FPR | 宏 PR-AUC | 宏 ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| full / SFT | 1,200 | 70.83% | 4.17% | 0.9316 | 0.9224 |
| window / SFT | 1,200 | 70.83% | 5.00% | 0.9298 | 0.9190 |
| memory / SFT | 1,200 | 63.17% | 3.83% | 0.9260 | 0.9163 |
| 分类 RL | 1,200 | 70.83% | 4.33% | 0.9316 | 0.9231 |
| 原版 A0（质量参考） | 1,200 | 65.17% | 3.83% | 0.9361 | 0.9290 |

| 模型 | 语言/目标角色 | 安全数 | 风险数 | 风险召回 | FPR | PR-AUC | 校准阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full / SFT | en/assistant | 200 | 200 | 68.50% | 3.50% | 0.9259 | 0.932051 |
| full / SFT | zh/assistant | 200 | 200 | 74.50% | 5.50% | 0.9383 | 0.930509 |
| full / SFT | zh/user | 200 | 200 | 69.50% | 3.50% | 0.9305 | 0.863911 |
| window / SFT | en/assistant | 200 | 200 | 68.50% | 5.50% | 0.9256 | 0.950355 |
| window / SFT | zh/assistant | 200 | 200 | 76.50% | 5.50% | 0.9438 | 0.975690 |
| window / SFT | zh/user | 200 | 200 | 67.50% | 4.00% | 0.9200 | 0.990612 |
| memory / SFT | en/assistant | 200 | 200 | 57.50% | 3.00% | 0.9156 | 0.993402 |
| memory / SFT | zh/assistant | 200 | 200 | 77.00% | 6.50% | 0.9394 | 0.983933 |
| memory / SFT | zh/user | 200 | 200 | 55.00% | 2.00% | 0.9231 | 0.995482 |
| 分类 RL | en/assistant | 200 | 200 | 67.00% | 3.50% | 0.9270 | 0.953814 |
| 分类 RL | zh/assistant | 200 | 200 | 75.50% | 5.50% | 0.9382 | 0.957026 |
| 分类 RL | zh/user | 200 | 200 | 70.00% | 4.00% | 0.9296 | 0.906959 |
| 原版 A0（质量参考） | en/assistant | 200 | 200 | 59.50% | 1.50% | 0.9323 | 0.933603 |
| 原版 A0（质量参考） | zh/assistant | 200 | 200 | 74.50% | 8.00% | 0.9400 | 0.604234 |
| 原版 A0（质量参考） | zh/user | 200 | 200 | 61.50% | 2.00% | 0.9360 | 0.887965 |

## SFT 与分类 RL 对比

以下只比较已固定起点架构及其 RL 版本。RL 采样分类动作，不生成文本 token；测试差值仅作观察。

| 版本 | 开发分数 | 测试宏召回 | 测试宏 FPR | 测试宏 PR-AUC | 长流门槛 |
| --- | --- | --- | --- | --- | --- |
| full / SFT | 0.6117 | 70.83% | 4.17% | 0.9316 | 通过 |
| 分类 RL | 0.6283 | 70.83% | 4.33% | 0.9316 | 通过 |

RL − SFT 的保留测试差值：召回 +0.00 个百分点；FPR +0.17 个百分点；PR-AUC +0.0000。召回/PR-AUC 越高越好，FPR 越低越好。

## L20 推理速度与状态内存

按相同初始 context 和 chunk 分组比较。ITPS 仅统计新输入 token；延迟包含分类结果返回 CPU。数据是预热后的单会话 token-ID 模型调用，不含分词、HTTP 或批处理。此处不引用旧实验速度，也不为 A0 补造同场速度。

| 初始 context | chunk | 模型 | ITPS | 分类次数/秒 | P50 ms | P95 ms | prefill 状态 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 1 | full / SFT | 54.0 | 54.0 | 18.47 | 18.70 | 24.844 |
| 512 | 1 | window / SFT | 52.4 | 52.4 | 19.08 | 19.13 | 24.832 |
| 512 | 1 | memory / SFT | 47.2 | 47.2 | 21.15 | 21.54 | 27.855 |
| 512 | 1 | 分类 RL | 57.1 | 57.1 | 17.36 | 18.49 | 24.844 |
| 512 | 8 | full / SFT | 393.5 | 49.2 | 20.26 | 20.99 | 24.844 |
| 512 | 8 | window / SFT | 389.3 | 48.7 | 20.49 | 20.72 | 24.832 |
| 512 | 8 | memory / SFT | 353.5 | 44.2 | 22.61 | 22.81 | 27.855 |
| 512 | 8 | 分类 RL | 421.2 | 52.6 | 18.99 | 19.19 | 24.844 |
| 512 | 32 | full / SFT | 1552.7 | 48.5 | 20.57 | 20.94 | 24.844 |
| 512 | 32 | window / SFT | 1539.3 | 48.1 | 20.77 | 20.97 | 24.832 |
| 512 | 32 | memory / SFT | 1390.3 | 43.4 | 22.98 | 23.16 | 27.855 |
| 512 | 32 | 分类 RL | 1658.7 | 51.8 | 19.24 | 19.57 | 24.844 |
| 8,192 | 1 | full / SFT | 53.8 | 53.8 | 18.49 | 19.12 | 114.844 |
| 8,192 | 1 | window / SFT | 52.3 | 52.3 | 19.09 | 19.22 | 24.832 |
| 8,192 | 1 | memory / SFT | 47.0 | 47.0 | 21.15 | 22.05 | 27.855 |
| 8,192 | 1 | 分类 RL | 57.3 | 57.3 | 17.38 | 17.54 | 114.844 |
| 8,192 | 8 | full / SFT | 392.1 | 49.0 | 20.32 | 20.52 | 114.844 |
| 8,192 | 8 | window / SFT | 389.7 | 48.7 | 20.44 | 21.37 | 24.832 |
| 8,192 | 8 | memory / SFT | 354.4 | 44.3 | 22.51 | 22.77 | 27.855 |
| 8,192 | 8 | 分类 RL | 416.6 | 52.1 | 19.13 | 19.29 | 114.844 |
| 8,192 | 32 | full / SFT | 1544.4 | 48.3 | 20.64 | 20.84 | 114.844 |
| 8,192 | 32 | window / SFT | 1536.1 | 48.0 | 20.77 | 20.95 | 24.832 |
| 8,192 | 32 | memory / SFT | 1391.0 | 43.5 | 22.98 | 23.12 | 27.855 |
| 8,192 | 32 | 分类 RL | 1636.9 | 51.2 | 19.49 | 19.62 | 114.844 |

状态 MiB 是记录的缓存底层存储量，不是全部 GPU 显存；full 的状态随历史增长，window/memory 的有界性以独立长流审计为准。

## 8K 长流审计

审计比较导出检查点在相同输入前缀上的一次性分类与增量分类，检查 513、1025、4097、8192 tokens、两种输入顺序、两组分块方案及两个角色头。长输入由开发样本拼接，仅验证数值和缓存，不证明自然长对话的风险识别能力。

审计命令完成状态：`True`；命令返回码：`0`；审计文件状态：`completed`；文件记录的全部候选通过：`False`。命令返回 0 或训练 completed 都不能替代逐候选审计通过。

| 模型 | 最大概率误差 | 容差 | 8K 记录数 | 缓存有界 | SHA | 完整证据门槛 |
| --- | --- | --- | --- | --- | --- | --- |
| full / SFT | 0.002535 | 0.0300 | 4 | 不适用 | 匹配 | 通过 |
| window / SFT | 0.003063 | 0.0300 | 4 | True | 匹配 | 通过 |
| memory / SFT | 0.032561 | 0.0300 | 4 | True | 匹配 | 未通过或证据未齐 |
| 分类 RL | 0.002354 | 0.0300 | 4 | 不适用 | 匹配 | 通过 |

- memory / SFT：该检查点未通过长流审计；长流概率一致性未通过；数值误差证据缺失或超出容差。

## 检查点与运行记录

收集器记录的 PVC：`safety-guard-base-compare-data`。权重保留在 PVC；本报告没有下载权重，也没有重新计算远端文件 SHA。

| 模型 | PVC 内路径 | 路径证据 | 训练记录的 SHA-256 |
| --- | --- | --- | --- |
| full / SFT | /work/output/round4/full/best.safetensors | 收集器已记录 | a3f6e5a7084be871553ef92feddfd3fae38a2e7a42fc1a38b52445aae3ebbaa2 |
| window / SFT | /work/output/round4/window/best.safetensors | 收集器已记录 | bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2 |
| memory / SFT | /work/output/round4/memory/best.safetensors | 收集器已记录 | 41f6143d0df5dee1e5107d9601257cc341f035fcb20235e4c2cf131b213cb283 |
| 分类 RL | /work/output/round4/classification_rl/best.safetensors | 收集器已记录 | 29fc3e5663eaa333235b1c2ba84d4c0a964f6ece4fd146c9fc6ec01761eaf222 |

冻结数据量：calibration 900 条；dev 1,200 条；sealed_test 1,200 条；train 33,311 条。

| 数据分割 | 冻结 SHA-256 |
| --- | --- |
| calibration | a39f9a961ddf99ee8a01ab8e148be1749a4d6a6ddca36024712fb7615e0f2663 |
| dev | 7c3ca2d4e99e4070ae9ec602de6c5a4beeb41114373c6c96ceb98f36734d02c9 |
| sealed_test | 2fbcbe2b3494ceaa0caa9d0191a7e107ae16ef2a0734f53e24d37a1b75423e73 |
| train | 9925114b148ee38df3e193a1e427e38b7dd8c6805c1bf28a8b67da868c7c5aa7 |

## 解释边界

- 本轮基于已有 Qwen3.5 文本主干和风险头做后训练，不是从零预训练。所有神经网络计算在 L20。
- 公共标签遵循来源政策；合成标签未独立核验并降权，benign 模板只在其构造情境内提供规则标签。
- 本轮硬标签为 safe/unsafe；类别头及第三档风险没有独立新增监督和验收，不能由二分类结果推断其能力。
- 公开参考测试缺少英文用户分层；精确去重不等于语义去重，也不能保证基础模型预训练从未见过测试内容。
- 阈值的 5% 预算只约束有限校准样本；开发/测试的真实 FPR 单独报告，不是生产 SLA。
- token-ID 流式一致性不能替代任意文本重分词回滚、HTTP 服务、并发吞吐或真实长期风险上下文验收。
- A0 与当前分类器使用各自输入格式，质量差异不能只归因于架构；本报告不使用保留测试重新选模。
- 所有候选都没有本报告授予的生产批准；未完成或失败的长流审计阻止候选进入下一步验收。

## 本报告读取的证据文件

- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/final_summary.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/risk_run_spec.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/long_stream_audit.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/post_training_audit_status.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/checkpoint_locations.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/full/sft_summary.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/full/exported_dev_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/full/sealed_test_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/full/runtime.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/window/sft_summary.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/window/exported_dev_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/window/sealed_test_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/window/runtime.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/memory/sft_summary.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/memory/exported_dev_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/memory/sealed_test_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/memory/runtime.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/classification_rl/summary.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/classification_rl/exported_dev_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/classification_rl/sealed_test_metrics.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/classification_rl/runtime.json`
- `/Users/liuzhihao/Work/safety-guard/round4/results/output/round4/a0_reference_metrics.json`
