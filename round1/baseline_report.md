# 原版 Qwen3Guard-Stream-0.6B 基线

固定 revision：419364a715de9840d47b1457982f64ff37f90ed4。Apple M5 / 32 GB，PyTorch MPS，BF16。中文数据为按词族隔离的 Luna 合成测试，不是人工金标。

| 目标 | test 数量 | argmax F1 | 校准召回 | 测试 block 误拦 | 正常 hold 数量 |
|---|---:|---:|---:|---:|---:|
| user | 446 | 0.9617 | 0.9921 | 0.0000 | 24 |
| assistant | 358 | 0.9780 | 0.9828 | 0.0041 | 0 |

阈值仅在 calibration 上选择，使经验 block FPR ≤5%。hold 单独报告。

| 官方分片 | 覆盖 | strict F1 | loose F1 | 定位 exact hit |
|---|---|---:|---:|---:|
| thinking | 1059/1059 | 0.8144 | 0.8175 | 0.2074 |
| thinking_loc | 569/569 | 0.8994 | 0.9026 | 0.2074 |
| response_loc | 813/813 | 0.9428 | 0.9496 | 0.8364 |

strict/loose 沿用官方打印定义，两个定位分片均为风险例，thinking_loc 与 thinking 重叠，不相加为独立样本。

| 总上下文 | 原包装器逐 token P95 ms | 正确 KV 逐 token P95 ms |
|---:|---:|---:|
| 256 | 49.10 | 16.63 |
| 1024 | 201.60 | 18.57 |
| 4096 | 1159.58 | 29.53 |
| 8192 | 3244.63 | 47.75 |

原版新进程首条 JSON 中位数 2.45s；OS/Metal 缓存未清空。热态逐 token 测试每格 96 个续流步，设备同步计时，不含输入到达等待。

原始预测、校准阈值、置信区间及覆盖率在 `baseline/`；完整时延、吞吐和 driver allocation 在 `performance/original_performance.json`。独立缓存/梯度验证见 `smoke_results.json`，上游指标一致性验证见 `official_metrics_parity.json`。
