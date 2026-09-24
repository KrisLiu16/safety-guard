# T022 反馈

- 对应任务卡：v1
- 状态：**完成，验收通过**
- 执行时间：2026-09-24 18:38–18:40（北京时间）
- 实际执行的命令：与任务卡相同。只读 Mac 上已有的文件，`model_calls`=0。

## 结果

1. 单测：`test_estimate_cpu.py` 3 项全过。
2. 脚本打印的全部内容：
   ```
   prefix_v2 train unsafe records: 6684, no red line: 1199
     political  likely      0   possible or likely    364
     R2         likely      0   possible or likely   2398
     R7         likely      0   possible or likely     78
     R8         likely      0   possible or likely   1375
     R9         likely      0   possible or likely      0
     R10        likely      0   possible or likely    275
     R11        likely    142   possible or likely    532
     R12        likely      0   possible or likely   3368
     R13        likely    251   possible or likely    251
   records without category: {'synthetic_context_weak': 110}  unmapped: {}
   Run A unsafe by topic proxy: {"calibration": {"mixed_or_unknown": 2464, "records": 2528, "political": 42, "R12": 14, "R11": 4, "R7": 4}, "dev": {"mixed_or_unknown": 2518, "records": 2594, "political": 44, "R12": 10, "R11": 12, "R7": 6, "R10": 4}, "train": {"mixed_or_unknown": 77268, "records": 79400, "political": 1364, "R12": 348, "R11": 264, "R7": 118, "R10": 38}}
   ```
3. `prefix_v2_unsafe` 的三行。每格是 likely / possible_or_likely；R1、R3–R6 合并在 political 里；R9（邪教）在对照表里没有对应的原始类别，所以两个值都是 0：

   | 行 | records | no_red_line | political | R2 | R7 | R8 | R9 | R10 | R11 | R12 | R13 |
   |---|---:|---:|---|---|---|---|---|---|---|---|---|
   | train/all | 6,684 | 1,199 | 0 / 364 | 0 / 2398 | 0 / 78 | 0 / 1375 | 0 / 0 | 0 / 275 | 142 / 532 | 0 / 3368 | 251 / 251 |
   | train/beavertails | 4,184 | 846 | 0 / 254 | 0 / 1855 | 0 / 78 | 0 / 1017 | 0 / 0 | 0 / 169 | 0 / 213 | 0 / 1889 | 61 / 61 |
   | train/nemotron_zh | 2,500 | 353 | 0 / 110 | 0 / 543 | 0 / 0 | 0 / 358 | 0 / 0 | 0 / 106 | 142 / 319 | 0 / 1479 | 190 / 190 |

4. `unmapped_categories`：空。`records_without_category`：`{"synthetic_context_weak": 110}`。
   - 只有合成那一档，符合验收。
   - 这一档的 unsafe 共 294 条（见 prefix_v2 manifest），其中 184 条通过 `sample_id` 连回了原始数据、带上了类别，所以没有类别的只有 110 条。
   - 按来源分：Nemotron 的 unsafe 共 2,610 条，其中 2,500 条连上、110 条没连上；BeaverTails 的 4,184 条全部连上。
5. `runA_unsafe_by_topic_proxy`（按词的来源组粗估，不和公开数据相加）：

   | split | records | mixed_or_unknown | political | R12 | R11 | R7 | R10 |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | train | 79,400 | 77,268 | 1364 | 348 | 264 | 118 | 38 |
   | dev | 2,594 | 2,518 | 44 | 10 | 12 | 6 | 4 |
   | calibration | 2,528 | 2,464 | 42 | 14 | 4 | 4 | 0 |

## 执行方观察（只陈述数字）

- **prefix_v2 train**：只有 R11（142 条，全部来自 Nemotron）和 R13（251 条）有 likely 正例，合计 393 条。其余各类的 likely 都是 0。possible 上限最大的是 R12（3,368）、R2（2,398）、R8（1,375）。
- **Run A train**：79,400 条 unsafe 回答中，约 97.3% 落在 mixed_or_unknown。political 话题 1,364 条，R12、R11、R7、R10 合计 768 条。

## 产物

- `round6/redline_estimate/estimate_v1.json`：SHA256 `8b4c795ced216adefb5a71ad5596231bb4b4fff1c49587f0e7979e59af327cb9`。已提交。
