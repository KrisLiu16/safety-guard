# 请求：数据平台停着期间，请设计方出不依赖平台的任务单

- 提出：执行方，2026-09-24 18:20（北京时间）。用户看过下面两条后回复：
  > 推吧，让云上出任务单
- 背景：现在执行方没有可跑的任务。T020 第 3 步起、T021 和 T018 v3 都要先用裁判模型打标签，而裁判走 Aster。L20 上目前没有 safety-guard 的作业在跑，节点空闲。

## 请求 1：服务引擎提速（T007）现在出任务单

- T007 只用 L20，不依赖标签和数据平台。
- 候选内容见 `round6/serving/RESULTS.md` 的“下一步”，按预计收益排序：
  1. GDN 状态改为 BF16；
  2. 注意力的环形缓冲改为原地读取；
  3. 权重改为 FP8；
  4. 把 A0 放到同一个引擎上对比测速。

## 请求 2：不调裁判，先粗估 13 类红线各有多少正例

用户想先知道缺多少数据。执行方核对到的事实如下：

- **prefix_v2 助手侧 train 的 unsafe 记录共 6,794 条**（见 `round6/redline_v1/input/prefix_v2_assistant.manifest.json`）。按 `source` 分：
  - BeaverTails round0 30k：4,184 条；
  - Nemotron-Safety-Guard-Dataset-v3（中文）：2,610 条；
  - 合成数据：约 294 条（manifest 里 `synthetic_context_weak` 那一档）。
- **这些记录本身没有类别字段**，只有 `source`、`source_revision`、`sample_id`，例如 `beaver-21199`。
- **原始公开数据在 Mac 上，带类别：**
  - `round3/data/raw/beavertails_round0_30k/train.jsonl.gz`：14 个类别布尔字段，例如 `self_harm`、`terrorism,organized_crime`、`sexually_explicit,adult_content`、`drug_abuse,weapons,banned_substance`、`controversial_topics,politics`；
  - `round3/data/raw/nemotron_safety_guard_v3_zh/{train,valid}.jsonl`：`violated_categories` 字段。
- **这些类别和 13 类红线不是一一对应的。** 例如 BeaverTails 的暴力类不区分是否血腥，`politics` 也不等于本口径的政治红线。所以只能粗估，映射表需要设计方来定。
- **Run A** 按 DATA_SPEC_v15 第 3 节，骚扰和造谣约占 70–80%，红线类很少。

## 不属于设计方的一条，记录备查

- 13 类红线的 S1 生成指令，按分工由用户和团队成员编写，设计方和执行方都不写。执行方已向用户建议，趁平台停着先写好。
