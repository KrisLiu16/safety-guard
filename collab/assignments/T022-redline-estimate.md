# T022 不调裁判，粗估现有训练数据里 13 类红线各有多少正例

- 版本：v1
- 目标：回答执行方请求 2（用户同意）：在红线标注（T020/T021）之前，先粗估 prefix_v2 公开数据和 Run A 里各类红线大概有多少正例，给 S1 缺口定个量级。代码 `round6/redline_estimate/estimate_positives.py`，对照表写在代码开头。
- 做法：
  - prefix_v2 的 unsafe 记录用 `sample_id` 连回原始公开数据取类别。`beaver-<行号>` 对应 BeaverTails 的第几行；Nemotron 经 `nemotron_assistant_candidates.jsonl` 的 `candidate_id → source_id` 对应到原始 `id`。
  - 每个原始类别由设计方定一个档：
    - `likely`：这个类别的有害内容通常就在红线内，例如自伤 → R13；
    - `possible`：只有一部分在红线内，例如武器类只有“制造方法”才算 R12，所以是上限；
    - `none`：不在红线内，例如隐私、诈骗、骚扰。按定稿口径，这些训练时标 safe。
  - 政治类的原始类别分不出 R1、R3–R6，合并报告为 `political`。
  - Run A 没有类别，只按词的词表来源组做话题粗估。这个数单独报，不和公开数据相加：v14 的 unsafe 回答不论什么词，大多是骚扰和造谣。
  - 真实数字以 T020/T021 的标注为准。
- 依赖：无（只读 Mac 上已有的文件，不调模型，不用平台）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/redline_estimate/test_estimate_cpu.py`，3 项全过。
  2. 运行（默认路径就是 Mac 上的文件；路径不同就用参数指定）：
     ```bash
     .venv/bin/python round6/redline_estimate/estimate_positives.py
     ```
     输出 `round6/redline_estimate/estimate_v1.json`，只有计数，没有原文。
  3. 如果 `unmapped_categories` 不为空（原始数据里有对照表没列到的类别名），照原样列在反馈里，由设计方补对照表后重跑；不要自己改表。
- 预期产物：提交 `estimate_v1.json`。
- 验收：单测全过；`records_without_category` 应基本只有合成数据那一档（约 294 条）。
- 需要用户决定：无。
- 反馈里必须报告：
  - 脚本打印的全部内容；
  - `prefix_v2_unsafe` 中 `train/all`、`train/beavertails`、`train/nemotron_zh` 三行的计数；
  - `unmapped_categories`、`records_without_category`；
  - `runA_unsafe_by_topic_proxy`；
  - 产物 SHA256。
