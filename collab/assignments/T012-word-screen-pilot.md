# T012 词表预筛：v2 复测，通过后全量

- 版本：v2（修订：v1 pilot 已完成，见反馈。v2 把结论从 yes/no/unsure 改成类别 insult / rumor / evasion / unsure / no，用同一批 1,003 个词复测；通过后直接全量筛。v1 的步骤不用重做）
- 目标：确认 v2 的分类与执行方 v1 的人工核对一致，然后对 Run A 和 Run B 的全部词做一次预筛，供 S2 放量（T013）、S5 安全一半放量（T015）、Run A 训练集清理和 R 切片重判使用。设计说明见 [round6/word_screen_v1/README.md](../../round6/word_screen_v1/README.md) 的 v2 一节。
- 依赖：T012 v1 完成（已完成）。
- 输入：`round6/word_screen_v1/`；Mac 上的 Run A 和 Run B 种子；本地 `known_positives.txt`（与 v1 相同）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/word_screen_v1/test_word_screen_cpu.py`，7 项全过。
  2. 按 v1 的人工核对，在 Mac 上写本地文件 `round6/word_screen_v1/manual_labels.tsv`，一行一个词，格式是“词<TAB>标签”：
     - v1 里 32 个 yes/unsure 词：A 类标 `insult`，B 类标 `evasion`，C 类标 `no`；
     - 随机看过的 100 个政治类 no 词：明确漏判的那个标 `rumor`，边界的那个标 `borderline`，其余 98 个标 `no`。

     确认 `git status` 里看不到它（已在 `.gitignore`），**不要提交**。
  3. 复测：用和 v1 完全相同的参数生成 pilot，输出到 `pilot_v2`：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --pilot 1000 --include round6/word_screen_v1/known_positives.txt --output round6/word_screen_v1/pilot_v2
     ```
     manifest 的 words 应当仍是 1,003。按 v1 的方式提交（luna，`--flow round6/word_screen_v1/flow`，尝试 1 次，并发拉满），抽取到 `pilot_v2/extracted`，然后对照：
     ```bash
     .venv/bin/python round6/word_screen_v1/compare_manual.py --manual round6/word_screen_v1/manual_labels.tsv --screen round6/word_screen_v1/pilot_v2/extracted/screen.jsonl
     ```
     抽取完成后删除这个 pilot 数据集（已定规则）。
  4. 复测验收（全部满足才做第 5 步，否则停下写反馈）：
     - 已知阳性全部是非 `no`；
     - `manual_positive_flagged` ≥ 0.95（人工标为 insult、rumor、evasion 的词里，被判为非 `no` 的比例）；
     - `manual_no_flagged` ≤ 0.10；
     - missing ≤ 1%。
  5. 全量：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --output round6/word_screen_v1/full
     ```
     （默认读 Run A 和 Run B 两个种子文件，每个不同的词判一次，约 1,700 个 Task。）提交方式同第 3 步，并发拉满。Task 超过 100 个，抽取前先冻结 attempt 列表：
     ```bash
     .venv/bin/python round6/response_v14/freeze_attempts.py <run_no> --out round6/word_screen_v1/full/attempts.json --expected-samples <Task 数>
     .venv/bin/python round6/word_screen_v1/extract_screen.py <run_no> --words round6/word_screen_v1/full/words.jsonl --out round6/word_screen_v1/full/extracted --attempts-json round6/word_screen_v1/full/attempts.json
     ```
     全量数据集属于正式放量，**不删除**。
- 预期产物：可提交 `pilot_v2/manifest.json`、`pilot_v2/run_no.txt`、`pilot_v2/extracted/summary.json`、`pilot_v2/dataset_rm.json`、`full/manifest.json`、`full/run_no.txt`、`full/extracted/summary.json`。`screen.jsonl`、`words.jsonl`、`manual_labels.tsv`、`known_positives.txt`、`attempts.json`、归档都含词本身或账号信息，不提交。
- 验收：第 4 步四项；全量 missing ≤ 1%。
- 需要用户决定：无（放量规则已定）。rumor 和 evasion 是否改标为 unsafe 是另一件事，不影响本任务。
- 反馈里必须报告：
  1. 单测结果；`manual_labels.tsv` 各标签的行数；
  2. 复测的 run_no、summary（`by_source_group` 只列有非 `no` 的组）、`compare_manual.py` 的输出全文；
  3. 如果做了全量：run_no、Task 数、summary 里的 `words`、`verdicts`、`batch_errors`、`known_positive_recall_flagged`，以及非 `no` 最多的 10 个来源组；
  4. 产物 SHA256。
