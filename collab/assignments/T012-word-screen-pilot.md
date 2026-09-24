# T012 词表预筛：v3 复测，通过后全量

- 版本：v3（修订：v2 复测召回不够（aster-dev-290，漏了 3 个人工阳性）。v3 写明“党和国家领导人”的范围，把“对具体领导人的贬损说法即使属于历史或政治话题也不算 no”写清楚，每个词跑两遍、按最严重的结论合并。第 1–2 步已做过的不用重做，从第 3 步开始）
- 目标：确认 v2 的分类与执行方 v1 的人工核对一致，然后对 Run A 和 Run B 的全部词做一次预筛，供 S2 放量（T013）、S5 安全一半放量（T015）、Run A 训练集清理和 R 切片重判使用。设计说明见 [round6/word_screen_v1/README.md](../../round6/word_screen_v1/README.md) 的 v2 一节。
- 依赖：T012 v1 完成（已完成）。
- 输入：`round6/word_screen_v1/`；Mac 上的 Run A 和 Run B 种子；本地 `known_positives.txt`（与 v1 相同）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/word_screen_v1/test_word_screen_cpu.py`，9 项全过。
  2. 按 v1 的人工核对，在 Mac 上写本地文件 `round6/word_screen_v1/manual_labels.tsv`，一行一个词，格式是“词<TAB>标签”：
     - v1 里 32 个 yes/unsure 词：A 类标 `insult`，B 类标 `evasion`，C 类标 `no`；
     - 随机看过的 100 个政治类 no 词：明确漏判的那个标 `rumor`，边界的那个标 `borderline`，其余 98 个标 `no`。

     确认 `git status` 里看不到它（已在 `.gitignore`），**不要提交**。
  3. 复测：用和 v1、v2 相同的词，输出到 `pilot_v3`（默认每个词跑两遍，Task 数应为 12）：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --pilot 1000 --include round6/word_screen_v1/known_positives.txt --output round6/word_screen_v1/pilot_v3
     ```
     manifest 的 words 应当仍是 1,003，passes 为 2。提交方式同前（luna，`--flow round6/word_screen_v1/flow`，尝试 1 次，并发拉满），抽取到 `pilot_v3/extracted`，然后对照：
     ```bash
     .venv/bin/python round6/word_screen_v1/compare_manual.py --manual round6/word_screen_v1/manual_labels.tsv --screen round6/word_screen_v1/pilot_v3/extracted/screen.jsonl
     ```
     人工标注以原来的 `manual_labels.tsv` 为准。v2 反馈里提到的两个“标 no 但按新定义可能算阳性”的词，只能按 v3 提示词里写明的范围和定义去改（例如：前任政治局委员不是常委，不在范围内，仍是 no），改后另存为 `manual_labels_v3.tsv`（同样不提交）。两份标注都跑一遍 `compare_manual.py`，都写进反馈，验收以原标注为准。
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
     全量同样每个词跑两遍（约 3,400 个 Task）。全量数据集属于正式放量，**不删除**。
- 预期产物：可提交 `pilot_v3/manifest.json`、`pilot_v3/run_no.txt`、`pilot_v3/extracted/summary.json`、`pilot_v3/dataset_rm.json`、`full/manifest.json`、`full/run_no.txt`、`full/extracted/summary.json`。`screen.jsonl`、`words.jsonl`、`manual_labels*.tsv`、`known_positives.txt`、`attempts.json`、归档都含词本身或账号信息，不提交。
- 验收：第 4 步四项；全量 missing ≤ 1%。
- 需要用户决定：无（放量规则已定）。rumor 和 evasion 是否改标为 unsafe 是另一件事，不影响本任务。
- 反馈里必须报告：
  1. 单测结果；`manual_labels.tsv` 各标签的行数；
  2. 复测的 run_no、summary（`by_source_group` 只列有非 `no` 的组；包括 `pass_agreement_flagged`）、`compare_manual.py` 对两份标注的输出全文，以及改过哪几行标注、依据的是哪条定义（不写词本身）；
  3. 如果做了全量：run_no、Task 数、summary 里的 `words`、`verdicts`、`batch_errors`、`known_positive_recall_flagged`，以及非 `no` 最多的 10 个来源组；
  4. 产物 SHA256。
