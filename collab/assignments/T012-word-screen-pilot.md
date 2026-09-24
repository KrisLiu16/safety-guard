# T012 词表预筛：v4 复测，通过后全量

- 版本：v5（修订：只改第 5 步。全量超过平台单个 Run 2,000 个 Task 的上限，make_batch 现在按每份不超过 2,000 个 Task 打包，每份一个 Run；抽取脚本一次接受多个 Run。如果 v4 的第 3–4 步已经做完，直接从第 5 步开始；单测现在是 12 项）
- 目标：确认 v2 的分类与执行方 v1 的人工核对一致，然后对 Run A 和 Run B 的全部词做一次预筛，供 S2 放量（T013）、S5 安全一半放量（T015）、Run A 训练集清理和 R 切片重判使用。设计说明见 [round6/word_screen_v1/README.md](../../round6/word_screen_v1/README.md) 的 v2 一节。
- 依赖：T012 v1 完成（已完成）。
- 输入：`round6/word_screen_v1/`；Mac 上的 Run A 和 Run B 种子；本地 `known_positives.txt`（与 v1 相同）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/word_screen_v1/test_word_screen_cpu.py`，12 项全过。
  2. 按 v1 的人工核对，在 Mac 上写本地文件 `round6/word_screen_v1/manual_labels.tsv`，一行一个词，格式是“词<TAB>标签”：
     - v1 里 32 个 yes/unsure 词：A 类标 `insult`，B 类标 `evasion`，C 类标 `no`；
     - 随机看过的 100 个政治类 no 词：明确漏判的那个标 `rumor`，边界的那个标 `borderline`，其余 98 个标 `no`。

     确认 `git status` 里看不到它（已在 `.gitignore`），**不要提交**。
  3. 复测：用和 v1、v2、v3 相同的词，输出到 `pilot_v4`（默认两遍分类加一遍是/否，Task 数应为 18）：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --pilot 1000 --include round6/word_screen_v1/known_positives.txt --output round6/word_screen_v1/pilot_v4
     ```
     manifest 的 words 应当仍是 1,003，passes 为 2，binary_passes 为 1。提交方式同前（luna，`--flow round6/word_screen_v1/flow`，尝试 1 次，并发拉满）。抽取时加 `--manual round6/word_screen_v1/manual_labels.tsv`，输出到 `pilot_v4/extracted`，然后对照（`compare_manual.py` 只看模型自己的结论，不看人工覆盖）：
     ```bash
     .venv/bin/python round6/word_screen_v1/compare_manual.py --manual round6/word_screen_v1/manual_labels.tsv --screen round6/word_screen_v1/pilot_v4/extracted/screen.jsonl
     ```
     人工标注用原来的 `manual_labels.tsv`（v3 反馈里按定义改过的 `manual_labels_v3.tsv` 也可以再跑一遍，写进反馈）。
     抽取完成后删除这个 pilot 数据集（已定规则）。
  4. 复测门槛（满足就做第 5 步，否则停下写反馈）：
     - `compare_manual.py` 的 `manual_positive_flagged`（只看模型）≥ 0.90；
     - `manual_no_flagged` ≤ 0.10；
     - missing ≤ 1%；
     - 已知阳性：模型判为非 `no`，或者已被人工标注覆盖（`verdict_source` 为 manual）。

     说明：这批词的人工阳性大多是 v1 自己标出来的，第三遍又用了 v1 的问法，所以复测召回会偏乐观。门槛因此定在 0.90，不再在这批词上反复调。已知的漏网词由人工标注兜底。
  5. 全量：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --output round6/word_screen_v1/full
     ```
     默认读 Run A 和 Run B 两个种子文件，每个不同的词判一次（两遍分类加一遍是/否，约 5,100 个 Task）。因为平台限制每个 Run 最多 2,000 个 Task，输出会是 `screen_part0.tar.gz`、`screen_part1.tar.gz`……每份不超过 2,000 个 Task，manifest 的 `archives` 列出每份的 Task 数。**每份单独上传为一个数据集、单独提交一个 Run**（luna，`--flow round6/word_screen_v1/flow`，尝试 1 次，并发 512），run_no 按份存到 `full/run_no_part<N>.txt`。每个 Run 都先冻结 attempt 列表，再一次性抽取：
     ```bash
     .venv/bin/python round6/response_v14/freeze_attempts.py <run_no_N> --out round6/word_screen_v1/full/attempts_part<N>.json --expected-samples <第 N 份的 Task 数>
     .venv/bin/python round6/word_screen_v1/extract_screen.py <run_no_0> <run_no_1> ... --words round6/word_screen_v1/full/words.jsonl --out round6/word_screen_v1/full/extracted --attempts-json round6/word_screen_v1/full/attempts_part0.json round6/word_screen_v1/full/attempts_part1.json ... --manual round6/word_screen_v1/manual_labels.tsv
     ```
     （run_no 和 attempts 文件按同样的顺序给。）全量数据集属于正式放量，**不删除**。
- 预期产物：可提交 `pilot_v4/manifest.json`、`pilot_v4/run_no.txt`、`pilot_v4/extracted/summary.json`、`pilot_v4/dataset_rm.json`、`full/manifest.json`、`full/run_no_part*.txt`、`full/extracted/summary.json`。`screen.jsonl`、`words.jsonl`、`manual_labels*.tsv`、`known_positives.txt`、`attempts.json`、归档都含词本身或账号信息，不提交。
- 验收：第 4 步的门槛；全量 missing ≤ 1%。
- 需要用户决定：无（放量规则已定）。rumor 和 evasion 是否改标为 unsafe 是另一件事，不影响本任务。
- 反馈里必须报告：
  1. 单测结果；`manual_labels.tsv` 各标签的行数；
  2. 复测的 run_no、summary（`by_source_group` 只列有非 `no` 的组；包括 `pass_agreement_flagged`、`flagged_only_by_binary_pass`、`manual_overrides`）、`compare_manual.py` 对两份标注的输出全文，以及改过哪几行标注、依据的是哪条定义（不写词本身）；
  3. 如果做了全量：run_no、Task 数、summary 里的 `words`、`verdicts`、`model_verdicts`、`flagged_only_by_binary_pass`、`batch_errors`、`known_positive_model_recall_flagged`，以及非 `no` 最多的 10 个来源组；
  4. 产物 SHA256。
