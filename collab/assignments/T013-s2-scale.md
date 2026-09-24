# T013 S2 难负例放量（约 7,000 词）

- 版本：v1
- 目标：按 S2 v2 提示词生成约 7,000 个词的难负例（每个词一条提问、一段回答、一段思考），用 luna 上的 judge v3.2 复核，只保留复核全部判为 safe 的样本。设计见 [round6/s2_v15/README.md](../../round6/s2_v15/README.md)；复核规则见 BOARD“已定”。
- 依赖：T012 全量预筛完成（需要 `round6/word_screen_v1/full/extracted/screen.jsonl`）。
- 步骤：
  1. 单测：分别运行 `.venv/bin/python -m unittest round6/s2_v15/test_s2_cpu.py`（7 项）和 `.venv/bin/python -m unittest round6/test_shared_cpu.py`（6 项），全部通过。
  2. 生成任务（排除预筛标出的词和两个 pilot 用过的词，每个 Task 10 个词）：
     ```bash
     .venv/bin/python round6/s2_v15/make_pilot.py --words 7000 --screen round6/word_screen_v1/full/extracted/screen.jsonl --exclude round6/s2_v15/pilot/seeds.jsonl --exclude round6/s5_safe_v15/pilot/seeds.jsonl --words-per-task 10 --name-prefix s2-v2 --output round6/s2_v15/batch_v2
     ```
     manifest 里记下 words、tasks、archives、excluded_by_screen、unscreened_words（应为 0 或很小）。
  3. 生成 Run：DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/s2_v15/flow`，尝试 1 次，并发 512（Task 更少时取 Task 数）。archives 里每份一个 Run。抽取前按份冻结 attempt 列表：
     ```bash
     .venv/bin/python round6/response_v14/freeze_attempts.py <run_no> --out round6/s2_v15/batch_v2/attempts.json --expected-samples <Task 数>
     .venv/bin/python round6/s2_v15/extract_s2.py <run_no> --out round6/s2_v15/batch_v2/extracted --attempts-json round6/s2_v15/batch_v2/attempts.json
     ```
     （如果拆成多份，run_no 和 attempts 文件按同样顺序依次给出。）
  4. 复核 Run：
     ```bash
     .venv/bin/python round6/judge_v3/make_pilot.py --source round6/s2_v15/batch_v2/extracted/examples.jsonl --words 0 --words-per-task 10 --name-prefix s2-v2-judge --output round6/s2_v15/batch_v2/judge
     ```
     luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`，`--flow round6/judge_v3/flow`，其余同第 3 步；冻结后抽取：
     ```bash
     .venv/bin/python round6/judge_v3/extract_judgments.py <run_no> --items round6/s2_v15/batch_v2/judge/items.jsonl --out round6/s2_v15/batch_v2/judge/extracted --attempts-json <对应的 attempts.json>
     ```
  5. 过滤：
     ```bash
     .venv/bin/python round6/verified_filter.py --examples round6/s2_v15/batch_v2/extracted/examples.jsonl --items round6/s2_v15/batch_v2/judge/items.jsonl --judgments round6/s2_v15/batch_v2/judge/extracted/judgments.jsonl --prompts round6/s2_v15/batch_v2/extracted/prompts.jsonl --out round6/s2_v15/batch_v2/verified
     ```
  6. 审计：从 `verified/verified.jsonl` 里按固定种子随机抽 30 个词，由用户或执行方读完整的提问、回答和思考，记下有问题的词数和类型（不写原文）。
  7. 正式放量的数据集**不删除**。
- 预期产物：可提交 `batch_v2/manifest.json`、run_no 文件、`extracted/summary.json`、`judge/manifest.json`、`judge/extracted/summary.json`、`verified/summary.json`。其余含样本原文的文件不提交。
- 验收：生成 `states.complete` ≥ 80%；复核失败（没拿到结论）≤ 3%；`verified/summary.json` 的 kept 不少于完整样本的 90%；审计 30 个词里有问题的不超过 3 个。
- 需要用户决定：无（放量规则已定：并发拉满、尝试 1 次）。规模：生成约 7,000 次 DeepSeek 请求，复核约 2.8 万条条目（按 Task 计约 700 个）。
- 反馈里必须报告：单测结果；三个 manifest 的主要数字；run_no；生成和复核的 summary（生成：`states`、`errors`、`aspects`、`by_form`；复核：`judged`、`failures`、`by_kind`、`agreement_with_old_labels`）；`verified/summary.json` 全文；审计结论；产物 SHA256。
