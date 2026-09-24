# T016 R 切片：Run A 的 dev 和 calibration 按国内口径重新判定

- 版本：v1
- 目标：用 judge v3.2 重新判定 Run A 的 dev 和 calibration 全部回答（整段和前缀探针），DeepSeek 为主、luna 兜底，再与词表预筛合并，得到国内口径的标签。设计说明见 [round6/r_slice/README.md](../../round6/r_slice/README.md)。
- 依赖：第 1–4 步无依赖，可以立即开始；第 5 步的最终合并要等 T012 全量预筛完成。
- 输入：Mac 上的 `round6/response_v14/batch_50k/extracted/trainable.jsonl`；`round6/judge_v3/`（v3.2）；`round6/r_slice/`。
- 步骤：
  1. 单测：分别运行 `.venv/bin/python -m unittest round6/judge_v3/test_judge_v3_cpu.py`（14 项）和 `.venv/bin/python -m unittest round6/r_slice/test_merge_cpu.py`（2 项），全部通过。
  2. 生成裁判任务：
     ```bash
     .venv/bin/python round6/judge_v3/make_pilot.py --source round6/response_v14/batch_50k/extracted/trainable.jsonl --splits dev,calibration --words 0 --name-prefix r-devcal --output round6/r_slice/devcal
     ```
     manifest 的 words 应为 2,561（dev 1,297 + calibration 1,264），items 约 2.4 万，每个词一个 Task。
  3. 主裁判 Run：DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/judge_v3/flow`，尝试 1 次，并发拉满。提交前 `aster runs plan`，提交后快照比对。Task 超过 100 个，抽取前先冻结 attempt 列表：
     ```bash
     .venv/bin/python round6/response_v14/freeze_attempts.py <run_no> --out round6/r_slice/devcal/attempts.json --expected-samples <Task 数>
     .venv/bin/python round6/judge_v3/extract_judgments.py <run_no> --items round6/r_slice/devcal/items.jsonl --out round6/r_slice/devcal/extracted --attempts-json round6/r_slice/devcal/attempts.json
     ```
  4. 兜底 Run：
     ```bash
     .venv/bin/python round6/judge_v3/make_fallback.py --items round6/r_slice/devcal/items.jsonl --judgments round6/r_slice/devcal/extracted/judgments.jsonl --controls 200 --output round6/r_slice/devcal_fallback
     ```
     luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`，`--flow round6/judge_v3/flow`，其余同第 3 步。抽取时加 `--reference-judgments round6/r_slice/devcal/extracted/judgments.jsonl`（Task 超过 100 个同样先冻结），这样 `cross_judge` 直接给出 luna 与 DeepSeek 的一致率。
  5. 合并（T012 全量完成之后）：
     ```bash
     .venv/bin/python round6/r_slice/merge_labels.py --trainable round6/response_v14/batch_50k/extracted/trainable.jsonl --items round6/r_slice/devcal/items.jsonl --primary round6/r_slice/devcal/extracted/judgments.jsonl --fallback round6/r_slice/devcal_fallback/extracted/judgments.jsonl --screen round6/word_screen_v1/full/extracted/screen.jsonl --out round6/r_slice/devcal/merged
     ```
     如果 T012 全量还没完成，先不带 `--screen` 跑一遍并报告，全量完成后再带上重跑。
  6. 人工：`disputed` 的回答全部由用户或执行方核对（预计几十条，主要是政治类），每条记人工结论和一句理由，不写原文。结论存在本地文件里，下一版合并会读取。
- 预期产物：可提交 `devcal/manifest.json`、`devcal/run_no.txt`、`devcal/extracted/summary.json`、`devcal_fallback/manifest.json`、`devcal_fallback/run_no.txt`、`devcal_fallback/extracted/summary.json`、`devcal/merged/summary.json`。其余含样本原文或账号信息的文件不提交。两个 Run 是正式数据，数据集**不删除**。
- 验收：主裁判的失败率不超过 6%（预计集中在政治类）；兜底之后 `missing` 不超过 1%；`cross_judge.control` 一致率不低于 0.9。
- 需要用户决定：无（放量规则已定）。规模：主裁判约 2.4 万次 DeepSeek 请求，兜底约 3 千次 luna 请求。
- 反馈里必须报告：
  1. 单测结果；manifest 的 words、items、kinds、tasks；
  2. 两个 run_no；两个 `extracted/summary.json` 里的 `judged`、`failures`、`by_kind`、`agreement_with_old_labels`，以及兜底 Run 的 `cross_judge` 全文；
  3. 合并 `summary.json` 全文（带和不带 `--screen` 各一份，如果都跑了）；
  4. 第 6 步：disputed 条数和人工结论的分布；
  5. 产物 SHA256。
