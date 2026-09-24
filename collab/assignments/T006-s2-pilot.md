# T006 S2 难负例 pilot（生成 + 另一个模型复核）

- 版本：v2（记录用户批准，内容其余未变）
- 目标：用 60 个词试跑 S2 生成（DeepSeek），再用 judge v3.1（luna）复核。看三件事：生成成功率和 skip 率、复核判为 safe 的比例、人工抽看的质量。设计说明见 [round6/s2_v15/README.md](../../round6/s2_v15/README.md)。
- 依赖：无。可以和 T009 并行，两者互不依赖。
- 输入：`round6/s2_v15/`；Mac 上的 `round6/response_v14/batch_rest/seeds.jsonl`（Run B 种子）；`round6/judge_v3/`（v3.1）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/s2_v15/test_s2_cpu.py`，6 项全过。
  2. 生成 pilot：`.venv/bin/python round6/s2_v15/make_pilot.py --words 60`，输出在 `round6/s2_v15/pilot/`。记下 manifest 里的 words、languages、forms、splits。
  3. **（需用户批准后再做）** 生成 Run：上传 `pilot/pilot.tar.gz`，模型 DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/s2_v15/flow`，尝试 1 次、重复 1 次，并发按用户批准的数值。提交前 `aster runs plan`，提交后下载平台快照与本地 `flow/` 逐字比对。run_no 存到 `pilot/run_no.txt`。
  4. 抽取：`.venv/bin/python round6/s2_v15/extract_s2.py <run_no> --out round6/s2_v15/pilot/extracted`。
  5. **（与第 3 步一起批准）** 复核 Run：
     ```bash
     .venv/bin/python round6/judge_v3/make_pilot.py --source round6/s2_v15/pilot/extracted/examples.jsonl --words 60 --output round6/s2_v15/pilot/judge
     ```
     上传 `pilot/judge/pilot.tar.gz`，模型 luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`，`--flow round6/judge_v3/flow`，其余同第 3 步。跑完后抽取：
     ```bash
     .venv/bin/python round6/judge_v3/extract_judgments.py <judge_run_no> --items round6/s2_v15/pilot/judge/items.jsonl --out round6/s2_v15/pilot/judge/extracted
     ```
  6. 人工抽看：由用户或执行方读 20 个词的完整三段文字（提问、回答、思考），每项写一句结论：内容是否正当、是否符合国内口径、思考过程是否真的复述了敏感请求、有没有拒答或说教过度。样本原文不写进反馈。
  7. 两个 Run 都抽取完之后，按 HANDOFF 第 5 节删除两个数据集，回执存为 `pilot/dataset_rm.json` 和 `pilot/judge/dataset_rm.json`。用户另有要求的以用户为准。
- 预期产物：可提交 `pilot/manifest.json`、`pilot/run_no.txt`、`pilot/extracted/summary.json`、`pilot/judge/manifest.json`、`pilot/judge/extracted/summary.json` 和两个删除回执；其余含样本原文的文件不提交。
- 验收：
  - 生成：`states.complete` 不少于 48 个词（80%），skip 和失败原因逐项列出；
  - 复核：`safe_whole` 和 `safe_mid_prefix` 判为 safe 的比例都不低于 95%；
  - 人工抽看：20 个词里不超过 2 个词有质量问题。
- 需要用户决定：无。**用户已批准（2026-09-24）：提交第 3 步（生成 Run）和第 5 步（复核 Run），并发拉满**，即每个 Run 的并发设为该 Run 的 Task 数。
- 反馈里必须报告：
  1. 单测结果；两个 manifest 的主要数字；
  2. 两个 run_no；`extracted/summary.json` 全文；
  3. 复核 `summary.json` 里的 `by_kind`、`agreement_with_old_labels` 和 `failures`；复核判为 unsafe 的条目按来源组和 `gbt_codes` 计数；
  4. 第 6 步的人工结论；
  5. DeepSeek 生成时被拦截或返回非 JSON 的词数，按来源组列出。
