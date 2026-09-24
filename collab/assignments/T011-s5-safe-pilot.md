# T011 S5 安全长思考 pilot（生成 + 另一个模型复核）

- 版本：v1
- 目标：用 40 个词试跑 S5 安全的一半（每个词两段 500–1,500 token 的安全思考），生成用 DeepSeek，复核用 luna 上的 judge v3.1。看生成成功率、长度是否达标、复核判为 safe 的比例，以及人工抽看的质量。设计说明见 [round6/s5_safe_v15/README.md](../../round6/s5_safe_v15/README.md)。
- 依赖：无，可以和 T006、T009 并行。如果 T006 已经生成了 `round6/s2_v15/pilot/seeds.jsonl`，make_pilot 会自动排除这些词；没有也不影响。
- 输入：`round6/s5_safe_v15/`；Mac 上的 `round6/response_v14/batch_rest/seeds.jsonl`（Run B 种子）；`round6/judge_v3/`（v3.1）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/s5_safe_v15/test_s5_safe_cpu.py`，6 项全过（和其他目录的测试分开跑，各目录都有同名模块）。
  2. 生成 pilot：`.venv/bin/python round6/s5_safe_v15/make_pilot.py --words 40`，输出在 `round6/s5_safe_v15/pilot/`。记下 manifest 里的 words、unsafe_prompt_language、shapes、splits、excluded_task_keys。
  3. 生成 Run：上传 `pilot/pilot.tar.gz`，模型 DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/s5_safe_v15/flow`，尝试 1 次、重复 1 次，并发拉满（等于 Task 数）。提交前 `aster runs plan`，提交后下载平台快照与本地 `flow/` 逐字比对。run_no 存到 `pilot/run_no.txt`。
  4. 抽取：`.venv/bin/python round6/s5_safe_v15/extract_s5.py <run_no> --out round6/s5_safe_v15/pilot/extracted`。
  5. 复核 Run：
     ```bash
     .venv/bin/python round6/judge_v3/make_pilot.py --source round6/s5_safe_v15/pilot/extracted/examples.jsonl --words 40 --output round6/s5_safe_v15/pilot/judge
     ```
     上传 `pilot/judge/pilot.tar.gz`，模型 luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`，`--flow round6/judge_v3/flow`，其余同第 3 步。跑完后抽取：
     ```bash
     .venv/bin/python round6/judge_v3/extract_judgments.py <judge_run_no> --items round6/s5_safe_v15/pilot/judge/items.jsonl --out round6/s5_safe_v15/pilot/judge/extracted
     ```
  6. 人工抽看：由用户或执行方读 10 个词的两段思考，每段写一句结论：是否真的复述了请求、有没有写出有害细节或先写后否定、是否符合国内口径、长度是否靠重复凑数。样本原文不写进反馈。
  7. 两个 Run 都抽取完之后删除两个数据集，回执存为 `pilot/dataset_rm.json` 和 `pilot/judge/dataset_rm.json`。
- 预期产物：可提交 `pilot/manifest.json`、`pilot/run_no.txt`、`pilot/extracted/summary.json`、`pilot/judge/manifest.json`、`pilot/judge/extracted/summary.json` 和两个删除回执；其余含样本原文的文件不提交。
- 验收：
  - 生成：`states.complete` 不少于 32 个词（80%）；`length_units_by_language_style` 的 p50 落在长度区间中部；
  - 复核：`risk_reasoning_long` 的整段和前缀判为 safe 的比例不低于 95%（这一类最容易夹带有害细节，是重点）；
  - 人工抽看：10 个词里不超过 1 个词有质量问题。
- 需要用户决定：无。用户 2026-09-24 要求推进 S5 安全的一半；并发按已定规则拉满，尝试 1 次。
- 反馈里必须报告：
  1. 单测结果；两个 manifest 的主要数字；
  2. 两个 run_no；`extracted/summary.json` 全文；
  3. 复核 `summary.json` 里的 `by_kind`、`failures`，以及判为 unsafe 的条目按 `response_style`（通过 sample_id 对应）和 `gbt_codes` 计数；
  4. 第 6 步的人工结论；
  5. DeepSeek 生成时被拦截、截断或返回非 JSON 的词数，按来源组列出。
