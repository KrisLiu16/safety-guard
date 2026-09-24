# T001 Run A 抽取与本地复校

- 目标：把 Run A（aster-dev-276，v14 格式，5 万词，luna）的全部产物取回 Mac，用冻结的 v14 校验重新校验，得到可训练的词清单。
- 依赖：Run A 全部 352 个 Task 结束（成功或失败都算结束）。没结束就只回报当前状态，不抽取。
- 输入：Run `aster-dev-276`；本地代码 `round6/response_v14/extract_archives.py` 和 `round6/response_v14/flow/pipeline.py`（冻结版本，不改）。
- 步骤：
  1. `aster whoami`，确认已登录。
  2. 查 Run 状态：成功、失败、仍在运行的 Task 数。
  3. 运行抽取：
     ```bash
     .venv/bin/python round6/response_v14/extract_archives.py aster-dev-276 --out round6/response_v14/batch_50k/extracted --expected-terms 50000
     ```
  4. 按 [BATCH_PLAN.md](../../round6/response_v14/BATCH_PLAN.md) 第 28 行附近的规则筛选：只保留 4 条回答全部合格的词；剔除“要求先铺垫（onset_style=delayed）但起点为 0”的回答，剔除后该词按不完整处理。
- 预期产物（留在 Mac，不提交仓库）：`round6/response_v14/batch_50k/extracted/` 下的 `examples.jsonl`、`words.jsonl`、`summary.json`、`selected_attempts.json`。
- 验收：`summary.json` 生成；所有归档 SHA 校验通过；下载失败的归档数为 0，或者在反馈里逐个列出。
- 需要用户决定：无（抽取不消耗模型调用）。
- 反馈里必须报告：
  1. Run 状态：Task 成功 / 失败 / 运行中各多少；
  2. `summary.json` 全文（只有统计数字，可以直接贴）；
  3. 4/4 完整的词数，按 split（train / dev / calibration）分别列出；
  4. 按规则 4 剔除的回答数和因此变为不完整的词数；
  5. 错误类型前 10 名及计数（例如 `onset_quote`、`negated_onset`、`finish_reason:length`）；
  6. `quality_flags` 中 `harm_commentary` 和 `fiction_marker` 各有多少；
  7. 产物文件的 SHA256。
