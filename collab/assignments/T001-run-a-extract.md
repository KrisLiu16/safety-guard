# T001 Run A 抽取与本地复校

- 版本：v2（v1 因 attempt 列表被截断在 100 条而阻塞；v2 先按 sample 逐个冻结 attempt，再用 `--attempts-json` 抽取，并补上训练过滤脚本）
- 目标：把 Run A（aster-dev-276，v14 格式，5 万词，luna）的全部产物取回 Mac，用冻结的 v14 校验重新校验，得到带 split 的可训练集。
- 依赖：无（Run A 已结束，352/352 成功）。
- 输入：Run `aster-dev-276`；`round6/response_v14/batch_50k/seeds.jsonl`（Mac 上，含 split 和 family）；代码 `round6/response_v14/` 下的 `freeze_attempts.py`、`extract_archives.py`、`make_trainable.py`。`flow/pipeline.py` 仍是冻结版本，这次没有改动。
- 步骤：
  1. `aster whoami`，确认已登录。
  2. 单测：`.venv/bin/python -m unittest round6/response_v14/test_extract_tools_cpu.py round6/response_v14/test_pipeline.py`，应为 15 项全过（其中 1 项跳过）。
  3. 冻结 attempt 列表（不调用模型）：
     ```bash
     .venv/bin/python round6/response_v14/freeze_attempts.py aster-dev-276 --out round6/response_v14/batch_50k/extracted/attempts.json --expected-samples 352
     ```
  4. 抽取：
     ```bash
     .venv/bin/python round6/response_v14/extract_archives.py aster-dev-276 --out round6/response_v14/batch_50k/extracted --expected-terms 50000 --attempts-json round6/response_v14/batch_50k/extracted/attempts.json
     ```
  5. 训练过滤（只保留 4/4 完整的词，剔除要求先铺垫但起点为 0 的回答，并挂上 split 和 family）：
     ```bash
     .venv/bin/python round6/response_v14/make_trainable.py --extracted round6/response_v14/batch_50k/extracted --seeds round6/response_v14/batch_50k/seeds.jsonl
     ```
- 预期产物（留在 Mac，不提交仓库）：`round6/response_v14/batch_50k/extracted/` 下的 `attempts.json`、`examples.jsonl`、`words.jsonl`、`summary.json`、`trainable.jsonl`、`trainable_summary.json`。
- 验收：`attempts.json` 有 352 条且 `truncated=false`；`summary.json` 中 `word_artifacts` 为 50000、`missing_word_artifacts` 为 0；所有归档 SHA 校验通过；`trainable_summary.json` 生成。
- 需要用户决定：无（冻结和抽取都不消耗模型调用）。
- 反馈里必须报告：
  1. 单测结果；冻结脚本输出的 samples 和 attempts 数；
  2. `summary.json` 全文（只有统计数字，可以直接贴）；
  3. `trainable_summary.json` 全文；
  4. 错误类型前 10 名及计数；
  5. `quality_flags` 中 `harm_commentary` 和 `fiction_marker` 各有多少；
  6. `examples.jsonl`、`trainable.jsonl` 的 SHA256。
