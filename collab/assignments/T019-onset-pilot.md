# T019 起点重定 pilot：200 条 Run A dev 不安全回答

- 版本：v1
- 目标：用 judge v3.2 在分句边界上二分，找出每条不安全回答里“第一个写完就变成 unsafe 的分句”，与 v14 的起点对比，量出 v14 起点偏差的大小和方向。设计见 [round6/onset_v1/README.md](../../round6/onset_v1/README.md)。
- 依赖：T001（`trainable.jsonl`）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/onset_v1/test_onset_cpu.py`，6 项全过（包括裁判文件与 judge_v3 逐字节一致）。
  2. 生成任务：`.venv/bin/python round6/onset_v1/make_onset_tasks.py --splits dev --count 200 --per-task 10`，输出 `round6/onset_v1/pilot/`。manifest 应为 200 条、20 个 Task，顺从和跑偏各半，中英各半。
  3. Run：DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/onset_v1/flow`，尝试 1 次，并发拉满。提交前 `aster runs plan`，提交后快照比对。
  4. 抽取：`.venv/bin/python round6/onset_v1/extract_onsets.py <run_no> --items round6/onset_v1/pilot/items.jsonl --out round6/onset_v1/pilot/extracted`（20 个 Task，不需要冻结 attempt 列表）。
  5. 人工：在 `located` 的结果里按固定种子抽 20 条（`v14_late`、`v14_early`、`overlap` 都要有），由用户或执行方判断“裁判定位的起点”和“v14 起点”哪个更准，写一句理由（不写原文）。
  6. 抽取完成后删除 pilot 数据集（已定规则）。
- 预期产物：可提交 `pilot/manifest.json`、`pilot/run_no.txt`、`pilot/extracted/summary.json`、`pilot/dataset_rm.json`；`items.jsonl`、`onsets.jsonl`、归档不提交。
- 验收：`statuses.located` ≥ 85%；`nonmonotonic` ≤ 5%；`judge_error` 预计集中在政治类，列出来源组。
- 需要用户决定：无（pilot 规则已定）。规模：约 200 × 7 ≈ 1,400 次 DeepSeek 请求。
- 反馈里必须报告：单测结果；manifest；run_no；`summary.json` 全文；`judge_error` 和 `whole_not_unsafe` 按来源组计数；第 5 步的人工结论（两种起点各判对几条）；产物 SHA256。
