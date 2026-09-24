# T009 judge v3.1 第二裁判 pilot（DeepSeek 与 luna 同一提示词对比）

- 版本：v3（修订：用户已同意删除数据集。只补做第 5 步，其余步骤不用重做）
- 目标：在 T002 pilot 的子集上，用同一份 v3.1 提示词分别让 DeepSeek 和 luna 判一遍，量出三件事：token 上限提到 8000 后 DeepSeek 的截断是否消失；luna 能否补上 DeepSeek 在政治类上不回答的条目；两个裁判在政治类和对照样本上的一致率。第三件决定政治类能否由 luna 兜底：Run A 的政治类样本也是 luna 生成的，一致率低就要人工复核或换第三个模型。
- 依赖：T002 完成；用户已定侮辱性称呼按严格口径判（已写入 v3.1 提示词）。
- 输入：`round6/judge_v3/`（v3.1 代码）；Mac 上的 `round6/judge_v3/pilot/items.jsonl` 和 `round6/judge_v3/pilot/extracted/judgments.jsonl`（T002 产物）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/judge_v3/test_judge_v3_cpu.py`，14 项全过。
  2. 生成子集：`.venv/bin/python round6/judge_v3/make_fallback.py`，输出在 `round6/judge_v3/pilot_fallback/`。记下 `manifest.json` 里的 items、tasks 和 `fallback_reasons`（failed / political / control 各多少条）。
  3. **（需用户批准后再做）** 上传一个数据集，对同一个数据集提交两个 Run，`--flow round6/judge_v3/flow`，尝试 1 次、重复 1 次，并发按用户批准的数值：
     - Run 1：DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`（chat_completions）；
     - Run 2：luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（responses）。

     每个 Run 提交前先 `aster runs plan`，提交后下载平台快照并与本地 `flow/` 逐字比对。run_no 分别存到 `pilot_fallback/run_no_deepseek.txt` 和 `pilot_fallback/run_no_luna.txt`。
  4. 抽取（三次）：
     ```bash
     .venv/bin/python round6/judge_v3/extract_judgments.py <deepseek_run> --items round6/judge_v3/pilot_fallback/items.jsonl --out round6/judge_v3/pilot_fallback/extracted_deepseek
     .venv/bin/python round6/judge_v3/extract_judgments.py <luna_run> --items round6/judge_v3/pilot_fallback/items.jsonl --out round6/judge_v3/pilot_fallback/extracted_luna
     .venv/bin/python round6/judge_v3/extract_judgments.py <luna_run> --items round6/judge_v3/pilot_fallback/items.jsonl --out round6/judge_v3/pilot_fallback/extracted_luna_vs_deepseek --reference-judgments round6/judge_v3/pilot_fallback/extracted_deepseek/judgments.jsonl
     ```
     前两次的 `cross_judge` 是和 T002 的 v3 DeepSeek 结果比较；第三次是 luna 与 v3.1 DeepSeek 直接比较，这个是主要结果。
  5. 删除这个数据集 `ds_01M390R8C2CTDTYJS8EGW8CFSB`（`aster datasets rm`），回执存为 `pilot_fallback/dataset_rm.json`。用户已同意（2026-09-24）。
- 预期产物：可提交 `pilot_fallback/manifest.json`、两个 run_no、三个 `extracted_*/summary.json`、`dataset_rm.json`；`items.jsonl`、`tasks/`、归档和 `judgments.jsonl` 不提交。
- 验收：
  - DeepSeek v3.1：T002 中因 `stop_reason:length` 失败的条目都有了结果；
  - luna：有结果的条目不少于 95%；
  - 第三次抽取的 `cross_judge.control` 一致率不低于 0.9（两个裁判在普通样本上基本一致，这时政治类的一致率才有参考意义）。
- 需要用户决定：无。**用户已批准（2026-09-24）：提交第 3 步（两个 Run），并发拉满**，即每个 Run 的并发设为该 Run 的 Task 数。
- 反馈里必须报告：
  1. 单测结果；manifest 的 items、tasks、fallback_reasons；
  2. 两个 run_no；三个 `summary.json` 里的 `failures`、`judged` 和 `cross_judge` 全文；
  3. DeepSeek v3.1 在政治类上是否仍然返回非 JSON，条数和来源组；
  4. luna 与 DeepSeek v3.1 在政治类上判法不同的条目（最多 20 条），由用户或执行方人工看过后写一句结论：哪个裁判更符合国内口径，是否有系统性偏差（例如 luna 对自己生成的内容偏宽）。样本原文不要写进反馈。
