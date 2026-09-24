# T020 红线标注 pilot：裁判 v4 + 规则表 + 两级起点

- 版本：v1
- 目标：按定稿口径（[POLICY-redline-scope.md](../feedback/POLICY-redline-scope.md)）先小规模验证新的标注流程：裁判 v4 只记事实，规则表定等级，两级二分找“有争议”和“风险”的起点。量出失败率、调用次数、两个裁判的一致率，并人工核对标签和起点。设计见 [round6/redline_v1/README.md](../../round6/redline_v1/README.md)。本任务取代 T016（R 切片）和 T019（起点 pilot）。
- 依赖：T001（`trainable.jsonl`）。**数据平台恢复后再做第 3 步起的 Aster 部分**；第 1、2 步现在就可以做。
- 步骤：
  1. 单测：
     ```bash
     .venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py round6/stage1_head/test_redline_stage1_cpu.py round6/stage1_head/test_stage1_cpu.py
     ```
     应为 21 + 4 + 6 = 31 项全过（`test_stage1_cpu.py` 里有 4 项需要 torch，Mac 的 .venv 有）。
  2. 导出 prefix_v2 助手侧记录。把 `round6/redline_v1/export_prefix_v2.py` 拷到 Pod 的 `/work/round6/redline_v1/`，在任何挂载了 PVC 的 Pod 里跑（T017 的 Pod 可以用，脚本只用标准库，几秒钟）：
     ```bash
     python /work/round6/redline_v1/export_prefix_v2.py --output /work/round6/redline_v1/input/prefix_v2_assistant.jsonl
     ```
     取回到 Mac 的 `round6/redline_v1/input/`。提交 `prefix_v2_assistant.manifest.json`（只有计数和 SHA），`.jsonl` 不提交。
  3. 建 pilot 任务（两个数据集，所以是两个 Run）：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/response_v14/batch_50k/extracted/trainable.jsonl --splits dev --count 300 --name-prefix redline-pilot-runA --output round6/redline_v1/pilot_runA
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/redline_v1/input/prefix_v2_assistant.jsonl --splits dev --count 100 --name-prefix redline-pilot-pv2 --output round6/redline_v1/pilot_pv2
     ```
     manifest 应分别为 300 条 / 30 个 Task、100 条 / 10 个 Task；Run A 的 `by_slot_language` 应大致均衡。
  4. Run：DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`，`--flow round6/redline_v1/flow`，尝试 1 次，并发拉满。提交前 `aster runs plan`，提交后快照比对。
  5. 抽取：
     ```bash
     .venv/bin/python round6/redline_v1/extract_probes.py <run_no> --items round6/redline_v1/pilot_runA/items.jsonl --out round6/redline_v1/pilot_runA/extracted_ds
     ```
     prefix_v2 同样处理，输出到 `pilot_pv2/extracted_ds`。
  6. 兜底：用 `failed_ids.txt` 建任务，交给 luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（Responses 接口），同样尝试 1 次：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --source <同上> --sample-ids round6/redline_v1/pilot_runA/extracted_ds/failed_ids.txt --name-prefix redline-pilot-runA-fb --output round6/redline_v1/pilot_runA_fb
     ```
     没有失败条目就跳过。抽取到 `pilot_runA_fb/extracted`。
  7. 第二裁判对照：Run A 里取前 100 条（与第 3 步抽中的前 100 条相同），交给 luna，抽取到 `pilot_runA_cross/extracted`：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/response_v14/batch_50k/extracted/trainable.jsonl --splits dev --count 100 --name-prefix redline-pilot-runA-x --output round6/redline_v1/pilot_runA_cross
     ```
  8. 定标签和对比：
     ```bash
     .venv/bin/python round6/redline_v1/apply_policy.py round6/redline_v1/pilot_runA/extracted_ds/probes.jsonl round6/redline_v1/pilot_runA_fb/extracted/probes.jsonl --out round6/redline_v1/pilot_runA/labels
     .venv/bin/python round6/redline_v1/apply_policy.py round6/redline_v1/pilot_runA/extracted_ds/probes.jsonl --out round6/redline_v1/pilot_runA/labels_ds_only
     .venv/bin/python round6/redline_v1/apply_policy.py round6/redline_v1/pilot_runA_cross/extracted/probes.jsonl --out round6/redline_v1/pilot_runA_cross/labels
     .venv/bin/python round6/redline_v1/compare_judges.py round6/redline_v1/pilot_runA/labels_ds_only/labels.jsonl round6/redline_v1/pilot_runA_cross/labels/labels.jsonl --out round6/redline_v1/pilot_runA_cross/compare.json
     ```
     prefix_v2 只做第一条（合并兜底）。
  9. 人工核对：
     ```bash
     .venv/bin/python round6/redline_v1/review_sample.py --labels round6/redline_v1/pilot_runA/labels/labels.jsonl --source round6/response_v14/batch_50k/extracted/trainable.jsonl --count 40 --out round6/redline_v1/pilot_runA/review
     .venv/bin/python round6/redline_v1/review_sample.py --labels round6/redline_v1/pilot_pv2/labels/labels.jsonl --source round6/redline_v1/input/prefix_v2_assistant.jsonl --count 20 --out round6/redline_v1/pilot_pv2/review
     ```
     由用户或执行方逐条回答两个问题：标签对不对（对 / 漏判红线 / 误判红线 / 档位错），起点分句对不对（对 / 偏早 / 偏晚 / 不适用）。判断依据是 POLICY 文件。反馈里只写计数和不写原文的一句话理由。
  10. 抽取完成后删除 pilot 数据集（已定规则）。
- 预期产物：可提交各目录的 `manifest.json`、`run_no.txt`、`extracted*/summary.json`、`labels*/summary.json`、`compare.json`、`dataset_rm.json`，以及 `prefix_v2_assistant.manifest.json`。`items.jsonl`、`probes.jsonl`、`labels.jsonl`、归档、`review/` 都不提交（前几项已被 .gitignore 覆盖）。
- 验收：
  - 单测全过；
  - DeepSeek 一轮后 `safe + excluded + located` ≥ 90%，加上 luna 兜底后 ≥ 98%；`nonmonotonic` ≤ 5%；
  - 两个裁判的截断一致率（`cut_agreement`，即“放行 / 截断”是否一致）≥ 0.90；
  - 人工核对：标签对 ≥ 90%；`normal` 分层里被判成非 safe 的必须逐条列出（口径要求正常内容绝不能判错）。
- 需要用户决定：无。规模：约 400 条回答；按每条非 safe 回答约 12 次调用估算，DeepSeek 约 2,500 次，luna 约 600 次。
- 反馈里必须报告：单测结果；两个 manifest；run_no；各 `extracted*/summary.json` 的 statuses 和 `judge_calls_per_response`；`labels/summary.json` 的 `labels_by_split`、`old_to_new`、`old_to_new_by_slot_style`、`stratum_by_label`、`rules_non_safe`、`onset_clause_chars`；`compare.json` 全文；人工核对的计数；产物 SHA256。
