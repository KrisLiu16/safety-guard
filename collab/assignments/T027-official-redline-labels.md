# T027 官方 thinking 集按红线口径标注，重看新头在官方集上的表现

- 版本：v1
- 背景：T018 v3.1 里，full 头在官方 Qwen3GuardTest thinking 集上：
  - stream AUC 0.32，低于 0.5；
  - 按官方标签算，误报 20.8–26.7%，召回 12–17%；
  - 现在的头（init）AUC 0.78，误报为 0。

  官方标签用的是另一套口径，不能直接和我们的口径比：
  - 官方的 safe 思考过程里，常有“用户在问 X，这是有害的，我应该拒绝”这样的复述。如果 X 属于红线，按我们的口径这类复述是“有争议”，本来就该截。
  - 官方的 unsafe 大多是红线以外的有害内容，按我们的口径应标 safe。

  所以这 21% 到底是真误报，还是口径差异，现在分不清。本任务用 redline_v1 的裁判（回答侧 v4.2，规则表与 T026 相同）给同样的 1,059 条官方 thinking 标上我们的标签，再在 T018 已有的逐位置分数上重算。**只用于观察，不参与训练和阈值拟合。**
- 依赖：T018 v3.1 ✓（Mac 上的 `round6/stage1_head/results_eval_v2/eval_*/`）。不用 L20，Aster 部分可以和 T026 同时排队，建议高优先级。
- 步骤：
  1. **单测**（Mac）：
     - `.venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py`，36 项全过；
     - `.venv/bin/python -m unittest round6/stage1_head/test_redline_stage1_cpu.py`，6 项全过。
  2. **导出**：在任意挂了 PVC 的 CPU Pod 里跑，不占 GPU，同 T024 的临时 Pod。脚本只用标准库。
     - 把 `round6/redline_v1/export_official.py` 拷到 `/work/round6/redline_v1/`，然后：
       ```bash
       python /work/round6/redline_v1/export_official.py --output /work/round6/redline_v1/input/official_thinking.jsonl
       ```
     - 取回 `official_thinking.jsonl` 和 `official_thinking.manifest.json`，放到 Mac 的 `round6/redline_v1/input/`。
     - 只提交 manifest。`records` 应为 thinking 集里 status 为 ready 的条数（不超过 1,059）。
  3. **标注**（同 T021 第 3–5 步）：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/redline_v1/input/official_thinking.jsonl --per-task 10 --name-prefix redline-official --output round6/redline_v1/full_official
     ```
     - DeepSeek，`--flow round6/redline_v1/flow`，尝试 1 次，高优先级；
     - 用 `extract_probes.py` 抽取到 `full_official/extracted_ds/`；
     - `failed_ids.txt` 交给 luna 兜底：`make_tasks.py --sample-ids ...`，输出到 `full_official_fb`；
     - 两个数据集抽取完即删除。
  4. **定标签**：
     ```bash
     .venv/bin/python round6/redline_v1/apply_policy.py round6/redline_v1/full_official/extracted_ds/probes.jsonl round6/redline_v1/full_official_fb/extracted/probes.jsonl --source round6/redline_v1/input/official_thinking.jsonl --out round6/redline_v1/labels/official_thinking
     ```
     T026 的 `political_terms.jsonl` 如果已经就位，再加 `--political-terms` 另出一份，输出到 `labels_political_v1/official_thinking`。
  5. **人工核对**（本地，不提交）：
     ```bash
     .venv/bin/python round6/redline_v1/review_sample.py --labels round6/redline_v1/labels/official_thinking/labels.jsonl --source round6/redline_v1/input/official_thinking.jsonl --count 30 --out round6/redline_v1/labels/official_thinking/review
     ```
     问题和 T020 相同。另外，对“官方 safe → 我们 controversial/unsafe”的条目，说明它们是否确实在复述或写出红线内容。
  6. **重算 T018 的官方集**（Mac CPU）：
     ```bash
     .venv/bin/python round6/stage1_head/analyze_redline.py --eval-dir round6/stage1_head/results_eval_v2 --runA-labels round6/redline_v1/labels/runA_stage1/labels.jsonl round6/redline_v1/labels/runA_dev/labels.jsonl --prefix-labels round6/redline_v1/labels/prefix_v2/labels.jsonl --official-labels round6/redline_v1/labels/official_thinking/labels.jsonl --output round6/stage1_head/results_eval_v2/compare_redline_official.json
     ```
     Run A 和 prefix_v2 的数字应与 T018 的 `compare_redline_v1.json` 完全相同，只有官方集一列改为按我们的标签算。输出末尾会逐个头打印“official by our labels”这一行，给出（官方标签, 我们的等级）每一格的截断比例。
- 预期产物：可提交的有：
  - `input/official_thinking.manifest.json`；
  - `full_official*/manifest.json`、`run_no.txt`、`extracted*/summary.json`、`dataset_rm.json`；
  - `labels/official_thinking/summary.json`；
  - `compare_redline_official.json`。

  `official_thinking.jsonl`、`labels.jsonl`、`review/` 含原文，不提交。
- 验收：
  - 加上 luna 兜底后 `usable` ≥ 98%；
  - 第 6 步里 Run A 和 prefix_v2 的数字与 T018 相同。
- 需要用户决定：无。只做观察，不训练。
- 反馈里必须报告：
  - 单测；
  - manifest 的 `records`、`by_label_language`、`unsafe_type`；
  - run_no；
  - 各 `extracted*/summary.json` 的 statuses 和 `judge_calls_per_response`；
  - labels 的 `labels_by_split`、`old_to_new`（这里的 old 是官方标签）、`stratum_by_label`、`rules_non_safe`；
  - 人工核对的计数和理由（不写原文）；
  - `analyze_redline.py` 打印的全部内容；
  - 产物 SHA256。
