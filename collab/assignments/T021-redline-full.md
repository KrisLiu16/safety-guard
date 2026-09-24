# T021 红线标注放量：第一阶段训练和评测要用的全部回答

- 版本：v1
- 目标：用 T020 验证过的流程，给第一阶段训练和评测要用的回答都打上红线标签：
  - Run A 第一阶段输入（`round6/stage1_head/input_v1/stage1_runA_v1.jsonl`：train 8,000 词 + 全部 calibration，37,056 条），T017 缓存用的就是它；
  - Run A dev（`trainable.jsonl` 的 dev，约 5,200 条）；
  - prefix_v2 助手侧的 train、calibration、dev（T020 第 2 步导出的文件）。
- 依赖：T020 验收通过；数据平台恢复。
- 步骤：
  1. 建任务，每个 Task 20 条回答：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/stage1_head/input_v1/stage1_runA_v1.jsonl --per-task 20 --name-prefix redline-runA-stage1 --output round6/redline_v1/full_runA_stage1
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/response_v14/batch_50k/extracted/trainable.jsonl --splits dev --per-task 20 --name-prefix redline-runA-dev --output round6/redline_v1/full_runA_dev
     .venv/bin/python round6/redline_v1/make_tasks.py --source round6/redline_v1/input/prefix_v2_assistant.jsonl --per-task 20 --name-prefix redline-pv2 --output round6/redline_v1/full_pv2
     ```
     超过 2,000 个 Task 的会拆成 `pilot_partN.tar.gz`，每个 part 一个 Run。
  2. Run：DeepSeek，`--flow round6/redline_v1/flow`，尝试 1 次，并发拉满。提交前 `aster runs plan`，提交后快照比对。
  3. 抽取：超过 100 个 Task 的 Run 先冻结 attempt 列表（`response_v14/freeze_attempts.py`），再 `extract_probes.py <run_no...> --attempts-json ...`，输出到各目录的 `extracted_ds/`。
  4. 兜底：各自的 `failed_ids.txt` 交给 luna（同 T020 第 6 步），抽取到 `*_fb/extracted/`。
  5. 定标签：
     ```bash
     .venv/bin/python round6/redline_v1/apply_policy.py round6/redline_v1/full_runA_stage1/extracted_ds/probes.jsonl round6/redline_v1/full_runA_stage1_fb/extracted/probes.jsonl --out round6/redline_v1/labels/runA_stage1
     ```
     另外两份输出到 `labels/runA_dev`、`labels/prefix_v2`。
  6. 把 `labels/runA_stage1/labels.jsonl` 和 `labels/prefix_v2/labels.jsonl` 拷到 PVC 的 `/work/round6/redline_v1/labels/` 下同名目录，用 `sha256sum` 核对。T018 v3 从这里读。
  7. 抽取完成后删除数据集（已定规则）。
- 预期产物：可提交各目录的 `manifest.json`、`run_no.txt`、`extracted*/summary.json`、`labels/*/summary.json`、`dataset_rm.json`；其余不提交。
- 验收：每份标签的 `usable` ≥ 97%；`nonmonotonic` ≤ 5%；Run A 的 `old_to_new_by_slot_style` 和 T020 pilot 的比例相近（差异大就停下报告）。
- 需要用户决定：**放量规模**。T020 会量出每条回答的平均调用次数；按 pilot 前的估算（约 30% 的回答不是 safe、每条约 12 次调用），约 6 万条回答需要 DeepSeek 约 25 万次调用。T020 反馈里按实测数字重估，用户确认后开始。
- 反馈里必须报告：三份 manifest；所有 run_no；每份 `extracted*/summary.json` 的 statuses 和 `judge_calls_per_response`；每份 `labels/*/summary.json` 的 `usable`、`labels_by_split`、`old_to_new`、`stratum_by_label`；第 6 步的 SHA256。
