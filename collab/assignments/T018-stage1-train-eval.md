# T018 第一阶段：按红线口径改标签，训练读出头并评测

- 版本：v3（取代 v2：口径改为“只审红线”，训练目标改为三类）
- 目标：复用 T017 缓存的特征，按红线标注结果重写每个位置的三类目标（safe / controversial / unsafe），训练两个读出头变体（`risk`、`full`），然后和现在的头（`init`）在同一次评测里对比。设计见 [round6/stage1_head/README.md](../../round6/stage1_head/README.md) 的 v2 一节。
- 依赖：T017 完成；T021 完成（`/work/round6/redline_v1/labels/runA_stage1/` 和 `labels/prefix_v2/` 就位）。训练已获用户批准（2026-09-24）；用户定稿口径后说“先训练，后生成数据”。
- 步骤：
  1. 提交作业：`kubectl apply -f round6/stage1_head/worker_redline_v2.yaml`。Pod 进入 Running 后拷入并用 `sha256sum` 核对：
     - `round6/stage1_head/*.py` → `/work/round6/stage1_head/`；
     - `round6/probe/*.py` → `/work/round6/probe/`；
     - `round6/redline_v1/flow/levels.py` → `/work/round6/redline_v1/flow/levels.py`。

     再 `touch /work/round6/stage1_redline_v2_input_ready`。作业依次：改标签（CPU，几分钟）→ 训练（几分钟）→ 三个头的评测（约 105 分钟）。
  2. 改标签完成后查看 `/work/output/round6/stage1_labels3_v2/report.json`：每个来源的 `kept` 和各项 `dropped_*`。`dropped_offsets` 应为 0，不为 0 说明重新分词和缓存时不一致，停下报告。
  3. 训练完成后查看 `/work/output/round6/stage1_train_v2/train_report.json`：`version` 为 `round6-stage1-head-v2-redline`，`calibration_score` 为 `cut`，`sources_left_out` 只应有 `s2`、`s5`；两个变体都有 `chosen_epoch`。
  4. 完成后 `stage1_redline_v2_exit_code.txt` 应为 `relabel=0 train=0 eval=0`。取回到 Mac：
     - `stage1_labels3_v2/report.json` → `round6/stage1_head/results_labels3_v2/`；
     - `stage1_train_v2/train_report.json` 和 `head_*.pt` → `round6/stage1_head/results_train_v2/`；
     - `stage1_eval_v2/report.json` 和三个 `eval_*/` 目录 → `round6/stage1_head/results_eval_v2/`。

     核对 SHA 后 `touch /work/output/round6/stage1_redline_v2_collected`。
  5. 对比（Mac CPU）：
     ```bash
     .venv/bin/python round6/stage1_head/analyze_redline.py --eval-dir round6/stage1_head/results_eval_v2 --runA-labels round6/redline_v1/labels/runA_stage1/labels.jsonl round6/redline_v1/labels/runA_dev/labels.jsonl --prefix-labels round6/redline_v1/labels/prefix_v2/labels.jsonl --output round6/stage1_head/results_eval_v2/compare_redline_v1.json
     ```
- 预期产物：提交三个 report、`train_report.json` 和 `compare_redline_v1.json`；`head_*.pt`、`*.jsonl.gz`、`*.npz` 留在 Mac 和 PVC。
- 验收：三步 exit code 都是 0；`dropped_offsets` 为 0；评测 `integrity_pass=true`、`max_prob_diff < 0.001`；官方集的 unique 序列数为 1872。
- 需要用户决定：无（训练已批准）。
- 反馈里必须报告：改标签 report 的每个来源的计数和 `old_to_new_class`；两个变体的 `chosen_epoch`、`chosen_calibration_mean_auc`、每个 epoch 的 calibration AUC、`train_positions_by_class`；评测 report 的 counts；`analyze_redline.py` 打印的全部内容；产物 SHA256。
