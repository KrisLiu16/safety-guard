# T018 第一阶段：训练读出头并评测

- 版本：v2（记录用户批准，内容其余未变）
- 目标：在 T017 的缓存上训练两个读出头变体（`risk`：只训风险层；`full`：投影加风险层），然后装回模型，在 Run A、prefix_v2 和官方 thinking 集上与现在的头对比。设计见 [round6/stage1_head/README.md](../../round6/stage1_head/README.md)。
- 依赖：T017 完成。**用户已批准（2026-09-24）：缓存完成后直接训练。**
- 步骤：
  1. 提交作业：`kubectl apply -f round6/stage1_head/worker_train_eval_v1.yaml`。Pod 进入 Running 后，把 `round6/stage1_head/*.py` 拷到 `/work/round6/stage1_head/`，用 `sha256sum` 核对，再 `touch /work/round6/stage1_train_eval_v1_input_ready`。作业先训练（几分钟），再用新头跑评测（两个变体共约 70 分钟）。
  2. 训练几分钟后查看 `/work/output/round6/stage1_train_v1/train_report.json`：两个变体都有 `chosen_epoch`，`dev_read` 为 false。
  3. 完成后 `stage1_train_eval_v1_exit_code.txt` 应为 `train=0 eval=0`。取回到 Mac：
     - `stage1_train_v1/train_report.json` 和两个 `head_*.pt`，放到 `round6/stage1_head/results_train_v1/`；
     - `stage1_eval_v1/report.json` 和两个 `eval_*/` 目录（全部 `*.jsonl.gz`），放到 `round6/stage1_head/results_eval_v1/`。

     按两个 report 的 SHA 核对后，`touch /work/output/round6/stage1_train_eval_v1_collected`。
  4. 对比（Mac CPU）：
     ```bash
     .venv/bin/python round6/stage1_head/analyze_stage1.py --current round6/probe/results_transfer_v1 --eval-dir round6/stage1_head/results_eval_v1 --output round6/stage1_head/results_eval_v1/compare_v1.json
     ```
- 预期产物：提交 `results_train_v1/train_report.json`、`results_eval_v1/report.json`、`results_eval_v1/compare_v1.json`；`head_*.pt` 和 `*.jsonl.gz` 留在 Mac 和 PVC。
- 验收：训练和评测的 exit code 都是 0；评测 `integrity_pass=true`、`max_prob_diff < 0.001`；官方集的 unique 序列数为 1872。
- 需要用户决定：无（已批准）。T017 验收通过后立即执行。
- 反馈里必须报告：两个变体的 `chosen_epoch`、`chosen_calibration_mean_auc` 和每个 epoch 的 calibration AUC；评测 report 的 counts；`analyze_stage1.py` 打印的全部内容；产物 SHA256。
