# T028 第一阶段：读出上限诊断（训练更久，另加一个宽读出做对照）

- 版本：v1
- 背景：T018 v3.1 里 full 头最好，但召回偏低：
  - Run A 有争议的召回 29–39%，风险的召回 15–22%；
  - 流式 AUC 0.83–0.86；
  - 位置级 calibration AUC 到第 4 轮（0.865）还在涨。

  这一步要回答：冻结主干、只重训读出头，还能提高多少？
  - 如果多训几轮或换更大的读出就能明显提高，第一阶段还有余地；
  - 如果都停在 0.87 左右，瓶颈就在主干最后一层的特征上，下一步要训练主干（第二阶段，需用户批准）。
- 依赖：T025 v1.1 跑完，让出 L20。缓存（`/work/output/round6/stage1_cache_v1`）和 T018 的三类目标（`/work/output/round6/stage1_labels3_v2`）已在 PVC 上，不用重新改标签。第一阶段训练已获用户批准。
- 作业内容（`worker_readout_v3.yaml`，7 份 GPU 时间片，与 T018 相同）：
  1. `stage1_train_v3a`：训练 `full` 和 `wide` 两个变体，各 16 轮，其余设置同 T018。
     - `wide` 是诊断用的：一个全新初始化的 1024→2048→512→3 读出，形状和运行时的头不同，**不部署、不评测**，只看 calibration AUC 能到多少。
  2. `stage1_train_v3b`：训练 `full`，16 轮，学习率 3e-3，不加向初始权重的 L2 约束。
  3. `stage1_eval_v3a`：只评测 v3a 的 `full`，与 T018 同一套评测，约 30 多分钟。
- 步骤：
  1. **单测**（Mac，小张量）：
     ```bash
     .venv/bin/python -m unittest round6/stage1_head/test_stage1_cpu.py round6/stage1_head/test_redline_stage1_cpu.py
     ```
     全过。`test_stage1_cpu.py` 里三类训练的那一项现在也会训练 `wide`。
  2. **提交**：`kubectl apply -f round6/stage1_head/worker_readout_v3.yaml`。Pod 进入 Running 后：
     - `round6/stage1_head/*.py` 拷到 `/work/round6/stage1_head/`；
     - `round6/probe/*.py` 拷到 `/work/round6/probe/`。

     用 `sha256sum` 核对后 `touch /work/round6/stage1_readout_v3_input_ready`。
  3. **完成后**：`stage1_readout_v3_exit_code.txt` 应为 `train_a=0 train_b=0 eval=0`。取回以下文件，核对后 `touch /work/output/round6/stage1_readout_v3_collected`：
     - 两份 `train_report.json` 和 `head_full.pt`，放到 `round6/stage1_head/results_train_v3a/`、`results_train_v3b/`；
     - `head_wide.pt` 放到 `results_train_v3a/`，不提交；
     - 评测的 `report.json` 和 `eval_full/`，放到 `round6/stage1_head/results_eval_v3a/`。
  4. **对比**（Mac CPU）：
     ```bash
     .venv/bin/python round6/stage1_head/analyze_redline.py --eval-dir round6/stage1_head/results_eval_v3a --variants full --runA-labels round6/redline_v1/labels/runA_stage1/labels.jsonl round6/redline_v1/labels/runA_dev/labels.jsonl --prefix-labels round6/redline_v1/labels/prefix_v2/labels.jsonl --output round6/stage1_head/results_eval_v3a/compare_redline_v1.json
     ```
     T027 的 `labels/official_thinking/labels.jsonl` 如果已经就位，再加上 `--official-labels` 另跑一次，输出到 `compare_redline_official.json`。
- 预期产物：可提交的有：
  - 两份 `train_report.json`；
  - 评测 `report.json`；
  - `compare_redline_v1.json`（以及 `compare_redline_official.json`）。

  权重和逐位置文件只放在 Mac 和 PVC。
- 验收：
  - exit code 全为 0；
  - 评测 `integrity_pass=true`、`max_prob_diff < 0.001`；
  - 两份 `train_report.json` 的 `sources_left_out` 都是 `["s2", "s5"]`，`wide` 的 `deployable` 为 false。
- 需要用户决定：无（第一阶段训练已批准）。
- 反馈里必须报告：
  - 三个训练结果（v3a full、v3a wide、v3b full）各自的 `chosen_epoch` 和 `chosen_calibration_mean_auc`，以及每一轮按来源分开的 calibration AUC；
  - 评测 counts；
  - `analyze_redline.py` 打印的全部内容；
  - 产物 SHA256。
