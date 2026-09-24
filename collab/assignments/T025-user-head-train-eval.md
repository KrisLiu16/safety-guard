# T025 提问侧：按红线口径改标签，训练用户头并评测

- 版本：v1
- 目标：在 T023 缓存的用户头特征上，按 T024 的提问侧红线标签写三类目标，训练两个读出头变体（`risk`、`full`），然后和现在的用户头（`init`）在同一次评测里对比。做法和 T018 v3 相同，只是换成用户头：
  - 改标签：`relabel_cache.py --role user`。Run A 提问的字符偏移直接取缓存里记下的 `char_ends`；prefix_v2 的 user 记录按原方式重新分词对齐。
  - 训练：`train_head.py --role user`，从 `head_user_init.pt` 热启动，来源是 prefix_v2 和 runA_prompts。
  - 评测：`eval_head_l20.py --role user`。在 Run A 的 calibration 和 dev 提问、prefix_v2 的 calibration 和 dev user 记录上逐位置打分。官方 thinking 集是回答侧的，这里跳过。
  - 对比：`analyze_redline.py --role user`。
- 依赖：T023 ✓；T024 放量完成（`/work/round6/redline_v1/labels/user_runA_stage1/`、`user_prefix_v2/` 就位）；评测和分析还要用 `labels/user_runA_dev/`（放在 Mac）。训练已获用户批准（与 T018 同属第一阶段）。
- 步骤：
  1. 单测（Mac，小张量）：`.venv/bin/python -m unittest round6/stage1_head/test_stage1_cpu.py round6/stage1_head/test_redline_stage1_cpu.py`，全过。
  2. 提交作业：`kubectl apply -f round6/stage1_head/worker_user_redline_v1.yaml`。Pod 进入 Running 后：
     - `round6/stage1_head/*.py` 拷到 `/work/round6/stage1_head/`；
     - `round6/probe/*.py` 拷到 `/work/round6/probe/`；
     - `round6/redline_v1/flow/levels.py` 拷到 `/work/round6/redline_v1/flow/`。

     用 `sha256sum` 核对后 `touch /work/round6/stage1_user_redline_v1_input_ready`。
  3. 改标签完成后看 `/work/output/round6/stage1_labels3_user_v1/report.json`：`dropped_offsets` 应为 0。
  4. 训练完成后看 `train_report.json`：
     - `role` 为 `user`；
     - `calibration_score` 为 `cut`；
     - `sources_left_out` 应为空；
     - 两个变体都有 `chosen_epoch`。
  5. 完成后 `stage1_user_redline_v1_exit_code.txt` 为 `relabel=0 train=0 eval=0`。取回三个 report、`train_report.json`、`head_*.pt` 和三个 `eval_*/` 目录，分别放到 `round6/stage1_head/results_labels3_user_v1/`、`results_train_user_v1/`、`results_eval_user_v1/`。核对后 `touch /work/output/round6/stage1_user_redline_v1_collected`。
  6. 对比（Mac CPU）：
     ```bash
     .venv/bin/python round6/stage1_head/analyze_redline.py --role user --eval-dir round6/stage1_head/results_eval_user_v1 --runA-labels round6/redline_v1/labels/user_runA_stage1/labels.jsonl round6/redline_v1/labels/user_runA_dev/labels.jsonl --prefix-labels round6/redline_v1/labels/user_prefix_v2/labels.jsonl --output round6/stage1_head/results_eval_user_v1/compare_redline_v1.json
     ```
- 预期产物：提交三个 report、`train_report.json`、`compare_redline_v1.json`；权重和逐位置文件留在 Mac 和 PVC。
- 验收：
  - 三步 exit code 都是 0；
  - `dropped_offsets` 为 0；
  - 评测 `integrity_pass=true`、`max_prob_diff < 0.001`。
- 通过标准：同 T018 v3，看用户头：
  - `normal` 层误报不高于现在的头，目标接近 0；
  - `non_redline_harm` 层误报明显下降；
  - “有争议”和“风险”的召回不低于现在的头。
- 需要用户决定：无。
- 反馈里必须报告：改标签 report 各来源的计数和 `old_to_new_class`；两个变体的 `chosen_epoch`、`chosen_calibration_mean_auc`、`train_positions_by_class`；评测 counts；`analyze_redline.py` 打印的全部内容；产物 SHA256。
