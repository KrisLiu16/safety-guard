# T014 探针迁移检验：Run A 上拟合的探针能否用到官方 thinking 集

- 版本：v1
- 目标：T004 的线性探针在 Run A 上远好于现在的头（AUC 0.995 对 0.791），但 Run A 全是 luna 按一个模板写的。本任务把同一个探针用到 prefix_v2 和官方 thinking 集（不是 luna 写的真实文本，只观察），和现在的头对比。结果决定第六轮先重训读出头，还是先补数据多样性。设计说明见 [round6/probe/README.md](../../round6/probe/README.md) 的“T004 结果与 T014”一节。
- 依赖：T004 完成，PVC 上 `/work/output/round6/probe_v1/` 的特征文件（`features_*.npz`、`features_*_records.json`、`probe_v1_results.json`）和 `/work/round6/probe/input_v1/` 都还在。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/probe/test_transfer_cpu.py`，3 项全过（需要 torch 和 numpy）。
  2. 提交作业：`kubectl apply -f round6/probe/worker_transfer_v1.yaml`。Pod 进入 Running 后，把 `round6/probe/*.py` 拷到 Pod 的 `/work/round6/probe/`（覆盖同名文件），在 Pod 里用 `sha256sum` 核对与本地一致，再 `touch /work/round6/probe_transfer_v1_input_ready`。
  3. 开始几分钟后查看 `/work/output/round6/probe_transfer_v1/report.json`：应当出现 `probe.calibration_auc_reproduced`（与 T004 的 0.99686 相差不超过 1e-4），`phase` 依次经过 runA、prefix_v2、official，`max_prob_diff` 小于 0.001。如果 `status` 是 `failed`，取回 `report.json` 和 `/work/output/round6/probe_transfer_v1.log`，写反馈，然后 `touch /work/output/round6/probe_transfer_v1_collected` 释放 GPU。
  4. 出现 `/work/output/round6/probe_transfer_v1_done` 后，`probe_transfer_v1_exit_code.txt` 应为 `transfer=0`。把 `report.json` 和全部 `*.jsonl.gz` 取回 Mac 的 `round6/probe/results_transfer_v1/`，按 `report.json` 的 `files` 核对 SHA256，然后 `touch /work/output/round6/probe_transfer_v1_collected`。
  5. 分析（Mac CPU）：`.venv/bin/python round6/probe/analyze_transfer.py round6/probe/results_transfer_v1`，输出 `round6/probe/results_transfer_v1/transfer_v1_results.json`。
- 预期产物：提交 `round6/probe/results_transfer_v1/report.json` 和 `transfer_v1_results.json`；`*.jsonl.gz` 留在 Mac。
- 验收：`status=completed`、`integrity_pass=true`、`max_prob_diff < 0.001`；探针 AUC 复现；counts 中 `official_unique_sequences` 为 1872，`runA_calibration`、`runA_dev` 与 T004 的记录数基本一致。
- 需要用户决定：无。单卡推理，预计 40 分钟左右，与 T004 同节点、同资源；节点被占用就等待，不要停别人的 Pod。
- 反馈里必须报告：
  1. 单测结果；作业名、开始和结束时间；`report.json` 的 `probe`、`counts`、`totals`、`max_prob_diff`、`elapsed_seconds`；
  2. `analyze_transfer.py` 打印的全部内容（两个打分器各自的温度、流式 AUC 和规则表）；
  3. 产物 SHA256。
