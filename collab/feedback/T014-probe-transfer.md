# T014 反馈

- 对应任务单版本：v1
- 状态：进行中（L20 作业在跑，预计约 40 分钟）
- 执行时间：2026-09-24 14:49 开始（北京时间）
- 实际执行的命令：步骤 1–2 与任务单相同。

## 目前的结果

1. 单测：`round6/probe/test_transfer_cpu.py` 3 项全过。
2. 作业：Job `safety-guard-probe-transfer-20260924-r1`，Pod `safety-guard-probe-transfer-20260924-r1-pckbc`，14:49:40 提交。提交前节点 GPU 分配为 0。
   - 已把 `round6/probe/*.py` 拷进 Pod，9 个文件用 `sha256sum` 核对，与本地一致。
   - PVC 上 T004 的特征文件（`features_*.npz`、`features_*_records.json`、`probe_v1_results.json`）和 `input_v1/` 都在。
   - 14:49:56 写入 `probe_transfer_v1_input_ready`。

冒烟检查结果和后续数字出来后更新本文件。
