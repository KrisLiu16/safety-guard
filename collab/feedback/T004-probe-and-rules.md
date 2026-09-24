# T004 反馈

- 对应任务单版本：v1
- 状态：进行中（L20 作业在跑）
- 执行时间：2026-09-24 14:10 开始（北京时间）
- 实际执行的命令：步骤 1–3 与任务单相同。

## 目前的结果

1. 单测：`round6/probe/test_probe_cpu.py` 8 项全过。
2. 输入 manifest（`round6/probe/input_v1/manifest.json`）：
   - records 16,244；words_by_split：calibration 1,264，dev 1,297，train 1,500；
   - records_by_split_label：calibration safe/unsafe 各 2,528，dev 各 2,594，train 各 3,000；
   - records_by_slot_style：0:neutral_information 1,016，0:safe_redirect 994，0:refusal 1,068，0:risk_reasoning 983，1:compliance 4,061，2:helpful_answer 2,010，2:helpful_reasoning 2,051，3:drift 4,061。
   - `probe_input_v1.jsonl` SHA256 `616b2c6e45f7acc9f68c5bcc827f8afe3590b5255c8f863b4f74c41eeb68bf6d`（和 manifest 里的 output_sha256 一致）。
3. 作业：Job `safety-guard-probe-20260924-r1`，Pod `safety-guard-probe-20260924-r1-pbrjk`，14:11:20 提交。提交前节点 GPU 分配为 0，没有别的作业占用。
   - 14:11 把 `round6/probe/*.py` 和 `input_v1/` 拷进 Pod，8 个文件在 Pod 里用 `sha256sum` 核对，与本地一致；14:11:46 写入 `probe_v1_input_ready`。
   - T005 需要的 v1 导出文件（`prefix_v2_*.jsonl.gz`、`official_*.jsonl.gz`）在 Mac 的 `round6/decision_rule/results_v1/` 里都有，第 6 步会带 `--v1-dump` 运行。

冒烟检查结果和后续数字出来后更新本文件。
