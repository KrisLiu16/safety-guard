# T017 反馈

- 对应任务单版本：v1
- 状态：进行中（L20 作业在跑，预计 3–3.5 小时）
- 执行时间：2026-09-24 15:52 开始（北京时间）
- 实际执行的命令：步骤 1–3 与任务单相同。

## 目前的结果

1. 单测：`round6/stage1_head/test_stage1_cpu.py` 5 项全过。
2. 输入 manifest（`round6/stage1_head/input_v1/manifest.json`，已提交）：
   - records 37,056；words_by_split 为 train 8,000、calibration 1,264；
   - records_by_split_label：train safe/unsafe 各 16,000，calibration safe/unsafe 各 2,528；
   - `stage1_runA_v1.jsonl` SHA256 `4e4afa5deefadaea694183b8988e919000b1c995cb62b0ca292f89499b30c8f9`，与 manifest 的 output_sha256 一致。
3. 作业：Job `safety-guard-stage1-cache-20260924-r1`，Pod `safety-guard-stage1-cache-20260924-r1-wp4dn`，15:52:33 提交。提交前节点 GPU 分配为 0。
   - 拷进 Pod 的文件：`round6/stage1_head/*.py`、`input_v1/`，以及覆盖到 `/work/round6/probe/` 的 `round6/probe/*.py`。共 17 个文件，用 `sha256sum` 核对，与本地一致。
   - 15:53:10 写入 `stage1_cache_v1_input_ready`。

冒烟检查和后续数字出来后更新本文件。T018 的训练要等用户确认后才开始。
