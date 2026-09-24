# T023 反馈

- 对应任务单版本：v1
- 状态：**进行中**。L20 作业在跑。
- 执行时间：2026-09-24 23:00 开始（北京时间）
- 实际执行的命令：第 1、2 步与任务单相同。
  - 单测：`test_stage1_cpu.py` 共 7 项，全过。
  - 作业：Job `safety-guard-stage1-cache-user-v1-r1`，Pod `safety-guard-stage1-cache-user-v1-r1-stv2h`，23:00 提交。提交前节点 GPU 分配为 0。
  - 拷进 Pod 的代码：`round6/stage1_head/*.py` 9 个、`round6/probe/*.py` 9 个，用 `sha256sum` 核对，都与本地一致。
  - PVC 上的 `input_v1/`：`stage1_runA_v1.jsonl` 的 SHA256 为 `4e4afa5d…9b30c8f9`，与 manifest 的 `output_sha256` 一致；`manifest.json` 的 SHA256 为 `472582e2…5767f0`，与本地一致。
  - 23:01:16 写入 `stage1_cache_user_v1_input_ready`。
- 完成后取回 report 并更新本文件。
