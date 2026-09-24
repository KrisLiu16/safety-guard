# T023 反馈

- 对应任务单版本：v1
- 状态：**完成，验收通过**。
- 执行时间：2026-09-24 23:00 – 23:45（北京时间）
- 实际执行的命令：第 1–3 步与任务单相同。
  - 单测：`test_stage1_cpu.py` 共 7 项，全过。
  - 作业：Job `safety-guard-stage1-cache-user-v1-r1`，Pod `safety-guard-stage1-cache-user-v1-r1-stv2h`，23:00 提交。提交前节点 GPU 分配为 0。
  - 拷进 Pod 的代码：`round6/stage1_head/*.py` 9 个、`round6/probe/*.py` 9 个，用 `sha256sum` 核对，都与本地一致。
  - PVC 上的 `input_v1/`：`stage1_runA_v1.jsonl` 的 SHA256 为 `4e4afa5d…9b30c8f9`，与 manifest 的 `output_sha256` 一致；`manifest.json` 的 SHA256 为 `472582e2…5767f0`，与本地一致。
  - 23:01:16 写入 `stage1_cache_user_v1_input_ready`；23:45 出现 `stage1_cache_user_v1_done`，`exit_code.txt` 为 `cache=0`。
  - 取回 `report.json` 到 `round6/stage1_head/results_cache_user_v1/`，两端 SHA256 一致，已写 `stage1_cache_user_v1_collected`。

## 结果

| 项 | 值 | 验收 |
|---|---|---|
| status | completed | ✓ |
| integrity_pass | true | ✓ |
| role | user | ✓ |
| max_prob_diff | 7.15e-7 | < 0.001 ✓ |
| elapsed_seconds | 2,620（约 44 分钟） | — |
| training_performed / generated_tokens | false / 0 | — |
| head_init_sha256 | `b2b1efb47a84677f3ebbabbfcfae12701c127ed7a4505f19db1701b993ebd53b` | — |
| inputs.runA | `4e4afa5d…9b30c8f9`（与 input_v1 一致） | — |

`stats`：

| 部分 | split | records | positions | shards |
|---|---|---|---|---|
| prefix_v2 | train | 27,456 | 310,110 | 2 |
| prefix_v2 | calibration | 300 | 1,384 | 1 |
| runA_prompts | train | 15,999 | 321,271 | 2 |
| runA_prompts | calibration | 2,526 | 50,868 | 1 |

- `runA_prompts` 跳过 train 1 条、calibration 2 条，所以 train 为 15,999（预期约 16,000），calibration 为 2,526（预期约 2,528）。✓
- prefix_v2 train 的源数据有 17,622 条 user 角色记录（与 T024 导出一致），缓存里是 27,456 个 records，每条约 1.56 个。T017 的 assistant 角色是 15,689 条对应 24,582 个，比例相同。calibration 两边都是一条对一条（300 / 300）。重写标签时，请按缓存的 records 文件对应。
- `totals`：46,281 个序列，2,888,896 个前向 token。

## 产物

- 已提交：`round6/stage1_head/results_cache_user_v1/report.json`，SHA256 `c787624b2a47750f0fadbe8c5683713100491ff8bd94ed2b571ebebdafedd052`。
- 缓存留在 PVC 的 `/work/output/round6/stage1_cache_user_v1/`，共 11 个文件约 2.1 GB，包括 `head_user_init.pt`，以及 prefix_v2 和 runA_prompts 的 train、calibration 分片和 records。
