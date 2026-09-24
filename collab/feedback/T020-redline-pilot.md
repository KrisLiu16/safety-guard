# T020 反馈

- 对应任务卡：v1
- 状态：**部分完成**。第 1、2 步已完成；第 3 步起要用 Aster，等数据平台恢复。
- 执行时间：2026-09-24 18:02–18:10（北京时间）
- 实际执行的命令：第 2 步与任务卡不同，改在 Mac 上跑，原因见下。

## 结果

1. **单测**：`test_redline_cpu.py`、`test_redline_stage1_cpu.py`、`test_stage1_cpu.py` 共 31 项，全部通过（Ran 31 tests, OK）。
2. **导出 prefix_v2 助手侧记录**：
   - 集群上没有挂着 PVC 的 Pod 可用：T017 的 Pod 已在 17:58 正常退出，其他 safety-guard 的 Pod 也都已结束。所以我没有为这一步新开 Pod，而是在 Mac 上对本地冻结副本运行：
     ```bash
     .venv/bin/python round6/redline_v1/export_prefix_v2.py --data-root round5/data/prefix_v2 --output round6/redline_v1/input/prefix_v2_assistant.jsonl
     ```
   - 本地三个文件的 SHA256 与 `round5/data/prefix_v2/manifest.json` 里的 `output_hashes` 逐一相同，确认是冻结版本：
     - train `77c31a0f…0772e`
     - calibration `8db0d6bf…c0808`
     - dev `53d7593f…3e211`
   - manifest `round6/redline_v1/input/prefix_v2_assistant.manifest.json` 已提交。
     - records 为 17,089。
     - `output_sha256` 为 `02bdacdad48557c2fba2d2c4e93f8796151ae509ca5f0061f58fc121e6b2a2a6`，与本地 `.jsonl` 一致。
     - `by_split_label_tier`：

       | split | 标签 | tier | 条数 |
       |---|---|---|---:|
       | train | safe | public_source_reference | 6,500 |
       | train | safe | rubric_benign | 2,048 |
       | train | safe | synthetic_context_weak | 347 |
       | train | unsafe | public_source_reference | 6,500 |
       | train | unsafe | synthetic_context_weak | 294 |
       | calibration | safe | public_source_reference | 300 |
       | calibration | unsafe | public_source_reference | 300 |
       | dev | safe | public_source_reference | 400 |
       | dev | unsafe | public_source_reference | 400 |
   - `.jsonl` 只放在 Mac 上，已确认被 .gitignore 覆盖。PVC 上的 `/work/round6/redline_v1/input/` 暂时没有这份文件。按现有代码，下游的 T018 v3 改标签读的是 PVC 上的 `/work/round5/data/prefix_v2` 和 `labels/prefix_v2/`，不读这份导出。如果需要 PVC 上也有，下次有 Pod 时我再拷过去。

## 待做

- 第 3–10 步要等数据平台恢复。恢复后按任务卡执行：DeepSeek 尝试 1 次，并发拉满。
