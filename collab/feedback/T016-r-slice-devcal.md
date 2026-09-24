# T016 反馈

- 对应任务单版本：v1
- 状态：进行中（主裁判的两个 Run 在跑）
- 执行时间：2026-09-24 15:10 开始（北京时间）
- 实际执行的命令：步骤 1–2 与任务单相同。**第 3 步有偏差**：主裁判拆成了两个 Run，原因和做法见下面。

## 目前的结果

1. 单测：`round6/judge_v3/test_judge_v3_cpu.py` 14 项全过，`round6/r_slice/test_merge_cpu.py` 2 项全过。
2. manifest（`round6/r_slice/devcal/manifest.json`）：words 2,561，items 24,390，tasks 2,561，prompt_version `guard-judge-domestic-v3.2`。kinds：safe_whole 5,122，safe_mid_prefix 5,122，unsafe_before_onset 3,902，unsafe_through_onset 5,122，unsafe_whole 5,122。
3. 主裁判（DeepSeek，judge v3.2，尝试 1 次）：
   - `aster-dev-292`：1,281 个 Task，来自数据集版本 `dsv_01M39434Z60857QKVFGE695EZZ` 中按 task_key 排序的前 1,281 个。
   - `aster-dev-293`：1,280 个 Task，来自数据集版本 `devcal_part2/dataset_version.txt`（其余 1,280 个）。
   - 两个 Run 提交前 `aster runs plan` 都通过，提交后平台快照与本地 `flow/` 逐字一致。

## 偏差：主裁判拆成两个 Run

平台有三条限制，依次碰到：

1. `--concurrency` 最大 512。“拉满”按 512 执行。
2. 一个 Run 里，同一个数据集版本最多选 2,000 个 Task，而且同一版本只能出现一次。
3. 一个 Run 总共最多 2,000 个 Task，报错原文提示“请拆成多个 Run”。

做法：

- 把后 1,280 个 Task 单独上传为第二个数据集 `guard-r-slice-devcal-v32-part2`。这 1,280 个 Task 的 `content_sha256` 与原数据集逐个比对，全部一致，内容没有改动。
- 两个 Run 都用同一份 v3.2 提示词和同一个 DeepSeek 模型。
- 抽取时把 `items.jsonl` 按两个 Run 的 task_key 拆成两份，分别冻结 attempt 列表并抽取，再把两份 `judgments.jsonl` 拼成 `devcal/extracted/judgments.jsonl`。每条 item 一行，拼接不会改变结果，第 4、5 步直接用拼好的文件。
- 汇总数字按两份结果分别报告，再给出合计。

两个数据集都是正式数据，不删除。第 4 步兜底如果超过 2,000 个 Task，也按同样的方式拆分。

冒烟和后续数字出来后更新本文件。
