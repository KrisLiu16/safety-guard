# T027 反馈：官方 thinking 集按红线口径标注（进行中）

- 对应任务单版本：v1（dc79364）
- 状态：**标注 Run 在跑**。

## 第 1 步：单测

- `round6/redline_v1/test_redline_cpu.py`：36 项全过。
- `round6/stage1_head/test_redline_stage1_cpu.py`：6 项全过。

## 第 2 步：导出

- 用不占 GPU 的临时 Pod 挂 PVC，把 `export_official.py` 拷进去（SHA `ef6a8dd7…`，与本地一致），再用系统自带的 python 运行。
- 取回 `official_thinking.jsonl` 和 manifest。两端 SHA256 一致：`b2675f58edcc9dd0c0a95a28fa42d4c4ba939b8d53fb9923e9a998677d47b2ad`。`.jsonl` 被 gitignore 忽略。
- manifest：
  - `records` 1,059，即 thinking 集里 status 为 ready 的全部条目；
  - `by_label_language`：safe:en 490，unsafe:en 569；
  - `unsafe_type`：
    - Non-violent Illegal Acts 195
    - Unethical Acts 175
    - Violent 118
    - PII 31
    - Sexual Content or Sexual Acts 22
    - Politically Sensitive Topics 18
    - Suicide & Self-Harm 7
    - Copyright Violation 3
    - safe 为空，共 490

## 第 3 步：标注

- `make_tasks.py --per-task 10`：1,059 条，106 个 Task。judge `guard-judge-redline-v4.2`，policy_digest `1f687a50f57ffaa3`，这是 T026 以后的规则表。
- 提交 **aster-dev-362**：DeepSeek，尝试 1 次，并发 106，高优先级。`runs plan` 没有 blocking，平台快照与本地一致。
