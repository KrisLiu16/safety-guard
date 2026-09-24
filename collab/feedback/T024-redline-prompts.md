# T024 反馈：提问侧红线标注试跑（进行中）

- 卡片版本：v1（a2e8407）
- 状态：试跑已提交，等待结果。

## 单测

`.venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py`：28 项全过。

## 第 2 步：prefix_v2 user 角色导出

`round6/redline_v1/input/prefix_v2_user.manifest.json`（已提交，`.jsonl` 不提交）：

- records 18,322，output_sha256 `a5d9fecc51bf7f27…`
- train 17,622（safe 9,848 / unsafe 7,774），calibration 300（150 / 150），dev 400（200 / 200）

## 第 3 步：试跑

| 目录 | 来源 | 条数 | Task | judge | policy_digest | Run |
|---|---|---|---|---|---|---|
| `user_pilot_runA` | Run A dev | 200（safe 100 / unsafe 100；中文 100 / 英文 100） | 20 | guard-judge-redline-user-v1 | 9b2667295fadc6f2 | aster-dev-339 |
| `user_pilot_pv2` | prefix_v2 dev | 100（safe 50 / unsafe 50） | 10 | guard-judge-redline-user-v1 | 9b2667295fadc6f2 | aster-dev-340 |

- DeepSeek，`--flow round6/redline_v1/flow`，尝试 1 次，并发 = Task 数，高优先级，SDK 4.3。
- 两个 Run 的 flow 快照与本地 `round6/redline_v1/flow` 逐字一致。
