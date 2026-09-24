# T002 反馈

- 状态：部分完成（步骤 1–2 完成；步骤 3 等用户批准并发）
- 执行时间：2026-09-24 13:42—13:45（北京时间）
- 实际执行的命令：步骤 1–2 与任务单相同。

## 结果

1. 单测：`.venv/bin/python -m unittest round6/judge_v3/test_judge_v3_cpu.py`，8 项全过（OK）。

   `manifest.json`：

   | 项 | 值 |
   |---|---|
   | items | 433 |
   | tasks | 49 |
   | words | 49 |
   | kinds | safe_whole 98，safe_mid_prefix 98，unsafe_before_onset 63，unsafe_through_onset 87，unsafe_whole 87 |
   | 来源组 | 26 个，每组 4–20 条 |
   | prompt_version | `guard-judge-domestic-v3` |
   | system_prompt SHA256 | `63c5c510e073459762469e62ac4d4821e998d09b1a96bbc969a6508126bbbc66` |

2.–5. 步骤 3 未提交，暂无结果。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/judge_v3/pilot/manifest.json` | `a44d05745a0b3c13fad631e0756bf20d23d2a31f2c3d3f895b1e359e734dcf68` | 仓库（本次提交） |
| `round6/judge_v3/pilot/pilot.tar.gz` | `2eaa1d304a0e8f00190427c2823e0031147149e915acbeb31831e88bcbe081aa` | 只在 Mac（含样本原文，已被 .gitignore 排除） |
| `round6/judge_v3/pilot/items.jsonl` | `70563f6aeff1faeb43eb196e8d0b76d6c3b26eeaa23d584e04a69df959b2e25b` | 只在 Mac（答案表，已被排除） |

`tasks/`、`*.gz`、`*.jsonl` 都已被 .gitignore 排除，`git status` 里只出现 `manifest.json`。

## 偏差与问题

1. **词数比任务单少。** 按 `--words 60` 运行，但来源 `round6/response_v14/pilot/extracted/examples.jsonl` 只有 49 个词（185 条回答），所以全部 49 个词都进了 pilot。
2. **请求数比估计多。** 实际是 433 次请求（每条 item 一次），不是任务单和 BOARD 里写的约 300 次。按 Task 计是 49 个，并发限制的是同时运行的 Task 数。
3. 待用户决定：步骤 3 的并发和是否提交。
