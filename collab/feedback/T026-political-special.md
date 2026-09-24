# T026 反馈：政治专项（进行中）

- 对应任务单版本：v1（4ea0271）
- 状态：**第一遍预筛在跑**。

## 第 1 步：单测

| 文件 | 结果 |
|---|---|
| `round6/political_screen_v1/test_political_screen_cpu.py` | 7 项全过 |
| `round6/redline_v1/test_redline_cpu.py` | 35 项全过 |
| `round6/word_screen_v1/test_word_screen_cpu.py` | 12 项全过 |

## 第 2 步：第一遍（全部词）

- `make_batch.py --output round6/political_screen_v1/full_p1`：manifest 为 `words` 340,883、`tasks` 1,705、`batch_words` 200，只有一个 `screen.tar.gz`，与卡片一致。
- 上传为一个数据集，提交 **aster-dev-359**：luna，`--flow round6/political_screen_v1/flow`，尝试 1 次，并发 512，高优先级。`runs plan` 没有 blocking，平台快照（`flow.py`、`pipeline.py`）与本地逐字一致。
- 完成后按卡片冻结 attempt 列表，再抽取。
