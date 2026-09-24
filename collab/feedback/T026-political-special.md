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

## 第 3 步：抽取第一遍（04:07）

- 冻结 attempt 列表：1,705 个样本，1,705 个 attempt。
- `extract.py aster-dev-359 --attempts-json … --words … --leader-screen … --out full_p1/extracted`。执行方另外按 attempt_id 并行预取了一部分归档，文件名和脚本用的一样，脚本遇到已存在的归档直接跳过，结果不受影响。
- `summary.json`（只报计数）：
  - `words` 340,638，`archives` 1,705；
  - **`words_without_verdict` 245（0.07%）**，≤ 1%，可以继续；
  - `batch_errors`：word_changed 43、missing 43、bad_index 2、request 1；
  - `verdicts`：no 289,583、other_political 20,233、leader 10,787、event 8,621、unsure 6,079、figure 3,518、org 1,817；
  - `ambiguous_by_verdict`：no 132,069、other_political 6,674、unsure 5,327、leader 1,639、event 1,475、figure 1,109、org 182；
  - `matched_as`：event 7,146（另有 ambiguous 1,475）、figure 2,409（1,109）、org 1,635（182）、leader_insult 1,606（226）、leader_rumor 2,080（121）、leader_evasion 1,074（271）、leader_unsure 2,688（700）；
  - `confirm_words` 37,983。
- 245 个缺的词按卡片补筛一次：`make_batch.py --only missing_words.txt --tag r1 --output full_r1`，245 个词、2 个 Task。

## 第 4 步：复查

- `make_batch.py --only full_p1/extracted/confirm_words.txt --tag c1 --output full_c1`：37,983 个词，190 个 Task。
- 两个 Run 都用 luna，尝试 1 次，高优先级，平台快照与本地一致：
  - r1：**aster-dev-363**（2 个 Task）；
  - c1：**aster-dev-364**（190 个 Task，超过 100，完成后先冻结 attempt 列表）。
