# T012 反馈

- 对应任务单版本：v1
- 状态：部分完成。步骤 1–7 全部执行，数据集已删除。验收四项里有三项达到，“yes/unsure 中误判 ≤ 20%”没有达到，见第 3 节。
- 执行时间：2026-09-24 14:43—14:50（北京时间）
- 实际执行的命令：与任务单相同。并发 6，即 Task 数。提交前 `aster runs plan` 通过，提交后平台快照与本地 `flow/` 逐字一致。本地已知阳性清单 `known_positives.txt` 共 3 个词，已被 `.gitignore` 排除，`git status` 里看不到，没有提交。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| 已知阳性判为 yes 或 unsure | 全部 | 3/3（yes 2，unsure 1） |
| missing（没拿到结论） | ≤ 1% | 0（1,003 个词都有结论，batch_errors 为空） |
| yes/unsure 中的误判 | ≤ 20% | **未达到**：严格口径下 15/32（47%）；如果把“领导人姓名的规避写法”算作命中，是 7/32（22%）。见第 3 节 |
| 随机 100 个政治类 no 中的漏判 | ≤ 2 个 | 1 个明确漏判，另有 1 个边界情况 |

## 结果

### 1. 单测与 manifest

- 单测：`round6/word_screen_v1/test_word_screen_cpu.py` 6 项全过。
- manifest：words 1,003，known_positives 3，tasks 6（每批 200 词），prompt_version `guard-word-screen-leader-v1`。

### 2. Run 与 summary

run_no：`aster-dev-289`（luna），6/6 个 Task 成功。

`extracted/summary.json`（`by_source_group` 以外的部分）：

```json
{
 "words": 1003,
 "verdicts": {"unsure": 13, "yes": 19, "no": 971},
 "batch_errors": {},
 "known_positives": 3,
 "known_positive_verdicts": {"unsure": 1, "yes": 2},
 "known_positive_recall_yes_or_unsure": 1.0,
 "run_no": "aster-dev-289",
 "run_status": "completed",
 "archives": 6
}
```

`by_source_group` 里 yes 或 unsure 不为 0 的组：

| 来源组 | yes | unsure | no |
|---|---|---|---|
| citizenlab/search | 6 | 2 | 28 |
| permissive/网易前端过滤敏感词库.txt | 4 | 1 | 31 |
| permissive/GFW补充词库.txt | 2 | 1 | 34 |
| permissive/反动词库.txt | 1 | 2 | 33 |
| permissive/政治类型.txt | 1 | 2 | 33 |
| citizenlab/wechat | 1 | 1 | 34 |
| citizenlab/livestream | 0 | 2 | 34 |
| citizenlab/chinese-games | 1 | 0 | 35 |
| citizenlab/qqmail | 1 | 0 | 8 |
| permissive/其他词库.txt | 1 | 0 | 35 |
| citizenlab/june-4 | 0 | 1 | 35 |
| permissive/零时-Tencent.txt | 0 | 1 | 34 |
| include-only（只来自已知阳性清单） | 1 | 0 | 0 |

### 3. 人工核对（执行方逐个看过）

**yes 和 unsure 共 32 个词**，我分成三类：

| 类别 | yes（19） | unsure（13） | 合计 |
|---|---|---|---|
| A. 明确贬损现任或前任国家领导人（辱骂、谐音蔑称、贬损性说法） | 15 | 2 | 17 |
| B. 领导人姓名的规避写法（同音字、换字、倒序、拆字），本身没有明显的贬损含义 | 4 | 4 | 8 |
| C. 与领导人无关，或者无法辨认（泛指性骂人话、低俗谐音、异体字乱码等） | 0 | 7 | 7 |

- 严格按任务单“其实不是贬损领导人的”算，误判是 B+C，共 15/32（47%）。如果把 B 算作命中，误判只算 C，是 7/32（22%）。两种算法都超过 20%。
- 分开看两档：**yes 档的精度高**，19 个里 15 个是 A，其余 4 个都是 B，没有 C。**unsure 档基本是噪声**，13 个里只有 2 个是 A，其中 1 个是已知阳性。
- 结论：luna 的 yes 很可靠。unsure 档混进了大量无关词和乱码，如果按“宁可多筛”把 unsure 也全部筛掉，每 1,000 词大约多筛掉 10 个不相干的词，代价不大。B 类要不要筛，要看设计方的口径：S2 v2 的 skip 规则已经把“代号、谐音”列进去了，所以按那个口径，B 算命中更合适。

**随机 100 个政治类来源组里判为 no 的词**（从 507 个里按固定种子抽取）：

- 明确漏判 1 个：一条针对前任中央政治局常委的贬损性传言，用名字加动作的形式写成。
- 边界情况 1 个：一条关于前任最高领导人的政治传言（说他被软禁）。不是辱骂，但带贬损意味。
- 其余 98 个都是正确的 no：主要是反党或政治事件类的词，还有中性提及领导人的写法（如尊称、倒装称呼）、违禁品广告、色情词、网址、乱码等。它们敏感，但不针对领导人个人。

结论：漏判率在验收线以内（1 到 2 个，要求不超过 2 个）。漏判的类型是“领导人名字加贬损性传言”这类短语，而不是蔑称。筛查提示词如果只问“侮辱”，这类会被放过；要不要把“贬损性传言”也算进去，请设计方决定。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/word_screen_v1/pilot/manifest.json` | `1fbe8fc62899a961a11257abcfeea2c2a80bcac67d763afee2c69ab57635adab` | 仓库 |
| `round6/word_screen_v1/pilot/extracted/summary.json` | `1a48db3847c83872f93b6f8cf6894ddf935bc4a0dc8398dc4453f66efc85dc41` | 仓库 |
| `round6/word_screen_v1/pilot/extracted/screen.jsonl` | `35376a2a3f64197a224673f886cf58eacefd3377ef973e41d77e5c4b1ce780dc` | 只在 Mac（含词本身） |
| `round6/word_screen_v1/pilot/words.jsonl` | `7d042a531b4bf845aa3f4882edcbf9440bcd3345e6ef6c7da4bda17e4b865fba` | 只在 Mac（含词本身） |

一并提交：`run_no.txt`、`dataset_id.txt`、`dataset_version.txt`、`flow_snapshot/`、删除回执 `dataset_rm.json`（`ds_01M392GRJ31DPFXZEA70GW3CZT`，ok=true）。

## 偏差与问题（需要设计方决定）

1. **误判率超线**，主要来自 unsure 档和 B 类，见第 3 节。建议的口径二选一：只把 yes 当命中，unsure 另行处理；或者把 yes 和 unsure 全部筛掉，接受每 1,000 词多筛约 10 个无关词。
2. **B 类（姓名规避写法）算不算命中**：按 S2 v2 的 skip 规则应该算，那样 yes 档的精度就是 100%。
3. **“领导人加贬损性传言”类短语会漏判**，见第 3 节。如果放量用在 Run A 清理和 R 切片上，可以考虑在筛查提示词里补上这一类。
