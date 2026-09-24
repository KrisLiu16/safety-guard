# T012 反馈

- 对应任务单版本：v2（v1 的反馈见本文件 git 历史，519ead9）
- 状态：**阻塞在第 4 步**。复测验收四项里有两项没有达到，按任务单停下，**没有做第 5 步全量**。
- 执行时间：2026-09-24 14:52—14:59（北京时间）
- 实际执行的命令：步骤 1–3 与任务单相同。复测用的参数和 v1 完全相同：`words.jsonl` 的 SHA256 与 v1 一致（`7d042a53…`），1,003 个词完全相同。并发 6（Task 数）。提交前 `aster runs plan` 通过，提交后平台快照与本地 `flow/` 逐字一致。复测数据集抽取后已删除。

## 复测验收（第 4 步）

| 项 | 要求 | 实际 |
|---|---|---|
| 已知阳性全部非 `no` | 3/3 | **2/3**：insult 2，no 1 |
| `manual_positive_flagged` | ≥ 0.95 | **0.8846**（26 个里有 23 个） |
| `manual_no_flagged` | ≤ 0.10 | 0.0286（105 个里有 3 个） |
| missing | ≤ 1% | 0（batch_errors 为空） |

## 结果

### 1. 单测与人工标注

- 单测：`round6/word_screen_v1/test_word_screen_cpu.py` 7 项全过。
- `manual_labels.tsv`（本地，已被 `.gitignore` 排除，没有提交）共 132 行：insult 17，evasion 8，rumor 1，borderline 1，no 105。这些标签来自 v1 反馈第 3 节的人工核对：A 类标 insult，B 类标 evasion，C 类和随机 100 个 no 里正确的 98 个标 no。

### 2. 复测

run_no：`aster-dev-290`（luna），6/6 个 Task 成功。

summary（`by_source_group` 以外的部分）：

```json
{
 "words": 1003,
 "verdicts": {"no": 954, "insult": 11, "evasion": 17, "rumor": 19, "unsure": 2},
 "batch_errors": {},
 "known_positives": 3,
 "known_positive_verdicts": {"no": 1, "insult": 2},
 "known_positive_recall_flagged": 0.6667,
 "run_no": "aster-dev-290",
 "run_status": "completed",
 "archives": 6
}
```

有非 `no` 结论的来源组：

| 来源组 | insult | rumor | evasion | unsure | no |
|---|---|---|---|---|---|
| citizenlab/wechat | 0 | 11 | 1 | 0 | 24 |
| citizenlab/search | 3 | 0 | 6 | 0 | 27 |
| permissive/网易前端过滤敏感词库.txt | 3 | 1 | 2 | 1 | 29 |
| permissive/政治类型.txt | 0 | 0 | 4 | 1 | 31 |
| citizenlab/LINE | 0 | 2 | 3 | 0 | 31 |
| citizenlab/TOM-Skype--Sina-UC | 0 | 2 | 0 | 0 | 34 |
| citizenlab/chinese-games | 0 | 1 | 1 | 0 | 34 |
| citizenlab/qqmail | 1 | 1 | 0 | 0 | 7 |
| citizenlab/june-4 | 0 | 1 | 0 | 0 | 35 |
| permissive/GFW补充词库.txt | 1 | 0 | 0 | 0 | 36 |
| permissive/其他词库.txt | 1 | 0 | 0 | 0 | 35 |
| permissive/反动词库.txt | 1 | 0 | 0 | 0 | 35 |
| include-only（只来自已知阳性清单） | 1 | 0 | 0 | 0 | 0 |

`compare_manual.py` 输出全文：

```json
{
 "table": {
  "insult": {
   "insult": 11,
   "no": 3,
   "rumor": 1,
   "evasion": 1,
   "unsure": 1
  },
  "rumor": {
   "rumor": 1
  },
  "evasion": {
   "evasion": 8
  },
  "borderline": {
   "rumor": 1
  },
  "no": {
   "no": 102,
   "unsure": 1,
   "rumor": 1,
   "evasion": 1
  }
 },
 "manual_positive_flagged": 0.8846,
 "manual_positive_n": 26,
 "manual_no_flagged": 0.0286,
 "manual_no_n": 105
}
```

### 3. 全量

未执行（第 4 步没有通过）。

## 分析：v2 漏掉的 3 个人工阳性

v1 对这 3 个词全都判了非 `no`（yes 2 个，unsure 1 个），v2 全判成 `no`：

1. 一个前任中央政治局常委名字的贬损性谐音（v1：yes）。
2. 一个前任中央政治局常委名字的贬损性谐音，就是 T006 里的那个已知阳性（v1：unsure）。
3. 一个把开国领袖和“整人”手段绑在一起的贬损说法（v1：yes）。按 v2 的类别，这更像 `rumor`。

可能的原因：

- **“国家领导人”的范围**：第 1、2 个都是前任政治局常委，不是最高领导人。v2 提示词只写了“国家领导人（现任或历任）”，luna 可能没把常委算进去。v1 的 yes/no 问法下却判了出来。
- **`no` 的定义把“其他政治话题”也包括了进来**，可能吞掉了第 3 类“历史人物加贬损评价”的说法。
- **同一个词两次运行的结论本身会变**：v1 和 v2 提示词不同，没法单独估计这部分；luna 走 reasoning，没有固定种子。

人工标为 `no` 却被标出的 3 个（`manual_no_flagged` 0.0286）：一个把前任总书记和历史事件放在一起的短语被判 `rumor`，一个前任政治局委员名字的变体写法被判 `evasion`，一个含义不明的词被判 `unsure`。按 v2 的定义，前两个说得通，只是我在 v1 里按“不是贬损”标了 `no`。

整体上，v2 标出的非 `no` 词比 v1 多（49 对 32），多出来的主要是 wechat 组的 rumor（11 个）和 search 组的 evasion。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/word_screen_v1/pilot_v2/manifest.json` | `7ea07ff936a5e92e0fb7c6aa8c22d06b9e55d21578ca1872cb10a2fada2b1126` | 仓库 |
| `round6/word_screen_v1/pilot_v2/extracted/summary.json` | `fa756e55d687afd431c02a84440724304eb962d1f2db656b28d3cc2663077fad` | 仓库 |
| `round6/word_screen_v1/pilot_v2/extracted/screen.jsonl` | `fd3d63e7b1781af52a0e1b35fbe308dd1e4097d170fa944e6305b1636c00d058` | 只在 Mac（含词本身） |
| `round6/word_screen_v1/pilot_v2/words.jsonl` | `7d042a531b4bf845aa3f4882edcbf9440bcd3345e6ef6c7da4bda17e4b865fba` | 只在 Mac（与 v1 相同） |

一并提交：`run_no.txt`、`dataset_id.txt`、`dataset_version.txt`、`flow_snapshot/`、删除回执 `dataset_rm.json`（`ds_01M3933SPMRJB9X7M3QYD5B7ZJ`，ok=true）。

## 需要设计方决定

1. **明确“国家领导人”的范围**：是否包括历任中央政治局常委、国家副主席等。两个漏判都是前任常委。
2. **是否用 v1 和 v2 的并集**：v1 的 yes/unsure 召回了全部人工阳性，v2 提供类别。两个一起跑、取并集，能同时满足召回和分类。另一个办法是 v2 同一批跑两次、取并集，来对冲运行之间的波动。
3. 改好后我按新任务单复测，再做全量。
