# T012 反馈

- 对应任务单版本：v3（v1、v2 的反馈见本文件 git 历史：519ead9、f6c1de7）
- 状态：**阻塞在第 4 步**。复测验收四项里仍有两项没有达到，按任务单停下，**没有做第 5 步全量**。
- 执行时间：2026-09-24 15:05—15:22（北京时间）
- 实际执行的命令：第 3 步与任务单相同。1,003 个词与 v1、v2 完全相同（`words.jsonl` 只是字段名从 `batch_key` 改成了 `batch_keys`，词集合逐一比对一致）。manifest：words 1,003，passes 2，tasks 12。并发 12（Task 数）。提交前 `aster runs plan` 通过，提交后平台快照与本地 `flow/` 逐字一致。复测数据集抽取后已删除。

## 复测验收（第 4 步，以原标注为准）

| 项 | 要求 | v2（aster-dev-290） | v3（aster-dev-291） |
|---|---|---|---|
| 已知阳性全部非 `no` | 3/3 | 2/3 | **2/3** |
| `manual_positive_flagged` | ≥ 0.95 | 0.8846 | **0.9231**（26 个里有 24 个） |
| `manual_no_flagged` | ≤ 0.10 | 0.0286 | 0.0476（105 个里有 5 个） |
| missing | ≤ 1% | 0 | 0（两遍都有结论的词 1,003/1,003，batch_errors 为空） |

## 结果

### 1. 单测

`test_word_screen_cpu.py` 9 项全过。`manual_labels.tsv` 与 v2 相同：insult 17，evasion 8，rumor 1，borderline 1，no 105。

### 2. 复测

run_no：`aster-dev-291`（luna），12/12 个 Task 成功。

summary（`by_source_group` 以外的部分）：

```json
{
 "words": 1003,
 "passes_seen": ["p0", "p1"],
 "verdicts": {"no": 934, "insult": 13, "rumor": 26, "evasion": 19, "unsure": 11},
 "words_with_all_passes": 1003,
 "pass_agreement_flagged": 0.9751,
 "batch_errors": {},
 "known_positives": 3,
 "known_positive_verdicts": {"no": 1, "insult": 2},
 "known_positive_recall_flagged": 0.6667,
 "run_no": "aster-dev-291",
 "run_status": "completed",
 "archives": 12
}
```

有非 `no` 结论的来源组：

| 来源组 | insult | rumor | evasion | unsure | no |
|---|---|---|---|---|---|
| citizenlab/wechat | 1 | 10 | 0 | 1 | 24 |
| citizenlab/LINE | 0 | 7 | 3 | 1 | 25 |
| citizenlab/search | 3 | 0 | 7 | 0 | 26 |
| permissive/政治类型.txt | 0 | 0 | 6 | 3 | 27 |
| permissive/网易前端过滤敏感词库.txt | 3 | 1 | 2 | 1 | 29 |
| permissive/反动词库.txt | 2 | 1 | 0 | 1 | 32 |
| citizenlab/qqmail | 1 | 1 | 0 | 0 | 7 |
| citizenlab/TOM-Skype--Sina-UC | 0 | 2 | 0 | 0 | 34 |
| citizenlab/chinese-games | 0 | 1 | 1 | 0 | 34 |
| citizenlab/june-4 | 0 | 1 | 0 | 1 | 34 |
| permissive/GFW补充词库.txt | 1 | 1 | 0 | 0 | 35 |
| permissive/其他词库.txt | 1 | 0 | 0 | 1 | 34 |
| citizenlab/olympics | 0 | 1 | 0 | 0 | 35 |
| citizenlab/livestream | 0 | 0 | 0 | 1 | 35 |
| permissive/民生词库.txt | 0 | 0 | 0 | 1 | 34 |
| include-only（只来自已知阳性清单） | 1 | 0 | 0 | 0 | 0 |

`compare_manual.py` 输出（原标注 `manual_labels.tsv`，验收以此为准）：

```json
{
 "table": {
  "insult": {
   "insult": 11,
   "no": 2,
   "rumor": 2,
   "evasion": 1,
   "unsure": 1
  },
  "rumor": {
   "rumor": 1
  },
  "evasion": {
   "evasion": 7,
   "unsure": 1
  },
  "borderline": {
   "rumor": 1
  },
  "no": {
   "unsure": 2,
   "no": 100,
   "rumor": 2,
   "evasion": 1
  }
 },
 "manual_positive_flagged": 0.9231,
 "manual_positive_n": 26,
 "manual_no_flagged": 0.0476,
 "manual_no_n": 105
}
```

`compare_manual.py` 输出（按 v3 定义改过的 `manual_labels_v3.tsv`）：

```json
{
 "table": {
  "insult": {
   "insult": 11,
   "no": 2,
   "rumor": 2,
   "evasion": 1,
   "unsure": 1
  },
  "rumor": {
   "rumor": 2
  },
  "evasion": {
   "evasion": 7,
   "unsure": 1
  },
  "no": {
   "unsure": 2,
   "no": 100,
   "rumor": 2,
   "evasion": 1
  }
 },
 "manual_positive_flagged": 0.9259,
 "manual_positive_n": 27,
 "manual_no_flagged": 0.0476,
 "manual_no_n": 105
}
```

**改动的标注**：只改了 1 行。v1 人工核对时标为 `borderline` 的那个词（一条关于前任最高领导人的负面传言），按 v3 的 `rumor` 定义（“把上述人物和负面传言……绑在一起的短语”）改成了 `rumor`。v2 反馈里提到的两个“标 no 但可能算阳性”的词没有改，仍是 `no`：
- 一个涉及的人物是前任政治局委员，不是常委，不在 v3 写明的范围内。
- 一个是把前任总书记和一个历史事件放在一起的短语，本身没有对这个人的贬损内容，不符合 `rumor` 的定义。

## 分析

- **漏判仍然是那 2 个**，两遍都判 `no`：都是前任中央政治局常委名字的贬损性谐音，其中一个是已知阳性。v3 已经把常委写进了范围，这两个仍然没有识别出来。原因更像是 luna 看不出这两个谐音对应的是谁，而不是范围写得不清楚。v1 的 yes/no 问法曾经识别出其中一个（另一个当时是 unsure）。
- **人工 `no` 被标出的 5 个**：2 个 `unsure`（含义不明）；2 个涉及前任政治局委员（不在 v3 范围内，但被判为 `rumor` 或 `evasion`）；1 个是上面那个“前任总书记加历史事件”的短语，被判为 `rumor`。这些多筛的代价小，在验收线以内。
- 两遍之间的一致率 `pass_agreement_flagged` 为 0.975。按“最严重的结论”合并后，非 `no` 从 v2 的 49 个增加到 69 个。

## 需要设计方决定

1. 这 2 个谐音漏判两遍都稳定地判 `no`，只改提示词可能很难收敛。可以考虑：
   - 把词表里已知的领导人谐音或蔑称做成本地清单直接命中，不依赖模型；
   - 验收时允许“已知阳性由本地清单兜底”；
   - 或者对 v1 的 yes/unsure 和 v3 的结论取并集（v1 在这 1,003 个词上召回了全部人工阳性）。
2. 改好后我按新任务单复测，再做全量。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/word_screen_v1/pilot_v3/manifest.json` | `155cee3ed3e12e1d6fe1cef66c8fd9a0c237306c93ece41e3b4d0ca922da5ab3` | 仓库 |
| `round6/word_screen_v1/pilot_v3/extracted/summary.json` | `568d6012754ecadf54599e8cea5e868edfadb34984cd427dbc8e7d4dc6616ad3` | 仓库 |
| `round6/word_screen_v1/pilot_v3/extracted/screen.jsonl` | `32d5c03ec6374e72d4d9b6579c972563281ecab4995c0ae4afea0700f957683f` | 只在 Mac（含词本身） |
| `round6/word_screen_v1/pilot_v3/words.jsonl` | `cf70957a4e4d7c793968c3be5d2ecce75e57370cea88585f4429e837105cb7fc` | 只在 Mac（含词本身） |

一并提交：`run_no.txt`、`dataset_id.txt`、`dataset_version.txt`、`flow_snapshot/`、删除回执 `dataset_rm.json`（ok=true）。`manual_labels.tsv` 和 `manual_labels_v3.tsv` 只在本地，已被 `.gitignore` 排除。
