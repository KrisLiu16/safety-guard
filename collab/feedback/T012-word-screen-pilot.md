# T012 反馈

- 对应任务单版本：v5（v1–v3 的反馈见本文件 git 历史：519ead9、f6c1de7、33422de）
- 状态：**进行中**。第 3–4 步复测（v4 提示词）四项门槛全部通过；第 5 步全量的 3 个 Run 已提交，正在跑。
- 执行时间：2026-09-24 15:27 开始（北京时间）
- 实际执行的命令：与任务单 v5 相同。复测 18 个 Task，并发 18；全量 3 个 Run，并发各 512。每个 Run 提交前 `aster runs plan` 都通过，提交后平台快照都与本地 `flow/` 逐字一致。
  - 全量 part0 第一次上传时连接被断开（`CLI_UNREACHABLE`，retryable=true）。对同一个草稿版本重传一次就成功了，没有产生多余的数据集。

## 复测（第 3–4 步）

- 单测：`test_word_screen_cpu.py` 12 项全过（v5 的单测数）。
- `manual_labels.tsv` 与 v2、v3 相同：insult 17，evasion 8，rumor 1，borderline 1，no 105。另一份 `manual_labels_v3.tsv` 也照 v3 反馈跑了一遍，只改了 1 行：原来的 borderline 按 `rumor` 定义改为 rumor。两份都只在本地，没有提交。
- run_no：`aster-dev-295`（luna），18/18 个 Task 成功。manifest：words 1,003，passes 2，binary_passes 1，tasks 18。词集合与 v1 逐一比对一致。复测数据集已删除。

| 门槛 | 要求 | 实际 |
|---|---|---|
| `manual_positive_flagged`（只看模型） | ≥ 0.90 | **1.0**（26/26） |
| `manual_no_flagged` | ≤ 0.10 | 0.0952（105 个里有 10 个） |
| missing | ≤ 1% | 0：1,003 个词都有结论。batch_errors 为 word_changed 6、missing 2，这是单遍单条的错误，占 3,009 条的 0.27%；有 997 个词两遍分类都齐全 |
| 已知阳性 | 模型判非 `no`，或被人工覆盖 | 模型 3/3 判非 `no`（insult、rumor、unsure 各 1） |

summary（`by_source_group` 以外的部分）：

```json
{
 "words": 1003,
 "passes_seen": [
  "b0",
  "p0",
  "p1"
 ],
 "verdicts": {
  "insult": 22,
  "rumor": 21,
  "no": 929,
  "evasion": 17,
  "unsure": 14
 },
 "model_verdicts": {
  "unsure": 29,
  "rumor": 28,
  "insult": 15,
  "no": 919,
  "evasion": 12
 },
 "manual_overrides": 132,
 "words_with_both_category_passes": 997,
 "pass_agreement_flagged": 0.9729,
 "flagged_only_by_binary_pass": 9,
 "batch_errors": {
  "word_changed": 6,
  "missing": 2
 },
 "known_positives": 3,
 "known_positive_model_verdicts": {
  "unsure": 1,
  "rumor": 1,
  "insult": 1
 },
 "known_positive_model_recall_flagged": 1.0,
 "runs": [
  {
   "run_no": "aster-dev-295",
   "status": "completed"
  }
 ],
 "archives": 18
}
```

有非 `no` 结论的来源组（覆盖后的结论）：

| 来源组 | insult | rumor | evasion | unsure | no |
|---|---|---|---|---|---|
| citizenlab/wechat | 2 | 8 | 1 | 1 | 24 |
| permissive/政治类型.txt | 2 | 0 | 6 | 3 | 25 |
| citizenlab/search | 3 | 0 | 6 | 1 | 26 |
| citizenlab/LINE | 0 | 6 | 2 | 1 | 27 |
| permissive/网易前端过滤敏感词库.txt | 5 | 1 | 1 | 1 | 28 |
| permissive/反动词库.txt | 2 | 0 | 0 | 2 | 32 |
| permissive/GFW补充词库.txt | 3 | 0 | 0 | 0 | 34 |
| citizenlab/TOM-Skype--Sina-UC | 0 | 2 | 0 | 0 | 34 |
| citizenlab/chinese-games | 0 | 1 | 1 | 0 | 34 |
| citizenlab/livestream | 0 | 0 | 0 | 2 | 34 |
| citizenlab/olympics | 1 | 1 | 0 | 0 | 34 |
| citizenlab/qqmail | 1 | 1 | 0 | 0 | 7 |
| permissive/其他词库.txt | 2 | 0 | 0 | 0 | 34 |
| citizenlab/june-4 | 0 | 1 | 0 | 0 | 35 |
| include-only | 1 | 0 | 0 | 0 | 0 |
| permissive/暴恐词库.txt | 0 | 0 | 0 | 1 | 34 |
| permissive/民生词库.txt | 0 | 0 | 0 | 1 | 34 |
| permissive/补充词库.txt | 0 | 0 | 0 | 1 | 34 |

`compare_manual.py` 输出（原标注 `manual_labels.tsv`，门槛以此为准）：

```json
{
 "table": {
  "insult": {
   "rumor": 3,
   "insult": 10,
   "evasion": 1,
   "unsure": 3
  },
  "rumor": {
   "rumor": 1
  },
  "evasion": {
   "unsure": 7,
   "evasion": 1
  },
  "borderline": {
   "rumor": 1
  },
  "no": {
   "unsure": 6,
   "no": 95,
   "rumor": 3,
   "evasion": 1
  }
 },
 "manual_positive_flagged": 1.0,
 "manual_positive_n": 26,
 "manual_no_flagged": 0.0952,
 "manual_no_n": 105
}
```

`compare_manual.py` 输出（`manual_labels_v3.tsv`）：

```json
{
 "table": {
  "insult": {
   "rumor": 3,
   "insult": 10,
   "evasion": 1,
   "unsure": 3
  },
  "rumor": {
   "rumor": 2
  },
  "evasion": {
   "unsure": 7,
   "evasion": 1
  },
  "no": {
   "unsure": 6,
   "no": 95,
   "rumor": 3,
   "evasion": 1
  }
 },
 "manual_positive_flagged": 1.0,
 "manual_positive_n": 27,
 "manual_no_flagged": 0.0952,
 "manual_no_n": 105
}
```

说明：
- 第三遍“是/否”问法单独多标出了 9 个词（`flagged_only_by_binary_pass`），这 9 个词只算 unsure。上一轮漏掉的两个谐音，这次都被模型标出来了。
- `manual_no_flagged` 从 v3 的 0.048 升到 0.095，接近 0.10 的门槛。这是多了一遍是/否问法的代价：多筛了一些人工标为 no 的词。

## 全量（第 5 步，进行中）

- manifest：words 340,883，tasks 5,115（passes 2，binary_passes 1），prompt_version `guard-word-screen-leader-v4`。
- 分 3 份打包，每份一个数据集、一个 Run：

| 份 | Task 数 | run_no |
|---|---|---|
| part0 | 2,000 | `aster-dev-298` |
| part1 | 2,000 | `aster-dev-299` |
| part2 | 1,115 | `aster-dev-300` |

这 3 个数据集都是正式数据，不删除。跑完后冻结 attempt 列表、一次性抽取（加 `--manual`），再更新本文件。

## 用户决定（2026-09-24 17:05）：暂停造数，先训练

用户说明数据平台在更新，要求先停掉 Aster 上正在进行的工作，造数晚点再做，先训练模型（T017 → T018）。已执行：
- T012 全量的 3 个 Run 已取消：`aster-dev-298` 完成 326 个 Task 后取消（其余 1,674 个取消），`aster-dev-299`（2,000 个）和 `aster-dev-300`（1,115 个）都还没开始。3 个全量数据集保留。
- T016 主裁判两个 Run 在此之前已经结束：`aster-dev-292` 1,281/1,281 成功；`aster-dev-293` 1,224/1,280 成功，56 个在约 16:55 平台宕机时中断（`LEASE_EXPIRED`，没有归档）。这 56 个是否重跑、两个 Run 的抽取，等平台恢复、用户同意造数后再做；已下载并校验了 305 个归档。
- Run B 上传暂停在第 1,760/1,943 个（`resume_upload_only.sh` 可以续传）。
- 不受影响：T017 在 L20 上继续，T018 在 T017 验收通过后开始。
