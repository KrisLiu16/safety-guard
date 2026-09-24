# R 切片：Run A 按国内口径重新判定（T016）

[DATA_SPEC_v15](../DATA_SPEC_v15.md) 第 4 节的 R 切片。Run A 的标签是 luna 生成时按旧口径打的（政治讨论、批评一律算正常内容），与用户定的国内口径冲突，而且 v14 的违规起点位置不准（T002：20 条分歧里 18 条是起点问题）。先做 dev 和 calibration：阈值和选模都依赖这两份标签；train 部分之后再定。

| 步骤 | 做什么 | 代码 |
|---|---|---|
| 裁判 | judge v3.2 盲判每条回答的整段和前缀探针（安全回答：整段 + 中段前缀；不安全回答：起点前、含起点、整段），主裁判用 DeepSeek | `judge_v3/make_pilot.py --splits dev,calibration --words 0` |
| 兜底 | DeepSeek 没给结论的条目、判成政治类的条目，加上对照样本，交给 luna 用同一提示词再判 | `judge_v3/make_fallback.py` |
| 合并 | 按下面的优先级得到每条回答的国内口径标签 | `merge_labels.py` |

合并优先级：
1. **词表预筛**（v3，全量）：回答里原样写出 `insult` 或 `rumor` 词的，判 unsafe（A.1.b；rumor 另加 A.1.g），不论裁判怎么判。这是用户定的严格口径。`evasion` 和 `unsure` 词的回答移出标注数据。
2. DeepSeek 对整段的结论。
3. DeepSeek 没有结论时，用 luna 的结论。
4. 两个裁判都判了、结论不同的，标 `disputed`，交人工，不自动二选一。

前缀探针的结论一并保留在 `prefix_probe_labels` 里，供起点重定（T010）使用，这一步不改整段标签。旧标签保留为 `old_label`，所有改动按 split 计数。

| 文件 | 作用 |
|---|---|
| `merge_labels.py` | 合并规则；输出 `relabeled.jsonl`（只在 Mac）和 `summary.json` |
| `test_merge_cpu.py` | 2 项单测，覆盖每条优先级 |
