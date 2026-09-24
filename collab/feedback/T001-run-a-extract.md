# T001 反馈

- 状态：阻塞（Run A 已全部结束，抽取在第 3 步报错退出，需要设计方改抽取脚本）
- 执行时间：2026-09-24 13:30—13:45（北京时间）
- 实际执行的命令：步骤 1–3 与任务单相同，没有改动。步骤 4 未执行（没有抽取结果）。

## 结果

1. Run 状态（`aster runs get aster-dev-276`）：status `completed`，352 个 Task 全部成功；失败 0、超时 0、取消 0、运行中 0。reward 合计 298.993（平均 0.849）。
2. `summary.json`：未生成。
3.–6. 未得到（依赖抽取结果）。
7. 产物：无。`round6/response_v14/batch_50k/extracted/` 已被脚本创建，但是空目录，没有下载任何归档。

## 偏差与问题（需要设计方决定）

**抽取脚本拿不到完整的 attempt 列表。** 步骤 3 在 `round6/response_v14/extract_archives.py:75` 抛出：

```
RuntimeError: Attempt listing truncated; freeze a per-sample attempt list first
```

原因：脚本用 `aster runs attempts aster-dev-276` 一次取全部 attempt，但这个接口最多返回 100 条。实测返回 `items` 100 条、`truncated: true`，而 Run A 有 352 个 sample。脚本检测到截断后主动退出，这是预期内的保护。

任务单写明 `extract_archives.py` 是冻结版本，这也不属于路径、拼写一类的小问题，所以我没有做 `[exec-fix]`。

可参考的做法：`round2/oneword2_v12/freeze_full_attempts.py` 处理过同样的问题。它按 100 条一页翻 `aster runs samples`，再对每个 sample 调 `aster runs attempts <run> --sample-id <id>`，选出唯一一个已完成且带归档的 attempt，写成冻结的 `attempts.json`（`truncated: false`）。它的文档说明输出交给 `extract_archives.py --attempts-json`，但当前的 `round6/response_v14/extract_archives.py` 只有 `run`、`--out`、`--expected-terms`、`--download-workers` 四个参数，没有 `--attempts-json`。

请设计方给出改好的抽取脚本（或单独的冻结脚本加 `--attempts-json` 参数），我按新任务单重跑。抽取不消耗模型调用。
