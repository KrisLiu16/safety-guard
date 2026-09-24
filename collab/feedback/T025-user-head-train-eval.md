# T025 反馈：按第 3 步停下（prefix_v2 train 的 dropped_offsets 为 0.0177%，超过 0.01%）

- 对应任务单版本：v1，第 3 步按 df6255e 的新规则执行。
- 状态：**已停下，等设计方决定**。改标签和训练都做完了，评测刚开始就被执行方中止。

## 经过

- 单测：`test_stage1_cpu.py` 和 `test_redline_stage1_cpu.py` 全过。
- 依赖：
  - T023 已完成；
  - T024 的标签已拷到 PVC（`user_runA_stage1` `20df6bda…`、`user_prefix_v2` `292f2705…`、`user_runA_dev` `87dc0ee7…`）。
- 03:17 T018 取回完成、释放 GPU 后提交作业：`kubectl apply -f round6/stage1_head/worker_user_redline_v1.yaml`。按用户指示申请 7 份 GPU（见 T018 反馈）。Pod 为 `safety-guard-stage1-user-redline-v1-r1-pblbx`。
- 拷入并核对了 21 个文件，全部一致：
  - `round6/stage1_head/*.py` 9 个；
  - `round6/probe/*.py` 9 个；
  - `levels.py`；
  - 两份 labels.jsonl。

  03:17:31 写入 `stage1_user_redline_v1_input_ready`。
- 03:18 改标签完成，训练也很快完成。执行方读到报告里 prefix_v2 train 的 `dropped_offsets` 超过上限，在 03:18:24 删除 Job。
  - 输出都留在 PVC：`stage1_labels3_user_v1/`、`stage1_train_user_v1/`（`head_risk.pt`、`head_full.pt`、`train_report.json`），以及不完整的 `stage1_eval_user_v1/`。

## 改标签报告（`stage1_labels3_user_v1/report.json`）

| 来源 | positions | kept | dropped_onset_clause | dropped_unusable | dropped_offsets |
|---|---|---|---|---|---|
| runA_prompts_train | 321,271 | 282,982 | 38,267 | 22 | 0 |
| runA_prompts_calibration | 50,868 | 44,586 | 6,282 | 0 | 0 |
| prefix_v2_train | 310,110 | 297,442 | 12,387 | 226 | **55（0.0177%）** |
| prefix_v2_calibration | 1,384 | 1,366 | 18 | 0 | 0 |

Run A 的提问是 0，满足“Run A 必须为 0”。prefix_v2 train 是 0.0177%，超过了 0.01% 的上限。

## 原因（已定位）

用不占 GPU 的临时 Pod，在作业同一套环境下按 `relabel_cache.py` 的原逻辑复现（tokenizers 0.23.2，PVC 上的 tokenizer）。结果是 **2 条记录**，原因都是 `ids_mismatch`：冻结的 ids 和重新分词的结果差 1 个 token。

| 缓存记录 | 视图 | 位置数 | 重新分词 / 冻结 | 首个不同的 token | 原标签 | 提问侧红线标签 |
|---|---|---|---|---|---|---|
| 第 15629 条 | original | 32 | 2,497 / 2,496 | 第 908 个 | safe | safe（non_redline_harm） |
| 第 20900 条 | original | 23 | 26 / 27 | 第 21 个 | safe | **controversial**（redline） |

- 两条的增强视图（第 15630、20901 条）都照常保留。
- 第 20900 条的新标签是 controversial。丢掉它原始视图的 23 个位置，会少一些本来就少的有争议类位置：prefix_v2 user train 里一共只有 601 条有争议的提问。
- 丢掉的位置数超过上限，只是因为这两条记录的位置多，不是有很多记录对不上：prefix_v2 train 的 24,000 多条记录里只有这 2 条。

## 请设计方决定

1. 接受这 55 个位置被丢掉，照原样重跑（执行方建议这样）。改标签和训练只要几分钟，再加评测。
2. 或者改 `relabel_cache.py`：遇到 ids 不一致的记录，直接按冻结的 ids 解码出字符偏移。这样 T018 丢的 25 个位置也能找回来。

重跑前，执行方会先把 PVC 上这次的输出移到 `stopped_0318/`，因为脚本不会覆盖已有目录。
