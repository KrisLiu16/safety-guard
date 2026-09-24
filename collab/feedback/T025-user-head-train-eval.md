# T025 反馈：完成（v1.1 重跑）

- 对应任务单版本：v1.1（dc79364）
- 状态：**完成，验收通过**。v1 那次在 03:18 按 0.01% 的规则停下，经过见下文“v1 那次”。v1.1 改为按记录数判断，接受了这 2 条记录；03:49 重跑，04:03 完成。结果见文末。

## v1 那次（03:17–03:18，已停）

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

## v1.1 重跑（03:49 – 04:03）

- 03:48 用临时 Pod 把 v1 的输出移到 `/work/output/round6/stopped_0318/`，并撤掉了就绪标记。
- 单测全过。重新提交作业，Pod 为 `safety-guard-stage1-user-redline-v1-r1-9xplj`，申请 7 份 GPU。
- 拷入**当前** `round6/stage1_head/*.py`，dc79364 改过 `train_head.py` 和 `analyze_redline.py`。21 个文件核对一致，03:49:01 写入 `input_ready`。
- 改标签结果和 v1 相同：prefix_v2 train 的 `dropped_offsets` 为 55，还是那 2 条 ids 对不上的记录；其余来源都是 0，符合 v1.1。
- 04:03 完成，`exit_code` 为 `relabel=0 train=0 eval=0`。取回 17 个文件，都和 PVC 上的 SHA 一致，已写 collected 标记。

### 验收

| 项 | 结果 |
|---|---|
| 三步 exit code | 全为 0 ✓ |
| `dropped_offsets` | Run A 提问为 0；prefix_v2 train 为 2 条记录共 55 个位置，符合 v1.1 ✓ |
| 评测 `integrity_pass` | true ✓ |
| `max_prob_diff` | 6.26e-7 < 0.001 ✓ |
| role | user；评测用时 720 秒 |

### 改标签 `old_to_new_class`（0 = safe，1 = unsafe，2 = controversial）

- runA_prompts_train：0→0 139,125，0→1 2,253，0→2 2,572，1→0 131,647，1→1 7,247，1→2 138。
- runA_prompts_calibration：0→0 22,181，0→1 333，0→2 443，1→0 20,458，1→1 1,163，1→2 8。
- prefix_v2_train：0→0 242,147，0→1 2,163，0→2 4,704，1→0 39,278，1→1 9,007，1→2 143。
- prefix_v2_calibration：0→0 1,202，0→1 14，1→0 115，1→1 35。

### 训练（`train_report.json`）

- role `user`，`calibration_score` 为 `cut`，`sources_left_out` 为 `[]`（来源只有 prefix_v2 和 runA_prompts）。
- `train_positions_by_class`（safe / unsafe / controversial）：prefix_v2 281,425 / 11,170 / 4,847；runA_prompts 270,772 / 9,500 / 2,710。
- calibration positions：prefix_v2 1,366，runA_prompts 44,586。

| 变体 | chosen_epoch | chosen_calibration_mean_auc | 每轮的 calibration AUC（prefix_v2 / runA_prompts） |
|---|---|---|---|
| risk | 4 | 0.8293 | 0：0.776 / 0.578；1：0.812 / 0.764；2：0.825 / 0.803；3：0.835 / 0.797；4：0.838 / 0.820 |
| full | 4 | 0.8956 | 0：0.775 / 0.578；1：0.856 / 0.851；2：0.878 / 0.840；3：0.880 / 0.880；4：0.893 / 0.898 |

### 评测 counts（三个头相同）

runA_prompts_calibration 2,526（跳过 2 条），runA_prompts_dev 2,594，prefix_v2_calibration 300，prefix_v2_dev 400。

### `analyze_redline.py --role user` 输出（全文）

```
init: T=10.0  stream AUC: runA_calibration 0.587, prefix_v2_calibration 0.752, runA_dev 0.580, prefix_v2_dev 0.765
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |     n/a    n/a
consecutive_2  |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |     n/a    n/a
ema_0.1        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |     n/a    n/a
ema_0.3        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |     n/a    n/a

risk: T=10.0  stream AUC: runA_calibration 0.517, prefix_v2_calibration 0.795, runA_dev 0.528, prefix_v2_dev 0.781
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |     n/a    n/a
consecutive_2  |     1.8%    3.0%   9.0%   1.0%   0.0% |    2.0%   4.5% |     n/a    n/a
ema_0.1        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.6%   0.0% |     n/a    n/a
ema_0.3        |     0.4%    0.5%   1.4%   0.0%   0.0% |    1.1%   2.3% |     n/a    n/a

full: T=5.2  stream AUC: runA_calibration 0.771, prefix_v2_calibration 0.838, runA_dev 0.767, prefix_v2_dev 0.853
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     1.5%    0.9%   9.0%   9.3%   2.4% |    2.0%  13.6% |     n/a    n/a
consecutive_2  |     2.0%    1.4%  13.1%  10.0%   0.0% |    2.0%  13.6% |     n/a    n/a
ema_0.1        |     0.1%    0.1%   3.5%   1.7%   0.0% |    2.8%  22.7% |     n/a    n/a
ema_0.3        |     0.4%    0.3%   4.8%   5.0%   0.0% |    1.4%   9.1% |     n/a    n/a
```

按分层的 FPR：

| 头 · 规则 | Run A dev：normal / non_redline_harm / redline_topic / other_sensitive | prefix_v2 dev：同左 |
|---|---|---|
| init · 各规则 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| full · threshold | 0.9% / 1.3% / 3.3% / 3.5% | 0 / 2.3% / 10.5% / 10.0% |
| full · consecutive_2 | 1.4% / 1.8% / 3.3% / 4.4% | 0 / 3.1% / 10.5% / 5.0% |
| full · ema_0.1 | 0.1% / 0.2% / 0 / 0 | 0 / 5.5% / 10.5% / 5.0% |

prefix_v2 dev 各分层的条数少：normal 189、non_redline_harm 128、redline_topic 19、other_sensitive 20。

### 执行方的读数（只陈述事实，结论请设计方下）

- **init（现在的用户头）** 的 stream AUC 为 Run A 0.58–0.59、prefix_v2 0.75–0.77。按“每个校准分层 FPR ≤ 5%”定阈值后（T=10.0），任何规则都不触发，召回和误报都是 0。
- **full** 最好，但召回低：
  - stream AUC 为 Run A 0.77、prefix_v2 0.84–0.85；
  - Run A 有争议的召回 3.5–13.1%，风险的召回 1.7–10.0%；prefix_v2 召回 9.1–22.7%；
  - Run A 的 normal 层误报 0.1–1.4%，prefix_v2 的 normal 为 0。
- **risk** 的 Run A 流式 AUC 只有 0.52–0.53，基本分不开。
- 对照卡片的通过标准：
  - normal 层误报不高于现在的头：现在的头是 0，full 在 Run A 上是 0.1–1.4%，**严格说不满足**；
  - non_redline_harm 误报明显下降：现在的头本来就是 0，无从比较；
  - 召回不低于现在的头：满足，但现在的头是 0。
- 位置级 AUC 在第 4 轮还在涨（full 0.8956），和 T018 一样，可能没训练够。T028 正在答这个问题，不过只针对回答侧。

### 产物 SHA256

| 文件 | SHA256 |
|---|---|
| `results_labels3_user_v1/report.json` | `b14d9639d604473f17186705d467aa4434fb6d7bebedeb1c20983b9d9a20b13f` |
| `results_train_user_v1/train_report.json` | `78e56410ba4d46fcd93385f96173e082b4e69fe70de25ad52a00b11692dda6fe` |
| `results_train_user_v1/head_full.pt` | `c07c87e3cf2aa1af347877e526a5d8d2541c22895714d648356488e984c873f4` |
| `results_train_user_v1/head_risk.pt` | `2698a3ae10de44d987d5e47ce6e314e781886b2bfb854b2d9d2eaf289e367345` |
| `results_eval_user_v1/report.json` | `ba9a53830158fee6879543dbe9ac7c29d32a8b92fc78a105a60394c740f783ca` |
| `results_eval_user_v1/compare_redline_v1.json` | `e98c55a7ba72307a77bfaf8fda0b98f9576d65b88858f31476384f810a5b1ece` |

权重和逐位置的 `.jsonl.gz` 只留在 Mac 和 PVC 上，不提交。init 头的 SHA `b2b1efb4…` 与 T023 的 `head_init_sha256` 一致。
