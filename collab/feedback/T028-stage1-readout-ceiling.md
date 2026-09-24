# T028 反馈：第一阶段读出上限诊断

- 对应任务单版本：v1（dc79364）
- 状态：**完成，验收通过**。
- 执行时间：2026-09-25 04:04 – 04:41（北京时间）

## 经过

- 单测：`test_stage1_cpu.py` 和 `test_redline_stage1_cpu.py` 全过。
- T025 取回以后，04:04 `kubectl apply -f round6/stage1_head/worker_readout_v3.yaml`，申请 7 份 GPU。Pod 为 `safety-guard-stage1-readout-v3-r1-k69pf`。
- 拷入 `round6/stage1_head/*.py` 9 个和 `round6/probe/*.py` 9 个，18 个文件都用 `sha256sum -c` 核对一致。PVC 上的 `stage1_labels3_v2` 还在，是 T018 的三类目标。04:04:37 写入 `input_ready`。
- 04:40 出现 done 标记，`exit_code` 为 `train_a=0 train_b=0 eval=0`。
- 取回 12 个文件：2 份 train_report、3 个 head、评测 report，以及 `eval_full/` 里的 6 个文件。都和 PVC 上的 SHA 一致，已写 collected 标记。

## 验收

| 项 | 结果 |
|---|---|
| exit code | 全为 0 ✓ |
| 评测 `integrity_pass` | true ✓ |
| `max_prob_diff` | 1.07e-6 < 0.001 ✓ |
| 两份 `sources_left_out` | 都是 `["s2", "s5"]` ✓ |
| wide 的 `deployable` | false ✓（v3a、v3b 的 full 都是 true） |
| 评测 counts | runA_calibration 5,052（跳过 4），runA_dev 5,188，prefix_v2_calibration 600，prefix_v2_dev 800，official_unique_sequences 1,872 |
| 评测用时 | 2,010 秒 |

## 三个训练结果（每轮的 calibration AUC，格式为 prefix_v2 / runA）

| 结果 | 设置 | chosen_epoch | chosen_calibration_mean_auc |
|---|---|---|---|
| v3a full | 16 轮，lr 1e-3，l2_to_init 1e-4 | 12 | **0.8740** |
| v3a wide | 16 轮，1024→2048→512→3，全新初始化 | 9 | **0.8729** |
| v3b full | 16 轮，lr 3e-3，不加 L2 | 12 | **0.8707** |
| 对照：T018 full | 4 轮 | 4 | 0.8650 |

- v3a full：
  - 第 0–4 轮：0.687/0.512，0.843/0.833，0.856/0.851，0.866/0.860，0.866/0.864；
  - 第 5–8 轮：0.860/0.861，0.856/0.867，0.863/0.856，0.870/0.872；
  - 第 9–12 轮：0.871/0.869，0.858/0.867，0.861/0.875，**0.875/0.873**；
  - 第 13–16 轮：0.868/0.872，0.872/0.875，0.867/0.872，0.841/0.872。
- v3a wide：
  - 第 0–4 轮：0.582/0.594，0.846/0.837，0.854/0.852，0.856/0.857，0.864/0.867；
  - 第 5–8 轮：0.865/0.868，0.864/0.870，0.862/0.870，0.872/0.870；
  - 第 9–12 轮：**0.876/0.870**，0.856/0.873，0.857/0.877，0.870/0.875；
  - 第 13–16 轮：0.862/0.877，0.869/0.873，0.864/0.875，0.854/0.877。
- v3b full：
  - 第 0–4 轮：0.687/0.512，0.843/0.840，0.856/0.852，0.857/0.852，0.861/0.866；
  - 第 5–8 轮：0.867/0.866，0.861/0.866，0.860/0.865，0.866/0.866；
  - 第 9–12 轮：0.865/0.870，0.859/0.871，0.859/0.875，**0.872/0.869**；
  - 第 13–16 轮：0.865/0.872，0.866/0.875，0.866/0.873，0.852/0.876。

## `analyze_redline.py --variants full` 输出（全文，v3a full）

```
full: T=2.4  stream AUC: runA_calibration 0.839, prefix_v2_calibration 0.883, runA_dev 0.850, prefix_v2_dev 0.860, official_thinking 0.416
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     2.5%    0.0%  28.2%  20.7%   4.9% |    2.5%  35.1% |   29.0%  27.6%
consecutive_2  |     1.5%    0.0%  25.1%  19.4%   5.4% |    2.8%  32.4% |   21.4%  28.3%
ema_0.1        |     0.8%    0.0%  21.3%  13.5%   2.3% |    2.8%  38.7% |   12.4%  27.6%
ema_0.3        |     1.2%    0.0%  24.1%  16.8%   3.9% |    2.3%  37.8% |   16.9%  26.2%
```

runA_dev 按分层的 FPR（threshold / ema_0.1）：normal 0 / 0，non_redline_harm 2.0% / 0.6%，redline_topic 5.4% / 1.2%，other_sensitive 5.1% / 2.0%。

## 执行方的读数（只陈述事实，结论请设计方下）

- **三种做法都停在 0.87 左右。**
  - 训练 16 轮只比 T018 的 4 轮高 0.009，最好的一轮在第 12 轮。
  - 换成大一倍的全新读出（wide）是 0.873。
  - 调大学习率、去掉 L2（v3b）是 0.871。
  - 第 4 轮以后，每轮之间的起伏（约 ±0.01）和总的提升差不多大。
- **和 T018 的 full（4 轮）比较，评测结果有得有失：**
  - 流式 AUC：runA_dev 从 0.832 升到 0.850，prefix_v2_dev 从 0.837 升到 0.860；
  - Run A 召回：有争议的从 29–39% 降到 21–28%，风险的从 15–22% 变成 14–21%；
  - prefix_v2 召回：从 23–33% 升到 32–39%；
  - normal 仍然是 0；
  - 官方 thinking 集：AUC 0.42（T018 是 0.32），FPR 12–29%（T018 是 21–27%）。
- 按卡片的说法，这个结果属于“都停在 0.87 左右”，也就是瓶颈在主干最后一层的特征上。下一步是否进入第二阶段（训练主干），需要用户批准。
- T027 的官方集标签出来以后，执行方会按卡片再跑一次 `--official-labels`，输出到 `compare_redline_official.json`。

## 产物 SHA256

| 文件 | SHA256 |
|---|---|
| `results_train_v3a/train_report.json` | `2d2e345cbaa333cbb026d225cfc0ade7e6220b73beb3d9eeb47f660a2aae6027` |
| `results_train_v3a/head_full.pt` | `3600bbd170b7b2d75034af021c271e045c4bf73e98684c09251ed9079021e4e9` |
| `results_train_v3a/head_wide.pt` | `71940802903f57315c1230787644723494141a93be8d19e5106c603d29231963` |
| `results_train_v3b/train_report.json` | `c86ba59214f93484ce870731743bcb9a737b49549f61a215f39d185044842ef2` |
| `results_train_v3b/head_full.pt` | `06eb2ff8d592f544010b0c62b49c7ec98ca6f63869702acd3636e12237e9f829` |
| `results_eval_v3a/report.json` | `a28f07ceda7fe463077e4150588c1d04697769c6ec9f9d81fdc55a71e8d92e8c` |
| `results_eval_v3a/compare_redline_v1.json` | `f1fb639972f87daf39b9ff8b7e97473f9c11dd2a7dbb2114f6f67947874dc261` |

权重和逐位置的 `.jsonl.gz` 只留在 Mac 和 PVC 上，不提交。
