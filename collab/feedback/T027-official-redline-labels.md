# T027 反馈：官方 thinking 集按红线口径标注

- 对应任务单版本：v1（dc79364）
- 状态：**完成**。有一项验收没过：加上兜底后 usable 为 97.1%，门槛是 98%，原因见下。另一项验收通过：Run A 和 prefix_v2 的数字与 T018 完全相同。

## 第 1 步：单测

- `round6/redline_v1/test_redline_cpu.py`：36 项全过。
- `round6/stage1_head/test_redline_stage1_cpu.py`：6 项全过。

## 第 2 步：导出

- 用不占 GPU 的临时 Pod 挂 PVC，把 `export_official.py` 拷进去（SHA `ef6a8dd7…`，与本地一致），再用系统自带的 python 运行。
- 取回 `official_thinking.jsonl` 和 manifest。两端 SHA256 一致：`b2675f58edcc9dd0c0a95a28fa42d4c4ba939b8d53fb9923e9a998677d47b2ad`。`.jsonl` 被 gitignore 忽略。
- manifest：
  - `records` 1,059，即 thinking 集里 status 为 ready 的全部条目；
  - `by_label_language`：safe:en 490，unsafe:en 569；
  - `unsafe_type`：
    - Non-violent Illegal Acts 195
    - Unethical Acts 175
    - Violent 118
    - PII 31
    - Sexual Content or Sexual Acts 22
    - Politically Sensitive Topics 18
    - Suicide & Self-Harm 7
    - Copyright Violation 3
    - safe 为空，共 490

## 第 3 步：标注

- `make_tasks.py --per-task 10`：1,059 条，106 个 Task。judge `guard-judge-redline-v4.2`，policy_digest `1f687a50f57ffaa3`，这是 T026 以后的规则表。
- 提交 **aster-dev-362**：DeepSeek，尝试 1 次，并发 106，高优先级。`runs plan` 没有 blocking，平台快照与本地一致。

### 抽取

| | Run | responses | safe | located | nonmonotonic | judge_error | 每条调用（均值 / 最大） | 交给下一步 |
|---|---|---|---|---|---|---|---|---|
| 主跑 | aster-dev-362（106 Task，0 失败） | 1,059 | 828 | 169 | 35 | 27 | 2.81 / 17 | 62 |
| luna 兜底 | aster-dev-365（7 Task，0 失败） | 62 | 11 | 20 | 27 | 4 | 8.87 / 17 | 31 条仍不可用 |

- 两个 Run 的 Task 和 flow 快照来自同一版代码（policy_digest 都是 `1f687a50…`），都在 5a21af9 之前生成、提交，所以不受 6e670e9 提醒的那个问题影响。
- 两个数据集抽取完都已删除，回执在 `dataset_rm.json`。
- **usable 为 1,028 / 1,059 = 97.1%，低于 98%。**
  - 官方 thinking 集的思考过程很长，平均每条要调用 2.8 次，兜底这批每条 8.9 次。
  - 二分定位时容易出现前后不一致，兜底后剩下的 31 条里有 27 条是这种情况。
  - 这份标签只做观察，不参与训练和阈值拟合，所以执行方照常做完了第 4–6 步。是否要补救（比如再换一个裁判重试），请设计方定。

## 第 4 步：定标签（`labels/official_thinking/summary.json`，policy_digest `53c771f3b583e841`）

- `labels_by_split`：safe 839，controversial 102，unsafe 87，不可用 31。
- `old_to_new`（旧的是官方标签）：
  - 官方 safe：→safe 401，→controversial **84**，→不可用 5；
  - 官方 unsafe：→safe **438**，→unsafe 87，→controversial 18，→不可用 26。
- `stratum_by_label`：safe 里 non_redline_harm 344、other_sensitive 311、normal 147、redline_topic 37；controversial 和 unsafe 都是 redline。
- `rules_non_safe` 前几名：
  - R2:restates 36，R2:refusal_naming 19，R2:writes 14；
  - B2:writes 33，B2:refusal_naming 7，B2:restates 7；
  - R12:writes 15；
  - R8:restates 12，R8:writes 11，R8:refusal_naming 10；
  - R11:restates 10；
  - B1:writes 9，B1:refusal_naming 8，B1:restates 7。
- `texts.length_mismatch` 为 0。
- `labels_political_v1/official_thinking` 没有出：T026 的政治词表还没通过核对。

## 第 5 步：人工核对（本地，不写原文）

`review_sample.py --count 30`，按官方标签到我们标签的变化分组，每组 6 条：safe→controversial、safe→safe、unsafe→controversial、unsafe→safe、unsafe→unsafe。

| 结果 | 条数 |
|---|---|
| 对 | 22 |
| 误判红线 | 2 |
| 漏判红线 | 0 |
| 档位错 | 0 |
| 口径存疑 | 6 |
| 剔除存疑后的正确率 | 22 / 24 = 91.7% |

- **误判红线 2 条**：都是请求讲“荤段子”，回答里的笑话并不露骨。裁判按 R11 的复述或拒绝点名判成了“有争议”，但请求本身并不露骨，不是红线内容。
- **口径存疑 6 条**：
  - 2 条是教人伤害动物的请求：裁判按 B2 判 unsafe，但 B2 原本针对的是对人下手。
  - 2 条是以种族为对象的内容：回答侧裁判没有 B7，还是按 R8 判的。B7 目前只在提问侧记录，按 B7 不算红线的话应为 safe。
  - 1 条是表达轻生念头、并询问方法的请求：回答是支持性的，给了热线；但思考过程复述了请求。“给热线的支持性回复放行”和“复述算有争议”两条规则在这里冲突。
  - 1 条是为了逃课谎称有爆炸物：属于假的公共安全警报。算不算 R10 或 R12，不确定。
- **起点分句**：10 条标签正确的非 safe 里，6 条对、4 条偏晚。偏晚的几条都是思考过程第一句就复述了请求，但裁判到后面拒绝或引用的那一句才开始判。
- **官方 safe → 我们 controversial 的 6 条**：
  - 4 条确实在思考过程里复述或引用了红线内容，涉及投毒、群体伤害、露骨的性请求和轻生方法；
  - 2 条就是上面说的“荤段子”误判。

## 第 6 步：按我们的标签重算官方集

### T018（`results_eval_v2`）

- Run A 和 prefix_v2 各项与 T018 的 `compare_redline_v1.json` 逐项比较，**0 处不同** ✓。
- 输出全文：

```
init: T=6.7  stream AUC: runA_calibration 0.598, prefix_v2_calibration 0.728, runA_dev 0.597, prefix_v2_dev 0.743, official_thinking 0.633
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     0.0%    0.0%   0.0%   0.0%    n/a |    0.4%   0.9% |    1.1%   2.6%
consecutive_2  |     0.0%    0.0%   0.0%   0.0%    n/a |    0.3%   0.0% |    0.2%   1.6%
ema_0.1        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |    0.0%   0.0%
ema_0.3        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |    0.0%   0.0%
official by our labels (threshold rule): official_safe:controversial n=84 fired=0.0%, official_safe:safe n=401 fired=0.0%, official_unsafe:controversial n=18 fired=0.0%, official_unsafe:safe n=438 fired=2.1%, official_unsafe:unsafe n=87 fired=5.8%

risk: T=2.1  stream AUC: runA_calibration 0.738, prefix_v2_calibration 0.795, runA_dev 0.744, prefix_v2_dev 0.816, official_thinking 0.784
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     2.6%    0.9%   8.9%   7.8%  20.3% |    1.9%  11.7% |    3.9%  17.5%
consecutive_2  |     1.0%    0.2%   3.2%   4.1%  20.7% |    1.7%  10.8% |    3.8%  18.0%
ema_0.1        |     0.1%    0.0%   1.6%   1.5%  16.7% |    2.3%  13.5% |    4.6%  20.6%
ema_0.3        |     0.6%    0.2%   3.2%   3.7%  14.8% |    2.2%  13.5% |    4.4%  21.7%
official by our labels (threshold rule): official_safe:controversial n=84 fired=14.3%, official_safe:safe n=401 fired=2.5%, official_unsafe:controversial n=18 fired=5.6%, official_unsafe:safe n=438 fired=5.2%, official_unsafe:unsafe n=87 fired=23.0%

full: T=2.1  stream AUC: runA_calibration 0.832, prefix_v2_calibration 0.855, runA_dev 0.832, prefix_v2_dev 0.837, official_thinking 0.819
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     1.9%    0.0%  28.9%  16.3%   3.0% |    1.7%  23.4% |    8.9%  45.0%
consecutive_2  |     2.7%    0.0%  36.5%  18.7%   4.5% |    2.3%  32.4% |   13.2%  54.5%
ema_0.1        |     1.5%    0.0%  30.5%  14.6%   3.1% |    1.9%  31.5% |   10.6%  51.3%
ema_0.3        |     2.7%    0.0%  39.4%  21.6%   4.9% |    2.5%  33.3% |   12.2%  55.6%
official by our labels (threshold rule): official_safe:controversial n=84 fired=52.4%, official_safe:safe n=401 fired=14.0%, official_unsafe:controversial n=18 fired=16.7%, official_unsafe:safe n=438 fired=4.3%, official_unsafe:unsafe n=87 fired=43.7%
```

### T028（v3a full，按卡片另跑一次）

```
full: T=2.4  stream AUC: runA_calibration 0.839, prefix_v2_calibration 0.883, runA_dev 0.850, prefix_v2_dev 0.860, official_thinking 0.812
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     2.5%    0.0%  28.2%  20.7%   4.9% |    2.5%  35.1% |   19.2%  61.4%
consecutive_2  |     1.5%    0.0%  25.1%  19.4%   5.4% |    2.8%  32.4% |   15.1%  61.9%
ema_0.1        |     0.8%    0.0%  21.3%  13.5%   2.3% |    2.8%  38.7% |   10.8%  56.1%
ema_0.3        |     1.2%    0.0%  24.1%  16.8%   3.9% |    2.3%  37.8% |   12.0%  58.7%
official by our labels (threshold rule): official_safe:controversial n=84 fired=56.0%, official_safe:safe n=401 fired=22.7%, official_unsafe:controversial n=18 fired=38.9%, official_unsafe:safe n=438 fired=16.0%, official_unsafe:unsafe n=87 fired=71.3%
```

### 执行方的读数（只陈述事实）

- **按我们的标签，官方集上的新头并没有反着判。**
  - T018 full 的流式 AUC 从 0.32（官方标签）变成 **0.82**（我们的标签），召回 45–56%；
  - risk 是 0.78，init 是 0.63。
- **口径差异能解释一部分误报，但不是全部。** 在“官方 safe、我们也判 safe”的 401 条上，阈值规则下仍然截断：
  - T018 full 截了 **14.0%**；
  - T028 v3a full 截了 **22.7%**；
  - init 截了 0。

  另外，“官方 unsafe、我们判 safe”的 438 条（红线外的有害内容）上，T018 full 截了 4.3%，T028 截了 16.0%。
- 按我们的标签，官方集上的 FPR（threshold / consecutive_2 / ema_0.1 / ema_0.3）：
  - T018 full：8.9% / 13.2% / 10.6% / 12.2%；
  - T028 full：19.2% / 15.1% / 10.8% / 12.0%。
- T028 训练更久，Run A 和 prefix_v2 的数字差不多，但在官方集的 safe 上误报更多（14.0% → 22.7%）。

## 产物 SHA256

| 文件 | SHA256 |
|---|---|
| `input/official_thinking.manifest.json` | `7b31c759c75e4306081eb4cc630eb5f2576623d2d1de05071b1eaae4a461efa8` |
| `labels/official_thinking/summary.json` | `d8c38b06691f9fc3f3b31e1ea63d6f4188d3572050b14e76bd41e18b8003b917` |
| `stage1_head/results_eval_v2/compare_redline_official.json` | `457d19e531a8f9cae538461eeb597854c5e8f915ab295f1cb7623f988f6afbe4` |
| `stage1_head/results_eval_v3a/compare_redline_official.json` | `1e433c04d83d926de087d09ce4bb13265d4a30b3a47363f65bb545581a88153e` |

`official_thinking.jsonl`、`labels.jsonl` 和 `review/` 都含原文，不提交。
