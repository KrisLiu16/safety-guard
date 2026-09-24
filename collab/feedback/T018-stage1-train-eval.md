# T018 反馈（v3）：评测进行中（接受 dropped_offsets = 25 后重跑）

- 对应任务卡：T018 v3.1（df6255e）。25 个位置占 prefix_v2 train 的 0.0077%，低于 0.01%，Run A 两个来源都是 0，符合 v3.1 第 2 步。卡片说按原 yaml 重新提交，但用户指定用 7 份 GPU，按用户的来。
- 状态：**完成，验收通过**。03:15 出现 done 标记，`exit_code` 为 `relabel=0 train=0 eval=0`。03:16 取回结果并写了 collected 标记，对比分析结果见文末“结果”一节。
- 01:09 第一次跑时，因为 prefix_v2 train 的 `dropped_offsets` 是 25（不是 0），按卡片停下报告。原因已经定位，见下文：只涉及 1 条 safe 记录。
- 用户 01:3x 指示继续，原话：
  > 继续完成你的训练工作吧 用7份就行了

  执行方理解为接受这 25 个位置被丢掉，重跑。
- **GPU 申请改成 7 份。** 两个 yaml（`worker_redline_v2.yaml`、`worker_user_redline_v1.yaml`）的 `nvidia.com/gpu` 从 8 改为 7。这台节点只有 1 块 L20，按 `timeslice-8x` 分成 8 份，剩下 1 份留给另一个会话部署模型做测试，是用户安排的。
  - 两边共用同一块物理卡，显存不隔离，算力分时轮流用，所以评测会比独占时慢一些。
  - 执行方已经提交了这个改动。
- 重跑前的准备：
  - 用临时 Pod 把上次的输出移到 `/work/output/round6/stopped_0109/`，撤掉了就绪标记，因为脚本不会覆盖已有的输出目录。
  - 01:34 重新提交作业，Pod 为 `safety-guard-stage1-redline-v2-r1-v4gr9`。
  - 21 个输入文件在 PVC 上重新核对，SHA 都一致，和 01:03 拷进去时相同。01:34:41 写入 `input_ready`。
- 改标签的结果和第一次完全相同：报告计数一样，4 个 `labels3_*.npz` 的 SHA 也一样。训练在 01:36 完成。
- 单测：`test_stage1_cpu.py` 和 `test_redline_stage1_cpu.py` 共 14 项，全过。

## 经过

- 2026-09-24 23:59 第一次提交作业，把代码和 prefix_v2 标签拷进 Pod，等 Run A 标签。
  - 这个 Job 在 01:02 之前被删掉了，不是执行方删的，见下面“另一个会话”。
- 01:02 重新提交。拷入并用 `sha256sum -c` 核对 21 个文件，全部一致：
  - `round6/stage1_head/*.py` 9 个；
  - `round6/probe/*.py` 9 个；
  - `levels.py`；
  - 两份 labels.jsonl：runA_stage1 `3c4a71b8…`，prefix_v2 `7b419ffb…`。

  PVC 上的 `input_v1/stage1_runA_v1.jsonl` 的 SHA 为 `4e4afa5d…`，与 manifest 一致。01:03 写入 `input_ready`。
- 01:03 和 01:05 两次都在作业里的 `apt-get update` 失败，退出码 100。报错是 `File has unexpected size (1318479 != 1320963). Mirror sync in progress?`，原因是 Ubuntu 的 security 源正在同步。用不占 GPU 的临时 Pod 确认源恢复以后，01:07 按 git 里提交的 yaml 重新提交。
- 01:07 开始改标签，01:08 训练完成，随后进入评测。执行方读到改标签报告里的 `dropped_offsets=25`，在 01:09 删除 Job，评测中止。
  - 改标签和训练的产物都留在 PVC：`/work/output/round6/stage1_labels3_v2/`、`stage1_train_v2/`。

## 改标签报告（`stage1_labels3_v2/report.json`）

| 来源 | positions | kept | dropped_onset_clause | dropped_unusable | dropped_offsets |
|---|---|---|---|---|---|
| runA_train | 686,727 | 660,365 | 24,736 | 1,626 | 0 |
| runA_calibration | 108,233 | 104,215 | 3,840 | 178 | 0 |
| prefix_v2_train | 326,460 | 321,367 | 4,537 | 531 | **25** |
| prefix_v2_calibration | 2,973 | 2,938 | 34 | 1 | 0 |

`old_to_new_class`（0 = safe，1 = unsafe，2 = controversial）：

- runA_train：0→0 465,755，0→2 25,110，0→1 9,000，1→0 141,551，1→1 18,116，1→2 833。
- runA_calibration：0→0 73,852，0→2 3,951，0→1 1,412，1→0 21,842，1→1 3,061，1→2 97。
- prefix_v2_train：0→0 272,627，0→2 4,764，0→1 2,045，1→0 33,184，1→1 8,109，1→2 638。

## dropped_offsets = 25 的原因（已定位）

在临时 Pod（不占 GPU）上，用作业同一套 Python（tokenizers 0.23.2）、PVC 上的 tokenizer（SHA `fe000e3e…`）和 prefix_v2 数据，按 `relabel_cache.py` 的原逻辑复现，结果如下：

- 只有 **1 条记录**：`cache_prefix_v2_train_records.json` 的第 22510 条。
  - 来源 rubric_benign，原标签 safe，中文，`view=original`。
  - 这 25 个位置就是它的全部位置。
- 原因是 `ids_mismatch`：把 messages 重新分词得到 57 个 token，冻结的 `ids` 有 59 个，从第 16 个 token 开始不同。缓存用的是冻结的 ids，所以这条记录的字符偏移无法靠重新分词找回来。
- 这条的红线标签是 safe（other_sensitive 分层），它的增强视图（第 22511 条）照常保留。所以影响只是少了 25 个 safe 位置，占 prefix_v2 train 的 0.008%。
- 其余 24,581 条记录都对得上。

**执行方建议**：接受这 25 个位置被丢掉，重新提交作业，改标签加训练只要几分钟，再跑完评测。如果要求严格为 0，就需要设计方改 `relabel_cache.py`，比如对 ids 不一致的记录按冻结 ids 解码出偏移。请设计方决定。

## 训练报告（已生成，评测前）

- version `round6-stage1-head-v2-redline`，role `assistant`，`calibration_score` 为 `cut`，`sources_left_out` 为 `["s2", "s5"]`，与卡片一致。
- `train_positions_by_class`：
  - prefix_v2：safe 305,811 / unsafe 10,154 / controversial 5,402；
  - runA：safe 607,306 / unsafe 27,116 / controversial 25,943。
- risk 变体：`chosen_epoch` 4，`chosen_calibration_mean_auc` 0.7932。
  - 每个 epoch 的 calibration AUC 均值：0.599、0.766、0.782，后面几个 epoch 的数字等作业重跑后一并报告。
- full 变体：等重跑后一并报告。

## 另一个会话

- 本机还有一个会话“Safety-guard GitHub 项目 (fork)”在同一目录里运行。
- 00:07:44 有人把 `worker_redline_v2.yaml` 和 `worker_user_redline_v1.yaml` 的 GPU 申请从 8 改成了 1，没有提交。用户看到后问这是什么意思；执行方这个会话没有改过这两个文件。
- 这台节点只有 1 块 L20，按 `timeslice-8x` 分成 8 份：申请 8 份就是独占整卡，申请 1 份则会和别的作业共用这块卡。
- 执行方重新提交时用的是 git 里提交的版本（8 份），本地这两个文件没有动，等用户决定。

## 结果（01:34 那次重跑）

### 验收

| 项 | 结果 |
|---|---|
| 三步 exit code | `relabel=0 train=0 eval=0` ✓ |
| `dropped_offsets` | prefix_v2 train 25（0.0077%，1 条记录），其余都是 0，符合 v3.1 ✓ |
| 评测 `integrity_pass` | true ✓ |
| `max_prob_diff` | 8.05e-7 < 0.001 ✓ |
| 官方集 unique 序列数 | 1,872 ✓ |
| 官方集 native ids | passed，2,441 行，mismatched 0 |

- 评测用时 `elapsed_seconds` 5,899（约 98 分钟），共 40,536 个序列，7,465,824 个前向 token。
- `train_report.json` 的 `version` 为 `round6-stage1-head-v2-redline`，`calibration_score` 为 `cut`，`sources_left_out` 为 `["s2", "s5"]`。

### 训练

`train_positions_by_class`（0 = safe，1 = unsafe，2 = controversial）：

- prefix_v2：305,811 / 10,154 / 5,402；
- runA：607,306 / 27,116 / 25,943。

calibration positions：prefix_v2 2,938，runA 104,215。

| 变体 | chosen_epoch | chosen_calibration_mean_auc | 各 epoch 的 calibration AUC 均值（0–4） |
|---|---|---|---|
| risk | 4 | 0.7932 | 0.599、0.766、0.782、0.790、0.793 |
| full | 4 | 0.8650 | 0.599、0.838、0.853、0.863、0.865 |

epoch 4 时分开看：risk 为 prefix_v2 0.794、runA 0.792；full 为 0.866、0.864。两个变体到最后一个 epoch 还在涨。

### 评测 counts（三个头相同）

runA_calibration 5,052（跳过 4 条），runA_dev 5,188，prefix_v2_calibration 600，prefix_v2_dev 800，official_unique_sequences 1,872。

### `analyze_redline.py` 输出（全文）

```

init: T=6.7  stream AUC: runA_calibration 0.598, prefix_v2_calibration 0.728, runA_dev 0.597, prefix_v2_dev 0.743, official_thinking 0.779
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     0.0%    0.0%   0.0%   0.0%    n/a |    0.4%   0.9% |    0.0%   2.5%
consecutive_2  |     0.0%    0.0%   0.0%   0.0%    n/a |    0.3%   0.0% |    0.0%   0.9%
ema_0.1        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |    0.0%   0.0%
ema_0.3        |     0.0%    0.0%   0.0%   0.0%    n/a |    0.0%   0.0% |    0.0%   0.0%

risk: T=2.1  stream AUC: runA_calibration 0.738, prefix_v2_calibration 0.795, runA_dev 0.744, prefix_v2_dev 0.816, official_thinking 0.470
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     2.6%    0.9%   8.9%   7.8%  20.3% |    1.9%  11.7% |    4.7%   9.3%
consecutive_2  |     1.0%    0.2%   3.2%   4.1%  20.7% |    1.7%  10.8% |    4.7%   9.1%
ema_0.1        |     0.1%    0.0%   1.6%   1.5%  16.7% |    2.3%  13.5% |    4.1%  12.1%
ema_0.3        |     0.6%    0.2%   3.2%   3.7%  14.8% |    2.2%  13.5% |    4.7%  11.4%

full: T=2.1  stream AUC: runA_calibration 0.832, prefix_v2_calibration 0.855, runA_dev 0.832, prefix_v2_dev 0.837, official_thinking 0.324
rule           | RunA FPR  normal  R-con  R-uns  early | pv2 FPR      R | off FPR      R
threshold      |     1.9%    0.0%  28.9%  16.3%   3.0% |    1.7%  23.4% |   20.8%  12.1%
consecutive_2  |     2.7%    0.0%  36.5%  18.7%   4.5% |    2.3%  32.4% |   26.7%  16.7%
ema_0.1        |     1.5%    0.0%  30.5%  14.6%   3.1% |    1.9%  31.5% |   21.4%  16.5%
ema_0.3        |     2.7%    0.0%  39.4%  21.6%   4.9% |    2.5%  33.3% |   26.1%  16.2%
```

runA_dev 按分层的 FPR（阈值规则 / ema_0.1）：

| 头 | normal | non_redline_harm | redline_topic | other_sensitive |
|---|---|---|---|---|
| init | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| risk | 0.9% / 0 | 2.2% / 0.1% | 6.0% / 0.3% | 3.8% / 0.3% |
| full | 0 / 0 | 1.1% / 0.7% | 4.5% / 2.7% | 4.5% / 4.1% |

### 执行方的读数（只陈述事实，结论请设计方下）

- **init（现在的头）** 在新标签下几乎分不开，stream AUC 为 Run A 0.60、prefix_v2 0.73–0.74。按“每个校准分层 FPR ≤ 5%”定出的阈值（T=6.7）实际上不会触发：Run A 召回 0，prefix_v2 召回 0.9%。
- **full** 最好：
  - stream AUC 为 Run A 0.83、prefix_v2 0.84–0.86；
  - normal 层误报为 0；
  - 有争议的召回 29–39%，风险的召回 15–22%，prefix_v2 召回 23–33%；
  - 早截（在有争议的起点分句之前就截断）只占 3–5%。
- **risk** 召回低：Run A 有争议 2–9%、风险 2–8%，prefix_v2 11–14%。早截也多，占 15–21%。
- **官方 thinking 集要注意。** 这个集只观察、不参与拟合，标签是另一套口径。
  - full 在上面的 stream AUC 是 0.32，低于 0.5；risk 是 0.47；init 是 0.78。
  - full 的官方集 FPR 是 20.8–26.7%，init 是 0。
  - 也就是说，新头在官方集的 safe 思考过程上误报很多。官方集的 safe 和我们的口径不完全一样，但这和“正常内容绝不能误判”的要求有冲突，建议设计方看一下误报的是哪一类内容。
- 两个新头在 epoch 4 时 AUC 还在上升，训练轮数可能不够。

### 产物 SHA256

| 文件 | SHA256 |
|---|---|
| `results_labels3_v2/report.json` | `b268bd0922404fff3d0936e6e9fbacde360dfe4b65cf650e8dc87cfed9d54959` |
| `results_train_v2/train_report.json` | `99eddc809fbcd3cce462d15bfe290a4676d2cb9f2e453269994f543101e78f26` |
| `results_train_v2/head_full.pt` | `add8645c3e5b09c1b938711da83a1e8c722870980b2388b6e9c334a4901d5cf1` |
| `results_train_v2/head_risk.pt` | `81100f848e5b4b931a0802252eea6afcd7c350329a5b3ba0ef1ee83fc3b93747` |
| `results_eval_v2/report.json` | `4c3a498d8e7bfee9ad0c6df2588916982ae0df757146448848a84c7102c50bb6` |
| `results_eval_v2/compare_redline_v1.json` | `736ebcf339ef7690154b8c8e66f5b0df5f1ae8383f6eed88c743e508af0add7f` |

- 取回的 23 个文件都和 PVC 上的 SHA 一致：3 个 report、train_report、2 个 head，以及 3 个 `eval_*/` 目录里的 18 个文件。
- 权重和逐位置的 `.jsonl.gz` 只留在 Mac 和 PVC 上，被 gitignore 忽略，不提交。
- init 头的 SHA 是 `166f6432…`，与 `init_sha256` 一致。

---

# 以下为 v2 时的记录（已被 v3 取代）

# T018 反馈：暂缓，用户更正了“复述”的口径

- 对应任务卡：T018 v2
- 状态：**未提交作业**。T017 的缓存只做推理，与标签口径无关，所以继续跑。T017 完成后，按它的步骤验收、取回，并标记 collected，然后停下，不 apply `worker_train_eval_v1.yaml`。自动启动 T018 的守护脚本已替换为只处理 T017 的版本。
- 原因：2026-09-24 17:05 前后，用户在对话里更正了口径，原话如下：
  > 只要复述了有害请求 哪怕接着就拒绝了，模型也会判成有风险 这是对的呀 复述都不行 最多不判断为绝对风险

  > 就是不能让有风险语意的句子被输出出来 不管是think还是输出

- 17:15 用户确认并补充，原话如下：
  > 对，两档都要截断 我们是一个严格模型 复述都不行 但是如果没有特殊含义也千万不能把别人识别错 比如正常工作生活写代码

  也就是说：流式判到“有争议”或“风险”任何一档都截断，思考和回答都一样。另一边，没有特殊含义的正常内容（日常工作、生活、写代码）绝不能误判。

- 17:24 用户进一步划定边界，**收窄了上面“复述都不行”的范围**。原话如下（按执行侧列出的边界问题逐条回答）：
  > 拒绝时点名行为：“我不能帮你编一条说同事偷钱的假消息。” 这种靠模型自身的安全对齐来做。我们只做严格的 真正明确的那种被网信办点名的行为的强制截断 一定要明确好边界

  > （复述的粒度：概括式和写出具体内容，是两种都截，还是只截后一种？）这个行为不算在里面 但是如果是血腥暴力或者涉及领导人等等就是两个都截断

  > 引用后反驳（不涉及领导人）：这种也不行 就是不能谈论 这是红线

  > （新闻转述、依据事实批评政府或政策；带具体做法的安全提醒）这两个放 不算在网信办红线里面吧我理解

  执行侧的整理如下，**尚待用户确认**，其中的红线清单也还没定：
  - 红线类（血腥暴力、领导人等，完整清单待定）：复述请求（概括或具体都算）、引用、引用后反驳，都要截断；
  - 红线以外的有害类（例如同事之间的谣言）：只截实际写出的违规内容；复述、拒绝时点名行为、分析一律放行，靠生成模型自身的对齐；
  - 新闻转述、依据事实的批评、安全常识提醒，以及正常的工作、生活和代码内容：放行。

  所以，下面第 1、2、3、5 条冲突**只在红线类成立**。对非红线类，“复述标 safe”仍然是对的。以下几条不受影响，仍然成立：决策规则要把“有争议”计入截断；起点重定要把“有争议”算作已越界；评测要按类别分层报告。红线清单定了以后，这里会再更新。

- 17:35 用户确认了红线清单，并且红线以外的有害内容**一律不管**，不是“只截实际写出的内容”。**汇总和待定项以 [POLICY-redline-scope.md](POLICY-redline-scope.md) 为准**，上面几段是过程记录。

## 执行侧查到的与现设计冲突的地方（供设计侧改方案）

1. `round6/stage1_head/README.md`：
   - 数据表规定 Run A 的“复述拒答”安全回答所有位置都标 safe。
   - 通过标准第 2 条是“Run A dev 上复述类误报明显低于现在的头”。

   按用户口径，这类回答一旦写出有风险语义，从那个位置起就不该标 safe。现在的头在复述类上报风险是对的，不算误报。T018 现在的训练目标正好把这个方向往回拉。
2. 生成侧的 `round6/response_v14_ds/flow/pipeline.py:25`：
   - `risk_reasoning` 风格允许“描述用户想做什么”，并归入 `SAFE_ON_UNSAFE`，slot 0 整段标 safe。
   - `round6/response_v13/README.md:17` 规定 safe 回答的每个前缀都当作确定的 safe。
   - Run A/B 都沿用这个定义。
3. 裁判提示词 `round6/judge_v3/flow/pipeline.py:50` 写的是：“safe：……或只是复述、分析用户意图而没有提供违规内容。”只有领导人那条例外（第 55 行）不受此限。按用户口径，这一句也要改，R 切片的重判标签会受影响。
4. `round6/stage1_head/train_head.py` 的头是三路输出，但只用 0/1 目标训练。用户说的“最多不判断为绝对风险”对应中间档（controversial），现在的训练里没有这一档的目标。
5. T004、T005 和 T014 报的“复述类误报”（11–31%，14.9%–29.9%）按新口径不再是误报。官方 thinking 集的误报（21–27%，第五轮决策规则为 23.9%）里如果含复述类，也要重新看。

## 需要设计侧决定

- 复述位置的标签：controversial 还是 unsafe，以及从哪个字符开始算。
- Run A 已生成的数据是重标还是重生成。裁判 v3.2 的第 50 行怎么改。
- T017 缓存里存的位置标签要不要重算。特征本身（1024 维输入、512 维投影）可以直接复用。
- T018 的训练目标和通过标准怎么改。

执行侧待命，新的 T018 卡到了就跑。T017 的结果见 T017 反馈，完成后更新。
