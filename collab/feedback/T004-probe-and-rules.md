# T004 反馈

- 对应任务单版本：v1
- 状态：完成。验收各项全部达到。
- 执行时间：2026-09-24 14:10—14:45（北京时间）
- 实际执行的命令：步骤 1–6 与任务单相同。T005 带了 `--v1-dump round6/decision_rule/results_v1` 运行，Mac 上 v1 的导出文件都在。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| `report.json` status | completed | completed |
| integrity_pass | true | true |
| max_prob_diff | < 0.001 | 5.66e-7 |
| 各 split records 与输入 manifest 一致 | 基本一致，跳过的记在 totals | calibration 5,052/5,056，dev 5,188/5,188，train 5,999/6,000；共跳过 5 条（`skipped_tokenizer_mismatch`） |
| `probe_v1_results.json`、`runA_rules_v1_results.json` | 生成 | 都已生成 |

## 结果

### 1. 单测与输入 manifest

- 单测：8 项全过。
- 输入 manifest：records 16,244。words_by_split：calibration 1,264，dev 1,297，train 1,500。records_by_slot_style：0:neutral_information 1,016，0:safe_redirect 994，0:refusal 1,068，0:risk_reasoning 983，1:compliance 4,061，2:helpful_answer 2,010，2:helpful_reasoning 2,051，3:drift 4,061。

### 2. 作业与 report.json

- 作业：Job `safety-guard-probe-20260924-r1`，Pod `safety-guard-probe-20260924-r1-pbrjk`。14:11:20 提交，14:11:46 输入就绪，约 14:40 写出 done 标记，14:41:19 取回产物后写 collected，Pod 正常退出（Succeeded），节点 GPU 分配回到 0。
- 冒烟检查：开始约 1 分钟后 `smoke_passed=true`，当时 max_prob_diff 3.3e-7。
- exit code：`dump=0 fit=0`。
- totals：`{"projection_calls": 58588, "records": 16239, "native_tokens": 1626447, "forward_tokens": 1874816, "skipped_tokenizer_mismatch": 5}`
- records：calibration 5,052，dev 5,188，train 5,999。
- feature_positions：calibration 119,808，dev 123,029，train 142,297。
- max_prob_diff：5.66e-7。
- hf_tokenizer_cross_check：true。
- elapsed_seconds：1,627（约 27 分钟）。

### 3. `fit_probe.py` 输出

`probe_v1_fit.log` 全文：

```
{
 "head": {
  "auc_post_vs_restating_risk_reasoning": 0.7913175086114175,
  "auc_all": 0.9102382899121242,
  "recall@5%": 0.6834778136937996
 },
 "probe_hidden": {
  "auc_post_vs_restating_risk_reasoning": 0.9947465215232167,
  "auc_all": 0.9966628035247675,
  "recall@5%": 0.990192477225977
 },
 "probe_projection": {
  "auc_post_vs_restating_risk_reasoning": 0.986752173804499,
  "auc_all": 0.9925193875972718,
  "recall@5%": 0.9711651483984719
 }
}
```

结论：冻结特征上的线性探针，区分“复述有害请求的安全推理”和“起点之后的真有害文本”，AUC 是 0.995（1024 维头输入）和 0.987（512 维投影）；现有风险头只有 0.791。5% 误报下的召回，现有头 68.3%，两个探针分别是 99.0% 和 97.1%。**主干特征本身区分得开，问题在读出头。**置信区间等完整数字见 `round6/probe/results_v1/probe_v1_results.json`（已提交）。

### 4. `analyze_runA_rules.py` 输出（T005）

在 Run A 校准集上拟合出的温度 **T=1.7**（端点 NLL 0.200）。决策规则 v1 的温度是 2.2，是在 prefix_v2 上拟合的。

```
T=1.7 (endpoint NLL 0.200)
rule                           tau |         dev FPR [CI]  dev R | restating FPR | early | off FPR  off R
threshold@T=1.0           0.997182 |   3.6% [ 2.9, 4.3]  57.4% |         24.0% |  0.03 |   34.7%  68.9%
threshold@T=1.7           0.968377 |   2.2% [ 1.7, 2.8]  53.4% |         14.9% |  0.01 |   27.1%  65.9%
threshold@T=2.2           0.929205 |   1.6% [ 1.2, 2.2]  49.8% |         10.7% |  0.01 |   21.2%  64.7%
consecutive_2@T=1.0       0.996452 |   3.6% [ 2.9, 4.3]  57.2% |         24.3% |  0.02 |   31.0%  69.2%
consecutive_2@T=1.7       0.964519 |   2.6% [ 2.0, 3.2]  53.5% |         16.9% |  0.01 |   24.5%  67.5%
consecutive_2@T=2.2       0.920567 |   2.9% [ 2.3, 3.6]  55.1% |         19.5% |  0.01 |   27.8%  68.2%
consecutive_4@T=1.0       0.995533 |   3.4% [ 2.7, 4.0]  53.6% |         23.7% |  0.02 |   29.6%  69.8%
consecutive_4@T=1.7       0.955332 |   4.4% [ 3.6, 5.1]  56.1% |         29.9% |  0.03 |   33.1%  71.4%
consecutive_4@T=2.2       0.910875 |   3.2% [ 2.5, 3.9]  54.3% |         22.7% |  0.02 |   30.2%  70.5%
consecutive_8@T=1.0       0.992921 |   4.2% [ 3.4, 5.0]  55.3% |         28.9% |  0.02 |   32.2%  73.8%
consecutive_8@T=1.7       0.943766 |   4.4% [ 3.7, 5.2]  56.4% |         30.8% |  0.02 |   34.5%  75.2%
consecutive_8@T=2.2       0.900000 |   2.9% [ 2.2, 3.5]  53.0% |         20.1% |  0.01 |   29.6%  71.9%
ema_0.3@T=1.0             0.995533 |   4.2% [ 3.4, 4.9]  56.9% |         29.5% |  0.02 |   32.9%  70.3%
ema_0.3@T=1.7             0.960189 |   2.8% [ 2.2, 3.4]  53.9% |         19.2% |  0.01 |   25.9%  68.2%
ema_0.3@T=2.2             0.910875 |   3.9% [ 3.2, 4.6]  57.5% |         27.6% |  0.02 |   33.9%  70.8%
ema_0.1@T=1.0             0.992057 |   3.7% [ 3.0, 4.4]  50.2% |         21.8% |  0.05 |   30.6%  75.4%
ema_0.1@T=1.7             0.943766 |   3.2% [ 2.6, 3.9]  48.8% |         19.5% |  0.04 |   28.6%  74.9%
ema_0.1@T=2.2             0.900000 |   2.2% [ 1.6, 2.8]  45.0% |         14.0% |  0.02 |   23.3%  72.4%
```

列含义（按脚本输出）：dev FPR 是 Run A 开发集上安全流的误报率，方括号里是按词族 bootstrap 的置信区间；dev R 是不安全流的召回；restating FPR 是复述式 risk_reasoning 的安全流误报率；early 是首次触发落在标注起点之前的比例；off FPR / off R 是官方 thinking 集上的误报和召回，只做观察，不参与拟合。

要点：阈值是在 Run A 校准集上按每种语言误报 ≤ 5% 拟合的。在 Run A dev 上，各规则误报 1.6%–4.4%，召回 45%–58%。复述式推理的误报 10.7%–30.8%，仍然是误报的主要来源。官方 thinking 集上，误报最低是 threshold@T=2.2 的 21.2%（召回 64.7%），和决策规则 v1 的 23.9% 相近，没有明显改善。和第 3 节合起来看：光靠换决策规则压不下复述式误报，要重训读出头。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/probe/results_v1/report.json` | `6c072456765141202290bb6c53c37c843d2fa0e8961b5e934c5dbca5f0573be9` | 仓库 |
| `round6/probe/results_v1/probe_v1_results.json` | `2873c57b514847ab582fb0640eca887a58872e94edb9467991a14f9a3658fa43` | 仓库 |
| `round6/probe/results_v1/runA_rules_v1_results.json` | `b5c019347efa5f17d6fa16e862b9f69be4a138d961b8f743cacfa87b8ed49de9` | 仓库 |
| `round6/probe/results_v1/runA_calibration.jsonl.gz` | `2fbb1c4d0e212af1e0d7ec0dc18cc59e5da36e435d3267f4441de1004e9a805a`（与 report.json 的 files 一致） | Mac |
| `round6/probe/results_v1/runA_dev.jsonl.gz` | `4167a473cdc272d2eae9b1267696510744095939fbcca1f0f825472447d3629a`（与 report.json 的 files 一致） | Mac |
| `round6/probe/results_v1/probe_v1_dump.log` | `078a81fe3d4d7238ff1a12d1114ed11c0a077a3554c417e1e07d1a917f2a8b90` | Mac |
| `round6/probe/results_v1/probe_v1_fit.log` | `4e386d79368e2f11152a68db260598cacf04d4f4dc0892450306bb917ee6b0c2` | Mac |

`*.npz` 特征文件和 `runA_train.jsonl.gz` 按任务单没有取回，留在 PVC 的 `/work/output/round6/probe_v1/`。

## 偏差与问题

没有偏差。跳过 5 条是分词校验不一致（`skipped_tokenizer_mismatch`），已记在 totals。
