# T014 反馈

- 对应任务单版本：v1
- 状态：完成。验收各项全部达到。
- 执行时间：2026-09-24 14:49—15:26（北京时间）
- 实际执行的命令：步骤 1–5 与任务单相同。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| status | completed | completed |
| integrity_pass | true | true |
| max_prob_diff | < 0.001 | 6.56e-7 |
| 探针 AUC 复现 | 与 0.99686 相差 ≤ 1e-4 | 0.996858（相差约 2e-6） |
| official_unique_sequences | 1872 | 1872 |
| runA_calibration / runA_dev 记录数 | 与 T004 基本一致 | 5,052 / 5,188（与 T004 完全相同；跳过 4 条） |

## 结果

### 1. 作业与 report.json

- 单测：`round6/probe/test_transfer_cpu.py` 3 项全过。
- 作业：Job `safety-guard-probe-transfer-20260924-r1`，Pod `safety-guard-probe-transfer-20260924-r1-pckbc`。
  - 14:49:40 提交，14:49:56 输入就绪。拷进 Pod 的 9 个 `.py` 文件用 `sha256sum` 核对，与本地一致。
  - 约 15:23 写出 done 标记；exit code 为 `transfer=0`。
  - 15:25:04 取回产物后写 collected，Pod 正常退出（Succeeded），节点 GPU 分配回到 0。
- `probe`：`{"l2": 0.001, "calibration_auc_reproduced": 0.9968581072891843, "features": "hidden (1024-d assistant head input)"}`
- `counts`：`{"runA_calibration": 5052, "runA_skipped": 4, "runA_dev": 5188, "prefix_v2_calibration": 600, "prefix_v2_dev": 800, "official_unique_sequences": 1872}`
- `totals`：`{"sequences": 13512, "native_tokens": 2281841, "forward_tokens": 2488608}`
- `max_prob_diff`：6.56e-7
- `elapsed_seconds`：2037（约 34 分钟）

### 2. `analyze_transfer.py` 输出全文

```
head: T=1.7  stream AUC (max score): runA_calibration 0.853, runA_dev 0.850, prefix_v2_dev 0.903, official_thinking 0.784
rule            | RunA dev FPR      R restating | prefix_v2 FPR      R | official FPR      R
threshold       |         2.2%  53.4%     14.9% |          2.5%  52.8% |        26.9%  65.7%
consecutive_2   |         2.6%  53.5%     16.9% |          2.8%  55.5% |        24.5%  67.5%
consecutive_4   |         4.4%  56.1%     29.9% |          3.0%  58.0% |        32.9%  71.4%
ema_0.1         |         3.2%  48.8%     19.5% |          4.8%  59.5% |        28.6%  74.7%
ema_0.3         |         2.8%  53.9%     19.2% |          2.8%  56.5% |        25.9%  68.2%

probe: T=0.7  stream AUC (max score): runA_calibration 0.999, runA_dev 0.999, prefix_v2_dev 0.839, official_thinking 0.889
rule            | RunA dev FPR      R restating | prefix_v2 FPR      R | official FPR      R
threshold       |         4.2%  99.7%     10.4% |         31.8%  83.5% |        35.1%  91.6%
consecutive_2   |         3.4%  99.8%     11.0% |         31.0%  85.8% |        40.0%  93.8%
consecutive_4   |         4.1%  99.8%     15.6% |         31.8%  87.8% |        46.1%  94.6%
ema_0.1         |         1.9%  99.7%      6.5% |         29.2%  87.2% |        35.5%  93.1%
ema_0.3         |         3.7%  99.8%     12.7% |         32.5%  87.2% |        42.4%  93.1%
```

## 结论

**探针只迁移了一部分，而且阈值不能迁移。**

- **排序能力**（流式 AUC，取整条流的最高分）：
  - 官方 thinking 集：探针 0.889，现有头 0.784，探针明显更好。
  - prefix_v2 dev：探针 0.839，现有头 0.903，探针反而更差。
  - Run A：探针 0.999，现有头 0.85。
- **误报**：阈值按 Run A 校准集拟合。
  - Run A dev：探针误报 1.9%–4.2%，召回 99.7% 以上，复述类误报降到 6.5%–15.6%（现有头是 14.9%–29.9%）。
  - 换到 prefix_v2 和官方集，探针的误报升到 29%–33% 和 35%–46%，比现有头还高（现有头分别是 2.5%–4.8% 和 24.5%–32.9%），召回则更高。

**解读**：线性探针学到的东西里，有一部分是 luna 这一个模板的写作风格。它在别的文本上仍能把“复述”和“真有害”排开一些（官方集 AUC 提高 0.1），但分数的尺度变了，在 Run A 上定的阈值换到别的分布就不适用。按任务单的说法，这说明第六轮不能只用 Run A 重训读出头，还需要补数据的多样性，至少让校准集覆盖非 luna 的文本；是否在重训头之前先补数据，请设计方决定。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/probe/results_transfer_v1/report.json` | `7c31161d41c4287f120a6ec53d9fb1b61d9e5e7d1621766ce67f3f521aea2878` | 仓库 |
| `round6/probe/results_transfer_v1/transfer_v1_results.json` | `fa1d2989cf75081476c962d0521c83a8288ade95a6f2db5098dd239de9e2546a` | 仓库 |
| `official_rows.jsonl.gz` | `037362eeed261ddd9d881edbc81131e3240f790196ff29dbb4bf83c305dd5399` | Mac（与 report.json 的 files 一致） |
| `official_sequences.jsonl.gz` | `6331706b2c83eedb738c44ad24ab5646980cdbce1680463897f8ccce1c7a8187` | Mac（一致） |
| `prefix_v2_calibration.jsonl.gz` | `306c787f2aca99ea3843ee9349f1787174088b7434f86f968e07583f633b50ef` | Mac（一致） |
| `prefix_v2_dev.jsonl.gz` | `4154baae3b6fbdea068f643a1ca32a7367fe52cc27b136959ecc4dc212547f82` | Mac（一致） |
| `runA_calibration.jsonl.gz` | `8429f2ae518e2325854373e2930027f23bc298f3909caf63e12aed9899072fda` | Mac（一致） |
| `runA_dev.jsonl.gz` | `d19727f972469c3ca2aee67ad07589e6934bf4df8eb5d9b571aefe56b682184c` | Mac（一致） |

## 偏差与问题

没有偏差。取回文件时第一次拷贝失败：zsh 不会对未加引号的变量分词，文件列表被当成了一个参数。改成逐行读取后重拷成功，6 个文件的 SHA256 都与 `report.json` 一致。
