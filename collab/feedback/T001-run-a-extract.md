# T001 反馈

- 对应任务单版本：v2
- 状态：完成（验收各项全部达到）
- 执行时间：2026-09-24 13:53—13:57（北京时间）。v1 的阻塞记录见本文件的 git 历史（0734b07）。
- 实际执行的命令：步骤 1–5 与任务单相同，没有改动。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| `attempts.json` | 352 条，`truncated=false` | 352 条，`truncated=false` |
| `word_artifacts` | 50000 | 50000 |
| `missing_word_artifacts` | 0 | 0 |
| 归档 SHA 校验 | 全部通过 | 352 个归档全部通过（不通过时脚本会报错退出），下载失败 0 |
| `trainable_summary.json` | 生成 | 已生成 |

## 结果

### 1. 单测与冻结

- 单测：15 项全过。任务单预期其中 1 项跳过，实际 0 项跳过。
- 冻结脚本输出：samples 352，attempts 352。

### 2. `summary.json` 全文

```json
{
  "run_id": "run_01M38P6E4HZ027ZSNG1CDX2VD3",
  "run_no": "aster-dev-276",
  "run_status": "completed",
  "prompt_version": "guard-luna-oneword-response-v14",
  "attempts_with_archive": 352,
  "attempts_without_archive": 0,
  "expected_terms": 50000,
  "word_artifacts": 50000,
  "missing_word_artifacts": 0,
  "complete_words": 42502,
  "raw_complete_words": 42502,
  "reward_over_expected": 0.85004,
  "valid_responses": 188946,
  "repaired_responses": 6368,
  "valid_by_slot": {
    "0": 49702,
    "1": 44614,
    "2": 49494,
    "3": 45136
  },
  "quality_flags": {
    "unsafe:fiction_marker": 1432,
    "safe:fiction_marker": 1542,
    "unsafe:harm_commentary": 251
  },
  "error_types": {
    "slot_error": 17926,
    "root_mismatch": 66,
    "request_or_parse": 8
  },
  "slot_errors": {
    "1:onset_quote": 4805,
    "3:onset_quote": 4695,
    "3:unsafe_basis": 3058,
    "1:unsafe_basis": 2990,
    "1:unsafe_refusal_opening": 1021,
    "1:missing_word": 401,
    "2:missing_word": 290,
    "0:not_english": 180,
    "1:not_english": 142,
    "2:not_english": 106,
    "3:missing_word": 77,
    "3:not_english": 49,
    "0:missing_word": 45,
    "2:response": 19,
    "2:not_chinese": 13,
    "1:not_chinese": 9,
    "2:safe_onset_or_basis": 8,
    "3:negated_onset": 6,
    "3:not_chinese": 5,
    "1:response_length": 3,
    "0:safe_onset_or_basis": 3,
    "3:unsafe_refusal_opening": 1
  },
  "model_refusal_words_by_group": {},
  "complete_by_group": {
    "citizenlab/livestream": "9002/10386",
    "citizenlab/open-source": "7292/8493",
    "citizenlab/chinese-games": "5202/6197",
    "permissive/houbb-sensitive-word-data": "5167/6197",
    "permissive/零时-Tencent.txt": "2902/3438",
    "citizenlab/SVP": "2707/3157",
    "citizenlab/search": "2266/2661",
    "permissive/网易前端过滤敏感词库.txt": "1818/2164",
    "permissive/GFW补充词库.txt": "1642/1945",
    "citizenlab/TOM-Skype--Sina-UC": "788/946",
    "citizenlab/olympics": "659/788",
    "citizenlab/microsoft-bing": "646/771",
    "permissive/补充词库.txt": "419/490",
    "permissive/反动词库.txt": "303/360",
    "citizenlab/june-4": "284/336",
    "citizenlab/LINE": "229/277",
    "citizenlab/apple": "192/235",
    "permissive/涉枪涉爆.txt": "186/227",
    "permissive/民生词库.txt": "161/190",
    "permissive/政治类型.txt": "141/164",
    "permissive/贪腐词库.txt": "104/114",
    "permissive/色情类型.txt": "69/85",
    "permissive/色情词库.txt": "72/81",
    "permissive/暴恐词库.txt": "64/76",
    "permissive/其他词库.txt": "65/75",
    "permissive/广告类型.txt": "52/60",
    "citizenlab/wechat": "41/50",
    "citizenlab/coronavirus": "12/16",
    "permissive/COVID-19词库.txt": "9/10",
    "citizenlab/qqmail": "6/9",
    "permissive/新思想启蒙.txt": "2/2"
  },
  "onset_fraction_by_slot_style": {
    "1:delayed": {
      "n": 21916,
      "p10": 0.20689655172413793,
      "p50": 0.3181818181818182,
      "p90": 0.4704225352112676,
      "at_zero": 64
    },
    "1:immediate": {
      "n": 22698,
      "p10": 0.0,
      "p50": 0.0,
      "p90": 0.0,
      "at_zero": 21245
    },
    "3:delayed": {
      "n": 45136,
      "p10": 0.36,
      "p50": 0.5037593984962406,
      "p90": 0.6365638766519823,
      "at_zero": 200
    }
  },
  "token_usage": {
    "input_tokens": 154242189,
    "output_tokens": 36882546,
    "total_tokens": 191124735
  },
  "tokens_per_word": {
    "input_tokens": 3084.84378,
    "output_tokens": 737.65092,
    "total_tokens": 3822.4947
  },
  "latency_seconds": {
    "mean": 36.63963422962549,
    "p50": 34.7086416920647,
    "p95": 64.55073097301647
  },
  "synthetic_labels_are_gold": false
}
```

### 3. `trainable_summary.json` 全文

```json
{
  "words_seen": 49924,
  "words_incomplete_at_extract": 7422,
  "responses_delayed_onset_zero": 248,
  "words_dropped_delayed_onset_zero": 241,
  "words_kept": 42261,
  "responses_kept": 169044,
  "words_by_split": {
    "train": 39700,
    "dev": 1297,
    "calibration": 1264
  },
  "responses_by_split_label": {
    "train:safe": 79400,
    "train:unsafe": 79400,
    "dev:safe": 2594,
    "dev:unsafe": 2594,
    "calibration:safe": 2528,
    "calibration:unsafe": 2528
  },
  "quality_flags_kept": {
    "unsafe:fiction_marker": 1354,
    "safe:fiction_marker": 1340,
    "unsafe:harm_commentary": 234
  }
}
```

4/4 完整、可以训练的词共 42,261 个：train 39,700，dev 1,297，calibration 1,264。每个 split 里 safe 和 unsafe 回答各占一半。
抽取时 4/4 完整的词有 42,502 个。按规则 4 剔除了 248 条“要求先铺垫但起点为 0”的回答，涉及 241 个词，这些词整词去掉。
`words_seen` 是 49,924，比 50,000 少 76 个：这些词一条合格回答都没有。其中 74 个是整词级错误（`root_mismatch` 66，`request_or_parse` 8）。

### 4. 错误类型前 10 名

按槽位和类型分（槽位顺序见 `flow/pipeline.py` 的 `slot_records`：0 = 有害提问的安全回答，1 = 有害提问的不安全回答（顺从式），2 = 正常提问的安全回答，3 = 正常提问的不安全回答（跑偏式））：

| 槽位:错误 | 条数 |
|---|---|
| `1:onset_quote` | 4805 |
| `3:onset_quote` | 4695 |
| `3:unsafe_basis` | 3058 |
| `1:unsafe_basis` | 2990 |
| `1:unsafe_refusal_opening` | 1021 |
| `1:missing_word` | 401 |
| `2:missing_word` | 290 |
| `0:not_english` | 180 |
| `1:not_english` | 142 |
| `2:not_english` | 106 |

合并槽位后按错误类型：

| 错误 | 条数 |
|---|---|
| `onset_quote` | 9500 |
| `unsafe_basis` | 6048 |
| `unsafe_refusal_opening` | 1022 |
| `missing_word` | 813 |
| `not_english` | 477 |
| `root_mismatch` | 66 |
| `not_chinese` | 27 |
| `response` | 19 |
| `safe_onset_or_basis` | 11 |
| `request_or_parse` | 8 |

`onset_quote` 和 `unsafe_basis` 两类占了不合格回答的大头，都出在不安全回答的两个槽位（1 和 3）。`finish_reason:length` 没有出现。

### 5. `quality_flags`

| 标记 | 抽取后全部合格回答 | 训练过滤后 |
|---|---|---|
| harm_commentary（仅 unsafe） | 251 | 234 |
| fiction_marker（unsafe） | 1432 | 1354 |
| fiction_marker（safe） | 1542 | 1340 |

### 6. 产物

全部留在 Mac 的 `round6/response_v14/batch_50k/extracted/`，没有提交仓库。

| 文件 | 行数 | SHA256 |
|---|---|---|
| `examples.jsonl` | 188,946 | `c91d4bc8d5500da82483da5c7e834d9661320e152007e2cafc939f62014c6f94` |
| `trainable.jsonl` | 169,044 | `2c207b4e093254edf7237ed7b7459f783219fb5064858b4c2d866188097926de` |
| `words.jsonl` | 50,000 | `0c577c9ee5e692defbeedaa616c7114606a1cdfd9c3e25f55bb85ccc817b136c` |
| `attempts.json` | — | `86c162e40c95af682ed87bb3d54a2e41044e63a71407beff8550c8f4f9f50c26` |
| `summary.json` | — | `e30fc322e4603bc6ff018ac7870326b6896e5d129ae7d12a25dd12a5e4872b1f` |
| `trainable_summary.json` | — | `9b5d78ee0ec0e7018b506b1b7a0302946d5e1d8ed94389fa4b12f340323bf0f5` |

`selected_attempts.json` 与 `attempts.json` 完全相同（SHA 相同）。归档在 `archives/` 下，每个都有对应的 `.sha256` 文件。

## 偏差与问题

1. 单测是 15 项全过、0 项跳过，任务单写的是“其中 1 项跳过”。不影响结果，供设计方核对。
2. 没有做 `[exec-fix]`。
3. T002 pilot 显示 v14 的起点标得偏晚：顺从式回答在起点之前就已经答应去做，转折句被当成起点。这会影响用 Run A 做前缀探针和重判（R 切片）时的标签，详见 T002 反馈第 5 节。
