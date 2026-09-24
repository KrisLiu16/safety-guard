# T011 反馈

- 对应任务单版本：v2
- 状态：部分完成。步骤 1–7 和 6b 都已执行，两个数据集已删除。验收里只有长度 p50 这一项没有达到：落在区间偏短的一侧，不在中部。其余各项都达到。
- 执行时间：2026-09-24 14:25—14:48（北京时间）
- 实际执行的命令：与任务单相同。并发按规则拉满：生成 Run 40，复核 Run 35。提交前 `aster runs plan` 都通过，提交后平台快照都与本地 `flow/` 逐字一致。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| 生成：`states.complete` | ≥ 32 个词 | 35 个词 |
| 生成：长度 p50 | 落在长度区间中部 | **偏短**：中文 p50 为 1,000–1,020 字，校验区间 700–2,400，位置约 18%；英文 p50 为 702–738 词，校验区间 380–1,400，位置约 32–35%。见下面“偏差与问题”第 1 条 |
| 复核：人工核对后 `risk_reasoning_long` 真正不安全的比例 | 整段和前缀各 ≤ 5% | 各 0%（0/35）。luna 判为不安全的 1 条，人工核对是误判 |
| 人工抽看 | 10 个词里 ≤ 1 个有问题 | 1 个（格式问题，见第 4 节） |

## 结果

### 1. 单测与 manifest

- 单测：`round6/s5_safe_v15/test_s5_safe_cpu.py` 6 项全过（单独运行）。
- 生成 manifest：words 40，tasks 40；unsafe_prompt_language 为 en 21、zh 19；shapes 为 context_check、draft_revise、interpretations、stepwise 各 10；splits 全部是 train（40）；excluded_task_keys 60（T006 用过的 60 个词已排除）。
- 复核 manifest：items 140，tasks 35，words 35；kinds 为 safe_whole 70、safe_mid_prefix 70。

### 2. Run 与生成 summary

- 生成 Run：`aster-dev-287`（DeepSeek），40/40 个 Task 成功。
- 复核 Run：`aster-dev-288`（luna），35/35 个 Task 成功。

`extracted/summary.json` 全文：

```json
{
  "prompt_version": "guard-s5-safe-long-thinking-v1",
  "words": 40,
  "states": {
    "complete": 35,
    "failed": 5
  },
  "assistant_rows": 70,
  "errors": {
    "unsafe_reasoning:length": 5
  },
  "length_units_by_language_style": {
    "en:helpful_reasoning_long": {
      "p10": 622,
      "p50": 702,
      "p90": 872
    },
    "en:risk_reasoning_long": {
      "p10": 598,
      "p50": 738,
      "p90": 841
    },
    "zh:helpful_reasoning_long": {
      "p10": 847,
      "p50": 1000,
      "p90": 1354
    },
    "zh:risk_reasoning_long": {
      "p10": 833,
      "p50": 1020,
      "p90": 1211
    }
  },
  "quality_flags": {
    "fiction_marker": 5
  },
  "by_shape": {
    "context_check": {
      "complete": 8,
      "failed": 2
    },
    "draft_revise": {
      "complete": 10
    },
    "interpretations": {
      "complete": 9,
      "failed": 1
    },
    "stepwise": {
      "failed": 2,
      "complete": 8
    }
  },
  "by_source_group": {
    "citizenlab/SVP": {
      "complete": 4
    },
    "citizenlab/TOM-Skype--Sina-UC": {
      "complete": 4
    },
    "citizenlab/chinese-games": {
      "failed": 1,
      "complete": 3
    },
    "citizenlab/livestream": {
      "complete": 3,
      "failed": 1
    },
    "citizenlab/open-source": {
      "complete": 4
    },
    "citizenlab/search": {
      "complete": 4
    },
    "permissive/GFW补充词库.txt": {
      "complete": 4
    },
    "permissive/houbb-sensitive-word-data": {
      "complete": 4
    },
    "permissive/网易前端过滤敏感词库.txt": {
      "complete": 2,
      "failed": 2
    },
    "permissive/零时-Tencent.txt": {
      "failed": 1,
      "complete": 3
    }
  },
  "synthetic_labels_are_gold": false,
  "judge_verified": false,
  "run_no": "aster-dev-287",
  "run_status": "completed",
  "archives": 40
}
```

5 个失败都是 `unsafe_reasoning:length`，而且都是**中文风险推理写得太短**：实测 562、640、669、670、693 字，都低于 700 字的下限。来源组：网易前端过滤敏感词库.txt 2 个，零时-Tencent.txt、chinese-games、livestream 各 1 个。

### 3. 复核 summary（luna，judge v3.1）

- judged 140/140，failures 为空。
- by_kind：safe_whole 为 safe 69、unsafe 1；safe_mid_prefix 为 safe 70。
- luna 的原始 safe 比例：safe_whole 98.6%，safe_mid_prefix 100%（单独报告，不作验收线）。
- 判为 unsafe 的 1 条：response_style 是 `risk_reasoning_long`，gbt_codes 为 A.4.e、A.4.f，qwen_category 为 PII，来源组 citizenlab/open-source。

### 4. 人工抽看（第 6 步，执行方读了 10 个词的两段思考，共 20 段）

10 个词按 SHA 顺序抽取，覆盖 4 种 shape、中英文和 8 个来源组。

- **是否真的复述了请求**：10 段风险推理都用自己的话概括了有害请求（骚扰、造谣、威胁公共场所、下药、人肉搜索、煽动等），再分析风险、推测用户的真实需要、决定拒绝，并规划替代回应。10 段正常推理都围绕提问梳理知识，并规划回答结构。
- **有没有写出有害细节，或者先写后否定**：没有。风险推理只停留在概括层面，没有出现步骤、话术、目标、地点、剂量或渠道；有几段还在结尾自检“任意位置截断都安全”。正常推理里，法医和药名类的内容停留在术语和鉴别层面，不能拿来操作。
- **是否符合国内口径**：符合。涉及民族人名、新疆地名、历史事件时，都按民族团结和主流史观表述。
- **是否靠重复凑长度**：没有明显凑字数。英文段落普遍比中文长，信息密度正常。
- **1 个词有格式问题**：一段英文正常推理里出现了字面上的 `\n\n`（反斜杠加 n，不是真的换行），全部 70 段里只有这一段。原因可能是模型在 JSON 里多转义了一次，建议在抽取或校验时处理。

### 5. 第 6b 步：luna 判为不安全的条目逐条核对

| response_style | luna 判为不安全 | 人工确认真不安全 | luna 误判 |
|---|---|---|---|
| risk_reasoning_long | 1（safe_whole） | 0 | 1 |
| helpful_reasoning_long | 0 | 0 | 0 |
| 合计 | 1 | 0 | 1（luna 的不安全判定 100% 是误判；占全部 140 条的 0.7%） |

这一条是对“公开他人住址和电话来恐吓别人”的请求的完整拒绝推理：只分析风险、没有写出任何个人信息，结尾明确说不会提供方法。luna 的理由是“大量复述并展开隐私侵害情境”，这正是 T009 里发现的偏差：把“复述有害请求并拒绝”的安全推理判成 unsafe。**S5 用 luna 复核时，要预期会误删少量合格样本**；这次 140 条里误删 1 条。

### 6. DeepSeek 生成时被拦截、截断或返回非 JSON 的词数

0 个。40 个词都有正常响应，没有接口拦截，没有 finish_reason=length，也没有非 JSON 返回。5 个失败都是程序校验的长度不够。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/s5_safe_v15/pilot/manifest.json` | `ed2fb0c101e6e75fdbe576594b26bf29eecd4eee0c310d2117d47f240ebf536d` | 仓库 |
| `round6/s5_safe_v15/pilot/extracted/summary.json` | `e12d5e4faef68dacc826f5e51d2dff75b667a6f57af7a4544580d5f4f6b80296` | 仓库 |
| `round6/s5_safe_v15/pilot/judge/manifest.json` | `209db07618f241d44a993f4bd4174854451e856b4ae8294a9de60a4e3b8e5483` | 仓库 |
| `round6/s5_safe_v15/pilot/judge/extracted/summary.json` | `c8464a24bd3580ee5900681c9ba66f3a05e86f6b514709692a438076d781035f` | 仓库 |
| `round6/s5_safe_v15/pilot/extracted/examples.jsonl` | `3ba152061011cc746c920a2d91585b808c42a6c907ea8435f476040acbc245ba` | 只在 Mac |
| `round6/s5_safe_v15/pilot/judge/extracted/judgments.jsonl` | `ad322d27e477b96212fc38be101f76ad23bf1d8565abe7ffb4f971e8fa7ae19c` | 只在 Mac |

一并提交：两个 `run_no.txt`、两个 `dataset_id.txt` 和 `dataset_version.txt`、两个 `flow_snapshot/`、两个删除回执 `dataset_rm.json`（`ds_01M391MNEYG9Q12DTJG0W7P3VH`、`ds_01M39229A2FZBBGFYJKTC0KGME`，都是 ok=true）。

## 偏差与问题（需要设计方决定）

1. **长度偏短**：DeepSeek 写的中文推理大多在 700–1,200 字，靠近下限；5 个失败都是中文风险推理不足 700 字。提示词写的是 700–2,000 字，校验区间是 700–2,400。如果要让 p50 落在中部，可以把提示词里的目标长度往上调，或者把下限放宽。
2. **luna 误判复述式拒绝推理**：本切片误删率低（1/140），但方向和 T009 一致。
3. **字面上的 `\n`**：70 段里有 1 段，建议在校验时拒收，或者在抽取时还原成换行。
