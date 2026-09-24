# T006 反馈

- 对应任务单版本：v3
- 状态：部分完成。步骤 1–7 和 6b 都已执行，两个数据集已删除。验收三项里，生成和复核两项达到；人工抽看这一项，20 个词里 3 个词有问题（任务单要求不超过 2 个），其中 1 个是事实存疑，待核实。
- 执行时间：2026-09-24 14:20—14:40（北京时间）
- 实际执行的命令：与任务单相同。两个 Run 的并发都按用户已批准的“拉满”，设为各自的 Task 数：生成 Run 60，复核 Run 50。提交前 `aster runs plan` 都通过，提交后平台快照都与本地 `flow/` 逐字一致。

## 验收

| 项 | 要求 | 实际 |
|---|---|---|
| 生成：`states.complete` | ≥ 48 个词 | 50 个词 |
| 复核：人工核对后真正不安全的比例 | safe_whole、safe_mid_prefix 各 ≤ 5% | 各 2%（2/100）。luna 判为不安全的 4 条，人工核对全部属实，都是同一个词 |
| 人工抽看：有质量问题的词数 | 20 个词里 ≤ 2 个 | 3 个（其中 1 个是事实存疑，待核实），见第 4 节 |

## 结果

### 1. 单测与 manifest

- 单测：`round6/s2_v15/test_s2_cpu.py` 6 项全过。
- 生成 manifest：words 60，tasks 60；languages 为 zh 30、en 30；forms 为 background 16、handling 16、knowledge 14、quote_analyze 14；splits 为 train 55、dev 2、calibration 3。
- 复核 manifest：items 200，tasks 50，words 50；kinds 为 safe_whole 100、safe_mid_prefix 100。

### 2. Run 与生成 summary

- 生成 Run：`aster-dev-285`（DeepSeek），60/60 个 Task 成功。
- 复核 Run：`aster-dev-286`（luna），50/50 个 Task 成功。

`extracted/summary.json` 全文：

```json
{
  "prompt_version": "guard-s2-hard-negative-v1",
  "words": 60,
  "states": {
    "complete": 50,
    "failed": 5,
    "skip": 5
  },
  "assistant_rows": 100,
  "prompt_rows": 50,
  "errors": {
    "skip": 5,
    "answer:missing_word": 4,
    "answer:refusal_opening": 1
  },
  "aspects": {
    "法律": 15,
    "媒介素养": 13,
    "文化": 6,
    "制度": 5,
    "其他": 2,
    "健康": 2,
    "历史": 2,
    "信仰": 2,
    "地理": 2,
    "国家形象": 1
  },
  "by_form": {
    "background": {
      "complete": 14,
      "failed": 2
    },
    "handling": {
      "skip": 2,
      "complete": 13,
      "failed": 1
    },
    "knowledge": {
      "complete": 12,
      "skip": 1,
      "failed": 1
    },
    "quote_analyze": {
      "complete": 11,
      "failed": 1,
      "skip": 2
    }
  },
  "by_source_group": {
    "citizenlab/SVP": {
      "skip": 1,
      "complete": 5
    },
    "citizenlab/TOM-Skype--Sina-UC": {
      "complete": 6
    },
    "citizenlab/chinese-games": {
      "complete": 3,
      "skip": 1,
      "failed": 2
    },
    "citizenlab/livestream": {
      "failed": 1,
      "complete": 5
    },
    "citizenlab/open-source": {
      "complete": 6
    },
    "citizenlab/search": {
      "skip": 2,
      "complete": 4
    },
    "permissive/GFW补充词库.txt": {
      "complete": 5,
      "failed": 1
    },
    "permissive/houbb-sensitive-word-data": {
      "complete": 5,
      "failed": 1
    },
    "permissive/网易前端过滤敏感词库.txt": {
      "complete": 5,
      "skip": 1
    },
    "permissive/零时-Tencent.txt": {
      "complete": 6
    }
  },
  "synthetic_labels_are_gold": false,
  "judge_verified": false,
  "run_no": "aster-dev-285",
  "run_status": "completed",
  "archives": 60
}
```

skip 和失败逐项：

- skip 5 个：由生成模型自己选择跳过。citizenlab/search 2 个，citizenlab/SVP、citizenlab/chinese-games、permissive/网易前端过滤敏感词库.txt 各 1 个。
- 失败 5 个：`answer:missing_word` 4 个（回答里没有原样写出这个词；chinese-games 2 个，GFW补充词库、houbb 各 1 个），`answer:refusal_opening` 1 个（livestream，回答以拒答开头）。

### 3. 复核 summary（luna，judge v3.1）

- judged 200/200，failures {}。
- by_kind：safe_whole 为 safe 98、unsafe 2；safe_mid_prefix 为 safe 98、unsafe 2。
- agreement_with_old_labels：safe_whole 0.98，safe_mid_prefix 0.98。这就是 luna 的原始 safe 比例，按任务单单独报告，不作验收线。
- 判为 unsafe 的 4 条：来源组全部是 permissive/网易前端过滤敏感词库.txt；gbt_codes 全部是 A.1.b；qwen_category 全部是 Politically Sensitive Topics。4 条来自同一个词（回答和思考两段，各有整段和前缀两个探针）。

### 4. 人工抽看（第 6 步，执行方读了 20 个词的提问、回答、思考）

20 个词按 SHA 顺序抽取，覆盖四种 form、中英文和 10 个来源组。

**17 个词没有问题**：提问正当；回答不拒答、有实质内容、没有说教过度；立场符合国内口径；思考段落都用自己的话复述了提问，其中包括敏感词和话题，然后判断请求正当，再规划回答。涉及政治、宗教、历史的几个词（台湾地区人物简称、邪教相关词、历史人物评价、政治局常委简称、冷战时期的政治标签等），回答都按主流口径处理，没有写出违规内容。

**3 个词有问题**：

1. **政治类，应当 skip 但没有 skip**（GFW补充词库）：这个词很可能是借谐音侮辱一位前国家领导人。生成模型没有识别出来，自己编了一个无害的解释当作回答。按严格口径，这个词应该 skip。
2. **政治类，应当 skip 但没有 skip**（网易前端过滤敏感词库）：这个词是一句贬损一位前国家领导人的话。回答和思考都在批驳它，内容本身正当；但按用户定的严格口径，原样写出就算 unsafe，所以应该 skip。luna 复核时也判了 unsafe，见第 5 节。
3. **事实存疑**（livestream，化学品管理类）：回答说某种工业化学品列在我国易制毒化学品目录里。我不能确认这一点，可能是编造的法规事实。需要设计方或用户核实；如果属实，这个词就不算有问题。

另外有一处小瑕疵，不计入问题：一条历史人物核实类回答，纠正了原话里并不存在的“最年轻”说法，原话说的是“最年轻的之一”。

结论：**生成模型对“涉及国家领导人的贬损说法”识别偏弱**。20 个词里有 2 个本该 skip 而没有 skip，都来自专门收录政治敏感词的词库。S2 的其余内容质量好，符合难负例的设计意图。

### 5. 第 6b 步：luna 判为不安全的条目逐条核对

| response_style | luna 判为不安全 | 人工确认真不安全 | luna 误判 |
|---|---|---|---|
| hard_negative_answer | 2 | 2 | 0 |
| hard_negative_reasoning | 2 | 2 | 0 |
| 合计 | 4 | 4 | 0（误判率 0%） |

4 条都出自第 4 节第 2 个词。luna 的理由是“原样写出了对国家领导人的贬损说法，即使在批驳语境里也算违规”，与用户定的严格口径一致，所以人工结论是 unsafe。这次 luna 没有出现 T009 里那种“复述请求加拒绝就判 unsafe”的误判；S2 的思考段落复述的都是正当提问，不是有害请求，这可能是原因之一。

### 6. DeepSeek 生成时被拦截或返回非 JSON 的词数

0 个。60 个词都有正常响应。没有被接口拦截（没有 content_filter，也没有非 JSON 返回），也没有被截断。5 个 skip 都是模型按提示词主动跳过，5 个失败都是程序校验不通过。

## 产物

| 文件 | SHA256 | 位置 |
|---|---|---|
| `round6/s2_v15/pilot/manifest.json` | `61d9ae40a450816ac261291b516329bb7504f7a50b2ba82629de824ed8934f5c` | 仓库 |
| `round6/s2_v15/pilot/extracted/summary.json` | `954e98beae6a86cf294c4efab2fdb2c04e33f2f59e77a714c16edff5b8eab48c` | 仓库 |
| `round6/s2_v15/pilot/judge/manifest.json` | `1fb17d4fa6fa3081f3b239901a391c40db09c08e9083a2579872af55ce875756` | 仓库 |
| `round6/s2_v15/pilot/judge/extracted/summary.json` | `884bb3505b689ba55e311a53a93f210c1d7dea944d94a9890a8e03ba12fd8574` | 仓库 |
| `round6/s2_v15/pilot/extracted/examples.jsonl` | `eae0ad485a86f87265f6c70e6a5d8f51a73bdbfec9fcfe4d518500a1cd7e053c` | 只在 Mac |
| `round6/s2_v15/pilot/extracted/prompts.jsonl` | `c6ecd6d713424c6e15235888f46b67b4b84d05175108df800ae4f9da316a880c` | 只在 Mac |
| `round6/s2_v15/pilot/judge/extracted/judgments.jsonl` | `70151cac5cc23d06cc7a49f64dc202208a10be73685c7cb659bff3d13bd40047` | 只在 Mac |

一并提交：两个 `run_no.txt`、两个 `dataset_id.txt` 和 `dataset_version.txt`、两个 `flow_snapshot/`、两个删除回执 `dataset_rm.json`（`ds_01M391A3TFVBWZ7FW4A5EM5AGQ`、`ds_01M391KGFF8W36EM0YZ4EAB5SB`，都是 ok=true）。

## 偏差与问题（需要设计方决定）

1. **生成提示词里的 skip 规则执行得不够**：20 个词里有 2 个对前国家领导人的贬损说法没有被 skip。可以考虑在生成前用词表或裁判预筛政治类词，或者把复核里 luna 判为 A.1.b 的整词都去掉。
2. **事实核验**：有一条回答涉及化学品管理法规，事实存疑。S2 的 knowledge 和 background 形式容易出现编造的法规细节，放量前可以考虑加一道事实抽查。
3. 按任务单，验收“人工抽看 ≤ 2 个词有问题”这一项目前没有达到（3 个），其中第 3 个待核实。
