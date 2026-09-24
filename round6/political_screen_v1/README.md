# 政治敏感词专项预筛（T026）

用户 2026-09-25 的决定：最后针对中国政治敏感词和话题做专项加强，模型遇到这类词要高度警觉。具体口径（用户选定）：

- **特殊事件一提就截**：提到政治敏感事件（包括代称、谐音、数字写法），不论语气，标“有争议”；
- **领导人**：正常新闻里用正式姓名、职务提到领导人照常放行；负面、嘲讽、谣言照旧标“风险”；用变体写法（谐音、换字、拼音缩写等）提到领导人，标“有争议”；
- **其他政治话题**：按现行红线规则判，另外补专门数据、收紧阈值（后续任务）。

裁判 v4 只记录“涉及哪条红线、怎么涉及”，分不出一个话题是不是特殊事件；事件的各种代称也不能指望裁判每次都认出来。所以和领导人预筛（T012）一样，这里对词表做一次预筛，再按词原样匹配到每一条训练文本上，不需要重判。

## 预筛

`flow/pipeline.py`（`guard-political-screen-v1`）只发送词本身，每个 Task 200 个词。每个词给出：

| verdict | 含义 |
|---|---|
| `event` | 中国政治敏感事件、运动、纪念日，或它们的代称、隐晦说法、谐音、数字写法 |
| `figure` | 政治敏感人物（不是领导人）：异见人士、维权人士、敏感事件的标志性人物、因政治原因被查处的官员 |
| `org` | 被取缔或被定性为分裂、恐怖、邪教的组织，以及它们的口号 |
| `leader` | 与领导人有关的任何写法（具体是哪一类由 T012 决定） |
| `other_political` | 普通政治词汇 |
| `no` / `unsure` | 与政治无关 / 拿不准 |

另外给出 `ambiguous`：这个字符串在普通文本里是不是常常另有意思（数字、常用词、常见人名），单凭它出现不能断定在说敏感内容。

两遍：第一遍（`p1`）筛全部 340,883 个词；第二遍（`c1`）换一种分批顺序，复查第一遍判为 event、figure、org、leader、unsure 的词，以及 T012 标出的领导人词（insult、rumor、evasion、unsure）。合并规则：取最警觉的 verdict；任一遍判为 ambiguous，就算 ambiguous。

## 怎么用到标签上

`redline_v1/apply_policy.py --political-terms political_terms.jsonl --source <行文件>`，对每条文本（回答或提问）：

| 词 | 条件 | 从词第一次出现处起 |
|---|---|---|
| event / figure / org，不 ambiguous | 出现就算 | 有争议（开关 `political_event` / `_figure` / `_org`） |
| leader，且 T012 判为 insult / rumor，不 ambiguous | 出现就算 | 风险 |
| leader，且 T012 判为 evasion / unsure，不 ambiguous | 出现就算 | 有争议（开关 `leader_variant`） |
| 上面任何一类，但 ambiguous | 只有裁判在这条文本里记了政治类红线（R1、R3–R9）时才算 | 最高到有争议 |
| 单个字 | 不匹配 | — |

词之前的位置保留裁判的等级，词内部的位置不训练。另外，规则表里裁判标为变体写法的 R1 条目现在也标“有争议”（`leader_variant`），Run A 种子词被 T012 判为 evasion / unsure 时同样标“有争议”（原来降为安全）。

## 文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `flow/pipeline.py`、`flow/flow.py` | Aster | 提示词、请求、解析；一个 Task 一次请求 |
| `make_batch.py` | Mac | 建 Task；`--only` 只筛给定的词，`--tag` 区分各遍（只能是字母和数字） |
| `extract.py` | Mac | 下载归档、按词合并各遍，输出 `political_terms.jsonl`、`confirm_words.txt`、`missing_words.txt`、`summary.json`；`--review-sample` 输出本地核对表 |
| `test_political_screen_cpu.py` | Mac | CPU 单测（单独跑：几个目录都有 `flow/pipeline.py`） |

词表、`political_terms.jsonl`、`confirm_words.txt`、`missing_words.txt` 和核对表都含词本身，**不提交**（已在 .gitignore）。可以提交的只有 manifest、run_no 和 `summary.json`。
