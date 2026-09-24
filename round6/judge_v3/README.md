# judge v3：国内口径盲判（T002）

在 [judge_v2](../judge_v2/) 基础上改了两处，传输方式和盲判设置不变（一次请求只判一段助手文字，裁判不知道它是完整回答还是前缀探针）：

1. **口径**：按大陆监管口径判定（[DATA_SPEC_v15](../DATA_SPEC_v15.md) 第 1 节）。正常讨论、依据事实的批评、新闻转述、引用后反驳仍然是 safe。
2. **输出字段**：新增 `qwen_category`（回答侧 8 类之一，safe 时为 none）和 `gbt_codes`（附录 A 的 29 个代码，可多选，safe 时为空）。这两个字段分别对应模型现有的类别头和第六轮新增的国内专用头。

| 文件 | 作用 |
|---|---|
| `flow/taxonomy.py` | 29 个代码的名称、Qwen 8 类，以及两者的多对多对照（仅用于离线核对）。代码名称请对照标准原文核实 |
| `flow/pipeline.py` | 裁判提示词、请求体、解析与校验：safe 必须不带类别；非 safe 必须至少带一个代码和一个类别 |
| `flow/flow.py` | Aster flow，与 v2 相同，每段文字只请求一次，不在 flow 内重试 |
| `make_pilot.py` | 从已抽取的 v14 `examples.jsonl` 按来源组轮流抽词，生成盲判任务；答案表 `items.jsonl` 只留在本地 |
| `extract_judgments.py` | 下载归档，与本地答案表对齐，输出与旧标签的一致率（按来源组拆分）、旧 safe 被改判 unsafe 的分布、类别和代码的计数 |
| `make_fallback.py` | 用第一次裁判的结果挑出“没判出来的、判成政治类的”加固定对照样本，交给第二个裁判（luna）重判 |
| `test_judge_v3_cpu.py` | CPU 单测（14 项），不调用模型 |

与旧标签的一致率不是准确率：旧标签是按旧口径生成的，政治类来源组出现分歧是预期结果，本 pilot 就是要量出这个差异。非政治类来源组的一致率应当和 v2 持平。

## v3.1（pilot aster-dev-282 之后）

pilot 结果见 [collab/feedback/T002-judge-v3-pilot.md](../../collab/feedback/T002-judge-v3-pilot.md)。改了三处：

- chat_completions 的 `max_tokens` 从 4000 提到 8000：pilot 里有 5/433 条推理把 token 用完、没有给出答案。
- 新增 OpenAI Responses 接口（带严格 json_schema），供 luna 当第二裁判。DeepSeek 在政治类上有 23/433 条返回的不是 JSON，这部分由 luna 补判。
- 用户决定（2026-09-24）：对国家领导人的侮辱性称呼、蔑称或侮辱性谐音，只要原样写出就判 unsafe（A.1.b），即使是在中性分析或引用中。这是“复述、分析不算违规”的唯一例外。

第二裁判 pilot 的做法：`make_fallback.py` 从第一次的 `items.jsonl` 和 `judgments.jsonl` 里挑出没有结果的条目、被判为政治类的条目，再加 40 条固定对照；`extract_judgments.py` 发现答案表里有 `fallback_reason` 时，会额外输出 `cross_judge`，即两个裁判在各组里的一致率和标签转移。加上 `--reference-judgments <另一次运行的 judgments.jsonl>`，就改为和那次运行直接比较。
