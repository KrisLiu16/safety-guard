# 第六轮造数 v13：回答侧 2×2 配对 + 风险起点

2026-09-24。写这版的原因见[第五轮诊断](../../round5/DIAGNOSIS.md)：模型把“提问危险”当成“回答危险”，训练数据里提问标签基本决定了回答标签。v12 的格式和产物不变，本目录是新版本。

## 格式约定

- 一词一次请求，输入是 v12 里同一个词的两条固定用户提问（一条 safe、一条 unsafe，一中一英），模型不改写提问。
- 每词写 4 条助手回答，按“提问 × 回答”做 2×2，回答语言跟随提问：

| slot | 提问 | 回答 | 写法（按 rotation 轮换） |
|---|---|---|---|
| 0 | unsafe | safe | refusal / safe_redirect / risk_reasoning（思考体）/ neutral_information |
| 1 | unsafe | unsafe | compliance；answer 或 reasoning × 立即或延迟起点 |
| 2 | safe | safe | helpful_answer 或 helpful_reasoning |
| 3 | safe | unsafe | drift：先正常回答再偏离，延迟起点 |

- **safe 回答**：从任意位置截断都安全，不允许先写有害内容再否定。因此它的每个前缀都能作为确定的 safe 标签。
- **unsafe 回答**：`onset_quote` 必须逐字取自回答。由程序定位字符位置 `onset_char`/`onset_end_char`，不让模型报数字。起点之前的文字是 safe，读完 `onset_quote` 之后是 unsafe，片段内部不打标签。
- **判定对象是回答已写出的内容**：复述或分析用户的有害意图算安全；实际提供有害内容或明确答应去做，才算有害。
- 完整 JSON Schema 和一个格式示例（`example_output`，语言顺序与真实请求一致）都放在 user instruction 里，同时通过 Responses strict `text.format` 执行。
- reward = 4 条全部合格的词数 / 词数。被模型拒绝、被过滤或只部分合格的词记 0；部分合格的记录另行保存。
- 全部标签为 `luna_synthetic_unverified`，不是金标。

## 文件

- [flow/pipeline.py](flow/pipeline.py)：提示词、Schema、槽位计划、校验。
- [flow/flow.py](flow/flow.py)：Aster 流程，一词一请求，不在流程内重试。
- [make_pilot.py](make_pilot.py)：从 v12 完整快照挑 50 词，按来源类别分层，排除 round4/round5 已用过的 6,320 个词。
- [test_pipeline.py](test_pipeline.py)：合同测试（CPU）。
- `pilot/`：50 词、27 个 Task（一类一个 Task），以及 `manifest.json` 中记录的 SHA、`example_request.json`。
