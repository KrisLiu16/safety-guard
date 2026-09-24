# S2：非拒答难负例（T006）

[DATA_SPEC_v15](../DATA_SPEC_v15.md) 第 4 节的 S2 切片：话题敏感、但内容正当的提问和回答。目标是现在误报最多的地方。[decision_rule](../decision_rule/RESULTS.md) 发现，校准之后误报集中在安全推理的中段：模型把“复述有害请求、但最后正常处理”的文字判得很危险。

每个词一次请求，产出三段文字，全部应当是 safe：

| 文字 | 说明 |
|---|---|
| 提问 | 用户以正当意图使用这个词。形式由程序轮换分配：分析一段引用、问背景知识、以审核员/老师/家长身份问怎么处理、问健康安全法律常识 |
| 正式回答 | 有实质内容、不拒答、包含这个词 |
| 思考过程 | 第一人称内部思考：先复述用户要什么（可以原样提到敏感词），判断请求正当，再规划回答。针对的正是上面那类误报 |

- 立场按国内口径：尊重制度、民族团结、英烈和历史定论，不写可执行的伤害细节。
- 如果这个词本身是对国家领导人的侮辱性称呼，或者无法在正当内容中出现，模型返回 skip（与 judge v3.1 的严格口径一致）。
- 词来自 Run B 的种子（与 Run A 按词族不重叠）。S2 按词族哈希自留约 3% dev、3% calibration。
- 生成后必须经 judge v3.1 用**另一个模型**复核（生成用 DeepSeek，复核用 luna）。被判为 unsafe 或 controversial 的直接丢弃，不改标签。

| 文件 | 作用 |
|---|---|
| `flow/pipeline.py` | 提示词、schema、请求体（chat_completions 和 responses 两种接口）、程序校验 |
| `flow/flow.py` | Aster flow，每个词只请求一次，flow 内不重试 |
| `make_pilot.py` | 从 Run B 种子按来源组轮流抽词，语言中英交替，形式轮换，每个词一个 Task |
| `extract_s2.py` | 下载归档并重新校验：`examples.jsonl` 是助手行（v14 行格式，judge_v3 的 `make_pilot.py --source` 能直接用），`prompts.jsonl` 是用户侧提问行 |
| `test_s2_cpu.py` | 6 项 CPU 单测，包括与 judge_v3 构建器的兼容性 |
