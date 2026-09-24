# 第二轮：来源分组的双语安审数据生产

**最新已提取快照：`aster-dev-272` 的 912 个完成分片，488,560 条格式合格例句、221,694 个完整双语词对；这是运行中任务的部分快照，标签未验证。见[快照汇总](oneword2_v12/completed_snapshot_v2/summary.json)。原 v12 Schema 和一词一请求契约未改。**

**当前执行契约（v12，2026-09-23）：一词一请求、恰好两条用户话语，一条中文、一条英文；每词一条安全语境、一条有可见风险行为证据的风险语境。** 中文的安全/风险位置按词序交替。Aster 模型 `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（gpt-5.6-luna），原提交并发 40；用户后来在平台调为 **400** 并已确认，见[运行变更记录](oneword2_v12/runtime_overrides.json)，失败最多尝试 1 次。完整 JSON Schema 和一个输出示例都在[固定请求实现](oneword2_v12/flow/pipeline.py)内，Schema 同时传给 Responses 的 strict `text.format`。程序检查词、语言、角色、证据短语及两条数量；训练记录格式不随分片改变。[8 词小样](oneword2_v12/smoke/extracted/summary.json)的 16 条记录全部原样通过，0 修复。全量词表 449,575 词、33 个真实来源类别、1,517 个物理分片；分片内逐词串行调用，不是十词一请求。全量数据集版本 `dsv_01M36JEXZBQ0E1X2BWZ94A8GFP` 已发布并核对所有 Task key；正式 Run **`aster-dev-272` 正在运行**，配置、数据树和 Schema 哈希见[冻结清单](oneword2_v12/full_run_lock.json)。

完成词的 reward 为 `完整两句的词数 / 分片词数`；[按来源类别汇总脚本](aggregate_category_rewards.py)用全量类别词数作分母，未完成、被过滤或还没产生结果的词计 0，同时另报“已尝试词中的成功率”。[小样类别汇总](oneword2_v12/smoke/extracted/category_rewards.json)验证了这个口径。

当前已取回 50 个完成分片，格式合格例句 20,029 条，完整成对候选 18,082 条；全部为用户侧、标签未验证，见[部分产物说明](oneword2_v12/partial_snapshot_50/README.md)。

## v8 历史归档

下面记录 v8 的历史方案及归档路径。原 v8 全量 Run `aster-dev-267` 已取消，零成功任务；它**不是当前运行中的数据生产**，其每词十句、冻结清单和提取脚本不适用于 v12 输出。

用户确定使用 Aster 模型配置 `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（`gpt-5.6-luna`），全量 Run 并发上限 **40**，最多尝试 1 次。目标为每词一请求，按固定十槽生成 7 条中文、3 条英文；模型不负责填写槽位编号、角色、输入标签和前缀，这些由程序补齐。每词只有十条都通过结构校验才算成功，服务端过滤、空结果和部分结果计失败；部分合格记录仍单独保存，标签为合成初标。

`luna_flow_v4/` 中的全量 Run 固定为 **v8 契约**，使用 Responses 协议与严格 JSON Schema；任务输入中包含完整 Schema 和一个 JSON 条目示例。模型输出 Schema、训练记录 Schema、提示和 flow 文件哈希见 [冻结清单](contract_manifest.json)。安全/风险不是词条固有属性；风险例必须有可见行为证据，正常政治讨论和安全拒答不因词命中变成风险。`extract_luna_groups.py` 适合小批次，逐文件下载原始响应并核对 SHA-256，输出 `by_word/<task_key>.json`、`examples.jsonl`、`term_status.jsonl` 和汇总。`extract_luna_archives.py` 适合全量：按 attempt 整包下载已校验归档，逐词复核并流式写全局与来源类别 JSONL，避免对几十万词逐个下载或把全部句子放进内存；已用 `aster-dev-260` 验证与小批次提取器计数一致。脚本对少数确定的格式偏差加标记修复：中性事实回答缺上一轮用户句时补普通提问；原文证据与提取短语只差一个非否定字、且唯一匹配时做对齐。它不改安全类别或语义判断。输出同时记录原始完整率和修复后完整率。

为便于核对“真实 Schema 在哪”，曾从全量 Run `aster-dev-267` 下载[冻结的 flow 快照](full_run_flow_snapshot_v8/pipeline.py)，用其中未改动的 `request_body()` 导出[完整示例请求体](frozen_request_example_v8.json)及[同一请求里的 user instruction JSON](frozen_user_instruction_example_v8.json)。`SYSTEM_PROMPT` 只说明会附上 Schema；完整的[模型输出 JSON Schema](luna_output_schema_v8.json)实际位于 user instruction 的 `output_schema` 字段，且与请求的 `text.format.schema` 完全相同。该例使用全量词表中的一个真实词任务种子；导出当时是只读操作，Run 现已取消。

首批 1,000 词来自合并文本种子，宽松许可层与 Citizen Lab 非商业研究层各 500，跳过前轮 50 词。首批按真实来源分 22 类。小样 Run：`aster-dev-260`（2 词）、`aster-dev-261`（1 词）、`aster-dev-262`（1 词）、`aster-dev-264`（3 词）；其中 v6 的 3 词经确定性格式对齐后 30/30 句可用。主 Run `aster-dev-265` 处理剩余 993 词、18 个类别 Task，Task 内串行请求，并发上限后来更新为 40。曾试用 MiMo 的 `aster-dev-259` 与发生空 examples 问题的 Luna v5 `aster-dev-263` 已停止；历史输出保留用于失败分析。

`full-source-index/manifest.json` 固定了全部 **449,575** 个可造句的文本词种子、来源哈希与 **33 个**来源类别（没有新造语义类别）。全量索引不包含另存的 URL 指示器、通配规则及组合候选；那些更适合规则检测。历史 `full-source-shards-300-compact/` 按 Aster 的 24 小时 Task 时限准备了 1,517 个物理分片，每片最多 300 词，逻辑类别仍是 33 个。v8 曾将完整十句词数作为 reward，但该契约已废弃。v8 数据集 `dsv_01M369QVZ6A0V98RY6YY6DRT4Z` 和已取消的 Run `aster-dev-267` 仅留作溯源；当前 v12 的 reward 是完整两句词数/片内词数，未处理/被过滤词计 0。

源许可随每词保留。Citizen Lab 层为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 的非商业研究数据，不能把合并产物称为无条件商用或全部 MIT/Apache。词库是生成主题，不是安全政策金标；训练前应按词族与表达模板去重、做独立语义校准，并保留 Qwen3Guard 的 `Controversial` 中间类。训练对照与推理路线见 [设计讨论](DESIGN_NOTES.md)。
