# 自有安审模型计划：L20 第四轮训练与流式分类交付

> 交接文档：第六轮最新见 [round6/HANDOFF.md](round6/HANDOFF.md)；第一到五轮完整历史见 [HANDOFF.md](HANDOFF.md)。

**当前状态（2026-09-24）：第五轮已跑完，但没有检查点通过质量门槛，保留的是第四轮 window/SFT 初始权重（bb16a3a6…），状态为 research_unvalidated_runtime，见[第五轮端到端结果](round5/END_TO_END_RESULTS.md)。复盘见[第五轮诊断](round5/DIAGNOSIS.md)：流式误报主要来自“提问风险泄漏到回答判定”，根源是训练数据中提问标签基本决定了回答标签；第五轮门槛的差异大多是阈值落点噪声。第六轮先补 assistant 侧配对数据和 unsafe 起点标注，方向待用户确认。下文第四轮的描述是历史记录。**

**第四轮（历史）：第四轮已启动，以真实风险识别、误报和漏报选模，不再以教师相似度选模。33,311 条训练记录、完整/窗口/可学习记忆三组同预算对照及离散分类 RL，见[第四轮执行计划](round4/README.md)。模型与数据原格式保持不变。**

当前状态、完成要求和实测产物分别见[实时状态](round4/LIVE_STATUS.json)、[完成验收清单](round4/COMPLETION_AUDIT.md)、[本轮训练结果](round4/ROUND4_RESULTS.md)及[端到端交付结果](round4/END_TO_END_RESULTS.md)。完整主干已完成本轮训练，窗口与记忆、RL 按流水线继续；实际文字分块回滚和独立模型包将在训练后另用同一 L20 验证。下文第三轮及更早内容保留为历史背景，以第四轮文件中的已运行证据为当前进度。

**当前执行顺序（2026-09-23）：以[当前执行计划](round3/ACTIVE_EXECUTION_PLAN.md)为准。主线是完整预训练主干 + 分类头预热/全参数后训练，再做流式结构改造、恢复训练和 RL。LoRA 仅作为可选对照；下文旧阶段的训练方式不代表当前默认。**

**当前硬件决定（2026-09-23）：第三轮的训练、蒸馏、RL 与正式性能验收只用腾讯云 L20。** 下文的 M5 首轮记录为已完成的历史实验；第三轮以[L20 执行边界](round3/L20_TRAINING.md)为准。

**第三轮架构与训练设计（2026-09-23）：**[流式分类器设计与数据/奖励方案](round3/ARCHITECTURE_TRAINING_PLAN.md)讨论定型概率输出、专用因果学生、公开语义与安审数据、分类概率训练和流式动作 RL；[H1/H2 固定状态高速架构](round3/FAST_STREAM_ARCHITECTURE.md)单列为实测实验。[分类服务输出 Schema](round3/guard-decision-v1.schema.json)独立于第二轮 v12 造数格式。

第三轮当前可执行模型是直接分类器：每个输入 chunk 立即给三档风险和类别分布，见[当前契约及 ITPS 定义](round3/CLASSIFICATION_AND_ITPS.md)。速度主指标为**完成分类的新增输入 token/s（ITPS）**与分类响应延迟，不存在生成 token TPS。目标策略 Schema 的证据充分、拒答与动作字段尚待训练和校准。

第三轮已开始执行；[DAG 与当前节点状态](round3/EXECUTION_DAG.md)记录开放中英语料审计、A0/C1 功能探针、14 层学生初始权重，以及 L20 上已完成的 1k Stage0 pilot 和 [20k/2k 续训](round3/l20/STAGE0_20K_REPORT.md)。开放语料的教师蒸馏不使用安全标签；最终政策安全对齐仍需独立金标与项目政策映射。

[首轮 L20 ITPS 与质量对照](round3/l20/L20_CLASSIFICATION_ITPS_REPORT.md)表明 C1 的输入速率约为 A0 两倍，但官方子集误报与漏判明显恶化；这版 C1 不进入服务主线，先做能力恢复，再比较速度。

**第二轮（2026-09-23）：**当前数据生产为来源分组、gpt-5.6-luna；用户已在平台将并发从原提交的 40 调为 **400**，见[运行变更记录](round2/oneword2_v12/runtime_overrides.json)。仍为每词一请求、一条中文和一条英文。已固定 449,575 个文本种子和 33 个来源类别；正式 Run [`aster-dev-272`](round2/oneword2_v12/full_run_lock.json) 运行中，已从 50 个完成分片提取 20,029 条格式合格记录，见[部分产物审计](round2/oneword2_v12/partial_snapshot_50/quality_audit.json)。原提交锁、数据格式与生成提示未变；每词十句的 v8 Run 已取消。

日期：2026-09-22。

**当前执行依据：[M5 第一轮端到端计划](round1/PLAN.md)及[配置规格](round1/config.json)。先在本机完成数据→基线→RL→评测→推理优化；Luna 后续并发为 50。下文为完整研究背景，其中早期 L20 前置要求、大范围实验矩阵和先前并发值已由第一轮计划覆盖。**
当前状态：M5 第一轮闭环已完成。Luna Run `aster-dev-254` 产出 4,950 条原始例句，清洗后保留 4,095 条完整记录及 74 条训练前缀；已完成 159 步 RL、全量官方评测、推理优化、权重导出和本地演示验收。主要收益是缓存提速，能力增益有限且有小幅退化，详见 [第一轮结果](round1/ROUND1_REPORT.md)。造数比目标少 50 条，失败任务未补跑。本轮结束，不自动开始第二轮。

**1. 本版决定与需要验证的假设**

采用用户指定的 Qwen/Qwen3Guard-Stream-0.6B 为唯一首版基座，以 RL 继续训练为主；不再要求先收集 10 万条人工 SFT 数据。用 Sensitive-lexicon 提供词汇和主题种子，构建可自动扩展的审查任务环境。核心公开能力观测统一使用 Qwen3Guard 官方评测与 Qwen3GuardTest。推理优化是最终必做交付，必须同时报告能力、延迟、吞吐、缓存内存及适用负载。[1][2][3]

需要验证的假设：已有 Guard 的能力足以支持少量校验样本驱动的奖励训练；词库派生的上下文任务能产生目标领域增益；在真实 L20 和目标并发下，缓存、kernel、调度与选择性计算能改善质量约束下的服务效率。它们均不是提前成立的结论。

RL 可以减少对人工示范回答的依赖，但仍需要输入分布、政策边界和可信反馈。词库只给候选主题，无法单独决定一句话是否违规。若奖励就是“包含词条则拦截”，模型最多学成昂贵的词匹配器，无法证明语义审查有改善。模型发布日期也不能直接说明更换 MoE/DSA 一定获益。

**2. 已核实的起点**

| 项目 | 本次核实值 | 对方案的影响 |
|---|---|---|
| 模型 revision | 419364a715de9840d47b1457982f64ff37f90ed4 | 初始基线固定版本，后续变更单独登记 |
| 结构 | 28 层稠密 Qwen3 主干；hidden_size=1024，intermediate_size=3072 | 现成权重不是 MoE |
| Attention | 16 个 query head，8 个 KV head，head_dim=128 | GQA；不是 MLA，不能直接用 MLA 权重格式 |
| 分类头 | 用户风险 3 类/类别 9 类；回答风险 3 类/类别 8 类 | 首轮复用已有头；不生成答案 token 或 CoT |
| 配置上下文上限 | max_position_embeddings=8192 | 所有历史、角色边界和本轮新增内容合计必须在 8192 内；不默认测 16k 为受支持能力 |
| guard_inner_size | 512 | 轻量分类层的开销也纳入 profiling |
| 词库 revision | d967c30b053fa40b06c5a0dddf0be493f2dfae46 | 固定词表、文件清单与来源，防止实验间漂移 |
| 词库实际目录 | 主要文件位于 Vocabulary/；Organized/政治类型.txt 在所查树中只有 1 byte | 不把 README 的“数万条”当清洗后样本量；下载后核验编码、分隔符、空文件与重复 |

结构信息来自模型 config 和实现，词库文件信息来自对应 Git 树。[3][4][5]

标准 BF16 KV 不计分页/对齐等额外开销时，每会话每 token 约为 2 × 28 × 8 × 128 × 2 = 114,688 bytes，即 112 KiB；8192 token 约 896 MiB。多会话缓存可能比约 0.6B 权重占用更快成为容量瓶颈。该数值是理论估算，实际按实现和实测显存报告。

**3. 成果定义和实验对象**

最终交付一个支持 batch/stream 的 Guard 服务，以及三个可比较版本：

- Original：原始权重、固定标签/阈值，建立真实能力基线。
- RL-Dense：同一稠密结构，完成 RL 能力优化。
- RL-Optimized：从 RL-Dense 出发完成推理优化；包含工程优化版本，以及通过质量门槛的结构优化版本。

区分 Original-PublishedWrapper 与 Original-CorrectCache：修复公开包装器重复计算后的版本才是比较新 kernel/结构的主要速度基线。不能把所有收益都归因于“新注意力”，实际只是修复缓存错误。[5]

政治公共事务主题标签与具体政策违规判断分开；人物、机构、历史名词命中不自动判违规。对正常新闻、历史讨论、引用、一般批评和具体有害行为设置反例。输入请求与输出回答独立判断，保证“危险问题＋安全拒绝回答”不会把回答也罚成危险。

**4. 从词库构建 RL 环境，而不是把词条当答案**

词库处理：固定 revision → 下载 Vocabulary 文件 → 保留原词、来源文件、原始类别 → 规范化副本 → 去重/别名分组 → 显式映射到本项目类别 → 人工抽样核对。不同来源的第三方词表保留来源标识，不能只靠顶层 MIT 就声称每项来源都核验完整。仓库目录名不是官方政策类别。[3]

词库是任务 seed，不是统一二元真值，也不要求把完整词库塞进模型上下文。保留 CPU 增量匹配器作为基线与可选特征来源。

每个合格 seed 生成多个上下文任务族：

| 任务族 | 环境构造 | 奖励所需证据 |
|---|---|---|
| 正常提及 | 新闻说明、历史讨论、定义、引用后反驳等 | 按明确政策允许的上下文；不能仅靠“新闻”字样判安全 |
| 确定违规行为 | 依据已审核的政策模板构造请求/回答 | 有可执行规则或经审核的语义标签 |
| 同词不同语境 | 保留相同关键词，改变行为意图和关系 | 独立评估两个实例，不要求同词同判 |
| 等义变体 | 繁简体、空格、零宽字符、拼写变化 | 仅当语义与政策判断确实保持时继承标签；不确定变体重评 |
| 多轮任务 | 前文定义实体、后文代词引用、跨轮累积信息 | verifier 阅读同等可见历史，给当前角色标注 |
| 流式任务 | 完整样本派生多个 chunk 切法、截断点和风险位置 | 每个前缀单独判断，允许未充分可判定 |
| 无词命中正常/风险 | 独立授权文本、人工种子或合成样本 | 防止“没词就放”“见词就拦”的捷径 |

数据不提前做满，先 1k–2k 审核过的环境实例，再运行 5k/20k/50k 独立实例的学习曲线。一个 seed 派生的翻译、改写、前缀和两种回答属于同源组，不算独立数据来源。同一静态输入重复抽样动作不等于生成了新的训练语料。

生成器不使用 Qwen3Guard-Stream 自己撰写文本，因为它不是生成模型。先用程序模板和已有获准文本；语义扩展可用本地获准生成模型或人工补写。收费 API 或外部私有数据并非默认前置条件。

完整样本必须覆盖：安全问/安全答、风险问/安全拒答、风险问/违规答、安全问/违规答。回答开头的安全拒绝措辞不能为后续违规内容免责。

**5. 奖励来源与小规模人工投入**

奖励可靠性优先级：

1. 有明确定义的可验证任务：规则/仿真器返回结果，适用于构造受控例子、缓存/切块测试、隐私标记等。
2. 小量人工政策种子和仲裁：确定业务边界，给语义 verifier 校验提供真值。
3. 冻结的独立语义评审器：仅用于已校验任务族，带低置信度弃权；高争议例转人工或不更新，不让 student 自己给自己发奖励。

最低建议约 600 条人工校验：200 条训练奖励锚点、100 条 dev、100 条校准、200 条冻结中文 sanity；按原始词/实体族和模板隔离。数量是低成本起点，不能据此证明千分位误拦率或稀有类别召回。该小集用于校验奖励与中文迁移，不恢复上一版的“大规模 SFT 数据前置要求”。如果只使用官方 bench，结论限定为官方测试分布，不能证明中国政治与业务政策审查达标。

verifier 不读取 student 的自报解释、奖励函数文本或它希望获得的答案；待审文本全部作为不可信数据处理。评审器版本、政策版本和提示固定，并与最终测试裁判隔离。规则和语义评审器分歧时记账，不能默默选能让奖励更高的一方。

词库命中自身不能获得主要正奖励。奖励必须对漏判、误拦、滥用 review/hold、前缀未完成时过早拦截、检测太迟分别记账。每轮记录 reward 与独立 dev 指标的相关性；奖励上升而 dev 恶化视为 reward hacking 警报。

**6. RL 算法：先有限动作，再流式序列决策**

阶段 RL-A：已有风险分类头视为有限动作策略 πθ(y|x)，y 为原生 safe/controversial/unsafe。x 包含合法角色边界、历史和当前目标消息。先不改主干架构，也不要求生成推理文本。类别头只有获得可靠的类别奖励时才训练，否则保留并监控漂移。

起点采用带参考 KL 的 policy gradient / PPO-style clipped 更新；优先训练 LoRA rank 16 和已有分类头，主干初始冻结或由 LoRA 调整。参考模型固定为原始 Guard；role 分别记录损失和奖励。选择具体优化器前做小型梯度/动作概率正确性检查，不直接套用面向长文本生成的 trainer。

分类任务拟议奖励（代价系数在 dev 上设定并冻结）：

```text
R_task = 正确判定收益
         - c_miss × 漏判
         - c_false_block × 正常内容误拦
         - c_wrong_review × 无根据地归为争议/复核
J = E[R_task] - beta × KL(pi_current || pi_reference)
```

R_task 不直接使用测试标签。安全与风险样本比例、代价权重、奖励分布分别报告，不能通过全拦或全放获得高平均奖励。对正确的真实争议标签不能一概处罚；只限制利用不确定标签回避决策的捷径。

静态动作数只有 3，同一输入 group 中重复抽样通常很便宜但多样性有限。GRPO 可以作为对照，group size 从 4/8 开始；不为每个 sampled label 重算一次整个主干。若组内 reward 全相同，记录 zero-variance group，不人为制造 advantage。动作空间可完全计算、且 verifier 能评价所有动作时，增加“精确期望奖励梯度”对照以降低采样方差；明确它是同一决策目标的可枚举优化，不包装成神奇 RL 增益。GRPO 的来源与适用边界参考 DeepSeekMath，但本任务并非长推理生成。[6]

成本敏感的动作奖励训练出的 π 不自动等于真实风险概率。先优化决策收益，再用独立 calibration 拟合温度/阈值并报告 Brier/NLL；小样本校准不作强保证。若引入 Brier 或 KL 蒸馏作为直接可微辅助项，明确称为“RL + 辅助目标”，不能把它伪称纯采样 RL。

阶段 RL-B：将流式控制作为真正的序列决策任务。

- 状态：审查器当前可见前缀的隐藏表示、当前角色、buffer 长度、已释放位置、历史决策状态。
- 动作：release / hold / block；最终不可无限 hold，超时进入有成本的 review。先对照固定阈值控制器，再训练额外轻量策略头。
- 转移：offline replay 环境提供下一块；block 终止，release 更新已展示范围，hold 增加缓冲。
- 回报：减少风险内容释放、降低正常误拦与等待成本、惩罚已具备可判定证据后的延迟；报告各分项而非只给总 reward。
- 终局回报可利用完整已标注样本评价结果，但绝不把未来 token 放进 policy 状态；也不因后文才出现风险，就处罚在此前正确放行的无害内容。

```text
R_episode = 正确处理收益
            - c_leak × 已释放风险内容量
            - c_fp × 正常会话错误阻断
            - c_wait × 用户实际等待时间
            - c_review × 复核成本
            - c_lag × 已可判定后的检测迟延
```

先使用按 token/时间建模的确定性离线成本，再在固定 GPU 环境实测端到端代价。不要把抖动很大的单次 GPU 计时直接作为主奖励。未检测的风险例不能从 lag 统计中消失。

训练必须保存 behavior log-prob、策略版本和 reference KL；PPO 多次更新需要按当前权重重算对应前向。训练步之间不得把旧权重下的 KV 缓存直接当新权重特征复用。RL 推理采样的缓存优化与可反传训练计算图分别实现。

**7. Bench：按用户选择收敛到官方体系**

首版唯一必选公开安全 benchmark 是 Qwen3GuardTest，使用官方 eval 脚本与定义。保持以下分片区别：[2]

| 分片 | 内容 | 观测 |
|---|---|---|
| thinking | 1,059 条带推理内容的样本 | 完整回答安全判定、官方 strict/loose 指标 |
| thinking_loc | thinking 中 569 条风险样本，首个风险句位置 | 带推理内容的流式定位与检测延迟 |
| response_loc | 813 条最终回答风险样本，首个风险句位置 | 普通回答流式定位与检测延迟 |

位置分片全部是风险样本，不能单独给出正常内容误拦率；分片有包含关系，不能相加成 2441 条互不重复独立测试。其主要覆盖不是本项目完整中文政策域，因此官方 bench 负责通用回归，小型中文 sanity 负责暴露领域偏移；没有后者时写明无法验证中文目标增益。

官方数据绝不进入 RL 环境、奖励模型训练、样本生成 seed、阈值选择或蒸馏目标。开发过程看独立合成 dev 和中文 dev；只在基线、RL 候选冻结、优化候选冻结三个里程碑看官方 test。反复看 test 做设计后，将相关成绩标记为开发观察，不再称独立最终泛化证据。

Qwen3GuardTest 标注为 CC BY-NC 4.0，使用范围按数据许可登记；公开权重许可不替代测试集许可。[2]

基线矩阵收敛为：

| 版本 | 意义 |
|---|---|
| B0：词库/规则 | 验证学习模型是否超过词匹配 |
| B1：Original-PublishedWrapper | 用户下载即用的完整基线 |
| B2：Original-CorrectCache | 相同权重、正确缓存；结构优化的公平速度起点 |
| R1：RL-A | RL 对静态能力和回归的贡献 |
| R2：RL-A + RL-B | 学习流式控制是否优于固定阈值 |
| O1：R2 + kernel/调度/量化 | 不大改模型架构的收益 |
| O2：R2 + 结构实验候选 | DSA-inspired/深度裁剪等，质量与性能一起验收 |

4B 仅为可选质量参照或冻结 verifier，不作为第一轮必跑依赖。CHiSafetyBench、SafeDialBench、FLAMES 等移到后续扩展，不占首版必选范围。

**8. 推理优化原则：先确定耗时在哪里**

分解总时延：

```text
T_total = 网络与排队 + 分词/规范化 + prefill
          + 新增块 Transformer + 分类/聚合
          + buffer 等待 + 调度/数据拷贝
```

分别测短句首判、长历史首判、长历史续流、高并发四种负载，不能用一个平均 tokens/s 代表所有场景。对于没有自回归答案生成的 Guard，主要目标是减少审查输入计算、缓存读取及服务等待；MTP/投机生成对它不是首选加速，因为它根本不生成自己的长答案。

应用 Amdahl 定律：若 attention 只占总时延 20%，即使消除全部 attention，最多也约 1.25×；若小矩阵、kernel launch、CPU 调度或 buffer 占主导，更换 DSA 不会自动解决。profiling 的结果决定结构实验的优先级。

性能实验与训练分卡或错峰，排除资源竞争。GPU、驱动、CUDA、kernel、batch、上下文长度、缓存命中率、精度与编译热身全部固定并记录。

**9. 推理优化路线 A：现有结构上的必做项**

| 项目 | 原理 | 验证条件 |
|---|---|---|
| 正确增量 KV Cache | 避免每步重复计算已处理的 token | 相同前缀缓存/全量输出一致；报告相对 published 与 correct-cache 两个基线 |
| 前缀共享/会话缓存 | 重复历史仅 prefill 一次，同会话分支复用 | 原始 token、角色模板、模型/adapter/位置编码一致；隔离会话；修改历史后正确失效 |
| SDPA/FlashAttention-2 等实际支持的 kernel | 避免中间 attention 矩阵落显存，改善访存与并行 | 在实际 GPU 与 128 head_dim 上测 kernel；不只更改配置字符串就认定生效 |
| torch.compile / CUDA Graph / 算子融合 | 减少 Python、launch 和小算子开销 | 形状分桶、动态 cache 兼容、编译耗时另报，端到端收益为准 |
| Continuous batching / 长度分桶 / deadline 调度 | 提高 GPU 利用率，限制短请求被长 prefill 阻塞 | 报相同 P95 SLO 下的可持续吞吐，不通过无限排队刷吞吐 |
| BF16 对照 INT8/低比特权重、KV 量化 | 减少带宽与驻留内存 | kernel 真支持当前 GPU、形状和 head；量化误差、校准漂移和漏检同时检查 |
| 自适应审查块大小 | 在风险和时延约束下减少每 token 调度 | 1/8/16/32 token 与按句对照；buffer 等待计入用户时延，检查跨块绕过 |
| 轻量/完整路径分流 | 高把握简单片段走低成本路径，疑难保留完整模型 | 简单规则不能单凭无词命中放行；评估复核比例和长尾风险 |

首版上下文长度网格为总长约 256/1k/4k/8k，8k 实验预留新增 token 和模板预算；8192 边界单测。并发 1/8/32，超过显存或 SLO 的点记为不可用，不静默缩短历史。先测同 tokenizer 的 token 流，后测异 tokenizer 的文本流、尾部重分词和 cache rollback。

block 调度内也保留必要的逐 token/逐句读出与风险聚合，不能只查看最后一个 token 的类别而遗漏中途风险。一次性和流式使用同一风险定义与消息结束处理；类别最终判定与“已释放过的内容风险”分别记录。

**10. 推理优化路线 B：参考 DeepSeek 的结构实验，必须做适配和消融**

结构研究列为必做可行性/瓶颈分析阶段；完整改造是否进入最终服务由实测胜负决定，不强迫上线更慢或更差的结构。

| 技术 | 从原理上节省什么 | 对现有 Qwen3Guard 的适用判断 |
|---|---|---|
| DSA：索引器选择少量历史 KV | 减少主要 attention 对全部历史的读取/计算 | 可研究 GQA 上的 DSA-inspired 选择器；不是直接复用 DeepSeek 原生 MLA/DSA 权重 |
| NSA：压缩、选择、局部窗口组合 | 兼顾硬件友好稀疏与局部/全局信息 | 可作为设计参考，需要训练适配和对应 kernel |
| MLA | 将 K/V 压缩到低维潜在表示，减少缓存 | 当前 GQA → MLA 涉及投影与位置处理改造，需结构转换/蒸馏；不是一行开关 |
| MoE | 在更大总参数容量下只激活少数专家 | 0.6B dense 改 MoE 不一定少读权重或少算，可能增加路由和碎片化；必须同质量、同实际延迟比较 |
| 减层/窄化/早退 | 降低有效层数与每 token 计算 | 可能更适合小模型低延迟；跨层早退会导致深层 KV 缺失，需补算/固定深度策略，不能直接跳层后继续复用不完整 cache |

DeepSeek V3.2 的 DSA 是学习型索引器 + top-k KV 选择，并经过 dense warm-up 和稀疏训练。主要 attention 的复杂度降低不代表整个系统严格线性：原始索引器仍需要比较历史 token。对最多 8k 的小模型，索引/选择成本可能抵消节省；应测实际交叉点。[7]

DSA-inspired 实验拟分四步：

1. 在 dense Guard 上采集训练环境中的参考 attention/输出分布；只在训练数据上做结构校准，不触碰官方 test。
2. 冻结主干训练小型 causal indexer，先尝试 top-k 128/256/512，配合保留当前块、近期窗口、角色边界和受保护的证据位置。词库线索只是附加，不是唯一保留策略。
3. 先局部替换部分层，做 output-KL/representation consistency 的结构适配，再用原任务 RL 恢复业务能力。该阶段显式包含蒸馏/一致性损失，不声称纯 RL，也不需要人工长答案 SFT 语料。
4. 在负载网格下比较完整 indexer + top-k + gather + sparse attention 的端到端成本，单列长距离否定、远处指代、引用反驳、多轮依赖漏检。质量或延迟不合格则保留 dense 路径。

仅在 PyTorch dense attention 上加一个稀疏 mask，不会自动省掉主要计算。必须有实际稀疏 kernel/稀疏 gather 路径；若所需 kernel 在 L20 上没有高效实现，就把结果记录为不适配，不用理论 FLOPs 冒充加速。

MoE 可做小型可行性实验：在部分 FFN 层采用 top-1 路由与专家，比较固定总参数和固定激活参数两种预算、路由均衡、激活量、kernel 开销与召回。如果只是复制全部 FFN 成多个专家，单次激活量未下降，不能预期比原始 dense 更快。该分支低于缓存、量化和稀疏可行性验证优先级。[9]

FlashMLA 当前官方支持矩阵主要列 SM90/SM100，不能把 H800/B200 的数据外推到用户的 L20。实际使用先查 GPU compute capability 与支持矩阵；FlashAttention-2 明确支持 Ampere/Ada/Hopper，是现有 GQA 路径优先实测候选。[10][11]

DeepSelect 的 top-k 优化可参考；它加速的是索引器/采样子步骤，只有 top-k 已成为我们实测瓶颈且 kernel 兼容时才引入，不能把其局部倍数当全服务倍数。[12]

**11. 能力、延迟和吞吐的统一成绩单**

| 指标组 | 必须保存的观测 |
|---|---|
| 官方能力 | 各分片 strict/loose F1、precision、recall；位置命中和 128-token 检测指标，遵循官方脚本定义 |
| 中文迁移 | 小型冻结 sanity 的误拦/漏检、同词不同语境、正常政治讨论与具体风险；报告样本数与置信区间 |
| RL 健康度 | reward 分项、KL、熵、zero-variance group 比例、allow/block/review 分布、奖励与独立 dev 的关系 |
| 流式质量 | Recall-by-token-budget、检测者条件 P50/P95 lag、漏检比例、每正常会话任意误拦概率、已释放风险量 |
| 响应延迟 | 冷/热首判、prefill、每新增块 GPU 与端到端时延，P50/P95/P99；buffer 等待及排队单列 |
| 吞吐 | 审查输入 tokens/s/GPU、messages/s、在固定 P95 SLO 下的最大可持续请求率；不报无意义的输出 token/s |
| 容量 | 权重/激活/KV 峰值显存、每会话缓存、缓存命中率、最大并发、OOM/超时率 |
| 质量约束下效率 | 同一阈值初测数值漂移；再在独立校准集重新匹配质量门槛，比延迟/吞吐 Pareto 前沿 |

第一轮验收采用相对目标而非预设硬件毫秒承诺：

- RL 能力目标：官方核心指标不显著退化（候选阈值：下降不超过 1 个百分点），中文 dev 在相同误拦/复核预算下改善；入围配置用 3 个 seed 复跑。
- 推理优化目标：相比同权重、correct-cache BF16 基线，目标负载 P95 延迟下降 30% 或固定 SLO 吞吐提高 50%，同时报告另一项是否退化。这是实验目标，不是承诺达成值。
- 同时要求关键类别漏检和正常误拦不突破预先冻结门槛；量化或稀疏模型不能只靠降低阈值/增加 review 掩盖退化。
- 若短文本低并发与长文本高并发的最优配置不同，交付两个部署 profile，明确适用边界，不把不同配置的最佳数字拼成一个不存在的系统。
- batch/chunk 改变带来的收益必须计入用户等待；质量和性能在同一配置、同一请求分布上测量。

所有性能门槛在基线 profiling 后冻结；后续不能看见 test 成绩后修改通过条件。小样本 sanity 只用来发现明显失败，不能宣称统计上证明 1% 以下的业务误拦。

**12. 执行阶段、停损条件和产物**

| 阶段 | 必做工作 | 产物 / 停损条件 |
|---|---|---|
| P0：固定输入 | 模型/词库/bench revision、政策 seed、许可与环境格式 | manifest；无可信 reward 的任务族先不训练 |
| P1：原版基线 | 官方能力、published-wrapper 与 correct-cache 速度、硬件 profile | baseline_report；增量结果不一致先修正确性 |
| P2：奖励验证 | 1k–2k 任务 pilot，少量人工锚点，B0 对照 | reward_audit；见词就拦、奖励与 dev 脱节则改环境 |
| P3：RL-A | LoRA + 分类头，小动作策略；5k/20k/50k 课程 | RL 权重、学习曲线；同等数据可枚举目标/GRPO对照；无收益先改 reward |
| P4：RL-B | replay 流式控制、hold/release/block 成本 | 流式策略与固定控制器对照；不能全 hold 或过早 block |
| P5：优化 A | KV、kernel、编译、调度、量化、缓存容量 | 必交性能报告和可部署 profile；质量不合格配置剔除 |
| P6：优化 B | DSA/NSA/MLA/MoE 原理与适配评估，优先一个有 profiling 依据的结构原型 | 完整端到端消融；无兼容 kernel/无交叉点则保留 dense 并报告证据 |
| P7：冻结验收 | 官方 bench、中文 sanity、性能负载矩阵、3 seed 稳定性 | 能力-延迟-吞吐 Pareto 报告、模型卡、推理服务与已知限制 |

P5/P6 是最终交付的一部分；不能在 RL 跑通后把推理优化留成一句“以后做”。结构优化需要的校准/蒸馏仅使用训练环境实例，和能力 RL 分开计预算与归因。

拟议首轮超参数：BF16、LoRA rank 16，主干 adapter LR 从 1e-6/3e-6 两点试起，分类头 LR 1e-5，gradient clip 1.0；PPO clip 0.1 起步，group size 4/8 仅用于 GRPO 对照；总 rollout/update budget 先按 200–500 step pilot profiling 再定。上述均为待验证起点，不按模型生成任务直接照搬默认值。

已有两张 L20 的资源和权限尚未实测确认：本地 API 网络连通，但当前凭证在已测试范围无法列节点/全部 Pod，也无法在 default 创建 Pod/Job。训练工作等待可调度 namespace、GPU request、镜像和存储落实；不修改云端权限作为本计划的一部分。资源可用后先每卡独立实验，不默认跨机 DDP。

**13. 最小交付文件**

- manifest.json：模型、词库、bench revision 和许可记录。
- reward_spec.md / reward_audit.json：政策定义、可验证条件、代价权重、评审器版本、抽检及分歧。
- environments.jsonl：同源 seed、任务族、可见上下文、隐藏评估结果、split；隐藏字段不可进入 policy 输入。
- baseline_report.md：官方能力与两种原版缓存实现的真实性能。
- rl_runs.jsonl：seed、策略版本、reference、rollout 数、有效独立实例数、奖励、KL、耗时与失败状态。
- inference_ablation.md：正确缓存→kernel→调度→量化→结构的逐项归因。
- release_report.md：同配置的能力、延迟、吞吐、显存、已释放风险量，及不支持范围。
- moderation_service：一次性/流式共用模型；seq_id、offset、结束标志、缓存回滚、超时与暂存区语义明确。

**14. 已授权的数据生产实施（2026-09-22）**

用户提供并允许使用 Aster 已有 `gpt-5.6-luna` 配置，明确首轮 50 个词、每词一组、并发 5、最多尝试 1 次。每组定义为 10 条安全/不安全各半的合成对照，目标 500 条。已提交 Run `aster-dev-253`，模型配置 ID `mdl_01M2TK4P8Z0DE5QCJYJ8RSK8YV`；API key 由平台管理，无需导出到本地。

生产实现位于 `pilot/`。每词一个稳定 task_key，独立保存词库 revision、来源类别和原始词。Aster 执行直接模型请求，不启动 coding agent，也不申请 GPU 沙箱。每个词最多一次本轮生成请求；格式失败保留原始产物，不自动重采样。

每词长期目标可在新 flow 快照中设为 100 或更多，按每批 10 条顺序调用，每批保存原始 JSON、token usage、校验结果。单词 Task 保持稳定，生成预算与并发独立冻结。扩大数量时需要同步提高 flow 总时限、估算 token 预算，并重新 plan；本轮授权不等同于自动启动无限生产。

数据契约：messages、target_role、input_label、label、policy_id、evidence_text、decidable_prefix、reason。提取脚本输出 `examples.jsonl`、`rl_tasks.jsonl`、`rejected.jsonl`、`review_required.jsonl` 和 `summary.json`。RL observation 仅含待审对话与角色，合成标签和证据放在独立 reward_spec 中，避免标签泄漏。

首轮质量抽检已发现：生成文本会出现证据在可判定前缀之后、角色/输入标签不符、以及把真实姓名改称虚构人物的含混表达。结构错误拒收；虚构语境和疑似输入标签冲突单列复核。机器校验通过的候选仍标记 `luna_synthetic_unverified`，不当作官方或人工金标。下一版生成提示须减少这些问题，且要对所有标签分支保持相近的文风与长度，避免靠“我会伪造”这种单一模板解决任务。

Luna 负责生成与初标；最终 benchmark 不交给它生成训练变体。独立质量抽检/冻结 verifier 仍保留，不能以生成器对自己样本的自评替代最终验证。高质量合成数据也可用于同数据预算的分类损失对照；主线仍为用户选择的 RL。

**15. 完整候选词表快照（2026-09-23）**

`lexicon/` 已固定下载同一 revision 的全部 17 个 `Vocabulary/` 文件，以及 TrChat JSON 格式、上游 README 和 MIT 许可原文。合计 89,452 条原始记录，NFKC 与大小写规范化去重后 51,085 个候选项；其中 36,295 个普通文本项已导出为稳定 task_key，另有 14,592 个网址指标和 198 个加号组合候选。每个候选项保留原写法、源文件与位置，`auto_block=false`，尚未自动生成例句或赋予违规标签。上一轮 50 个 seed 与已经完成的 M5/L20 验收仍保持冻结；扩大生成预算前先抽检新增词与类别，并单独确定每词样本数、并发和次数。完整来源与校验见 `lexicon/source_lock.json`、`lexicon/derived/d967c30b/summary.json`。

**16. 扩展来源与独立评测（2026-09-23）**

在不改变上一轮基线的前提下，扩展候选词表已合并 houbb/sensitive-word-data 和 SpaceGather 中文词表：总量 107,282 个规范化候选项，相对第一版新增 56,197 个；其中 92,492 个普通文本项、14,592 个网址指标、198 个加号组合候选。houbb 标签代码和白名单保留为来源元数据，不转换成项目安全标签。fwwdn/sensitive-stop-words 与原主表完全重叠，作为来源镜像留档；COLDataset 的 37,480 条有标注评论独立保存，可用于后续误拦/漏判测试。所有候选仍 `auto_block=false`，没有启动新模型生成或训练。其他许可不同或未明确的研究词表暂不混入；清单见 `lexicon/RESOURCE_CATALOG.md`。

**17. 非商业研究候选层（2026-09-23）**

用户确认项目是非商业开源研究，故无需逐词人工审核才可进入候选库。Citizen Lab 的 CC BY-NC-SA 4.0 数据以独立研究层保存，固定 revision、归档 SHA-256 和全部来源；解析 1,968 个明确的词表/规则文件，得 477,165 个去重候选项，与原宽松许可基础表合并为 492,640 项，并生成 357,083 个新增普通文本任务种子。平台审查观察不等于本项目的违规标签，模型输入侧/回答侧仍须依语境与政策边界判定，所有词条 `auto_block=false`。本层没有启动新 Luna 请求或 RL 训练。非商业数据的署名、相同方式共享和用途限制见 `lexicon/research_nc/README.md`。

推理引擎由助手负责选型。首版确定为 PyTorch + Transformers 的自定义分类服务，直接管理增量 forward/KV Cache，SDPA/兼容 FlashAttention 作为首个高效后端；进一步做 compile/CUDA Graph、形状分桶和动态 batching。vLLM/SGLang/TensorRT 只在确认支持该分类/流式读出路径且 L20 实测有净收益后切换，不为了使用引擎名称牺牲正确性。

试跑最终报告见 [pilot/PILOT_REPORT.md](pilot/PILOT_REPORT.md)。50 次调用成功，41 条结构拒收，107 条待语义复核；未进行自动补采样。

**一手参考资料**

1. [Qwen3Guard-Stream-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B)
2. [Qwen3GuardTest 数据卡](https://huggingface.co/datasets/Qwen/Qwen3GuardTest)；[官方评测说明](https://github.com/QwenLM/Qwen3Guard/blob/main/eval/README.md)
3. [Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon)
4. [固定 revision 的模型配置](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B/blob/419364a715de9840d47b1457982f64ff37f90ed4/config.json)
5. [固定 revision 的模型实现](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B/blob/419364a715de9840d47b1457982f64ff37f90ed4/modeling_qwen3_guard.py)
6. [DeepSeekMath：GRPO](https://arxiv.org/abs/2402.03300)
7. [DeepSeek-V3.2：DSA 架构与继续训练](https://arxiv.org/abs/2512.02556)；[官方 V3.2-Exp 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
8. [Native Sparse Attention：硬件对齐的可训练稀疏注意力](https://arxiv.org/abs/2502.11089)
9. [DeepSeek-V2：MLA 与 MoE](https://arxiv.org/abs/2405.04434)
10. [FlashMLA：kernel 与硬件支持矩阵](https://github.com/deepseek-ai/FlashMLA)
11. [FlashAttention：实现与支持硬件](https://github.com/Dao-AILab/flash-attention)
12. [DeepSelect：DSA 索引器 top-k 优化](https://github.com/deepseek-ai/DeepSelect)


## 第五轮：前缀监督与统一执行路径

前缀专项 SFT/RL 已按 [第五轮协议](round5/PREFIX_TRAINING_PROTOCOL.md) 冻结并在 L20 启动；[实际状态](round5/EXECUTION_STATUS.json)记录进度。整段与缓存数值差异的修复采用[统一 32-token 块计算方案](round5/CANONICAL_EXECUTION_PLAN.md)，服务端校准与原训练选模证据分别保存。
