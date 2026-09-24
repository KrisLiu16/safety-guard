# 流式安审分类器：架构与训练设计 v0.1

**架构优先级更新（2026-09-23）：** 用户允许 <1B 总参数，并要求效率来自计算结构。当前主线见[架构 v2](HYBRID_MOE_ARCHITECTURE_V2.md)：完整 24 层混合预训练主干，再分别验证有限窗口与部分层 MoE；下文 C1/C2 及 12 层 H2 是历史候选，C14/C21 已有的负面结果见对应报告。安全标签、分类契约和训练/评测隔离原则继续适用。

日期：2026-09-23。状态：设计与实验规格，尚未宣称新架构优于现有模型。第二轮生成数据已切换到[一词一请求、两句的 v12 契约](../round2/oneword2_v12/flow/pipeline.py)；本文定义的是下游分类器的另一份契约。v8 全量 Run 已取消。

## 决定什么，输出什么

目标是一个**只读取、不生成文本**的模型。输入为有角色边界的多轮对话和当前用户/助手消息，可一次性传入，也可逐 chunk 追加；每次只判断当前目标消息，历史仅作理解上下文。**当前可执行的 C1** 直接输出三档风险 logits 和原版单选类别 logits：用户侧 9 类、助手侧 8 类，每个输入 chunk 末尾立刻返回[分类 JSON](guard-classification-v1.schema.json)；类别中的 `Political` 只是主题，不等于违规。当前 Stage0 分数未校准，接口不伪造动作、证据充分或拒答概率。**目标服务**才会增加经训练的证据充分/拒答等头，独立校准并经固定政策产生 `allow/hold/block`，契约见 [guard-decision-v1.schema.json](guard-decision-v1.schema.json)及[示例](guard-decision-v1.example.json)。两者都没有词表 softmax、文本 decoder、答案 token 或 CoT 输出。已开源的 Qwen3Guard-Stream 本身也已经是因果主干上的分类头，源码中没有 `lm_head`；我们要改变的是主干尺寸、任务头与运行时，不应把“删除 LM head”虚报为创新。

`controversial` 是内容风险等级，`hold` 是服务动作，不能混作同一标签。词库命中、政治话题、政府人物、政策批评、历史叙述、新闻引用和和平表达都不是自动违规；风险要由内容中可指出的行为或意图支持。用户的危险提问与助手的安全拒答分别判断。不同应用政策可改变处置阈值，但不得把来源词表直接当金标。

JSON Schema 固定字段、枚举和概率范围；运行时另外检查 `risk_probabilities` 三项和为 1（容忍浮点误差）、用户侧 `refusal_probability=null`、助手侧非空，以及所有值有限。动作由版本化政策控制器计算，不能让分类主干任意填写。每次模型或策略更新都记录 `model_version` 和 `policy_version`。

Jev 可借鉴的只有官方已公开的接口取向：非结构化状态输入、并行产生带校准概率的定型决策、无文本生成，及 RLCD 名称。TypeSafe 没公布足以复现 Jev 的层数、注意力类型、损失或训练数据；这里的流式网络与 RL 奖励是**我们的设计假设**，不是声称复刻 Jev。[TypeSafe 官方说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## 推理架构与对照

| 分支 | 实体 | 流式更新 | 用途 |
|---|---|---|---|
| A0 | 原版 Qwen3Guard-Stream-0.6B，修正增量 KV | 只前向新 token | 能力与速度基线；第一轮已有数据 |
| A1 | 同权重，优化分类读出、增量缓存内核、量化与调度 | 同 A0 | 工程效率上限，先于改网络；原模型已无 LM 输出层 |
| B0 | `bge-small-zh-v1.5` 24M 或 Chinese RoBERTa 的双向 encoder + 分类头 | 新 chunk 到来时重算目标窗口 | 短文本/批处理的低成本对照；不能声称精确 O(1) 流式 |
| C1 | 专用 **causal classifier**：从 Qwen3Guard-Stream-0.6B 保留 tokenizer、嵌入和约一半主干层，再蒸馏到新头，约 0.35–0.45B 的待测范围 | KV 仅追加，最后隐状态直接分类 | 首个“自己的”流式分类网络候选 |
| C2 | 新建约 0.2B 的 16 层、宽 768、GQA 主干，仅分类头；不带 LM 输出层 | 同 C1 | 仅在 C1 成功且仍受成本约束时研究；随机初始化需额外语义训练 |
| H1/H2 | 从 Qwen3.5-0.8B-Base 混合主干出发，先保留完整预训练层，再研究 9 Gated DeltaNet + 3 局部注意力裁剪 | 固定大小递归状态 + 有界局部 KV | 长上下文/多会话高速分类的结构实验；细节与验收见[专门规格](FAST_STREAM_ARCHITECTURE.md) |

BERT/RoBERTa/BGE 可以给完整短句提供强的双向语义基线，必要时把其概率或表示作为 C1 蒸馏目标；但其注意力看到右侧上下文，直接把 BERT 权重改成因果掩码不能保证原能力仍在。精确增量流式模式仍以预训练因果主干为主，用 B0 的端到端实测决定是否有值得保留的短句通道。

C1 层裁剪先尝试均匀保留 28 层中的 14 层，保留原隐藏宽度与分词器，便于继承权重；实际参数、显存、质量以脚本计数和基准为准。C2 若沿用 Qwen 大词表，输入 embedding 本身约 1.17 亿参数，不能把“16 层”误当超小模型。层裁剪、半精度和量化都不能代替性能实测。Qwen 技术报告证明“预训练因果主干 + 用户/回答分类头”是可行起点，但 C1/C2 的规模和训练目标由我们自行验证。[Qwen3Guard 技术报告，§4](https://arxiv.org/html/2510.14276) · [当前 0.6B 源码](../round1/export/base_model/modeling_qwen3_guard.py)

每个会话保存 tokenizer 增量边界、角色位置、位置索引和 KV。现有 `append(chunk)` 只前向新 token（必要时回退重分词后缀），**立即返回该 chunk 末尾的风险等级与类别分布**；如需 chunk 内逐 token 定位，应显式打开全位置分类读出并计入计算成本。`finalize(message)` 使用同一状态和末位读出。若上游 LLM 分词器不同，用 UTF-8 文本在 Guard 端稳定增量分词，保留未闭合字节/子词，检查任意 chunk 切法与一次性输入的末态一致。不要把上游 token ID 强行当本模型词表。短暂缓存共享仅限消息历史完全相同、模型/词表/政策版本相同的请求；不同租户默认不共享状态。窗口截断或压缩历史另作为质量实验，不能隐式丢上下文。

当前输出头在最后隐状态上并行计算：角色条件化的 3 类风险头和原版类别单选头，不把类别概率当成多标签危害概率。目标版本再训练独立的危害多标签头（暴力、非暴力违法、性剥削/不当性内容、隐私、自伤、定向骚扰、欺骗/伪造、提示注入、其他明确危害）、证据充分头与助手拒答头。类别定义和映射必须在 `policy_version` 中冻结；不同来源标签只在确认语义相同时映射，不能把 `politically sensitive` 字面值机械映成 `unsafe`。

服务端每个 token 计算概率，但可合并传输和调度；对可见输出设少量 token 的 holdback buffer，危险证据首次出现后才能停发。保留一条 `prefix_state` 轨迹：风险分布、证据充分概率、动作、已释放 token 数、已等待 token 数。常规路由可先规则/词表召回或 B0 过滤低风险候选，但词命中不直接阻断，且任何两级路由都计入端到端漏判与延迟。

## 语义能力与数据来源

“词库 × 2 例”最多是定向覆盖，不等于预训练。先使用已有预训练权重获得语义理解，再以开放安审数据和独立自然文本补齐语言与意图。公开 Qwen3Guard 权重和 Qwen3GuardTest 已发布；技术报告提到的约 119 万条原训练样本并未在其官方仓库作为完整训练集发布，测试集绝不入训。[官方模型仓库](https://github.com/QwenLM/Qwen3Guard) · [官方测试集](https://huggingface.co/datasets/Qwen/Qwen3GuardTest)

| 来源 | 作用与可直接用的字段 | 当前已核对条件/问题 |
|---|---|---|
| [XGuard-Train-Open-200K](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K) | 20 万条 `q/r/qr`、风险标签与政策；最接近输入/回答分类 | Apache-2.0；既有中文也有英文预览，先统计语言、标签和空值，再映射 |
| [NVIDIA Nemotron Safety Guard Dataset v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) | 有 `zh/train.jsonl`，prompt/response 分开的标签 | CC BY 4.0；需核实中文分片实际行数及标签来源，避免把全库 451k 当中文量 |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | 86,759 条训练记录，用户危害、回答危害、拒答三个头 | ODC-By；访问需接受提供方条件，部分标注来自 GPT-4，空标签跳过 |
| [COLDataset](https://github.com/thu-coai/COLDataset) | 37,480 条中文冒犯/反偏见评论，专测“引用/反偏见不误拦” | Apache-2.0；范围仅冒犯语言，不可当通用安审 |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 大量英文问答及回答安全标签，补回答侧场景 | CC BY-NC 4.0；问答安全标签不可无核验地复制为用户提问标签 |
| [OpenGuardrailsMixZh 97k](https://huggingface.co/datasets/openguardrails/OpenGuardrailsMixZh_97k) | 机器翻译中文安审样本，补中文表达 | 汇编卡标 Apache-2.0，但注明需检查原数据许可；与 WildGuard/BeaverTails 重叠，先做来源交集与许可过滤 |
| [FineWeb2 中文分片](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 仅在语言适应实验中采样无标签自然中文/英文文本 | ODC-By；网页噪声、PII/重复/基准污染要过滤。它不是安全标签数据 |
| [Chinese RoBERTa WWM](https://huggingface.co/hfl/chinese-roberta-wwm-ext) / [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | B0 双向 encoder 权重，不必从空白训练 BERT | Apache-2.0 / MIT；BGE 为中文小模型，不应假设英语同等好 |
| 当前 Luna v12 数据 | 词汇、中英语境对照和风险证据短语 | 初标是 `luna_synthetic_unverified`；每词一中一英两句，先去重、质检和分组，不能把 449,575 词直接视作金标 |

第一轮建议训练混合不是按“条数”照单全收：先用开放安审数据建立自然语境，再取已完成的 Luna 记录做难例；按来源、词族、模板和标签分层采样，单一词/模板设上限。补充**无词命中**的安全与风险对话、正常公共事务讨论、误导/讽刺/引用、跨轮指代，以及“危险提问 + 安全拒答”和“安全提问 + 有害回答”。保留每项原始许可、源 ID、原标签、映射规则、语言、生成器版本与置信层。无标签 FineWeb2 用于语言适应/蒸馏而不自动给安全真值。公开数据 test/dev 与 Qwen3GuardTest、CHiSafetyBench 必须排除出训练和教师伪标。

起步人工金标建议按词族/来源/语言/角色/类别分层做 800–2,000 条，其中开发、校准、冻结测试各有独立实体族。数量是实验起点，不能以此证明万分级误拦。对争议政治内容、真实人物、隐私和政策边界优先双人标注及仲裁；标注指南明确“话题本身无罪”，记录证据起点和“未足以判断”前缀。扩张时优先主动采样模型分歧与高损失自然样本，而不是平均扩造每个词。

## 训练阶段与损失

**0. 基线冻结。** 用相同消息格式、分词器、阈值与硬件测 A0、B0，并记录原始完整文本和流式前缀。第一轮在 M5 上已经发现正确 KV 缓存带来约 10.8× 逐 token P95 改善，但原小集流式用户误拦仍约 9.3%；新实验必须以修正缓存的 A0 为速度基线。[第一轮报告](../round1/ROUND1_REPORT.md)

**1. 语义迁移。** A1/C1 继承预训练权重。已在 L20 完成 1k 开放语料 pilot 与 [20k/2k 续训](l20/STAGE0_20K_REPORT.md)：在没有项目金标的 Stage0，仅用 A0 同一**可见前缀**软分类分布与隐状态蒸馏，不读取原来源安全标签；教师匹配损失下降，不能推断项目安审能力提高。最终安全对齐再训练新分类头/LoRA，并用独立金标锚定 4B/0.6B 教师可靠性；不得让教师读到未来后缀。最终损失候选为 `L = λ_gold CE(y,p) + λ_teacher T² KL(q_T || p_T) + λ_category BCE + λ_refusal BCE + λ_evidence BCE + λ_consistency D(p_full,p_chunk_end)`。各项只在字段可用时计算，Gold 权重大于伪标；保留原三类，不把不确定样本强压二元。chunk 一致性要求不同切法的同一前缀输出一致。4B 教师只对在人工金标上可靠的分布增权，分歧样本先仲裁。

**2. 无标签语言适应作为可证伪消融。** 对 C1/C2 取 FineWeb2 中英对话化片段与公开合法文本，用冻结教师的中间表征/前缀分布、句对一致性与有限语义任务做蒸馏；B0 可单独做 MLM 适应。因果学生可用教师表征蒸馏和对比损失，无须在部署时保留 LM head。与“不做语言适应”同预算对比；若独立自然文本/意图测试不升，停止追加预训练。C2 随机初始化只在明确提供足够算力与语义数据后尝试，不能期待词库例句让它学会语言。

**3. 分类概率校准。** 在独立 calibration 集按角色、语言和类别拟合温度/阈值。CE/软 KL 是小动作空间的主要学习信号；Brier/log loss 评估概率质量，避免通过猜极端标签虚增名义 reward。教师不是真值，固定的人工金标是校准锚。[严格适当评分规则](https://doi.org/10.1198/016214506000001437)

**4. 真正的 RL 用在“什么时候放行/暂挂/拦截”。** 对每条完整对话生成多种到达 chunk 切法，标注最早能从当前前缀判定的证据位置 `t*`，当前策略在每个时间步采样动作 `allow/hold/block`。`block` 终止，`hold` 暂存输出并继续观测，`allow` 释放已缓存的安全内容；从用户实际可见内容计算漏发/误挡。先以固定分类头和枚举/动态规划找到离线近优控制策略，再与可微期望风险、带参考 KL 的小动作 PPO/actor-critic 对比；只有独立流式集上更好才接受 RL。分类能力本身不靠生成式 RL 来“学理解”。

一条轨迹的负成本可写成：

```text
C = c_fp·1[安全消息被 block]
  + c_early·max(0, t* - block_token)·1[有害且过早 block]
  + c_fn·1[风险内容已释放且到结束仍未 block]
  + c_exposure·(t* 之后实际放给用户的风险 token 数)
  + c_delay·1[有害且实际 block]·max(0, block_token - t*)
  + c_hold·(安全 token 被额外等待的 token/毫秒数)
  + c_unresolved·1[最终仍 hold、没有结论]
  + c_compute·前向计算成本
R = -C - λ_kl·KL(π || π_ref)
```

`c_*` 在开发集根据真实使用成本设定，再冻结到测试；报告每项原值而非只报总 reward。没有可信 `t*` 的样本只用于整句分类/校准，不拿来奖励“早拦”。对误拦公共事务讨论、拒答误判和类别漂移设硬约束。一个可审计的反事实：相同词、不同意图的两句应产生不同概率；若只靠词匹配也能取得同样 reward，数据/奖励设计不合格。

离线动作模拟器已落在 [stream_reward.py](stream_reward.py)：`allow` 会释放已暂挂的全部前缀，`hold` 继续缓存，`block` 终止；因此不能靠“先 hold、最后放出风险内容”逃避暴露成本，也不能靠永远 hold 免费避开漏判。当前成本系数仍为实验参数，测试文件只检验边界语义，不代表最终政策权重。

**Aster TITO 边界。** [平台 TITO 文档](http://aster.naiveai-dev.com/docs/config-run-tito) 写明它采集生成模型的原始 token 轨迹，且需要 sglang 的 `/tokenize`、`/generate`。本模型只消费上游 token 并输出分类向量，因此不能直接靠 TITO 得到分类 RL 轨迹。可用 Aster Run 组织数据生成、教师审阅与离线评测；分类 RL 训练器自己保存 `prefix_token_ids, logits, calibrated_probs, action, action_logprob, value, t*, released_tokens, reward_parts, model/policy/data_version`。若将来用生成式教师做探索，TITO 采的是**教师生成**轨迹，不能混称学生的分类轨迹。

## 评测、效率和进入下一阶段的门槛

公开回归：Qwen3GuardTest 的 `thinking` 分类及 `thinking_loc/response_loc` 风险起点；[官方评测代码](https://github.com/QwenLM/Qwen3Guard/blob/main/eval/README.md) 的定位分片全是风险样本，不能估计误拦。[CHiSafetyBench](https://github.com/UnicomAI/UnicomBenchmark/blob/main/CHiSafetyBench/README.md) 是问答/选择题基准，要适配成分类输入后才比较，拒答生成分数不能直接当分类准确率。另以 COLD 和按原来源隔离的中文公共事务人工金标测误拦、正常批评召回、用户/回答交叉四象限。合成同模板 test 只做回归，不能作为独立泛化结论。

每个模型同阈值策略报告：分角色三类 macro-F1、Unsafe PR-AUC、固定安全集 FPR 下的召回、固定召回下的误拦、Brier/NLL/ECE、类别 F1；流式报告最早证据点后的检测延迟分布、提前拦截率、每千安全 token 误拦、放给用户的风险 token、hold 等待。按语言、来源、政治话题、真实人物、拒答与长度分层，并给实体族 bootstrap 区间。

性能正式验收只在 L20 用真实到达回放测：**ITPS（input tokens/s）= 在测量窗口内已接收并完成分类的新增输入 token 数 / 墙钟秒数**；另报实际模型前向 token 数，以暴露重分词/回退开销。它没有生成 token TPS。分别测完整输入到首个分类结果的 P50/P95/P99、单 token 与 chunk 1/8/16 的每次分类延迟、含排队的首次决策时延、分类结果/s、并发会话 1/8/32/64、显存峰值与每会话 KV、缓存命中率。将纯模型前向、含分词/JSON 的本机 HTTP、真实网络与队列三层分开报告，避免把模型理想值当服务值。过去的 M5 数值只作历史探索，不与 L20 数值混算加速比。用同一请求流在 A0/A1/B0/C1 上做质量相同的 Pareto 对比。早退、小模型门控、量化或新注意力仅在独立质量门槛保持时进入主线；MoE/DSA/MLA 不作为预设胜利。

推进门槛：C1 至少在独立中文测试上不显著劣于 A0 的误拦与漏判，流式风险暴露更少或相同，且 L20 含排队 P95/吞吐有可重复的净收益；否则保留 A1。RL 控制器必须优于同一分类器的最佳静态阈值/防抖基线，且正常公共事务讨论的误拦不升。C2 只有在 C1 的误差与计算瓶颈被定位后立项。

## 首轮执行顺序

1. 固定 `guard-decision-v1` 输出契约与风险政策/来源映射；从公开数据各取小样做语言、标签、许可、重复与污染清单。保留 Luna v12 原始来源和未验证状态。
2. 建 800–2,000 条实体族隔离的人工锚点与若干带 `t*` 的流式轨迹；冻结测试。跑 A0/B0 基线。
3. 在 L20 做 A1 分类头/LoRA、B0 对照与 C1 14 层结构蒸馏；相同数据预算下比较 A1/C1。语言适应只做有消融的短试。模型训练不在 M5 执行，详见[L20 执行边界](L20_TRAINING.md)。
4. 固定概率校准后训练/搜索流式控制策略，再做小动作 RL 消融。输出完整能力、延迟、吞吐、成本及失败案例报告，决定是否继续 C2。
