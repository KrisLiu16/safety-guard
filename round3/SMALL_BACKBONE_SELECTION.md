# 小模型基座选型：Llama 与 Qwen

核查日期：2026-09-23。范围为用户此前提出的 <1B 总参数、中英安全分类、流式输入和 L20。此次完成官方资料及权重元数据核查，没有新做 Llama/Qwen Base 的分类训练或同卡性能对照；文中的推荐是选型判断，不能当作已测排名。

## 结论

可以省略从零预训练，直接使用完整预训练文本主干，训练分类头并适配主干。对于严格的 Qwen3 与 Llama 小型号二选一，推荐 **Qwen3-0.6B-Base**。如果把 Qwen3.5 纳入，推荐 **Qwen3.5-0.8B-Base** 作为新的混合主干实验候选，完整 **Qwen3Guard-Stream-0.6B** 继续作为已有安审能力和 L20 性能对照。

这不是说 Qwen3.5 Base 已证明比 Qwen3Guard 更准或更快。Base 没有现成安审头，仍需监督/蒸馏。若目标是最快得到可用的项目安审模型，完整 Guard 的后训练比从普通 Base 重新训练风险头更直接。

## 核查结果

| 候选 | 参数与结构 | 与本项目的关系 |
|---|---|---|
| Qwen3-0.6B-Base | 596,049,920 参数；28 层 GQA 因果 Transformer | 完整主干低于 1B；可复用现有 Qwen3 流式实现思路，新增/训练分类头 |
| Qwen3.5-0.8B-Base | 文件总计 873,438,784 参数；纯文本主干 752,393,024；24 层，其中 18 GDN / 6 注意力 | 不裁层即可继承混合架构；预计更适合长会话状态管理，需 L20 内核和分类质量验证 |
| Llama-3.2-1B | HF tensor metadata 为 1,235,814,400 参数；Meta 模型卡标 1.23B；GQA Transformer | 超过 <1B 上限；官方支持 8 种语言，不含中文，不能据此说完全不会中文 |
| Qwen3Guard-Stream-0.6B | 本地实数 597,111,296 参数；28 层已有用户/助手分类头 | 已在 L20 验证；有安审后训练能力，但尚未按本项目政策校准 |

参数元数据保存于 [research/base_comparison](research/base_comparison)，Qwen3.5 纯文本形状见 [此前固定的 tensor header](research/qwen35_08b_source_metadata.json)。取消 tied LM 输出投影不等于减去输入嵌入参数。Llama 是自定义 Llama 3.2 Community License；上述 Qwen 为 Apache-2.0。Llama 的官方 HF 元数据标记 gated/manual，本次仅读取公开卡片与元数据，没有下载受限权重。

本次查询官方模型目录：Meta 的 1B 量级文本主干仍以 Llama-3.2-1B 为候选；Qwen <1B 的通用文本候选为 Qwen3-0.6B-Base 和 Qwen3.5-0.8B-Base。没有把语音、Embedding/Reranker、量化体积或 MoE 激活参数冒充满足总参数上限的通用 Base。更大的 1.7B/2B 版本不纳入当前上限内排名。

## 能力证据及限制

- Qwen3 论文表 8 报告 Qwen3-0.6B-Base 的 MMLU 为 52.81（5-shot）；Meta 官方 Base 表报告 Llama-3.2-1B 为 32.2（5-shot、acc_char）。这是不同作者的评测管线，只作方向性参考，不能说我们在统一环境测得了 20.61 分差距。
- Meta 的官方支持语言不含中文，而 Qwen 对中文/多语言有明确训练与评测投入。结合规模预算和现有 Qwen Guard 的中文安审能力，Qwen 更适合作为本项目的优先起点；中文能力最终仍需同任务实测。
- Qwen3.5-0.8B 主模型卡展示的是后训练版本的成绩，不能把这些成绩贴到 `-Base`。本次没有找到三款 Base 在同一中文安审训练/测试配置下的权威完整对照，不能宣布绝对最强基座。
- 参数少不自动意味着低延迟。深度、宽度、KV、递归状态、batch 和内核共同影响速度；没有 Llama 的 L20 同负载实测，就不报其 ITPS 或声称它一定比 Qwen 慢。

## 对流式成本的意义

完整 Qwen3 的缓存随历史长度增长。Qwen3.5 只有 6 层注意力存完整 KV，其余 18 层用递归状态，仍不是完全固定状态模型。

按原始 BF16 KV 和配置估算，8,192 token 的 Qwen3-0.6B KV 约 896 MiB/会话；Qwen3.5-0.8B 的注意力 KV 约 96 MiB，加约 18 MiB FP32 GDN 主状态及约 0.84 MiB 卷积状态。这是张量形状估计，不含工作区、回滚快照、分配器与激活，也不是 L20 速度实测。短上下文未必更省。

Qwen3.5 需要匹配的递归内核；官方 Transformers 文档说明缺少 FLA/causal-conv1d 时会回退到较慢路径。不能根据架构名或生成 TPS 承诺分类速度。若把其 6 个注意力层也全部替换，已属于结构迁移，必须恢复训练和验收，不能认为保留原权重即可等价工作。

## 建议的验证顺序

1. 保留完整 A0 Guard 作为现成质量对照；不再通过直接删层定义主线收益。
2. 对完整 Qwen3-0.6B-Base 与 Qwen3.5-0.8B-Base 加相同任务语义的分类头，在相同开放安审训练数据、标签映射与预算下训练。可以另测 Qwen3.5 后训练权重，但必须单独命名，不和 Base 混算。
3. 每个新头先训练到能够输出有意义的分类，再比较固定误报率下的召回、中文/英文、用户/助手、引用/否定和跨轮风险。不能拿随机头来给 Base 排安审能力。
4. L20 上对相同文本流测首次/每次分类 P50/P95/P99、ITPS、分类数/s、会话状态。不同 tokenizer 额外报告共同参考计数和字符/字节吞吐。
5. 第一阶段保留全部文本层，只调整分类头/适配器或必要的主干参数。量化、batch、算子融合先做独立对照；MoE 或纯循环改造属于后续结构实验。

若选完整基座路线，自然语言从零预训练不再是前置条件；已下载自然语料可用于表示蒸馏、回归或有明确目的的继续预训练对照，不需要先消耗 100B token 才开始安审训练。此次是研究建议，未提交新的预训练、后训练或部署任务。

## 官方来源

- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen3 报告，表 8](https://arxiv.org/html/2505.09388v1)
- [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Qwen3.5-0.8B 后训练版卡片](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Meta 官方模型卡及 Base 评测](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)
- [Transformers Qwen3.5 内核说明](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)
