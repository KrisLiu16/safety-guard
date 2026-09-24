# 同场分类推理速度对照（protocol v2）

`benchmark_pair.py` 只在一张可见 CUDA L20 执行模型。候选使用冻结 `MODEL_MANIFEST.json` 的实际结构和 SHA-256，经现有 `infer_classifier.load_classifier` 加载；保留当前 short recurrent 优化与 `Classifier.step` 用户分类读出。A0 使用官方 Qwen3Guard-Stream 权重、SDPA、显式无配置 `DynamicCache()`，从专用 `query_risk_level_logits` 读用户风险，绝不取语言模型词表 logits。发现本地 A0 风险头覆盖文件会拒绝执行。

候选的 `infer_classifier.py` 与 `memory_attention.py` 从 **benchmark 脚本自身所在目录**精确加载，已导入的同名旧模块若来自其他路径会报错；基础依赖仍来自 `/work/input` 与 `/work/window`。结果记录实际加载模块的路径和源码 SHA-256，防止旧 `/work/round4` 文件覆盖验证包中的版本。

```sh
/work/modern/bin/python /work/round4/benchmark_pair.py \
  --kind candidate \
  --candidate-manifest /work/validation/MODEL_MANIFEST.json \
  --output /work/output/final_validation/speed_candidate_v2.json

/work/legacy/bin/python /work/round4/benchmark_pair.py \
  --kind a0 \
  --candidate-manifest /work/validation/MODEL_MANIFEST.json \
  --output /work/output/final_validation/speed_a0_v2.json
```

两组应在同一 L20 上串行执行。`research_candidate` 清单尚未形成时拒绝启动；输出不覆盖已有文件。脚本逐组写结果，只有全部六组完成后 `status` 才为 `completed`。

计时外调用一次 `nvidia-smi` 查询物理 UUID 与型号。只有恰好一个物理设备且是 L20 才据此补足或核验 CUDA UUID；两种 UUID 冲突直接失败，多物理设备不会猜映射而是标记 `unavailable`。查询不可用时，可保留 CUDA 属性直接提供的合法 UUID。输出保存身份来源与查询证据，配对提速比只允许用于两份具体、非 `unavailable` 且相同的 UUID。

固定 batch=1、单会话；初始上下文 **512 / 4,096** 个原生 token，增量大小 1 / 8 / 32。每组先用独立缓存完整跑一遍同样的 prefill 和 128 个增量块，丢弃状态，再从全新缓存重复该轨迹进行 128 次计时。每个块都计算用户风险概率并复制回 CPU。流式 ITPS 的分子是 `128 × chunk` 个新增输入 token，分母包含前向、分类读出、CPU 可见结果、同步和循环开销；prefill 单独记录，不计入增量 ITPS。零生成 token。

P50 / P95 使用 `sorted[ceil(n*q)-1]` 最近秩，保留全部 128 个延迟与三档用户风险概率便于审计。输出包括分类次数/秒、设备与库版本、输入及检查点哈希。缓存存储从完整对象图遍历并按 `(device, storage pointer)` 全局去重，包含 GDN 状态和根节点上的 memory 矩阵/归一化状态；每个唯一底层存储保留 owner 路径及实际字节数，分别记录 prefill 和结束时状态。它不等于模型总显存或 CUDA allocator 保留量。

两种 tokenizer 对**同一份固定中英无害重复文本**分别分词，不经过各自 chat template。输入分词和 GPU 输入准备均不计时。原生 token 边界及同等 token 数所覆盖的文字长度可能不同，因此这是相同协议的 token-ID 运行时对照，不是同文本边界的端到端文本服务压测。

两边消费的结果都是用户风险概率，但实际头部工作量不同：官方 A0 前向仍计算用户和助手两套 risk/category 头；`Classifier.step` 只计算用户 risk/category 头。这里保留已有入口，属于**当前运行时的对照**，不能把全部速度差归因于注意力结构，也不能称为相同头部工作量下的纯主干对照。

最长轨迹从 4,096 增长到 **8,192 tokens**。本地 A0 配置的 `max_position_embeddings` 确认为 8,192；原 v1 使用 8,192 prefill 后继续追加至 12,288，真实验证因此在模型位置上限检查处失败。**保留 v1 原产物，两边都重跑 v2**，不能把 v1 候选数字与 v2 A0 配对；结果的 `protocol_version` 必须同为 `matched-classifier-token-id-v2`。

该对照仅检查纯 token-ID 计时路径，不构成长上下文风险能力验收。不能把结果称为 HTTP、动态批处理或整套文本回滚运行时吞吐。提速比例应配对本脚本的 candidate/A0 同协议结果，也不能沿用旧实验的 24-call 数字作分母。
