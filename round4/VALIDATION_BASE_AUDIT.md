# 基线最终验证产物验收

审计日期：2026-09-24。只读取 [已收集基线产物](validation_results/output/round4_validation/)、冻结 validation 包、基准数据和对应代码，进行 CPU 哈希、分词和指标复算。没有运行模型、访问集群或改变模型选择、数据、包、阈值与数值门槛。此报告不包含正在准备补跑的修复版结果，也不把 Job 完成当成所有验收项通过。

## 结论及证据状态

固定质量候选仍是 **classification_rl/full**，SHA：

`29fc3e5663eaa333235b1c2ba84d4c0a964f6ece4fd146c9fc6ec01761eaf222`

四分支文本会话测试、固定 window 的 chunk=8 CUDA Graph 实验、固定 full 候选的独立包加载及 JSONL 行为测试均有真实通过产物。官方三分片两边均完整评估、无排除，逐例混淆矩阵与保存指标一致。

**质量结论并非全部提升。** 学生在 thinking 分片的两连续风险判决有 90.41% FPR，strict F1 为 0.711649，低于 A0 的 0.812609。学生的终点 F1 更高，但终点误报也更多；审计追加计算的终点 AP 为 0.927407，低于 A0 的 0.948029。当前结果不支持把未校准的中途风险输出直接当成可靠拦截策略。

[stage_status.json](validation_results/output/round4_validation/stage_status.json) 记录 10 个阶段中 8 个成功、2 个失败；`all_required_processes_completed=false`。失败为 **benchmark_a0** 和 **keyword_candidate**。不得以 `status=completed`、报告已收集或 Job 结束覆盖这些失败。

对照 [FINAL_EVIDENCE_MAP.md](FINAL_EVIDENCE_MAP.md)：

| 验收项 | 本次实际结论 |
|---|---|
| 固定候选身份、文本追加、事务、会话隔离 | 已验证对应候选 SHA，实际案例通过 |
| 官方三片覆盖、身份、标签、strict/loose/endpoint 指标 | 完整覆盖并逐例复算通过；定位协议未复现，质量有明显取舍 |
| 独立包父子进程、结构、RoPE、双头概率、JSONL | 已验证；Python 审计及离线加载成立，不冒充 OS 沙箱 |
| window/chunk8 CUDA Graph 正确性和成对速度 | 已验证，约 4.19 倍；不能归给 full 候选或完整文本 API |
| 最终候选与 A0 的同场六组效率比较 | **未完成**：A0 超出配置长度而失败 |
| 候选与 A0 的固定词表诊断 | **未完成**：候选 tokenizer 校验失败；只有 A0 结果 |
| 本地完整模型包交付与权重重新哈希 | **尚未完成**：本地报告包含 11 个清单文件，但未含 classifier.safetensors；不能由远端测试通过推断已下载完整权重 |

## 官方 Qwen3GuardTest：覆盖与质量

按实际阶段命令定位输入为冻结包的 `official_adapted_v1` 与 `qwen3guardtest`。逐片验证本地源文件、准备数据的 SHA，与已运行的 validation tar 内对应文件逐字节一致；原数据版本为 `8a8391a583d65dc5f2be59e815404fe54289fa7c`。

| 分片 | 请求数 | 学生已评估 | A0 已评估 | 安全 / 风险标签 | 排除 |
|---|---:|---:|---:|---:|---|
| thinking | 1059 | 1059 | 1059 | 490 / 569 | 两边均 0 |
| thinking_loc | 569 | 569 | 569 | 0 / 569 | 两边均 0 |
| response_loc | 813 | 813 | 813 | 0 / 813 | 两边均 0 |

六份预测 JSONL 各自 sample_id 唯一，按 row_index 与冻结输入逐条核对 sample_id、split、unique_id 和标签，均一致。每条概率有限、在 [0,1] 内且和接近 1；endpoint_decision 与保存概率 argmax 相同。所有序列及评估起点合法，学生长度/起点与准备数据精确一致。两边均有 **569 次完全相同输入和起点的复用，1872 次独立 forward**；所有复用行的判决、终点概率、长度和起点逐字段一致。2441 是分片记录数，不是 2441 个独立内容。

strict、loose、endpoint 三种口径均从逐例保存判决重新计算 TP/FN/FP/TN、recall、FPR、precision 和 F1，与 metrics 一致。逐位置 logits 未归档，本次没有在 CPU 重新执行神经网络来重建“两连续”判决；其执行依据是冻结评估代码和已完成的 L20 记录。

| 判决口径 | 模型 | TP | FN | FP | TN | 风险召回 | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 两连续，strict | 固定学生候选 | 559 | 10 | 443 | 47 | 98.24% | 90.41% | 0.711649 |
| 两连续，strict | A0 | 464 | 105 | 109 | 381 | 81.55% | 22.24% | 0.812609 |
| 两连续，loose | 固定学生候选 | 559 | 10 | 443 | 47 | 98.24% | 90.41% | 0.711649 |
| 两连续，loose | A0 | 467 | 102 | 109 | 381 | 82.07% | 22.24% | 0.815721 |
| 终点 argmax unsafe | 固定学生候选 | 447 | 122 | 57 | 433 | 78.56% | 11.63% | 0.833178 |
| 终点 argmax unsafe | A0 | 343 | 226 | 5 | 485 | 60.28% | 1.02% | 0.748092 |


thinking 中，学生 strict 召回较 A0 高 **16.70 个百分点**，但误报多 **334 条 / 490 条安全样本**，FPR 高 **68.16 个百分点**，F1 低 **0.100961**。学生终点 F1 高 **0.085086**，但 FPR 为 11.63%，A0 为 1.02%。不能只报召回或只报终点 F1。

为比较排序质量，本审计额外使用 thinking 保存的终点 unsafe 概率计算非插值 average precision（AP）：学生 **0.9274074334**，A0 **0.9480287107**，差 **−0.0206212773**。该 AP 是审计补充的终点指标，不是原报告中存在的指标，也不是“两连续”协议的 AP；未据此改选或调阈值。

| 分片 | 判决口径 | 学生 TP / FN | A0 TP / FN | 学生召回 | A0 召回 | 学生 F1 | A0 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| thinking_loc | 两连续，strict | 559 / 10 | 464 / 105 | 98.24% | 81.55% | 0.991135 | 0.898354 |
| thinking_loc | 两连续，loose | 559 / 10 | 467 / 102 | 98.24% | 82.07% | 0.991135 | 0.901544 |
| thinking_loc | 终点 argmax unsafe | 447 / 122 | 343 / 226 | 78.56% | 60.28% | 0.879921 | 0.752193 |
| response_loc | 两连续，strict | 770 / 43 | 725 / 88 | 94.71% | 89.18% | 0.972836 | 0.942783 |
| response_loc | 两连续，loose | 770 / 43 | 736 / 77 | 94.71% | 90.53% | 0.972836 | 0.950291 |
| response_loc | 终点 argmax unsafe | 727 / 86 | 713 / 100 | 89.42% | 87.70% | 0.944156 | 0.934469 |


两个 loc 分片无安全样本，**FPR 未定义，不能写 0%**。thinking_loc 与 thinking 的 569 条风险内容重叠；不汇总独立总分。学生采用训练 canonical 序列化和 Qwen3.5 原生 token；A0 采用原 chat template、保留尾换行及 Qwen3 token。二者是适配观测，不是控制住 token/template 的纯架构比较；原 token 定位、首 128 token 延迟等指标没有复现，`location_metrics=null`。

学生官方 metrics 的 checkpoint SHA 与固定候选一致。`official_a0/metrics.json` 自身 SHA 字段为空，不能说它完成了独立 A0 权重身份核验；同一 Job 的 A0 词表评估对共同模型目录提供了权重旁证 `e0a2eac6cc79cca5bf35bf8fb356f94c333dd9715f4f0dd58883b01f2fe33419`。CPU 使用本地 A0 tokenizer 重新执行所有 2441 条模板分词，长度/评估起点全部吻合，其中 1951 条有原 annotation IDs 的记录也逐 ID 相同；得到同样的 1872 个独立输入及起点。

## 四分支真实文本流

每个分支有 41 个实际案例，均含真实 192→191 BPE 收缩、跨角色、>1024-token prefill、交错会话、溢出拒绝、实际 forward 后注入失败并恢复和测速终点。全部记录的概率误差有限且低于原门槛 **0.03**；各分支 SHA 与主训练导出一致。会话当前缓存及两个快照的 storage 隔离检查、溢出事务和失败事务均通过。

| 模型 | 41 个案例最大概率误差 | 输入 ITPS | P50 ms | P95 ms | 会话状态早期→后期 MiB |
|---|---:|---:|---:|---:|---:|
| full | 0.000608861 | 216.85 | 25.66 | 46.88 | 124.02 → 152.73 |
| window | 0.000555456 | 214.70 | 25.86 | 47.86 | 74.50 → 74.50 |
| memory | 0.000264585 | 189.72 | 29.18 | 54.38 | 83.57 → 83.57 |
| classification_rl | 0.000324488 | 224.06 | 24.74 | 45.39 | 124.02 → 152.73 |


四条测速轨迹均从 2001 个原生 token 开始，完整相同轨迹独立预热后追加 128 次，每次净增 6 个原生 token，最终为 2769；净新增/实际 forward 均为 **768**，replay 为 0。CPU 以已交付 tokenizer 和可注入的文本状态机重新构造这条轨迹，计数完全一致。ITPS 按 **768 / 保存的秒数** 独立复算通过，没有把回放或预热算作新输入。

这里是文本会话层的内部 `wall_seconds`：包含重分词、缓存/快照拷贝、模型、CPU 可见概率等主要路径；计时点在最后结果字典构造和提交赋值之前，不含 JSON 输出、HTTP 或等待输入。报告未保存 128 条逐次原始延迟，因此 P50/P95 是已记录的聚合值，核对了冻结代码的 median / nearest-rank P95 计算方式，但无法从该文件独立重建分位数。

表中状态大小是审计早期与晚期的当前 cache 加快照总和，不是单份 KV，也不是总显存。window/memory 两次相同；full 和最终 RL/full 随上下文增长，不能宣称有界。**memory 此文本测试通过不覆盖主训练 8K 审计的那条失败**：其 8192-token 助手头误差 0.03256136178970337 > 0.03，仍然失败。

## 固定 window、8-token 图实验

[graph_stream_audit.json](validation_results/output/round4_validation/graph_stream_audit.json) 使用 window SFT 权重：

`bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`

20 项检查均通过：warm/capture 后种子缓存逐字节复原、连续变化块、从快照恢复分支、交错两个会话、arena 返回无 alias，以及测速末状态。每项完整库存为 18 个 GDN conv state、18 个 recurrent state、6 份 K 和 6 份 V；逻辑 metadata/seen 一致。固定概率门槛为 0.001，BF16/FP32 状态门槛保持原值；本次各检查双头概率差记录为 0，没有调高门槛。

| 同一 window 权重、同一输入轨迹 | Eager | CUDA Graph |
|---|---:|---:|
| 起始 context / chunk / 追加调用数 | 1024 / 8 / 128 | 1024 / 8 / 128 |
| 净新增输入 token | 1024 | 1024 |
| 计时秒数 | 3.013620 | 0.718848 |
| 输入 ITPS | 339.79 | 1424.50 |
| P50 ms | 23.52 | 5.61 |
| P95 ms | 23.65 | 5.63 |

128 条原始延迟均为有限正数。独立复算 ITPS、median P50、nearest-rank P95 及比例完全吻合：ITPS **4.1923 倍**，P95 降低 **76.19%**。capture 初始化为 **0.306918 秒**，未混入稳态吞吐。

两边都返回独立外部缓存与 CPU 可见双角色风险概率。Eager 计入输入缓存 deepcopy；graph 计入锁、arena copy-in、replay、copy-out、position/seen 更新及 alias 检查，因此不是只计 `replay()` 内核。各自完整相同轨迹先预热。该测量不含分词、文本 BPE 回滚选择或服务并发，也不是 A0 比较。**不能把它转记为最终 RL/full 候选提速，不能推导 1–32 长度或新文本 graph 入口均已通过**；扩展快包有独立后续验收。

## 候选与 A0 的效率比较尚不闭合

旧版 token-ID benchmark 的候选侧六组数据及每组 128 次预热/测量完整，已从原始延迟重算 ITPS/P50/P95；所有输入计数为 128×chunk。缓存库存每次都有 48 个不重复 owner，18 conv + 18 recurrent + 6 K + 6 V，storage 去重字节求和吻合，不只统计 attention KV。该 full 候选无 memory 矩阵，不应借此声称审计了 memory 分支库存。

| 起始→终止 context | chunk | 候选 ITPS | P50 ms | P95 ms | 单份状态起始→终止 MiB |
|---|---:|---:|---:|---:|---:|
| 512 → 640 | 1 | 54.77 | 18.22 | 18.55 | 24.84 → 26.34 |
| 512 → 1536 | 8 | 397.50 | 20.09 | 20.44 | 24.84 → 36.84 |
| 512 → 4608 | 32 | 1569.18 | 20.36 | 20.66 | 24.84 → 72.84 |
| 8192 → 8320 | 1 | 54.49 | 18.28 | 18.76 | 114.84 → 116.34 |
| 8192 → 9216 | 8 | 395.95 | 20.16 | 20.43 | 114.84 → 126.84 |
| 8192 → 12288 | 32 | 1556.14 | 20.55 | 20.76 | 114.84 → 162.84 |


这是单边结果，包含超过文本 API 8192 上限的纯 token-ID 轨迹，不构成扩展文本能力验收。`benchmark_a0.log` 明确在模型配置长度检查时报错：最长轨迹 8192+128×32=12288，超过 A0 最大 8192；没有 `benchmark_a0.json`。因此 **不存在可算候选/A0提速比的成对结果**。候选侧具体设备 UUID 已记录，但只有单边身份仍不够。

补跑需要两边使用同一修复协议。若上下文改为 512/4096，使最长轨迹不超过 8192，应重跑候选与 A0 两边；不能把新 A0 结果与本报告旧版 8192 起始轨迹拼成速度比较。A0 与学生 readout 执行的头数不同，其现有 runtime 的实测速率也不能被归因成纯 attention 差异。

## 独立包父子验收及本地交付边界

父、子报告均 `pass=true`、子进程退出码 0，父内嵌 portable_child 与 [独立 child JSON](validation_results/output/round4_validation/bundle_audit.child.json) 完全相同。reference 文件本地 SHA 与父报告 `reference_sha256` 一致，selection 与固定 manifest 完全一致，父子均绑定固定 RL/full SHA。

独立子进程在 L20、离线条件下完成加载。四个禁止根 `/work/models`、`/work/input`、`/work/window`、`/work/output/qwen35_full` 各有被拒绝的显式探测；unexpected_denials 为空、网络连接拦截计数为 0。实际运行模块路径全部位于交付包目录。结合已核验加载器可确认它不读取旧底座/H24权重或外部实验代码；这是 Python 审计事件和离线加载证据，不是防御任意 native syscall 的 OS 沙箱。

新旧加载器参数结构完全匹配：**753,456,471 参数、336 个参数 tensor、完整 24 层、无 LM head**；320 个 backbone tensor 为 BF16，16 个分类头 tensor 为 FP32。两份 FP32 RoPE buffer 的 dtype/shape/SHA 与参考完全相同。六个 fixture 均完成：中英 user/assistant 各一个及 1108、1111-token 长输入；按保存的双头概率重新计算最大误差为 **0.0003705024719238281**，小于固定 0.005。六例的 canonical token IDs、输入长度和序列化 SHA 也在本机 CPU 重建一致。

JSONL 验收为 6 个 whole 案例、9 个 stream 案例；重新核算所有新旧长度差及净 token 计数，均一致。真实 BPE rollback 2 次，包括 **192→191，net_new_tokens=−1，forward/replay=191**，以及长输入末尾 merge；后续增长至192仅 forward=1。最大 whole/stream 概率误差为 **0.00019042566418647766**，低于 0.03；跨角色、空追加、会话释放和 generated_tokens=0 成立。所有已校准阈值都与固定 manifest 同一分层一致；英文 user 保持 null 阈值/null 判决，没有借用英文 assistant 校准。把整段阈值应用于中途前缀仍明确标为未验证。

本地 [bundle 目录](validation_results/output/round4_validation/bundle/) 的 11 个现存清单文件均重新计算 SHA 和字节数，与 BUNDLE_MANIFEST 一致；MODEL_MANIFEST 也逐字节一致。但 **classifier.safetensors 未包含在这份报告下载目录**。清单声称它为 1,509,079,036 字节及固定 SHA，远端子进程已通过清单校验；本次 CPU 验收不能代替随后下载完整权重后的本地哈希验证。

运行依赖版本必须保留：实际子进程为 Transformers 5.17.0、Tokenizers 0.23.2。额外 CPU 检查发现，本机较旧 Transformers 4.55.0 / Tokenizers 0.21.4 读取包 tokenizer 时，官方 thinking 与 thinking_loc 各有 3 行（重叠内容）token IDs 与冻结准备结果不同；6 个包 fixture 则一致。不能用这些 fixture 推导任意旧版本兼容。已执行官方学生评估在其固定现代环境逐条核对 token IDs 后成功完成，这一旧环境差异不推翻现有执行证据，但说明版本锁与逐输入兼容校验有实质意义。

## 两项真实失败及下一次验收要求

1. **A0 速度阶段失败**：原轨迹最大 12288 > A0 配置上限 8192。保留旧失败日志；按同一修复版协议重跑候选和 A0，核对代码/输入轨迹/设备 UUID，再计算比例。
2. **候选词表诊断失败**：`keyword_candidate.log` 报 `Candidate tokenizer differs from the frozen probe tokenizer`，原始 tokenizer JSON SHA 确实不同，候选侧没有预测/指标产物。不能跳过校验或宣称全局 tokenizer 等价；兼容证明应在实际固定库版本下对全部 1024 个冻结 probe 输入逐 ID 一致，并记录模型资产/后端指纹，评估保留冻结原 IDs。是否真正通过等待补跑产物。

A0 词表侧有完整 1024 条预测、256 个词族，CPU核对身份/标签/来源及冻结 probe、校准文件 SHA 成立。768 条有同分层校准，TP=256、FP=30；256 条英文 user 未校准且保持未评分。没有候选结果，当前不能作两模型词表效果差值或宣称问题解决。它是受控模板诊断，不等同自然对话总体质量。

此外，完整本地模型权重交付尚待实际文件验收。候选仍属研究状态，`production_approval=false`；safe/unsafe 以外的独立风险细类、controversial 校准、英文用户公开测试和前缀安全放行未由这些通过项证明。

## 审计输入指纹

| 实际读取产物 | SHA-256 |
|---|---|
| text_stream_audit.json | `7df8207fb6adb35069a3646cce073d04d13a0c8bb863e3c98e6c68fe8992b445` |
| graph_stream_audit.json | `4167eb98afb70d5894c4e9f29da4e4e3fcf14cbfe9b0a011c5b3022af99c7d16` |
| bundle_audit.json | `3a22696cd54ea610f15b0042042d4be098b4a6683c9699045f7bee5fa686f444` |
| bundle_audit.child.json | `572c1a507295bc28bed794a323dd2d4b8b8f092fae5d71b2c6c10e77c59b6606` |
| bundle_audit.reference.json | `d7464e7d731c3b310f6b975c743aba0abe2919d4a1682c2b9500963f775639dc` |
| benchmark_candidate.json | `b3cc91ed055080d4274b61c7bdcfc1ceff9037f53b81000698cc744d6b3114c9` |
| stage_status.json | `0f884bc81260759090d16d397e26dbb0975689117bffb81eb9bcd169420c3ab5` |
| official_student/metrics.json | `78c025fc8dee30fe99c28f6f8d75bc9d9469736f38a368bfcecf193feb0af7a5` |
| official_a0/metrics.json | `9ff0b35279092bd89337e0c4920ffad027e62deaeb0a8f56b37c2ff760e06b65` |


