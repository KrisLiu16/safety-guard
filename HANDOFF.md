# 安审分类模型 · 工作交接

更新：2026-09-24。本文由上一段 Codex 会话的完整记录整理而来，去掉了工具调用细节，只保留决策、结果和待办。上一段会话因额度用尽中断，中断时第五轮自动流水线仍在后台运行，后来已全部跑完；本文第 4 节是跑完后的真实结果。

**先读这一节，再读第 1 节的用户决策；其余按需查阅。**

## 0. 一句话现状

我们在 Qwen3.5-0.8B-Base 完整 24 层主干（18 层 Gated DeltaNet + 6 层注意力，约 0.753B 参数，无 LM head）上训练了一个**直接输出风险分类、支持流式输入**的安审模型。已经完成五轮实验。第五轮的前缀监督 SFT（4,164 步）和分类 RL（256 步）都跑完了，但**没有检查点通过质量门槛**，最终保留的是第四轮 window/SFT 初始权重（SHA `bb16a3a6…`）。当前最大问题是**流式前缀误报过高**：在官方 thinking 分片上，按“连续两次 Unsafe”规则计算的 FPR 约 90%，原版 Qwen3Guard（A0）约 22%。

## 1. 目标与用户已定的决策（必须遵守）

- **模型类型**：分类模型，不做自回归文本输出。每次追加输入都要立即返回风险等级和概率。可以参考 Jev 的“直接给出概率化决策”思路。
- **速度指标**：输入 token 速率 ITPS、分类延迟（P50/P95）、吞吐和会话状态显存。**不用生成 TPS。**
- **规模**：1B 以内。要从计算原理上提效，不能只靠删层缩参数。可以考虑 MoE、稀疏或有限窗口注意力、循环状态，甚至自行预训练；实验后已收敛为继承完整小基座再改造。
- **训练硬件**：只用腾讯云 L20。本机 M5 不再做神经网络训练或推理，只做数据准备、校验和报告。
- **选模依据**：看真实风险指标（召回、误报、漏报），**不看与教师模型的相似度**。用户原话：教师“不是专门做安审的”。
- **训练方式**：完整主干全参数后训练，再做分类 RL；LoRA 只作可选对照。一次实验失败不等于路线错误，可以放手尝试更好的方案。
- **难负例**：大量补充“含敏感词但语境无害”的样本（新闻、历史、引用、文字处理、安全拒答），作为降低误报的 rubric，也用于 RL 奖励。
- **数据用途**：非商业开源研究。词表候选**不需要逐词人工审核**即可入库。Citizen Lab 数据采用 CC BY-NC-SA 4.0，需保留署名和许可。
- **造数格式**：**一词一次请求，输出一条中文和一条英文**（v12），并按来源类别分组。格式冻结后，运行中途不得修改。提示词里要附带 Schema 示例，不能只用文字描述。十词一请求（v9/v10）会让模型注意力涣散，已放弃。
- **Aster 规则**：采样次数、并发和尝试次数由用户决定。`aster-dev-272` 的并发是用户自己调到 400 的。
- **基座对照**：用户要求 Qwen 和 Llama 都试一下。Llama-3.2-1B 权重受访问限制，本机没有已授权的 HF 登录，目前还没做；需要用户执行 `hf auth login`。
- **集群资源**：可以停止占卡但 GPU 实际空闲的 Pod，前提是多次采样显存 0 MiB、利用率 0%，并保留其 PVC。仍在工作的 Pod 不能动。time-slice 节点可以拿来当整卡用，做法见第 2 节。
- **沟通**：用中文，说人话。用户期望工作 solid：提交大规模任务前先想清楚、先做小样验证。上一段会话曾因格式和任务粒度没想好而返工，被用户批评过。

## 2. 运行环境与资源

| 项 | 值 |
|---|---|
| 用户本机项目根 | `/Users/liuzhihao/Work/safety-guard`（即本仓库；权重、jsonl 数据、词表原文均在 `.gitignore` 中，只存在于 Mac 或 PVC） |
| 本机 Python 环境 | `safety-guard/.venv`（Python 3.12、torch 2.14、transformers 4.55，仅用于 CPU 校验） |
| TKE 集群 | `cls-og1rjus2`，namespace `default`（内网 API 地址见本机 kubeconfig） |
| kubeconfig（Mac） | `~/.kube/cls-og1rjus2-private-latest`；kubectl 在 `~/.local/bin/kubectl` |
| 腾讯云凭证（Mac） | `~/.config/tencent-cloud/tencent-test.env`（600 权限，**不要读出或粘贴值**） |
| 主用 L20 节点 | `172.19.1.144`，device plugin 配置 `timeslice-8x`。当前凭证**不能修改 Node**（RBAC 拒绝），所以做法是让 Job 申请 `nvidia.com/gpu: 8`，占满同一张物理 L20 的全部逻辑份额 |
| 其他 L20 | `172.19.0.92`、`172.19.0.130`，独占模式，常被其他成员的 Pod 占用 |
| PVC | `safety-guard-base-compare-data`（StorageClass `naive-sandbox-ssd`，已扩到 80Gi），各轮权重和报告都在 `/work/output/...` |
| PVC 上的 Python 环境 | `/work/modern`（transformers 5.17 + fla，用于 Qwen3.5）和 `/work/legacy`（transformers 4.55，用于原版 A0） |
| Aster | 已登录 CLI（上一段会话里执行过 `aster update`）。`gpt-5.6-luna` 用于 v12：`mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`；pilot/round1 用 `mdl_01M2TK4P8Z0DE5QCJYJ8RSK8YV`；MiMo `mdl_01M33V8C39GGNX33TFS5ETDNR3`（mimo-v2.6-flash）被 content_filter 大量拒绝，已弃用 |

**集群作业模式**（第三到第五轮都这样做）：先建 Job，Pod 起来后等待 `*_ready` 标记；本地打 tar 包，`kubectl cp` 上传后校验 SHA256，再写 ready 标记。训练结束后，本地收集脚本（如 `round5/collect_results.py`、`round5/complete_delivery.py`）取回报告，写 ack 标记让 Job 退出、释放 GPU。权重下载单独起**只用 CPU 的 artifact Job**，并逐文件校验 SHA。

## 3. 工作历史（按阶段）

### 3.1 调研与计划（2026-09-22）

- **Jev**（TypeSafe）是输出概率化决策的分类模型，官方没有开源权重。社区复现 NanoJev 的结构是 Qwen3-0.6B 主干加决策评分头。
- **网信办官方完整敏感词库没有公开下载**，只有规范（TC260-003、GB/T 45654-2025）。社区词表只能当主题线索，不能当官方标准。
- **Qwen3Guard-Stream**（0.6B/4B/8B）做 token 级分类，有 query/response 两个头。公开包装器默认关闭缓存，每一步都重算完整序列。其 119 万条训练数据未公开，公开的只有 `Qwen3GuardTest`（CC BY-NC）。
- 可用的开源数据：SafeConv、ShieldLM、CValues、Safety-Prompts、BeaverTails、PKU-SafeRLHF、Nemotron-Safety-Guard v3（中文）、COLD。
- 计划从 v0.1 迭代到 v0.4（`TRAINING_PLAN.md`，旧版在 `archive/`）。用户在这期间定下：以 RL 为主线、用 Qwen3Guard-Stream-0.6B 起步、用 Qwen3GuardTest 做 bench、用 Sensitive-lexicon 造数、推理优化要从原理层面入手。

### 3.2 Pilot 与第一轮（M5，已完成）

- **Pilot**（`pilot/`）：`aster-dev-253`，50 词 × 10 条，并发 5。得到 500 条原始例句、459 条结构合格、352 条 RL 候选。
- **Round1**（`round1/`）：`aster-dev-254`，50 词 × 100 条，并发 50，得到 4,950 条、清洗后保留 4,095 条。在 M5 上用 MPS 做了 159 步 LoRA + 风险头 RL。
  - 提速：改成真正的 KV 缓存后，1K 上下文逐 token P95 从 201.6 ms 降到 18.7 ms（约 10.8×）。
  - 能力：官方 F1 最多下降 0.36 个百分点，流式 user 误拦约 9.3%，能力收益有限。
- **首次上 L20**（`l20/`）：腾讯云权限开通后，用户授权删除两个空闲占卡 Pod（保留 PVC）。单卡试跑 16/16 一致，1K 缓存 P95 为 19.49 ms。

### 3.3 词表（`lexicon/`，已完成）

- Sensitive-lexicon 全部 17 个文件去重后得到 51,085 项，其中文本 36,295、网址 14,592、`+` 组合 198。“维尼”“圣上”都在其中。
- 加入 houbb（Apache-2.0）和 SpaceGather（MIT）后扩到 107,282 项。
- 加入 Citizen Lab（CC BY-NC-SA 4.0）后，非商业研究层共 **492,640 项**，生成 **449,575 个文本任务种子**。
- COLDataset 的 37,480 条只用于评测。

### 3.4 第二轮：批量造数（`round2/`）

- 经历过 MiMo 被拒、每词十句、十词一请求等版本，最终定为 **v12：一词一请求、每词一中一英**（一安全一风险，均为 user 侧）。Schema 和提示词冻结在 `round2/oneword2_v12/full_run_lock.json`。
- 全量 Run **`aster-dev-272`**（`run_01M36NWG2G4NF17NYJAX95HPN5`）：449,575 词，33 个来源类别，切成 1,517 个物理分片（受 Aster 单 Task 24 小时限制）。max_attempts=1，并发由用户从 40 调到 400。
- **已完成**：1,517/1,517 分片成功，reward_total 1242.04。reward 是每个分片“两句都合格的词数 ÷ 总词数”，平均约 0.82。
- **已提取到本地（Mac）**：
  - `partial_snapshot_50`：20,029 条。
  - `completed_snapshot_v2`：912 个分片，268,912 个词产物、488,560 条格式合格例句、221,694 个完整中英词对。
- **全部 1,517 个分片的提取还没做完**：`full_snapshot_final_v1` 刚开始时额度就用完了。
- 所有标签都是 `luna_synthetic_unverified`，且全部为 user 侧，不能当金标，也不能代表 assistant 侧数据。

### 3.5 第三轮：架构探索（`round3/`）

- **删层路线已放弃**：
  - C1（14 层）速度约 2×，但官方 thinking F1 从 0.819 降到 0.603。
  - C21（21 层）严重漏判。
- 研究过 4B→0.6B 蒸馏、纯循环从零预训练（R24 约 0.619B）、MoE 升级（约 0.951B）、DSA/MLA 等方案，文档都在 `round3/*.md`，均未实施。
  - 已下载 FineWeb2/FineWeb-Edu 各 1 万篇作为从零预训练的接入样本（约 1,938 万 token），但用户后来改为“继承完整底座”。
- **完整主干对照**：Qwen3-0.6B-Base 与 Qwen3.5-0.8B-Base（H24）各做 1,280 步，先 64 步只训分类头，再全参训练。用户质疑 LoRA 后，默认改为全参。
- **H24 有限窗口（W512）**：保留全部 24 层，只把 6 层注意力改成窗口 512。
  - 8K 历史下会话状态从 114.8 MiB 降到 24.8 MiB。
  - 短 chunk 递归内核把 ITPS 从 263 提到 377，P95 从 30.6 ms 降到 21.4 ms。这个收益来自内核分派，**窗口本身没有提速**。
  - 两轮窗口恢复训练都是按教师相似度选模，结果不理想。之后用户要求改成按风险指标选模。
- 分类输出契约：`round3/guard-classification-v1.schema.json`。

### 3.6 第四轮：按风险指标训练（`round4/`）

- 数据 `risk_v2`：
  - 训练 33,311 条：公开数据 20,000、同词弱标签 8,192、无害 rubric 4,096、长背景 1,023。
  - 校准 900、开发 1,200、保留测试 1,200。
- 三组各做 4,164 步全参 SFT：full、window（W2048→1024→512 渐进缩窗）、memory（窗口外加可学习的压缩历史）。
- 在 full 上做了 256 步分类 RL：每个输入采样 4 个动作，奖励为正确 +1、误报 −3、漏报 −4，外加 KL 和 CE 锚定项，**不生成任何 token**。
- memory 的 8K 长流一致性误差为 0.0326，超过容差 0.03，**已排除，没有放宽容差**。
- 主候选是 **RL/full**（SHA `29fc3e…af222`，取第 192 步），保留测试结果：

| 模型 | 召回 | FPR |
|---|---:|---:|
| RL/full | 70.83% | 4.33% |
| A0 | 65.17% | 3.83% |

- **官方 thinking 流式 FPR 为 90.41%，A0 为 22.24%**，这是质量阻碍。原因是训练只监督了消息结尾，没有监督前缀。
- 与 A0 同协议对比速度：RL/full 在 6 组配置中有 5 组更慢，比值为 0.85–1.02×。
- window 的 CUDA Graph（token-ID、chunk 8）：ITPS 1,425，eager 为 340。但真实文本接口的 whole/cache 一致性审计失败，bulk 与 cached 的差值为 0.0403。这一问题促成了第五轮的 canonical32。
- 关键词诊断：安全模板上误报为 0；中文诈骗类召回 100%；en/user 没有校准阈值。
- 第四轮完整模型包已下载到 Mac 的 `round4/release/`（1,509,079,036 字节，SHA 已核对）。

### 3.7 第五轮：前缀监督与 canonical32（`round5/`，流水线已跑完）

- **数据**：`prefix_v2` 共 85,399 个视图，全部用现代分词器核对一致。风险起点未知的中途前缀不强加硬标签。25 个中文 UTF-8 字节未输入完整的标签位置已剔除。
- **独立终测**：`fresh_holdout_v1` 共 390 条，来自 390 个不同词族，6 个分层各 65 条，不参与选模。
- **canonical32 执行契约**：从序列起点每 32 token 算一块；未满的块只在未来位置补 padding，不提交该块的缓存；BPE 回滚时回到块边界重算。这样整段输入与分批输入走同一条计算轨迹。设计见 `round5/CANONICAL_EXECUTION_PLAN.md`。
- 从第四轮 window/SFT（`bb16a3…`）出发，做 SFT 4,164 步 + RL 256 步，共 8,336 秒。**两个阶段都没有产生合格检查点**，结果见第 4 节。

## 4. 第五轮最终结果（权威数据在 `round5/results/output/round5/`）

- `delivery_status.json` 的状态：
  - 训练 Job、验证 Job 和 CPU 下载 Job 都已完成。
  - `validation_pipeline_integrity_pass=true`，`canonical_quality_gate_pass=false`。
  - 模型包已下载到 Mac 的 `round5/release/`，第二次下载成功。
- **最终清单** `validation/MODEL_MANIFEST.json`：
  - candidate 为 `initial_window_fallback`，状态 `research_unvalidated_runtime`，推理引擎 eager。
  - `production_approval=false`；第三档风险和类别头**未验证**。
- **选模门槛**：各流式分层 FPR ≤ 5%，且整段宏召回相比初始下降不超过 2 个百分点。

开发集结果（1,200 条；整段 = 整段宏召回，流式 = 流式宏召回，最后一列为 en/assistant、zh/assistant、zh/user 三层的流式 FPR）：

| 检查点 | 整段 | 流式 | 流式 FPR（三层） |
|---|---:|---:|---|
| 初始（bb16a3） | 70.2% | 65.2% | 6.5 / 6.5 / 9.5% |
| SFT epoch1 | 64.5% | 64.2% | 6.5 / 4.5 / 6.0% |
| SFT epoch2 | 64.5% | 61.7% | 4.5 / 6.0 / 5.0% |
| RL 64–256 步（从初始出发） | 70.2–70.3% | 65.0–65.7% | 5–6 / 6.5–7 / 9.5–11% |

**要点**：SFT 第 2 个 epoch 的流式 FPR 几乎达标（只有 zh/assistant 为 6.0%），但整段召回下降了 5.7 个百分点。RL 则基本没有改变模型行为。

- **fresh390**（初始权重，canonical32，阈值已锁定）：
  - 整段：宏召回 72.8%，FPR 6.7%。
  - 流式：宏召回 67.7%，FPR 9.2%。其中 zh/assistant 流式 FPR 为 13.8%，en/assistant 流式召回只有 44.6%。
- **官方适配评测**（2,441 行）：

| 模型 | thinking 连续两次 Unsafe（召回 / FPR） | 端点 argmax（召回 / FPR） |
|---|---|---|
| 学生 | 98.4% / 89.8% | 66.8% / 5.1% |
| A0 | 81.5% / 22.2% | 60.3% / 1.0% |

- **canonical32 核心审计通过**：到达方式不变性、未来 padding 的因果性、graph 与 eager 一致，以及 padding 缓存不会被提交。
- **真实文本速度**（单会话、上下文 705→1,665、128 次追加，含重分词、回放和 padding 计算）：

| 引擎 | ITPS | P50 | P95 |
|---|---:|---:|---:|
| eager | 229.6 | 27.9 ms | 52.4 ms |
| CUDA Graph | 706.3 | 9.4 ms | 16.0 ms |

  前向计算的 4,928 个 token 中有 1,984 个是 padding。**本轮没有在 canonical32 协议下与 A0 做同场速度对比**，第四轮的速度比例也不能套用到这里。
- 独立模型包审计通过：在新进程中加载，不依赖旧的基座或初始权重。

**注意**：`round5/END_TO_END_RESULTS.md` 是训练中途生成的，**内容已过时**，全是 pending。需要用 `round5/render_report.py` 按最终产物重新生成。`TRAINING_PLAN.md` 顶部的状态也还停留在第四轮。

## 5. 未完成事项与已知问题

1. **流式前缀误报**是核心问题：官方 thinking FPR 约 90%，fresh390 的 zh/assistant 为 13.8%。“降低前缀误报”和“保住整段召回”在当前训练设置下互相冲突。
2. **assistant 侧训练数据不足**：v12 生成的全部是 user 侧；assistant 侧只有公开数据和 4,096 条 rubric。
3. 第三档风险（controversial）、类别头和 en/user 阈值都没有得到有监督训练，也没有验收。
4. `aster-dev-272` 还有约 605 个分片没有提取；已提取的也没有做语义质检和去污染（别名、模板、与外部 benchmark 的重叠）。
5. 最终权重还没跑同词不同意图专项测试（`round5/final_checks/evaluate_keyword_canonical_l20.py` 已写好，没有运行），也没做 canonical32 与 A0 的同场测速。
6. Llama-3.2-1B 因权重访问受限，还没有做对照。
7. 服务层还没做：连续批处理、HTTP 并发压测（1/8/32 会话）、P95 SLO 下的吞吐。早期 HTTP 测试中 32 个客户端基本都在排队。
8. MoE、纯循环预训练等方案只写了设计文档，都没有实施。

## 6. 建议的下一步（供参考，开工前请与用户确认方向）

1. 用 `round5/render_report.py` 重新生成第五轮报告，并更新 `TRAINING_PLAN.md` 顶部状态，然后向用户汇报第五轮结果：门槛没过，最终保留初始权重。
2. 用 `round2/oneword2_v12/extract_archives.py` 提取 `aster-dev-272` 全部 1,517 个分片（先分页取 samples，再逐个查 attempt，最后用 `runs archive` 下载），然后做质检、词族去重，并记录与 benchmark 的去污染情况。
3. 针对前缀误报设计第六轮。可以考虑以下方向：
   - 分开整段读出和流式读出的损失权重；
   - 为流式单独设置读出头或阈值；
   - 补 assistant 侧的无害和拒答前缀；
   - 以 SFT epoch1/epoch2 为起点，加入召回保持约束；
   - RL 奖励直接以 episode 级别的流式误报和漏报计算。

   注意：官方集和 fresh390 都不能参与训练或调阈值。
4. 在 L20 上跑最终权重的关键词专项测试，以及 canonical32 与 A0 在同协议下的 ITPS/P95 对比。
5. 如果用户完成 HF 登录，补上 Llama-3.2-1B 对照；是否再做 MoE 由用户决定。

## 7. 目录地图

| 路径 | 内容 |
|---|---|
| `TRAINING_PLAN.md` | 总计划，多版叠加，顶部状态已过时；旧版在 `archive/` |
| `pilot/` | 50 词试跑的 flow、提取脚本和报告 |
| `round1/` | M5 首轮：计划、runtime、训练、评测、导出，`ROUND1_REPORT.md` |
| `l20/` | 首次 L20 试跑 |
| `lexicon/` | 词表构建和校验脚本、`RESOURCE_CATALOG.md`、`source_lock.json`（原文未进仓库） |
| `round2/` | 造数各版 flow；`oneword2_v12/` 是最终冻结版（lock、提取器、快照汇总） |
| `round3/` | 架构调研、基座对照、窗口改造、各报告（`l20/*_REPORT.md`）、分类 Schema |
| `round4/` | 风险训练（`train_risk.py`、`memory_attention.py`）、验证、`fast_runtime/`（CUDA Graph），`PRIMARY_RESULT_AUDIT.md`、`VALIDATION_BASE_AUDIT.md` |
| `round5/` | 前缀训练 `train_prefix.py`、协议 `PREFIX_TRAINING_PROTOCOL.md`、`runtime/`（canonical32 引擎、`run_guard.py` 独立入口）、评测与校准脚本、`results/output/round5/`（最终结果）、`COMPLETION_REQUIREMENTS_AUDIT.md` |

## 8. 操作要点与踩过的坑

- **冻结的东西不要改**：v12 Schema 和提示词、`risk_v2`/`prefix_v2` 数据、fresh390、各轮选定的 SHA。要改就另起新版本、新目录。
- **测试集隔离**：Qwen3GuardTest 已被多次观测，只能用于回归观察；选模只看开发集，阈值只用校准集拟合。
- **Aster**：
  - 大 Task 上传会报非标准 400，每个归档要小（一个归档放 1 个 Task 最稳）。
  - `runs attempts` 最多返回 100 条，需要按 sample 逐个查询。
  - 分片受单 Task 24 小时时限约束。
- **A0 上限**：原版 A0 的 `max_position_embeddings=8192`，测速轨迹不能超过这个长度。
- **分词器差异**：本机与集群的 transformers/tokenizers 版本不同，评测前要逐条核对 token IDs。
- **加载器陷阱**：旧加载器可能从上一轮目录误加载代码，验收时要核对实际加载文件的哈希。
- **完成 ≠ 通过**：流程跑完、进程退出码 0、下载成功，都不代表质量通过；汇报时要分开说。
- **凭证**：只引用位置，不输出内容。
