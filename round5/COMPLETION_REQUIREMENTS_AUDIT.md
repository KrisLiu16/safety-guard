# 最终需求与证据完成性审计

2026-09-24。本次仅检查本地计划、实现、已归档结果及状态快照；没有查询集群、运行模型或修改冻结输入。以下“等待”不是已通过。路径相对项目根 `/Users/liuzhihao/Work/safety-guard`。

判定优先级：用户最后确认的方向 → Round5 冻结协议/执行契约 → 实际 JSON/JSONL、权重 SHA 与独立审计；历史计划不是完成证据。`TRAINING_PLAN.md` 顶部仍有 Round4 运行中的旧文字，正文也保留首轮 LoRA/RL-B/三 seed 等拟议规格，不能将它们全部当成本轮同时必须执行的事项。

本地快照 `round5/LIVE_STATUS.json` 的观测时间为 `2026-09-23T20:07:42.318286+00:00`，记录 SFT 1856/4164；`delivery_status.json` 为 `waiting_for_training`。本次检查时 Round5 最终 summary、workflow_status 和本地 release 清单均尚未出现。不能由脚本写完、源码冻结或旧轮结果推断第五轮完成。

## 明确要求与验收证据

| 要求 | 当前判定 | 权威证据与最终核验方式 |
|---|---|---|
| **小于 1B，直接分类，不生成答案 token** | 历史结构已证明；最终导出待验 | `round4/results/output/round4/window/setup.json` 为 753,456,471 参数；`round3/l20/base_compare/train_base.py:Classifier` 与 `round5/runtime/standalone_model.py` 是双角色分类头。最终须在 `round5/results/output/round5/validation/bundle_audit{,.child,.reference}.json` 对齐实际参数/24 层/无 LM head/生成 token=0，而非只看模型名“0.8B”。 |
| **不是单纯删层提速；网络支持高效增量状态** | W512/GDN 原理与旧 L20 实测已证明；新运行待验 | `round3/l20/H24_WINDOW_PROBE_REPORT.md`、`round4/results/output/round4/window/setup.json`、`round5/runtime/canonical_block_engine.py`：18 GDN + 6 有限窗口注意力、保留全部 24 层。最终 `validation/canonical_core_audit.json` 核对权重 SHA、固定 32 块、全部状态、未来 padding 因果性与 8192 边界。神经状态有界不等于分词、快照和回滚的端到端成本恒定。 |
| **whole 与实际文字流都可输入，用户/助手分别分类** | 旧轮通过；新 canonical32 等当前 L20 | `round5/runtime/run_guard.py` 提供 `whole/begin_message/append_text/end_session`；`canonical_text_runtime.py` 管理真实 BPE 回滚、角色切换及事务。须查 core 与独立 bundle 父/子审计的细项；比较相同 canonical32 路径。旧 bulk 与 cached 的 `0.0403082` 失败反例仍保留于 `round4/fast_runtime/diagnostics/CONTROL_RESULTS.md`，不能用新执行定义改写旧失败。 |
| **输出风险等级** | 二元风险有监督；三档全能力未证明 | 三概率接口是 safe/unsafe/controversial，当前硬标签与阈值验收仅 safe/unsafe。`round5/PREFIX_TRAINING_PROTOCOL.md`、`runtime/standalone_model.py` 明确第三档与类别头未验证；英文 user 无独立分层阈值。不能把三个数等同三档都训练并验收，也不能借用助手阈值。 |
| **L20 训练，完整主干后训练并实际做 RL** | Round4 已完成；Round5 等结束 | `round4/PRIMARY_RESULT_AUDIT.md` 与 `results/output/round4/{full,window,memory}/sft_summary.json`、`classification_rl/summary.json` 已证明三 SFT 各 4164、分类 RL 256。Round5 以 `results/output/round5/prefix_v2/{summary.json,sft/summary.json,classification_rl/summary.json}`、更新日志与导出 SHA 验 4164+256；`train_prefix.py` 的 `action_traces` 可复算奖励。它继承 Qwen3.5 预训练主干，不是从零预训练，也不是教师表示对齐或自回归文本 RL。 |
| **奖励与真实风险相关，不能只看 teacher 相似或总 reward** | 设计/CPU 证据已齐；最终能力等 L20 | Round5 reward 正确 +1、FP −3、FN −4，固定自身 SFT reference；`classification_rl/trajectory_metrics.jsonl` 应保留动作、概率、标签、source weight、KL/CE 与归一化系数。最终按已固定 dev 门选择，SFT/RL 的结果、选中 step 或保留初始 fallback 均要报告。完成 RL 不等于 RL 有收益；fresh/official 不准改选。 |
| **敏感词可出现在无风险语境，支持同词不同意图/安全拒绝** | 数据与旧模板诊断已证明；最终新权重专项复验缺失 | `round4/data/risk_v2/manifest.json`、`round5/data/prefix_v2/manifest.json` 含 4096 benign rubric、同词弱标签和公开正常样本；未知 unsafe 中途 prefix 不强标。历史 1024 条/256 词族诊断见 `round4/fast_runtime/BASELINE_RECHECK_AUDIT.md` 及其预测。**当前 Round5 workflow 没有重跑该 probe**；fresh390 是公共数据，不保证覆盖这些词族/模板，更不是中国公共事务政策金标。 |
| **词表下载、固定 JSON 格式、每词一中一英的批量产物** | 词表/Schema 和大量部分产物已证明；全量完成未证明 | `lexicon/source_lock.json`、`lexicon/RESOURCE_CATALOG.md`、`round2/oneword2_v12/full_run_lock.json`、`runtime_overrides.json`；用户调并发 400 的记录存在。较大本地快照 `completed_snapshot_v2/{summary,audit}.json` 为 268912 个词产物、221694 个完整中英词对、488560 条格式有效例句，仍 `partial_snapshot=true` 且语义未金标核验。不能称固定 449575 seeds 全部成功，更不能把这批新增语料说成已全部进入本轮 33311 条训练。 |
| **Qwen 与 Llama 都试试** | Qwen 两种已实训；Llama 未完成 | `round3/l20/FULL_BACKBONE_COMPARISON_REPORT.md` 与 `base_compare/output/` 证明 Qwen3-0.6B 与 Qwen3.5-0.8B 各 1280 步完整主干探针。`round3/l20/base_compare/models_lock.json` 中 Llama-3.2-1B 为 `access_blocked=true`、无已授权 HF 登录。此为历史访问证据，不是当前权限再探测；未找到 Llama 训练/速度产物，本轮全绿也不会补齐。不得声称小底座全面比较后“Qwen 最好”。 |
| **用户指定 Qwen3GuardTest 基线与最终能力观测** | A0/旧候选全片历史证据已齐；新候选等 L20 | `round4/validation_results/output/round4_validation/official_{a0,student}/` 各 2441 行；`round5/evaluate_canonical_official_l20.py` 应产出 `results/output/round5/canonical32_official/{metrics.json,*_predictions.jsonl}`，逐片 1059/569/813，1872 唯一输入、569 复用。thinking_loc 与 thinking 重叠，两个 loc 无 safe 分母。当前是 native-token 适配两连续判决，**不是官方原 Qwen3 token 定位、128-token lag 的完全复现**。历史已看官方数据仅作回归。 |
| **独立校准、低误报流式与不泄漏最终测试** | 分离/冻结/现代分词 CPU 证明齐；质量等当前 L20 | `round5/native_tokenizer_proof.json` 与 `data/prefix_v2/`，cal900/dev1200；`data/fresh_holdout_v1/{manifest.json,episodes.jsonl}` 固定 390/390 families、六格各65。最终 `canonical32_calibration/{initial_window,selected}/calibration.json` 拟合 strict `>` whole/stream 阈值；各 strata stream dev FPR≤.05、whole 宏召回较同 canonical 初始下降≤.02。`canonical32_fresh390/` 只应用锁定阈值。即使流程完整，quality gate 也可能 false；390 例不证明未知政策/分布下的 FPR SLA。 |
| **速度用 ITPS、延迟、吞吐并计入流式真实成本** | 旧轮明确结果已证明；新单会话结果等 L20；服务负载缺失 | 新 core 的 `actual_text_timings` 包含同轨迹独立预热、各128次 append、净新增 token、padding/replay/forward 分开、原始延迟/P50/P95、CPU 可见分类和初始化单列。旧同场 RL/full-A0 六组证据见 `round4/fast_runtime/BASELINE_RECHECK_AUDIT.md`，不能移作 Round5 权重/执行路径的提速比例。本轮没有新候选-A0同协议同场速度矩阵，也没有连续 batching、HTTP/排队、并发1/8/32固定 P95 SLO 吞吐或 P99/峰值容量完整验收；单会话 ITPS 是计算吞吐，不是最大服务吞吐。 |
| **完整可加载产物、可复查、GPU 释放后本地交付** | Round4 已有完整本地包；Round5 等流程及本地验收 | `round4/release/classifier.safetensors` 实际存在，清单记录 1509079036 bytes / SHA `29fc3e…af222`，本审计没有再次全文件哈希。Round5 最終需 `results/output/round5/validation/{MODEL_MANIFEST.json,bundle_audit*.json,bundle/BUNDLE_MANIFEST.json}` 与本地 `round5/release/` 全文件 SHA。`complete_delivery.py` 绑定两次 training summary 与四方 checkpoint SHA，训练 Job 完成→验证 Job→完成→CPU下载；`delivery_status.json` 保留 `validation_pipeline_integrity_pass` 和 `canonical_quality_gate_pass`。下载成功不能替代质量或全流程通过。 |

## 流水线全绿仍不能声称完成的范围

1. **效果全面提升或已达业务政策要求。** 当前 pipeline 的完整性 pass 与开发质量 gate 分离；允许保留初始权重，也允许打包 `research_unvalidated_runtime`。此前 thinking 两连续风险 FPR 90.41% 的失败是真实的，本轮改善与否必须读新结果。没有独立中国公共事务政策金标，不能由词表模板/通用公开集推断目标域达标。
2. **完整三档、风险类别、中英四角色覆盖。** safe/unsafe 是当前有监督的核心；controversial/category 与 en/user 阈值仍缺相应独立证据。
3. **任意输入切分均具有同样风险召回或固定误报 SLA。** canonical 的数值一致性测试是有界样例；cal 统计 native target 全前缀 + 最多8个重新分词切点。服务 `append_text` 返回每次追加末位置概率，并非自动执行所有内部位置的累计拦截策略。无自然 risk-onset gold，也没有 release/hold/block 的 episode 策略、已泄漏风险量或“证据出现到检测”的真实语义延迟验收。
4. **最终模型与 A0 相比全面更快、更省。** 当前新 core 只比较同权重 canonical eager/graph 的一个文本轨迹。旧 window/chunk8 的4.19×、旧 RL/full 与 A0 六组都不能拼到 Round5 最终包。理论有界缓存与小规模隔离测试不是高并发服务容量证据。
5. **Llama 已试过、全量造数已完成、最终新权重已通过同词正常语境专项测试。** 分别只有访问受限记录、部分生成快照、旧权重诊断；本轮 workflow 不自动完成这三项。

这些缺口应在最终交付中明确保留；是否扩展研究另列范围，不得为了把清单全部改绿而重用旧模型通过结果、扩大指标口径、放宽容差或看 fresh 重调模型。

## 不应误加为当前硬性要求的候选讨论

- 从零语言预训练、JEPA/JEV 式专用网络、先训练 4B 再蒸馏、MoE、DSA/MLA、移除所有注意力，均有讨论/设计；后续已收敛到继承完整小基座、风险后训练与 W512/GDN。没有同时实施这些路线的最终确认，不能把未做它们称为本轮执行失败。
- 早期计划的 LoRA 主线已被完整主干全参数训练取代；早期每词10/100句已被每词一中一英 v12 取代；先用 M5 已被用户明确改成 L20。不可反过来要求恢复旧规格。
- 早期计划中的量化、连续 batching、HTTP SLA、三 seed、流式控制 RL 等是未完成的后续/部署研究。它们不能作为已经交付的能力；对用户明确要求的“更快、更省、吞吐高”，至少要报告目前测试范围和没有达成/没有测量的部分，不能仅用训练完成作答。
- “开源研究模型”不等于用户已指示立即公开发布到外部平台。当前验收对象是可分享的本地完整研究包及来源证据；本轮没有公共托管发布证据，不自动执行发布。

## 收尾时必须给用户的一页结果

固定最终 SHA/实际选择（SFT、RL 或保留初始）、继承的基座及 24 层结构；SFT/RL 完成与选模证据；同 canonical 初始/最终的 cal/dev/fresh 指标；官方三片与 A0 的明确改善和退化；同权重 eager/graph 真文本 ITPS/P50/P95/启动与状态成本；包的实际绝对路径、独立加载与本地 SHA 核验、GPU Job 终态；以上缺口。必须说明这是分类器，既不生成文本，也没有证据表明三档/所有语言角色/任意政策都已验收。
