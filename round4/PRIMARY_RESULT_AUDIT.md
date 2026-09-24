# Round 4 主训练真实产物验收

审计日期：2026-09-24。范围为本地已收集的 [主训练产物](results/output/round4/)、冻结数据、代码和 validation 包；只运行 CPU 解析、哈希与指标复算，没有运行模型、访问集群、修改阈值或重新选模。本文不替代另行运行的最终 L20 验证。

## 结论

三条 SFT 均完成 4,164 次更新；RL 完成 256 次更新，并依开发集记录固定选择第 192 步。最终候选仍为 **classification_rl/full**，checkpoint SHA-256：

`29fc3e5663eaa333235b1c2ba84d4c0a964f6ece4fd146c9fc6ec01761eaf222`

已记录的 checkpoint 身份在 RL summary、8K 审计、CLI 审计、四份 CLI 输出 metadata、[MODEL_MANIFEST.json](MODEL_MANIFEST.json) 及冻结 validation 包内的 manifest 闭合一致。候选通过本轮 token-ID 8K 增量一致性和 CLI 整段/分块一致性检查。**memory 的 8K 审计失败：唯一超限的助手头概率误差为 0.03256136178970337，超过原定 0.03。没有放宽容差。**

在 1,200 条保留测试中，候选相对 A0 多识别 34 条风险、增加 3 条误报；相对 full SFT，识别出的风险总数相同、增加 1 条误报。因此开发选模规则得到的固定候选有依据，但不能把这次 RL 描述为保留测试质量全面提升。

## 训练完整性、数据和架构

| 分支 | 实际连续更新 | epoch | 实际累计输入 token | 开发集选中步数 | 参数量 | 导出开发分数 |
|---|---:|---:|---:|---:|---:|---:|
| full SFT | 1–4164 | 2 | 7,466,070 | 2048 | 753,456,471 | 0.611667 |
| window SFT | 1–4164 | 2 | 7,466,070 | 4096 | 753,456,471 | 0.610000 |
| memory SFT | 1–4164 | 2 | 7,466,070 | 2560 | 753,554,823 | 0.601667 |

逐个读取三份 losses.jsonl：各 4,164 行、更新号无缺失或重复、loss 全部有限且非负；每个 epoch 2,082 次更新。按冻结训练数据及代码的长度排序、256 行分组、每个 epoch 固定随机种子、16 行 batch 重建顺序，**全部 4,164 个累计输入 token 值逐步吻合**。每轮 33,311 个样本、3,733,035 个输入 token；每个分支训练两轮。window/memory 各有 1,041 次 W2048、1,041 次 W1024、2,082 次 W512 更新。三条分支均完成全部预定更新，而非只保存早期检查点后终止。

训练数据分为公开来源参考标签 20,000 条、合成弱标签 8,192 条、无风险 rubric 4,096 条、上下文弱标签 1,023 条。校准集 900 条、开发集和保留测试各 1,200 条。[risk_run_spec.json](results/output/round4/risk_run_spec.json) 中的数据 manifest 与本地冻结 manifest 完全相同；[training_files.sha256](training_files.sha256) 的 10 项代码/数据文件重新计算后全部吻合，收集到的 worker 日志也记录了训练前的 SHA 检查成功。

| 冻结输入 | 行数 | SHA-256 |
|---|---:|---|
| train.jsonl | 33311 | `9925114b148ee38df3e193a1e427e38b7dd8c6805c1bf28a8b67da868c7c5aa7` |
| calibration.jsonl | 900 | `a39f9a961ddf99ee8a01ab8e148be1749a4d6a6ddca36024712fb7615e0f2663` |
| dev.jsonl | 1200 | `7c3ca2d4e99e4070ae9ec602de6c5a4beeb41114373c6c96ceb98f36734d02c9` |
| sealed_test.jsonl | 1200 | `2fbcbe2b3494ceaa0caa9d0191a7e107ae16ef2a0734f53e24d37a1b75423e73` |

tokenizer SHA-256：`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`。

归档的 backbone_config、冻结加载代码和执行记录一致指向 **完整 24 层 Qwen3.5 语言骨干：18 层 GDN/linear attention + 6 层 attention，hidden size 1024**。三条分支没有截层；window/memory 替换对应 attention 实现。setup 和最终 summary 均记录 `has_lm_head=false`，冻结 Classifier 加载代码抽取完整 language_model 并只保留分类读出；SFT 是全参数风险分类后训练。最终 full 候选的六层 attention 仍为完整注意力，不能宣称其已取得固定窗口状态复杂度。

该模型使用已有 Qwen3.5 预训练底座及上一轮 H24 初始化；不是从零预训练。H24 初始化 SHA 为 `8e01e428cc9f2455df75fe48cf556b85ad3a69bb3fc3593b06280077ac344d1c`。

**证据边界：本地报告目录没有下载的 safetensors。** 本次没有独立枚举远端权重 tensor 或对实际权重字节重新计算 SHA；架构确认来自冻结代码、配置、参数量与已执行报告，checkpoint 身份来自各独立加载/审计回执。保留测试仅证明在这一冻结划分上的表现；弱标签、仅精确去重及上游底座/历史训练暴露的不完全可知性仍是限制。

## RL 轨迹与开发选择

[trajectory_metrics.jsonl](results/output/round4/classification_rl/trajectory_metrics.jsonl) 有连续的 1–256 更新，每项统计有限、概率/比例范围有效。每输入采样 4 个分类动作，生成文本 token 为 0；reward 为正确 +1、误报 −3、漏报 −4，采用来源权重。优化为 REINFORCE、每输入精确期望奖励基线、SFT 参考策略 KL（系数 0.1）和 CE 锚定（系数 0.2），不是用通用教师的措辞相似度当正确性。

使用冻结训练池、种子、每步 8 个输入与每输入 4 个动作，独立验证了 **全部 256 条聚合 reward/FP/FN 记录在这些奖励和来源权重下均存在合法离散动作计数**。平均加权 reward 为 0.4795166063，范围 −0.75 到 1。日志没有保存每个输入的具体采样动作，因此这属于聚合一致性校验，不能声称逐动作复现。不同步的样本不同，不把前后均值变化当成独立质量证据。

| RL 更新步数 | 开发选择分数 |
|---:|---:|
| 0（full SFT 导出） | 0.611667 |
| 64 | 0.611667 |
| 128 | 0.616667 |
| **192（已固定）** | **0.628333** |
| 256 | 0.620000 |

第 192 步是既有开发曲线最高点，导出开发分数相同。未使用保留测试调整候选、更新步数或阈值。

## 保留测试独立逐例复算

五份预测各有 1,200 个唯一 sample ID，与冻结保留测试集合一致；逐条标签、语言、角色、family 一致。所有概率有限、在 [0,1] 范围内，概率和误差不超过 1e−5。按 `p_unsafe >= threshold` 重算 TP/FN/FP/TN，并独立重算带并列值处理的 ROC-AUC、非插值 average precision（本报告称 PR-AUC/AP）和 Brier；与存档报告在 1e−12 容差内吻合。

每一分层均为 200 条风险、200 条安全，总计 600+600。召回/FPR/AP/ROC 均为三个分层的宏平均；AP 并非跨分层混合概率后的 pooled AP。以下表格使用各模型 sealed_test_metrics 记录的原阈值；固定候选改用 manifest 已冻结阈值时，1,200 个判决全部不变。

| 模型 | TP | FN | FP | TN | 风险召回 | FPR | 宏 PR-AUC/AP | 宏 ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full SFT | 425 | 175 | 25 | 575 | 70.83% | 4.17% | 0.931553 | 0.922367 |
| window SFT | 425 | 175 | 30 | 570 | 70.83% | 5.00% | 0.929800 | 0.919025 |
| memory SFT | 379 | 221 | 23 | 577 | 63.17% | 3.83% | 0.926036 | 0.916283 |
| 固定候选：RL/full | 425 | 175 | 26 | 574 | 70.83% | 4.33% | 0.931577 | 0.923067 |
| A0 原版 | 391 | 209 | 23 | 577 | 65.17% | 3.83% | 0.936109 | 0.929033 |

| 模型 | 分层 | TP | FN | FP | TN | 风险召回 | FPR | PR-AUC/AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full SFT | en/assistant | 137 | 63 | 7 | 193 | 68.50% | 3.50% | 0.925871 |
| full SFT | zh/assistant | 149 | 51 | 11 | 189 | 74.50% | 5.50% | 0.938337 |
| full SFT | zh/user | 139 | 61 | 7 | 193 | 69.50% | 3.50% | 0.930450 |
| window SFT | en/assistant | 137 | 63 | 11 | 189 | 68.50% | 5.50% | 0.925583 |
| window SFT | zh/assistant | 153 | 47 | 11 | 189 | 76.50% | 5.50% | 0.943818 |
| window SFT | zh/user | 135 | 65 | 8 | 192 | 67.50% | 4.00% | 0.919998 |
| memory SFT | en/assistant | 115 | 85 | 6 | 194 | 57.50% | 3.00% | 0.915577 |
| memory SFT | zh/assistant | 154 | 46 | 13 | 187 | 77.00% | 6.50% | 0.939426 |
| memory SFT | zh/user | 110 | 90 | 4 | 196 | 55.00% | 2.00% | 0.923106 |
| 固定候选：RL/full | en/assistant | 134 | 66 | 7 | 193 | 67.00% | 3.50% | 0.927000 |
| 固定候选：RL/full | zh/assistant | 151 | 49 | 11 | 189 | 75.50% | 5.50% | 0.938179 |
| 固定候选：RL/full | zh/user | 140 | 60 | 8 | 192 | 70.00% | 4.00% | 0.929552 |
| A0 原版 | en/assistant | 119 | 81 | 3 | 197 | 59.50% | 1.50% | 0.932339 |
| A0 原版 | zh/assistant | 149 | 51 | 16 | 184 | 74.50% | 8.00% | 0.940035 |
| A0 原版 | zh/user | 123 | 77 | 4 | 196 | 61.50% | 2.00% | 0.935953 |


固定候选对 full SFT：风险召回差 **0 个百分点**，FPR **+0.1667 个百分点**；宏 AP 仅 +0.00002438，宏 ROC-AUC +0.000700。三个分层召回依次为 −1.5、+1.0、+0.5 个百分点。不能由此宣称 RL 全面提升，亦未据此反向改变已冻结候选。

固定候选对 A0：宏风险召回 **+5.6667 个百分点**，宏 FPR **+0.5000 个百分点**；宏 AP **−0.00453165**，宏 ROC-AUC **−0.00596667**。更高的既定工作点召回伴随更多误报，排序指标没有超过 A0；这不是显著性检验结论。A0 在这里仅作质量参考，不把本表用于速度或架构归因。

### 阈值记录的一处差异

导出开发报告/manifest 与重载后 sealed_test_metrics 的阈值有轻微数值差异，不能在报告中把两者混作完全相同。候选如下：

| 分层 | sealed_test_metrics 记录 | 固定 MODEL_MANIFEST |
|---|---:|---:|
| en/assistant | 0.9538143277168275 | 0.9539467692375184 |
| zh/assistant | 0.9570255875587464 | 0.9570242166519166 |
| zh/user | 0.9069585800170900 | 0.9069603681564332 |

将已有固定导出阈值直接应用于保存的测试分数：候选、full、window 和 A0 均无判决变化；memory 的 en/assistant 一条风险由 TP 变 FN（该层 TP 115→114，总体宏召回 63.1667%→63.0000%），其 8K 不通过的结论不变。本次不更改任何阈值。

A0 保存了 900 条校准预测，按原校准规则重新推导阈值后完全一致。学生模型未归档逐条校准概率，因此本次可核验记录间对应关系及固定阈值下的测试判决，不能独立从原始校准分数重建学生阈值。

## 8K 与 CLI 审计对齐

[long_stream_audit.json](results/output/round4/long_stream_audit.json) 的 `status=completed` 不等于所有模型通过，实际 `all_candidates_pass=false`。每个候选 16 个案例，覆盖两个输入顺序 × 两种 chunk 序列 × 513/1025/4097/8192 四个前缀，含两个角色头，共 32 个标量概率比较。chunk 序列为 [257,1,7,127,31,512] 与 [1024,32,8,1]。门槛保持原定 **最大绝对概率误差严格小于 0.03**。

| 候选 | 8K 审计最大绝对概率误差 | 原门槛结果 |
|---|---:|---|
| full SFT | 0.002535104751586914 | 通过 |
| window SFT | 0.0030629634857177734 | 通过 |
| memory SFT | **0.03256136178970337** | **失败** |
| 固定候选 RL/full | 0.002354443073272705 | 通过 |

memory 仅有一项超限：`input_order=forward`、`schedule=[1024,32,8,1]`、`prefix=8192`、`role=assistant`；同案例 user 误差 0.01387256383895874 通过。不能以其他 31 项通过、bounded-state 通过或整个审计完成为由覆盖这次失败。

[inference_cli_audit.json](results/output/round4/inference_cli_audit.json) 记录两例均通过、generated_tokens=0。重新读取四份 CLI JSONL，确认相同候选 SHA、分块新增 token 累计、最终两头概率及审计误差全部一致：

| CLI 案例 | token 数 | 流式调用数 | user 头整段/分块误差 | assistant 头误差 |
|---|---:|---:|---:|---:|
| user | 20 | 3 | 0.0002766847610473633 | 0.00010366551578044891 |
| assistant | 50 | 7 | 0.00004696287214756012 | 0.000011274591088294983 |

CLI metadata 的校准报告 SHA 与 classification_rl/exported_dev_metrics 一致。该 CLI 冷启动时间包含初始化/编译开销，不作为稳态 ITPS 结论。任意文本增量的 BPE 回滚、真实神经缓存隔离、同场 A0 速度及 CUDA Graph 实验属于另外的最终 L20 验证；此主训练验收不预先宣称它们通过。

## 产物身份与使用边界

实际远端 checkpoint 位置为 `/work/output/round4/classification_rl/best.safetensors`，PVC 为 `safety-guard-base-compare-data`。本地 manifest 与冻结 validation tar 内同名文件字节完全一致。关键审计输入 SHA-256：

| 文件 | SHA-256 |
|---|---|
| MODEL_MANIFEST.json | `ac55fc33463f1bf1c0be87cbac2419ff07a4c40d431e02f06505ffcfc6e90bcd` |
| validation_package_v1.tar.gz | `2625a539908be823073826283a536d402bfaa091375e2652edd30cf2a6a6dfa3` |
| classification_rl/sealed_test_predictions.jsonl | `81d68bbcdc754dda16ac55dce0b83185ac535a5dd8d4ce794274c99b9e5d2a72` |
| full/sealed_test_predictions.jsonl | `b9c3b0c7f985dbe641246f1205cc7a7e7e8d62065333172e35440900fd910fbc` |
| a0_sealed_test_predictions.jsonl | `1d64dc9d10b19284681f1196e52d78bdd2e35945bcda6ea19b8988682d0149ab` |
| long_stream_audit.json | `364933f5428a03de43bdc968fb6c494905803d187e1e7373544b24630995873d` |
| inference_cli_audit.json | `65386adf5893a97d26a909e35e7e234d1779751cbdb6607d12e0937769d6cf34` |

模型仍是研究候选，manifest 中 `production_approval=false`。本轮硬监督为 safe/unsafe；虽风险头保留 safe/unsafe/controversial 三个输出，controversial 未获得本轮独立硬标签验证，category 头也未验证。已校准分层只有英文助手、中文助手和中文用户，不能把结果外推为已验证英文用户或所有风险细类。全量安全造数质量、外部基准覆盖和最终运行时性能仍须按各自验收产物单独报告。

