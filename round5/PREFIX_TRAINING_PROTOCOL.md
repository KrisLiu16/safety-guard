# Round 5：前缀专项后训练协议

状态：设计与 CPU 数据审视，未运行模型、训练或集群，未联网造数。保留原 v12 生成格式、原始产物与“直接分类、生成 token=0”的输出目标。本协议冻结实验边界，不保证训练一定优于零更新基线。

## 缺陷、参考与范围

真实缺陷是官方 thinking 适配观测中，两连续 Unsafe argmax 的学生召回为 98.24%，同时 FPR 为 90.41%、strict F1 低于 A0。[现有 risk_logits](../round4/train_risk.py:53) 仅取末 token，SFT 与分类 RL 都使用 endpoint CE/动作；缓存数值一致性通过不等于每个中间位置已被监督或校准。

主代理已核对 [Qwen 技术报告 §4](https://arxiv.org/html/2510.14276v1#S4)：query 使用末 token CE，assistant 使用每 token CE；unsafe onset 由 rollout 与 judge 确认。项目尚无可直接复用的对应自然前缀金标集。可以参考监督位置，不能把整段标签伪装成相同质量的 onset 标注，也不做教师表示/措辞相似度对齐。

限定一个 window 起点、一次 SFT（最多 2 epoch）和一次分类 RL（256 更新）。固定起点 SHA：

bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2

完整 24 层（18 GDN + 6 attention）、W512、无 LM head。没有删层/MoE/窗口课程、没有多 seed 追分或新生成任务。Round 4 的 RL/full 质量候选与 manifest 不变，Round 5 独立记录候选。

## CPU 数据审视

| 划分 | 原始记录 | family 数 | safe / unsafe | token 中位数 / 最大值 |
|---|---:|---:|---:|---:|
| train | 33311 | 18243 | 18743 / 14568 | 45 / 3960 |
| calibration | 900 | 782 | 450 / 450 | 79 / 2833 |
| dev | 1200 | 1002 | 600 / 600 | 84 / 1005 |

公开训练参考 10000 safe + 10000 unsafe；v12 弱标签 4096 + 4096；rubric 4096 safe；context 弱标签 551 + 472。四种 rubric 各 1024 条。15689 条 assistant 记录全部有先前 user 消息；last role 均匹配 target_role，无空目标。记录数不是独立 family 数，多前缀也不能成为新的独立统计分母。

冻结输入继续使用 round4/data/risk_v2，SHA 分别为：

- train：9925114b148ee38df3e193a1e427e38b7dd8c6805c1bf28a8b67da868c7c5aa7
- calibration：a39f9a961ddf99ee8a01ab8e148be1749a4d6a6ddca36024712fb7615e0f2663
- dev：7c3ca2d4e99e4070ae9ec602de6c5a4beeb41114373c6c96ceb98f36734d02c9

保留 33311 条原始 endpoint 记录，只增加独立下游 sidecar/view，不改 v12。v12 evidence_text 字符串对齐不等于独立确认了最早 unsafe onset，本轮不用它制造 hard prefix 标签。

## 标签规则

对象是“完整前文作为条件下，最后目标角色已可见的内容”，不是预测未来意图。先前危险 user prompt 只能作为上下文，不能被当成 safe assistant 的负样本位置。

| 位置 | 标签与 confidence | 用途 |
|---|---|---|
| 原始完整 endpoint | 原 source_label、原来源权重 | 始终保留 endpoint CE |
| source-safe 未完成目标内容 | 弱 safe，0.1 | SFT，单列弱标签 loss |
| rubric-safe 未完成目标内容 | 同样弱 safe，0.1 | 本轮没有逐前缀模板证明，不升为 hard |
| source-unsafe 未完成内容 | unknown，mask | 不给 safe/unsafe 标签，不进 RL reward |
| 完整原目标后追加中性 suffix 的位置 | 继承原标签，1.0×原来源权重 | SFT 明确锚点、RL 可用位置 |
| 目标前文、其他角色、header、padding | mask | 只提供条件，不承受当前目标 CE |

原来源权重仍是公开 1.0、v12 0.2、rubric 0.8、context 0.4。suffix confidence=1 不把 v12 弱标签提升成人工金标。

最终 safe 不能证明所有 prefix 都 safe：引用说明可能尚未出现、否定或转折可能在句末。0.1 是可错的继承假设，不是自然 prefix gold；明确歧义可以 mask，规则须训练前冻结，不能依据已看官方/旧 sealed 失败例逐条定制。原 unsafe 的未知中途位置也绝不能全设 unsafe 或全设 safe。

## 中性 suffix 与边界

每个 record 每 epoch 50% 增强，选择仅由稳定哈希/seed 决定，与标签无关；epoch 翻转，使两轮内每条恰一次增强。safe/unsafe 共用冻结中性 suffix 池、概率和长度档，报告语言×标签×来源增强分布；中英文 marker 都不能成为某个标签专属形式。不得出现“以上虚构/撤回/仅供反驳”等改写原语义的后置说明，避免 suffix 本身成为风险捷径。

suffix 在同一最后目标消息内，原文逐字保留。已完整看到 unsafe 目标后，中性记事文字不撤销此前可见风险，因此可以继承原标签；safe 做相同操作。这是受控的证据保持增强，不是自然早期 onset 金标。

原始 endpoint **始终在原始 view 单独 forward**；增强 view 只贡献 prefix。所有 augmented IDs 重新序列化/分词，按字符边界证明完整原目标已可见，不硬套旧末 token 索引。多个 UTF-8 byte token 可能共享同一字符 end offset；仅 offset.end 到结尾仍不充分，还须验证 decode(augmented_ids[:end], skip_special_tokens=False).startswith(original_serialized_text)。截断、overflowing、超长、非法 ID 必须显式拒绝，不静默切短。

sidecar 至少记录 base_sample_id/base_family/base_split、source_label/source_weight、target_role、original_ids_sha256、target_content_start/end_char；view 记录 view_id/augmentation_kind/ids/ids_sha256；anchor 记录 token_end_exclusive/label/confidence/label_origin。监督 tensor 下标是 token_end_exclusive−1。

字符坐标为 Python Unicode code point，不是 UTF-8 字节。角色边界从消息结构累计，不能搜索内容中的 USER:/重复句子猜边界。序列化继续为 ROLE:\ncontent，消息间 \n\n。使用实际部署的 HF tokenizer 初始化和版本锁；原 JSON 哈希不能替代 added tokens/backend/逐输入 IDs 验证。

一遍因果 forward 的中间 hidden state 只代表该 native token 前缀。若要称为某个真实字符切点，必须验证“前缀重编码 IDs == 完整 IDs 的对应前缀”；不同则独立 view forward 或 mask。BPE 合并时禁止复用错位 hidden state。

## SFT loss 与预算

每个原 record 来源权重 w_i、原 endpoint CE 为 E_i、当次有效 prefix anchors 数 K_i、confidence c_ij：

    L_i = w_i × [E_i + 0.5 × (Σ_j c_ij CE_ij / K_i)]

K_i=0 时 prefix 项为 0。分母是 anchor 数 K_i，**不是 Σc_ij**，否则自然 safe 的 0.1 会被归一化回 1。先在同一原 record 内对所有 view/位置求均值，再按原 record 求 batch 均值；不能展开前缀后把长样本/增强样本当成多条独立训练记录。source weight 只乘一次。

有效 batch=16、microbatch=4、最多 2 epoch；按原分组对应最多 4164 次更新。W512 始终固定。主代理已冻结沿用原 backbone LR 8e−6、risk/projection heads LR 5e−5、原 AdamW/裁剪框架，减少本轮方法变量；类别头不训练，第三类不添加伪造标签。导出 BF16 权重后按实际导出文件评估，不能混用 FP32 训练阈值。

SFT 只评 0 / epoch1 / epoch2。记录原 record/来源/anchor 类型计数、有效权重总和、原始与增强 token 数、loss、SHA。数值失败可修复明确的实现错误后从同固定状态恢复，不以失败为理由新增 seed、epoch 或超参数搜索。

## 唯一一次分类 RL

固定 256 更新、每步 8 条原 record、microbatch 4、LR 5e−7；来源仅公开/rubric/v12，原权重保留，排除 context 弱增强。自然 safe 内部的 0.1 弱标签不进 hard reward，只能用完整 endpoint 和完整目标后的 suffix 可信锚点。无文本生成、无 teacher-similarity reward。

主代理已固定本轮采用**逐可信锚点分类 RL**，保留原 reward：正确 +1、FP −3、FN −4，避免同一轮另加事件 reward 变量。每可信锚点采样 4 个 Bernoulli 分类动作；只在原 record 内平均锚点，再对 record 平均并乘原来源权重，不能把锚点当新样本。未知 early-unsafe 位置不计 reward，也不惩罚其“延迟”。

设 p 为 unsafe 动作概率，r0/r1 为 nonrisk/risk 的 rubric reward，精确期望 baseline=(1−p)r0+p*r1，在 REINFORCE 中 detach。保留固定 SFT policy 的 KL=0.1 和每个可信锚点的 CE=0.2（包含原 endpoint）；参考策略用于限制漂移，不是正确性教师。记录逐动作/位置/标签来源/reward/baseline，而非只记录均值。

这不是 episode/停止策略 reward。stream-level 仅通过下面的校准和开发门槛处理；采样动作正确率不是部署事件 FPR。事件 reward 或精确期望 reward 可作为未来独立实验，本轮不实现、不事后切换。

RL 只评 0/64/128/192/256。若 SFT 无可行点，可按开跑前声明从原 window 起点做这唯一一次 RL；不会为追分另开第二次实验。

## 流式校准接口

每条原始 cal/dev record 输出：

- endpoint_p_unsafe：原完整输入末位置分数。
- native_target_max_p_unsafe：全因果输入中最后目标内容全部 native 位置的最大分数。
- text_cut_max_p_unsafe：预声明真实文本切点重分词后的末位置分数最大值。
- stream_max_p_unsafe：上述两个 prefix max 的最大值。

真实文本每条最多 8 个 cut：1、2、floor(L/4)、floor(L/2)、floor(3L/4)、L−2、L−1、L；clamp 到 1..L，去重排序。完整前文不变，只截最后目标内容，逐 cut 重新序列化/encode。空目标单独记录，不能编造 prefix 标签。

同一 record 所有 native/text cuts 先取 max，再计一次事件，不能以 token/cut 数稀释 FPR。当前 API 每 append 只返回末位置；因此策略 v1 用 max-any-risk，不用尚未提供真实逐 token 状态的“两连判”。官方 two-consecutive argmax 单列回归观察。

对同一 token-ID 轨迹，chunk ends 是所有 native 前缀的子集，因此 all-native max 对其保守。真实字符流 BPE 重分词不一定属于该集合；8 cuts 只支撑**预声明轨迹的经验 FPR**，不能声称任意切分/服务到达方式的 SLA。family 聚类相关性和长度/来源变化也限制分布外保证。

## 阈值、选择与隔离

whole/stream 各自只用 calibration safe 分数校准，分别保存，不共用旧 endpoint 阈值。当前 round5 metrics 约定明确的 strict “score > threshold”，按每条记录分数选择经验 FPR≤0.05 的最小合法阈值，ties 与 score=1 必须按同一比较符处理。τ=1 可表示完全不触发，其召回代价由门槛暴露。若服务仍用旧 >=，必须统一运算符或验证兼容转换后才能部署，不能复制阈值数值就称等价。保存逐条校准分数、阈值、实际 FP 数与比较符。

dev 选择固定为：

1. 各已校准 language/role 的 stream FPR≤0.05。
2. 宏 whole recall ≥同协议初始 window 基线宏 whole recall−0.02。
3. 可行候选中宏 stream recall 最高；平分选更早，零更新平分保留零更新。
4. 无可行候选就记录失败，不放宽门槛、改成本或追跑。

whole retention 是宏门槛，stream FPR 是逐分层门槛。公开支持仍仅 en/assistant、zh/assistant、zh/user，不虚构 en/user 公开校准。逐 record 经验指标与 family 数分开报告，不从相关前缀伪造独立样本置信度。

unsafe 的 stream max 只能证明“在整段目标中曾触发”，不能证明最早 onset 或伤害发生前拦截。source-safe 事件误报是操作指标，不代表所有未完成 prefix 都有自然 gold。单独报 endpoint AP/F1、stream recall/FPR、来源/长度分层，不概括为全部能力提升。

只有冻结 train 可梯度更新；cal/dev 只做规定用途。一个 base family 的原始/suffix/cuts 保持同划分，不按 prefix 行随机切分。官方、已看 Round4 sealed、关键词诊断及其失败输出不得进入训练/校准/选择。可使用既有 opaque ID/内容哈希排除清单做污染检查，不把测试反例改写后入训；若冲突，准备阶段显式失败/隔离并报告监督数，不能为凑33311隐藏重叠。精确去重不等于语义无污染。

主代理最终冻结 fresh 公共 holdout **390 条 / 390 唯一 family**：3 个已支持分层×safe/unsafe各65，来自已有 public_pools 的 test 池。[manifest](data/fresh_holdout_v1/manifest.json) 保存历史项目/官方数据的精确排除范围与预检历史；episodes SHA 为 f571f2a93c0c5352a7835e7b522dffd48139063eee3daa229efea409c6a56cd1。原拟600条和中间64/cell贪心预检因可用 family 及跨标签冲突失败，没有把这些失败当作已冻结集合；最终以纯数据可用性上界及增广路分配得到最大均衡65/cell（英文两个cell共131个family为瓶颈）。规模在本轮模型未运行前确定，没有按模型得分改采样；不再追加抽样追分。

fresh holdout 不进入训练、阈值或选择，只在最终候选冻结后评一次。官方/旧 sealed 同样只做一次回归观察，不称新的盲测，也不按结果改选。fresh 是相对列出的项目划分未曝光，仍不能排除基础预训练或未完整追溯的旧实验暴露；不为追分再抽另一批 holdout。

## 替代方案与交付门槛

**更稳健的零训练对照是保留 window 权重，仅做 stream max 校准。** 它直接处理 argmax 中途判决未校准的问题；若开发门槛达成且不劣于训练点，就保留零更新结果。不能预先保证只校准能保持足够召回。

已知二元分类标签时，精确期望 reward 或加权 CE 比采样 RL 更易复算、方差更低；本次 RL 是有界验证而非必需的提升路径。真正补足自然 earliest-onset 能力需要独立人工/可靠 judge 前缀标注，不能靠未知位置强标替代。

导出新 checkpoint 后重跑同 SHA 的 8K、BPE 收缩/快照、事务、跨会话与有界状态检查；新模型对象加载后重新捕获 CUDA Graph，不复用绑定旧权重地址的图。报告净输入 ITPS、P50/P95、forward/replay、session 状态和共享图启动成本，不报生成 TPS。

仍不能声称自然逐 token 金标充分、任意切分安全 SLA、未来意图识别、原官方 token 定位/延迟复现、第三类/细类独立校准、英文用户公开泛化或生产准入。
