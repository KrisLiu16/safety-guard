# 独立控制结果：Graph 保持 baseline，whole/cached 漂移会影响后续分类

**Graph 在本次同轨迹控制中保持 ordinary eager 的计算结果，内部状态和双头概率最大差均为 0。原有 whole/cached 状态差则确实影响未来输出：出现超过原概率容差的偏差及两次目标头 argmax 翻转。原 fast audit 的 `pass=false` 继续保留，未修改模型、阈值或原规格。**

控制由主任务已提交的 L20 Job 串行执行，本子任务只轮询、只读取回文件并做 CPU 分析，没有另启神经工作或干预后续训练。

## 证据身份与覆盖

- Job/Pod：`safety-guard-prefix-round5-20260924-r1` / `safety-guard-prefix-round5-20260924-r1-t9m2k`。
- 控制耗时 36.7032 秒；`status=completed`、`diagnostic_completed=true`。这两个字段表示实验结束，不表示模型通过交付验收。
- 完整原始产物：`prefill_stream_control.original.json`，594,953 bytes，SHA-256 `879a7a224e4680babd4779c533728a955cd5ec7f68cfff22c70ab1a03ceb1228`。远端与本地 SHA 一致，记录在 `control_retrieval.json`。
- 实验脚本 SHA：`09fac65fb3c9c9f48b8628598d3405c12d0689ace496f8d9a573095024f3a798`；原 helper 与运行代码 SHA 也在执行时核对，容差未变。
- 模型固定为 window/SFT `bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`；不是本轮正在继续训练的新权重。
- 前缀长度 512、4096、7680；9 个 1/8/32-token baseline 对照；6 个真实 dev 样本 × 3 个长前缀，共 18 条 continuation、57 个后续分块步骤。目标头非饱和概率实际覆盖 33 步，不只是孤立样本的筛选承诺。

CPU 可重现聚合见 `analyze_control_result.py` 和 `control_analysis.json`；原始 audit `fast_text_audit.original.json` 的 SHA 和 false 状态未改。

## 支持哪些结论

| 问题 | 实际结果 |
|---|---|
| 同一 whole 前缀重复计算是否稳定 | 512/4096/7680 三组内部状态均逐位相同 |
| 不使用 Graph 的 ordinary eager 流式是否也不同于 whole 状态 | 9/9 组重现原状态门槛失败 |
| 同一分块轨迹的 Graph 是否改变 baseline | 9 组控制和 57 步后续中，状态及双头概率最大差均精确为 0 |
| 不同来源历史状态接相同真实后缀，概率是否受影响 | 是；最大差 0.04030824，2 个步骤超过原 `<0.03` 门槛 |
| 原有校准阈值的拦截判决是否在这些步骤翻转 | 没有；计数为 0。这些阈值很高，不能据此推导输出无差异 |
| 默认三类 argmax 是否受影响 | 是；两个实际目标 assistant 头发生 safe/unsafe 翻转 |
| 与从头计算完整最终序列的 fresh whole 是否完全一致 | 否；18 个终点有 1 个概率门槛失败，最大差 0.03515422 |

这支持“CUDA Graph 加速保留了本次已测 ordinary cached 计算”的限定结论。它不支持“whole 与 cached 的状态差无害”或“所有未来后缀的分类相同”。目前仍未把差异精确归因到某一个 FLA/GDN 内核；BF16 投影/attention 的形状差异、chunk/recurrent 的数值路径差异及普通路径中的潜在实现问题仍须区分。

## 具体未来输出差异

两次超过原概率门槛的步骤：

| dev sample ID | 起始/当前 token 位置 | 实际目标 | 最大双头概率差 | 说明 |
|---|---|---|---:|---|
| `a0e28d5cf85a14b977aa7427c9d38cd822a37f8d3a737f3052768cdebb3c5c14` | 520 → 648 | zh/user | 0.03769439 | 目标 unsafe 概率：eager-history 0.40198869，whole-origin 0.36429450 |
| `beaver-6980` | 4104 → 4168 | en/assistant | 0.04030824 | 最大差位于诊断 user 头；实际 assistant 头差为 0.01914102。en/user 没有校准阈值，仍为空 |

两次实际目标头 argmax 翻转：

| dev sample ID | token 位置 | eager-history unsafe | whole-origin unsafe | argmax |
|---|---:|---:|---:|---|
| `beaver-6980` | 4200 | 0.48067313 | 0.50730664 | safe → unsafe |
| `beaver-23677` | 7769 | 0.50125968 | 0.49223751 | unsafe → safe |

两例的既有 en/assistant 校准阈值都是 0.95198792，因此校准判决均未翻转。**概率差即使小于 0.03，也可能在 argmax 边界上改变标签**；不能用有限概率容差或高阈值下零翻转来替代具体协议的分类一致性检查。用户指定公开基准采用连续 argmax 判决，本控制没有重算该基准，不能推断整套 benchmark 的变化幅度。

另有 fresh-whole 终点对照失败：上述中文 user 样本在 4104-token 起始前缀后，目标 user 头最大差 0.03515422。说明差异并非只存在于一次中间状态观测。

## 保留的边界

这些 continuation 是将真实 dev 内容附加到受控长背景上的数值诊断，不是新的自然分布质量金标；样本挑选用于非饱和压力测试，不参与选模型、训练或调阈值。已观察的 Graph 数值等价证据与原有 whole/cached 问题应分别报告。原 fast 总验收失败、正式前缀误报问题、正在执行的新训练均不因本次诊断结束而被改写成成功。
