# T004 线性探针 + T005 决策规则重拟合（Run A）

两件事共用一次 L20 前向，都不训练模型：固定第五轮权重 `bb16a3a6…`，走和 [decision_rule](../decision_rule/) 相同的 canonical32 eager 路径。

**T004 要回答的问题**：冻结的特征能不能把“复述有害请求的安全推理”（Run A 槽位 0 的 risk_reasoning 风格）和“真正有害的内容”（起点之后的位置）分开？
- 线性探针明显好于现在的头：只重训读出层或头就能压低前缀误报，成本低。
- 探针也分不开：说明主干特征里没有这个区分，要靠新数据训练主干。

**T005**：在 Run A 校准集（100–300 token 的回答）上重新拟合温度和阈值，替代 decision_rule v1 用的约 50 token 的 prefix_v2 校准集。

## 位置标签

按 token 结束位置相对于助手回答开头的偏移（`probe_common.py`）：

| 标记 | 含义 | 探针中的用法 |
|---|---|---|
| S | 安全回答的任意位置（v14 要求安全回答从任何位置截断都安全） | 负例 |
| P | 不安全回答中，结束于 `onset_char` 之前 | 负例 |
| O | 落在起点引用内部 | 丢弃 |
| U | 结束于 `onset_end_char` 之后 | 正例 |

T002 已经发现 v14 起点位置不准，所以报告会把起点附近的 P/U 单独列出，不把它们当成可靠的真值。

## 文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `make_probe_input.py` | Mac | 从 `trainable.jsonl` 取全部 dev、calibration 词和 1,500 个 train 词（每词 4 条回答） |
| `dump_runA_l20.py` | L20 | 前向；导出每个位置的三类 log 概率；用 hook 在每条记录最多 24 个位置取头的输入（1024 维）和投影（512 维）特征。hook 的特征必须能重算出运行时给出的概率，否则第一条记录就报错退出 |
| `fit_probe.py` | L20（也能在 CPU 上跑） | 用 train 拟合逻辑回归探针，在 calibration 上选 L2，只在 dev 上报告；与头自身的 p(unsafe) 对比，带词族自助采样的置信区间 |
| `analyze_runA_rules.py` | Mac | T005：温度和阈值只在 Run A 校准集上拟合，在 Run A dev 上报告；安全流误报按槽位和风格拆开；不安全流统计首次触发落在起点之前的比例；prefix_v2 dev 和官方 thinking 集只做观察 |
| `worker_v1.yaml` | 集群 | L20 作业：等输入就绪，先 dump 再拟合探针，写完成标记后等待回收 |
| `test_probe_cpu.py` | Mac | 8 项单测；其中 2 项用假引擎把 dump 主循环整个跑一遍，需要 torch |

结果只提交数字文件（`report.json`、`probe_v1_results.json`、`runA_rules_v1_results.json`）。`*.jsonl.gz` 和特征文件 `*.npz` 留在 PVC 或 Mac 上。

## T004 结果与 T014（迁移检验）

T004（[反馈](../../collab/feedback/T004-probe-and-rules.md)）：冻结特征上的线性探针区分“复述有害请求的安全推理”和“起点之后的真有害文本”，Run A dev 上 AUC 0.995 [0.994, 0.996]；现在的头只有 0.791 [0.777, 0.806]。中英两种语言都成立。T005：只换决策规则，复述类推理的流式误报仍有 11–31%。结论是区分信息已经在主干里，问题出在读出头。

但探针只在 Run A 上拟合和检验过，而 Run A 全部是 luna 用 v14 模板写的，探针可能学的是生成风格，而不是危害本身。T014 用同一个探针（在作业里按 T004 的特征和 L2 重新拟合，必须复现 T004 的校准 AUC）给以下三组数据的每个位置打分，并与现在的头对比：

- Run A calibration 和 dev；
- prefix_v2 calibration 和 dev（只取 assistant 分层）；
- 官方 1,872 条序列（只观察，不参与拟合）。

温度和阈值只在 Run A 校准集上拟合。

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `transfer_l20.py` | L20 | 重新拟合并核对探针；一次前向，逐位置输出头的 log 概率和探针 logit，自检方式与 T004 相同 |
| `analyze_transfer.py` | Mac | 头和探针走同一套温度、规则和 bootstrap 代码（探针 logit 写成三元组 `[0, logit, -50]`，softmax 恰好等于 sigmoid）；输出三组数据上的流式误报和召回，以及不依赖阈值的流式 AUC |
| `worker_transfer_v1.yaml` | 集群 | L20 作业 |
| `test_transfer_cpu.py` | Mac | 3 项单测：重新拟合能否复现、三个数据源的主循环（假引擎，需要 torch）、分析脚本端到端 |
