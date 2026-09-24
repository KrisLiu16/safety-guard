# Round5 端到端结果

生成时间：2026-09-23T20:13:06.096221+00:00

这是 CPU 文件报告；本脚本不加载 tokenizer/模型，不执行训练、评测或云操作。缺失字段标为 pending；`status=completed`、进程退出 0、`integrity_pass` 和权重下载均不等于安全质量通过。running/failed 产物中保留的数值只能视为局部观察，不能称为完成的基准结果。

当前交付状态：**waiting_for_training**。训练总报告：pending：尚无可读产物。canonical32 质量门槛：**pending / 未提供**。

**训练最终选择尚未完成/取回，后续结果 pending；不能宣称已训练完成或已可交付。**

最近本地状态快照（不是完成证明）：2026-09-23T20:12:32.914169+00:00，Pod=Running；SFT step=2048/4164；RL update=pending / 未提供。

## 训练、固定选择与 fallback

起点是既有 Qwen3.5 window/W512 分类器，不是从零预训练。训练先 SFT，再分类动作 RL；whole 为完整输入末点，stream 为该记录预定 native target tokens 和至多八个真实文本切点的最大风险。

固定初始权重 SHA：`bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`。最终训练选择 SHA：`pending / 未提供`。

MODEL_MANIFEST：status=pending / 未提供，candidate=pending / 未提供；training_promoted_from_initial=pending / 未提供，promoted_from_initial=pending / 未提供。

是否实际保留初始权重：pending / 未提供。选择说明：pending / 未提供。初始 fallback 或 serving gate 失败不算训练提升。

| 阶段 | 状态 | 完成步数 | 选中 stage/step | eligible | 选中权重 SHA |
| --- | --- | --- | --- | --- | --- |
| SFT | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| classification RL | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |

训练原 bulk dev（用于训练过程固定选择；不能替代下面的 canonical32 dev）：

| 权重 | 模式 | macro recall | macro FPR |
| --- | --- | --- | --- |
| initial | whole | pending / 未提供 | pending / 未提供 |
| initial | stream | pending / 未提供 | pending / 未提供 |
| SFT best | whole | pending / 未提供 | pending / 未提供 |
| SFT best | stream | pending / 未提供 | pending / 未提供 |
| RL selected | whole | pending / 未提供 | pending / 未提供 |
| RL selected | stream | pending / 未提供 | pending / 未提供 |

## canonical32 校准与独立 dev gate

pending：尚无可读产物

阈值只在 900 条 calibration 上拟合；1200 条 dev 检查：每个 stream stratum FPR≤5%，whole macro recall 比 canonical 初始基线最多下降 2 个百分点。whole/stream 各有独立阈值，严格 `>`；canonical gate 不重新选权重，失败仍须保留原记录。

initial_window：gate=pending / 未提供；gates=pending / 未提供；cal/dev records=pending / 未提供/pending / 未提供；calibrated_strata=pending / 未提供。

selected：gate=pending / 未提供；gates=pending / 未提供；cal/dev records=pending / 未提供/pending / 未提供；calibrated_strata=pending / 未提供。

| 权重 | 模式 | stratum | n | safe | unsafe | recall | FPR | TP | FP | 固定阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_window | whole | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | whole | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | whole | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |

## Fresh 390：冻结的留出评估

pending：尚无可读产物

预定覆盖 390 records / 390 recorded families；三个语言/角色层，每层 safe/unsafe 各 65。以下为各权重自己的固定 calibration 阈值，不在 fresh 上重拟合、不用 fresh 选权重。

initial_window：pending：尚无可读产物；coverage=pending / 未提供；observations_reused_from=pending / 未提供。

selected：pending：尚无可读产物；coverage=pending / 未提供；observations_reused_from=pending / 未提供。

| 权重 | 模式 | stratum | n | safe | unsafe | recall | FPR | TP | FP | 固定阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_window | whole | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | whole | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | whole | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| initial_window | stream | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | whole | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | zh/user | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | zh/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| selected | stream | en/assistant | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |

selected − initial（描述性差值）：pending / 未提供。

旧 bulk fresh 对照单独保留：pending：尚无可读产物。不把旧 bulk 的 threshold/概率当成 canonical32 结果。

## 官方 adapted benchmark：2441 来源行

pending：尚无可读产物

计划保留 thinking 1059、thinking_loc 569、response_loc 813 行；同一权重中只对完全相同的 native IDs 与 eval_start_index 复用预测：1872 条 unique forward sequences + 569 cache hits = 2441 来源行。重复行仍计分，不删行伪装更高覆盖；thinking 与 thinking_loc 重叠，禁止合并为总分。

决策为原 unsafe 优先的“连续两个 argmax 类别”规则；strict 只把 unsafe 算阳性，loose 也把 controversial 算阳性。它不使用 calibration 阈值。location splits 全是 Unsafe，FPR 未定义；原 Qwen3 token 定位下标不能直接用于 Qwen3.5，定位指标不报。

实际来源行=pending / 未提供；实际执行计数=pending / 未提供。

| split | 规则 | selected n | selected recall | selected FPR | A0 n | A0 recall | A0 FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thinking | native_two_consecutive_strict | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| thinking | native_two_consecutive_loose | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| thinking | endpoint_argmax_unsafe | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| thinking_loc | native_two_consecutive_strict | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| thinking_loc | native_two_consecutive_loose | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| thinking_loc | endpoint_argmax_unsafe | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| response_loc | native_two_consecutive_strict | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| response_loc | native_two_consecutive_loose | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| response_loc | endpoint_argmax_unsafe | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |

A0 是历史原始 Qwen3Guard-Stream 结果，不冒充本轮重新推理：reference SHA=pending / 未提供，本轮 A0 model calls=pending / 未提供。tokenizer/template/粒度不同，所以是 adapted 比较。

## canonical core 与真实文本速度

pending：尚无可读产物

全32位置双头/状态=pending / 未提供；future pad 因果性=pending / 未提供；到达顺序不变=pending / 未提供；不提交 dummy cache=pending / 未提供。

padding case 数=pending / 未提供；跨 aligned 边界 BPE=pending / 未提供；608-token 实际 dispatch=pending / 未提供。

ITPS 分子是测量期净新增原生 tokens（末长度−初长度）；重放及未来 padding 不能冒充新增输入，但其计算时间保留在分母。128 次真实文本 append，先完整预热同一 schedule；耗时含重分词、缓存复制/回滚、分类输出到 CPU 和同步，排除启动捕获。

| engine | ITPS | P50 ms | P95 ms | 净新增 | physical | real含重放 | padding | replay | wall秒 | 复核 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |
| window_cuda_graph | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 | pending / 未提供 |

eager 启动：seconds=pending / 未提供，capture=pending / 未提供，auxiliary tokens=pending / 未提供，arena bytes=pending / 未提供。
graph 启动：seconds=pending / 未提供，capture=pending / 未提供，auxiliary tokens=pending / 未提供，arena bytes=pending / 未提供。

core 是固定 checkpoint 的计算一致性验收，不是安全质量评估；旧 bulk 仅诊断。graph core 通过也不会自动把 eager 便携包宣称为 graph 交付。原 Round4 fast audit 的失败不被本报告覆盖。

## 便携包、工作流与本地交付

workflow：pending：尚无可读产物；portable：pending：尚无可读产物。

独立 child：pending：尚无可读产物；无需原基座权重=pending / 未提供；无需 H24=pending / 未提供；canonical loader 最大概率差=pending / 未提供。

child 隔离=pending / 未提供；HF offline=pending / 未提供；参数结构精确匹配=pending / 未提供；RoPE dtype/SHA 匹配=pending / 未提供；未知依赖拒绝记录=pending / 未提供；en/user 阈值未伪造=pending / 未提供。

包 checkpoint SHA=`pending / 未提供`；权重 bytes=pending / 未提供；包 inference_engine=pending / 未提供；包 graph_validation_passed=pending / 未提供。

本地交付记录路径：pending / 未提供；download attempts=pending / 未提供；artifact job completed=pending / 未提供。

本地 release 清单尚未出现：pending；不可声称模型已经下载。

## 固定数据与结论限制

- Fresh 390 是 390 个已记录 family；准备/审计保留 26 个短消息文本重合。长度≥20 字符的归一化精确文本排除规则，不等于所有长度零重复或语义去污染。
- en/user 缺少独立 calibration，不借用 zh/user 或 en/assistant 阈值，不报告其已校准覆盖。
- 第三风险类别 controversial、细分类别 category、自然风险最早出现位置及 safe-prefix 放行均未验收。
- 预定 native/text-cut 轨迹上的 episode FPR，不构成任意 BPE 到达 schedule 的线上 FPR 保证。
- 官方与 fresh 都是描述性评估，不用于改权重/选 checkpoint/调阈值；生产批准始终不由本报告推导。

## 证据完整性与来源

发现的文件一致性/计账问题：0。以下检查只处理已有文件，不补写不存在的结果。
已有字段未发现跨文件身份或速度计账冲突；缺失证据仍为 pending，不能据此通过总验收。

缺失/尚未取回：`prefix_v2/summary.json`、`prefix_v2/sft/summary.json`、`prefix_v2/classification_rl/summary.json`、`prefix_v2/sft/evaluation_0/metrics.json`、`validation/workflow_status.json`、`validation/MODEL_MANIFEST.json`、`validation/bundle/BUNDLE_MANIFEST.json`、`validation/canonical_core_audit.json`、`validation/bundle_audit.json`、`canonical32_calibration/audit.json`、`canonical32_calibration/initial_window/calibration.json`、`canonical32_calibration/selected/calibration.json`、`canonical32_fresh390/audit.json`、`canonical32_fresh390/initial_window/metrics.json`、`canonical32_fresh390/selected/metrics.json`、`canonical32_official/metrics.json`、`fresh390_final/audit.json`


| 证据 | 实际读取路径 | SHA256 |
| --- | --- | --- |
| delivery_status.json | /Users/liuzhihao/Work/safety-guard/round5/delivery_status.json | 7411d011d557dc1eee12f4fe8c42401252864157d0555f1089c50996567916cf |
| LIVE_STATUS.json | /Users/liuzhihao/Work/safety-guard/round5/LIVE_STATUS.json | f8394e1193b556f2999e3d415bae5152ddb80b248ca87ee823e79c9ee15cccbc |
