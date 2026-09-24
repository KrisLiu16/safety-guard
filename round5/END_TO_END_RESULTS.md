# Round5 端到端结果

生成时间：2026-09-24T02:10:15.831108+00:00

这是 CPU 文件报告；本脚本不加载 tokenizer/模型，不执行训练、评测或云操作。缺失字段标为 pending；`status=completed`、进程退出 0、`integrity_pass` 和权重下载均不等于安全质量通过。running/failed 产物中保留的数值只能视为局部观察，不能称为完成的基准结果。

当前交付状态：**bundle_delivered**。训练总报告：completed（仅进程/产物状态）。canonical32 质量门槛：**false**。

最近本地状态快照（不是完成证明）：2026-09-23T20:15:38.634534+00:00，Pod=Running；SFT step=2082/4164；RL update=pending / 未提供。

## 训练、固定选择与 fallback

起点是既有 Qwen3.5 window/W512 分类器，不是从零预训练。训练先 SFT，再分类动作 RL；whole 为完整输入末点，stream 为该记录预定 native target tokens 和至多八个真实文本切点的最大风险。

固定初始权重 SHA：`bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`。最终训练选择 SHA：`bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`。

MODEL_MANIFEST：status=research_unvalidated_runtime，candidate=initial_window_fallback；training_promoted_from_initial=false，promoted_from_initial=false。

是否实际保留初始权重：true。选择说明：Research classifier with diagnostic thresholds; serving gates failed。初始 fallback 或 serving gate 失败不算训练提升。

| 阶段 | 状态 | 完成步数 | 选中 stage/step | eligible | 选中权重 SHA |
| --- | --- | --- | --- | --- | --- |
| SFT | completed | 4164 | {"checkpoint_sha256": "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2", "source_checkpoint": "/work/output/round4/window/best.safetensors", "stage": "initial", "step": 0} | false | bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2 |
| classification RL | completed | 256 | {"checkpoint_sha256": "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2", "source_checkpoint": "/work/output/round5/prefix_v2/sft/best.safetensors", "stage": "sft_selected", "step": 0} | false | bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2 |

训练原 bulk dev（用于训练过程固定选择；不能替代下面的 canonical32 dev）：

| 权重 | 模式 | macro recall | macro FPR |
| --- | --- | --- | --- |
| initial | whole | 70.17% | 6.83% |
| initial | stream | 65.17% | 7.50% |
| SFT best | whole | 70.17% | 6.83% |
| SFT best | stream | 65.17% | 7.50% |
| RL selected | whole | 70.17% | 6.83% |
| RL selected | stream | 65.17% | 7.50% |

## canonical32 校准与独立 dev gate

completed；integrity_pass=true

阈值只在 900 条 calibration 上拟合；1200 条 dev 检查：每个 stream stratum FPR≤5%，whole macro recall 比 canonical 初始基线最多下降 2 个百分点。whole/stream 各有独立阈值，严格 `>`；canonical gate 不重新选权重，失败仍须保留原记录。

initial_window：gate=false；gates={"each_stream_stratum_fpr_at_most_5pct": false, "whole_macro_recall_drop_at_most_2pp": true}；cal/dev records=900/1200；calibrated_strata={"stream": ["en/assistant", "zh/assistant", "zh/user"], "whole": ["en/assistant", "zh/assistant", "zh/user"]}。

selected：gate=false；gates={"each_stream_stratum_fpr_at_most_5pct": false, "whole_macro_recall_drop_at_most_2pp": true}；cal/dev records=900/1200；calibrated_strata={"stream": ["en/assistant", "zh/assistant", "zh/user"], "whole": ["en/assistant", "zh/assistant", "zh/user"]}。

| 权重 | 模式 | stratum | n | safe | unsafe | recall | FPR | TP | FP | 固定阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_window | whole | zh/user | 400 | 200 | 200 | 64.00% | 8.50% | 128 | 17 | 0.990597 |
| initial_window | whole | zh/assistant | 400 | 200 | 200 | 79.00% | 7.00% | 158 | 14 | 0.974964 |
| initial_window | whole | en/assistant | 400 | 200 | 200 | 67.50% | 5.00% | 135 | 10 | 0.95063 |
| initial_window | stream | zh/user | 400 | 200 | 200 | 61.00% | 9.50% | 122 | 19 | 0.996167 |
| initial_window | stream | zh/assistant | 400 | 200 | 200 | 80.00% | 6.50% | 160 | 13 | 0.993289 |
| initial_window | stream | en/assistant | 400 | 200 | 200 | 54.50% | 6.50% | 109 | 13 | 0.995532 |
| selected | whole | zh/user | 400 | 200 | 200 | 64.00% | 8.50% | 128 | 17 | 0.990597 |
| selected | whole | zh/assistant | 400 | 200 | 200 | 79.00% | 7.00% | 158 | 14 | 0.974964 |
| selected | whole | en/assistant | 400 | 200 | 200 | 67.50% | 5.00% | 135 | 10 | 0.95063 |
| selected | stream | zh/user | 400 | 200 | 200 | 61.00% | 9.50% | 122 | 19 | 0.996167 |
| selected | stream | zh/assistant | 400 | 200 | 200 | 80.00% | 6.50% | 160 | 13 | 0.993289 |
| selected | stream | en/assistant | 400 | 200 | 200 | 54.50% | 6.50% | 109 | 13 | 0.995532 |

## Fresh 390：冻结的留出评估

completed；integrity_pass=true

预定覆盖 390 records / 390 recorded families；三个语言/角色层，每层 safe/unsafe 各 65。以下为各权重自己的固定 calibration 阈值，不在 fresh 上重拟合、不用 fresh 选权重。

initial_window：completed（仅进程/产物状态）；coverage={"evaluated": 390, "excluded": {}, "requested": 390, "unique_families": 390}；observations_reused_from=未复用（null）。

selected：completed（仅进程/产物状态）；coverage={"evaluated": 390, "excluded": {}, "requested": 390, "unique_families": 390}；observations_reused_from=initial_window。

| 权重 | 模式 | stratum | n | safe | unsafe | recall | FPR | TP | FP | 固定阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_window | whole | zh/user | 130 | 65 | 65 | 81.54% | 9.23% | 53 | 6 | 0.990597 |
| initial_window | whole | zh/assistant | 130 | 65 | 65 | 81.54% | 6.15% | 53 | 4 | 0.974964 |
| initial_window | whole | en/assistant | 130 | 65 | 65 | 55.38% | 4.62% | 36 | 3 | 0.95063 |
| initial_window | stream | zh/user | 130 | 65 | 65 | 80.00% | 6.15% | 52 | 4 | 0.996167 |
| initial_window | stream | zh/assistant | 130 | 65 | 65 | 78.46% | 13.85% | 51 | 9 | 0.993289 |
| initial_window | stream | en/assistant | 130 | 65 | 65 | 44.62% | 7.69% | 29 | 5 | 0.995532 |
| selected | whole | zh/user | 130 | 65 | 65 | 81.54% | 9.23% | 53 | 6 | 0.990597 |
| selected | whole | zh/assistant | 130 | 65 | 65 | 81.54% | 6.15% | 53 | 4 | 0.974964 |
| selected | whole | en/assistant | 130 | 65 | 65 | 55.38% | 4.62% | 36 | 3 | 0.95063 |
| selected | stream | zh/user | 130 | 65 | 65 | 80.00% | 6.15% | 52 | 4 | 0.996167 |
| selected | stream | zh/assistant | 130 | 65 | 65 | 78.46% | 13.85% | 51 | 9 | 0.993289 |
| selected | stream | en/assistant | 130 | 65 | 65 | 44.62% | 7.69% | 29 | 5 | 0.995532 |

selected − initial（描述性差值）：{"stream": {"macro_f1": 0.0, "macro_fpr": 0.0, "macro_recall": 0.0}, "whole": {"macro_f1": 0.0, "macro_fpr": 0.0, "macro_recall": 0.0}}。

旧 bulk fresh 对照单独保留：completed；integrity_pass=true。不把旧 bulk 的 threshold/概率当成 canonical32 结果。

## 官方 adapted benchmark：2441 来源行

completed；integrity_pass=true

计划保留 thinking 1059、thinking_loc 569、response_loc 813 行；同一权重中只对完全相同的 native IDs 与 eval_start_index 复用预测：1872 条 unique forward sequences + 569 cache hits = 2441 来源行。重复行仍计分，不删行伪装更高覆盖；thinking 与 thinking_loc 重叠，禁止合并为总分。

决策为原 unsafe 优先的“连续两个 argmax 类别”规则；strict 只把 unsafe 算阳性，loose 也把 controversial 算阳性。它不使用 calibration 阈值。location splits 全是 Unsafe，FPR 未定义；原 Qwen3 token 定位下标不能直接用于 Qwen3.5，定位指标不报。

实际来源行=2441；实际执行计数={"eager_block_calls": 34430, "exact_prediction_cache_hits": 569, "graph_block_calls": 0, "padding_tokens": 28758, "physical_block_forward_calls": 34430, "physical_forward_tokens": 1101760, "real_forward_tokens": 1073002, "unique_forward_sequences": 1872}。

| split | 规则 | selected n | selected recall | selected FPR | A0 n | A0 recall | A0 FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thinking | native_two_consecutive_strict | 1059 | 98.42% | 89.80% | 1059 | 81.55% | 22.24% |
| thinking | native_two_consecutive_loose | 1059 | 98.42% | 89.80% | 1059 | 82.07% | 22.24% |
| thinking | endpoint_argmax_unsafe | 1059 | 66.78% | 5.10% | 1059 | 60.28% | 1.02% |
| thinking_loc | native_two_consecutive_strict | 569 | 98.42% | undefined（无 safe） | 569 | 81.55% | undefined（无 safe） |
| thinking_loc | native_two_consecutive_loose | 569 | 98.42% | undefined（无 safe） | 569 | 82.07% | undefined（无 safe） |
| thinking_loc | endpoint_argmax_unsafe | 569 | 66.78% | undefined（无 safe） | 569 | 60.28% | undefined（无 safe） |
| response_loc | native_two_consecutive_strict | 813 | 95.45% | undefined（无 safe） | 813 | 89.18% | undefined（无 safe） |
| response_loc | native_two_consecutive_loose | 813 | 95.45% | undefined（无 safe） | 813 | 90.53% | undefined（无 safe） |
| response_loc | endpoint_argmax_unsafe | 813 | 86.35% | undefined（无 safe） | 813 | 87.70% | undefined（无 safe） |

A0 是历史原始 Qwen3Guard-Stream 结果，不冒充本轮重新推理：reference SHA=9ff0b35279092bd89337e0c4920ffad027e62deaeb0a8f56b37c2ff760e06b65，本轮 A0 model calls=0。tokenizer/template/粒度不同，所以是 adapted 比较。

## canonical core 与真实文本速度

completed；pass=true

全32位置双头/状态=true；future pad 因果性=true；到达顺序不变=true；不提交 dummy cache=true。

padding case 数=62；跨 aligned 边界 BPE=[{"common_prefix_tokens": 703, "native_tokens": 704, "old_aligned_cache_tokens": 704, "old_snapshot_positions": [640, 672], "previous_tokens": 705, "restore_position": 672}, {"common_prefix_tokens": 703, "native_tokens": 704, "old_aligned_cache_tokens": 704, "old_snapshot_positions": [640, 672], "previous_tokens": 705, "restore_position": 672}]；608-token 实际 dispatch={"eager": {"attempted_calls": 19, "attempted_forward_tokens": 608, "eager_calls": 19, "eager_forward_tokens": 608, "exported_full_caches": 19, "failed_calls": 0, "forward_calls": 19, "forward_tokens": 608, "graph_calls": 0, "graph_forward_tokens": 0, "padding_tokens": 0, "partial_calls": 0, "phase": "ready", "real_forward_tokens": 608}, "graph": {"attempted_calls": 19, "attempted_forward_tokens": 608, "eager_calls": 16, "eager_forward_tokens": 512, "exported_full_caches": 19, "failed_calls": 0, "forward_calls": 19, "forward_tokens": 608, "graph_calls": 3, "graph_forward_tokens": 96, "padding_tokens": 0, "partial_calls": 0, "phase": "ready", "real_forward_tokens": 608}}。

ITPS 分子是测量期净新增原生 tokens（末长度−初长度）；重放及未来 padding 不能冒充新增输入，但其计算时间保留在分母。128 次真实文本 append，先完整预热同一 schedule；耗时含重分词、缓存复制/回滚、分类输出到 CPU 和同步，排除启动捕获。

| engine | ITPS | P50 ms | P95 ms | 净新增 | physical | real含重放 | padding | replay | wall秒 | 复核 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager | 229.592 | 27.8661 | 52.3585 | 960 | 4928 | 2944 | 1984 | 1984 | 4.18133 | CPU计账一致 |
| window_cuda_graph | 706.331 | 9.43908 | 16.0214 | 960 | 4928 | 2944 | 1984 | 1984 | 1.35914 | CPU计账一致 |

eager 启动：seconds=2.99886e-07，capture=0，auxiliary tokens=0，arena bytes=0。
graph 启动：seconds=7.22228，capture=0.284039，auxiliary tokens=640，arena bytes=26038272。

core 是固定 checkpoint 的计算一致性验收，不是安全质量评估；旧 bulk 仅诊断。graph core 通过也不会自动把 eager 便携包宣称为 graph 交付。原 Round4 fast audit 的失败不被本报告覆盖。

## 便携包、工作流与本地交付

workflow：completed；pipeline_integrity_pass=true；portable：completed；pass=true。

独立 child：completed；pass=true；无需原基座权重=true；无需 H24=true；canonical loader 最大概率差=0。

child 隔离=true；HF offline=true；参数结构精确匹配=true；RoPE dtype/SHA 匹配=true；未知依赖拒绝记录=[]；en/user 阈值未伪造=true。

包 checkpoint SHA=`bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`；权重 bytes=1509079036；包 inference_engine=eager；包 graph_validation_passed=false。

本地交付记录路径：/Users/liuzhihao/Work/safety-guard/round5/release；download attempts=[{"attempt": 1, "returncode": 1}, {"attempt": 2, "returncode": 0}]；artifact job completed=true。

本地清单实际存在：[/Users/liuzhihao/Work/safety-guard/round5/release/BUNDLE_MANIFEST.json](/Users/liuzhihao/Work/safety-guard/round5/release/BUNDLE_MANIFEST.json)，SHA=`f5f9c343fcd3a9696f1206ecfb9676cee29bd6319bfe81eaaad916ddb44aa213`。本地权重存在=true；实际 bytes=1509079036。本报告不重新读取大权重做 SHA；完整权重 SHA 验证依据 downloader/portable 产物，不能仅凭目录存在证明可用。

| 阶段 | status | process completed | returncode | artifact quality gate |
| --- | --- | --- | --- | --- |
| canonical_core | completed | true | 0 | pending / 未提供 |
| canonical_calibration | completed | true | 0 | false |
| prepare_manifest | completed | true | 0 | false |
| package | completed | true | 0 | false |
| portable_bundle | completed | true | 0 | pending / 未提供 |
| canonical_fresh390 | completed | true | 0 | pending / 未提供 |
| canonical_official2441 | completed | true | 0 | false |
| bulk_reference_fresh390 | completed | true | 0 | pending / 未提供 |


## 固定数据与结论限制

- Fresh 390 是 390 个已记录 family；准备/审计保留 26 个短消息文本重合。长度≥20 字符的归一化精确文本排除规则，不等于所有长度零重复或语义去污染。
- en/user 缺少独立 calibration，不借用 zh/user 或 en/assistant 阈值，不报告其已校准覆盖。
- 第三风险类别 controversial、细分类别 category、自然风险最早出现位置及 safe-prefix 放行均未验收。
- 预定 native/text-cut 轨迹上的 episode FPR，不构成任意 BPE 到达 schedule 的线上 FPR 保证。
- 官方与 fresh 都是描述性评估，不用于改权重/选 checkpoint/调阈值；生产批准始终不由本报告推导。

## 证据完整性与来源

发现的文件一致性/计账问题：0。以下检查只处理已有文件，不补写不存在的结果。
已有字段未发现跨文件身份或速度计账冲突；缺失证据仍为 pending，不能据此通过总验收。

缺失/尚未取回：无预定 JSON 缺失。


| 证据 | 实际读取路径 | SHA256 |
| --- | --- | --- |
| delivery_status.json | /Users/liuzhihao/Work/safety-guard/round5/delivery_status.json | 79f458fca08cf94a28892330b1ce133e7f18017c2188ea3267c16c2cb4270596 |
| LIVE_STATUS.json | /Users/liuzhihao/Work/safety-guard/round5/LIVE_STATUS.json | 74a4a0667e52027fb462a2449e6edb60b9489df9a54b5ae2ead474c01b21f333 |
| prefix_v2/summary.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/prefix_v2/summary.json | 255fe4246c84041fbcadfb4e280c8dcfd1da7de6d0c283b4ed7c19b32d7fbd00 |
| prefix_v2/sft/summary.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/prefix_v2/sft/summary.json | ae004fd941f5f50a8bf4f85af4390c9286795fe065c9a9a5965ecdfc49450bae |
| prefix_v2/classification_rl/summary.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/prefix_v2/classification_rl/summary.json | a8e348c00d12677a2b97e65bd74c43b1413e272e93976c925fea72b0e2b512b2 |
| prefix_v2/sft/evaluation_0/metrics.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/prefix_v2/sft/evaluation_0/metrics.json | d9d0513e481ea76d84784e1df78ea1e2bb8ffb77abfc5d9220ac0d0ef13b8738 |
| validation/workflow_status.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/validation/workflow_status.json | 34e671fbe74c16022c58732612bf5e03b125073bc23bcc6fbd629ab35b70eff4 |
| validation/MODEL_MANIFEST.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/validation/MODEL_MANIFEST.json | 0f9003aeb923789993eb38f13edf99de6c8807532e0606d273fba1337cfd06d2 |
| validation/bundle/BUNDLE_MANIFEST.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/validation/bundle/BUNDLE_MANIFEST.json | f5f9c343fcd3a9696f1206ecfb9676cee29bd6319bfe81eaaad916ddb44aa213 |
| validation/canonical_core_audit.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/validation/canonical_core_audit.json | 8952b5fe499fee5a9ba8884c10c69cb36e3f13e30c15029e388e5dc693fc1cc7 |
| validation/bundle_audit.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/validation/bundle_audit.json | 760393629762fcded857142bbd3e4a405a4973509d8b1a7fa647c42976c6ebfc |
| canonical32_calibration/audit.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_calibration/audit.json | 15cb854d8e90b5624d60fcf875f5c19ba0406c26f7bc0f1e5abe5be6c3ad6a52 |
| canonical32_calibration/initial_window/calibration.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_calibration/initial_window/calibration.json | 22ca4ab6e007a9bc23a827ef450800745a3b60f13180b04bb5d6d072d90a5604 |
| canonical32_calibration/selected/calibration.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_calibration/selected/calibration.json | 9b68c3d0e1b1d301b0688c367160d5c10ad7a53d27c30c412742f1cf082189cb |
| canonical32_fresh390/audit.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_fresh390/audit.json | 0004cbb6b2c7f033d66b639143b42aea45755932f9dcf12fe01053d00a86c8a3 |
| canonical32_fresh390/initial_window/metrics.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_fresh390/initial_window/metrics.json | 10a7197f269f2ed1b8f085346a122a3bb43ad456339834a0f9eed29aaaa0417f |
| canonical32_fresh390/selected/metrics.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_fresh390/selected/metrics.json | eb0c3c9d3a5f4d87e1ef85cdc5c575ed8a4db7fde90a8cd6d292efb2b2d0d012 |
| canonical32_official/metrics.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/canonical32_official/metrics.json | 8986f0baebc4bad5f85b83de5ef1887c71d1f90ffb4960dc241ba1b4b3589ca8 |
| fresh390_final/audit.json | /Users/liuzhihao/Work/safety-guard/round5/results/output/round5/fresh390_final/audit.json | c7452156dd3113dcbe3a1b6e33cec63e9376dd08ac9308fa815c9d7602169e2f |
