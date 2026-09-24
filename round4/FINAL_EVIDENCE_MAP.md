# 最终验收证据映射

**状态：待实际结果。** 本表仅按六个指定文件的当前实现审查验收路径，不表示任何尚未产生的模型结果已通过。未运行模型、未访问集群；`COMPLETION_AUDIT.md` 的手填进度不是当前运行状态证据。

路径约定：

- `R` = `/Users/liuzhihao/Work/safety-guard/round4`
- `T` = `R/results/output/round4`
- `V` = `R/validation_results/output/round4_validation`
- `B` = 最终下载后解包的模型根目录；对应远端 `/work/output/round4_validation/bundle`。实际验收必须记录本地绝对路径，不能只验远端。
- `C` = `R/MODEL_MANIFEST.json` 的 `candidate`，`H` = 其 `checkpoint_sha256`。全部候选证据按 `C/H` 联结，不能借用另一个分支的通过结果。
- `<data-dir>`、`<source-dir>`、`<calibration-metrics>` 是验证任务真实命令参数；评估脚本没有给它们固定默认路径。必须从实际命令定位，不能假定某个目录就是本次输入。

## 必须实际读取的文件和字段

| 验收项 | 精确文件与字段 | 足够证据的条件 |
|---|---|---|
| 固定最终候选 | `R/MODEL_MANIFEST.json`：`status`, `candidate`, `variant`, `checkpoint`, `checkpoint_sha256`, `window`, `thresholds`, `calibrated_strata`, `selection_read_sealed_test` | `status=research_candidate`；实际权重 SHA 为 `H`；开发集选模来源及长流门槛成立。它只标识研究候选，不等于质量或生产准入通过。后续所有检查须匹配 `C/H`。 |
| 候选实际文本流 | `V/text_stream_audit.json`：`results[C].checkpoint_sha256`, `.variant`, `.pass`, `.all_text_parity_pass`, `.max_probability_error`, `.tolerance`, `.cases[]` | SHA 为 `H`；架构与 manifest 一致；`pass=true`、所有实际误差有限且 `<0.03`。检查 cases 包含 `contraction_192_to_191`、`assistant_role`、`next_user_role`、`long_prefill`、`long_variable_append`、`retry_after_failure`、`benchmark_end`。确认真实 BPE 收缩、>1024-token 文本、跨角色均被执行。 |
| 文本会话独立、失败可恢复 | 同文件：`results[C].cache_isolation`, `.overflow_transaction_pass`, `.failed_forward_transaction_pass`, `.bounded_session_state_pass`, `.session_state_bytes_before`, `.session_state_bytes_after` | 两个事务字段为 true，隔离检查存在且未抛异常；window/memory 必须有界且前后字节数相同，full 的有界字段为 null 不算失败、也不能宣称内存有界。隔离结构细节需结合运行时检查实现解释，不从一个总 `pass` 猜测。 |
| 新增输入 ITPS 与延迟 | 同文件：`results[C].text_stream_performance.{accepted_appends,initial_native_tokens,net_new_native_tokens,actual_forward_tokens,replay_tokens,rollback_events,seconds,itps_net_native_input,p50_ms,p95_ms,scope}` | 当前脚本应完成 128 次预热后追加；核算 `ITPS=net_new_native_tokens/seconds`，时间和结果均有限、正值。回放不能计为新输入；记录 P95 和状态大小。该结果不含 HTTP、等待输入和服务端批处理，不能与纯 token-ID 内核测速混称。 |
| 独立便携加载 | `V/bundle_audit.json`：`status`, `pass`, `candidate`, `variant`, `checkpoint_sha256`, `child_returncode`, `portable_child.pass`, `.isolated_python_process`, `.hf_offline`, `.device` | 父/子均通过，返回码为 0，`C/H` 一致，隔离新进程为 true、设备为 L20。还必须读取子项，不以“独立加载”文字字段代替检查。 |
| 无旧权重/代码依赖 | 同文件：`portable_child.dependency_audit.{forbidden_roots,denied_probes,unexpected_denials,network_connections_blocked,scope}`、`.runtime_module_sources`；旁证 `V/bundle_audit.child.json`、`.child.log` | 四个禁止根都完成拦截探测；实际 `unexpected_denials=[]`、网络连接数为 0；运行模块来源均在 `B` 的远端对应目录。这是 Python audit 事件、离线配置及已核验加载器共同提供的证据，不声称是阻挡任意原生系统调用的 OS 沙箱。 |
| 便携模型与原权重真正相同 | `V/bundle_audit.json`：`portable_child.{parameter_structure_exact_match,rope_buffer_dtype_and_sha_exact_match,structure,loader_cases,max_old_loader_probability_error,whole_probability_tolerance}`；`V/bundle_audit.reference.json`：`selection`, `structure`, `cases[].canonical_token_ids`, `.risk_probabilities_by_role`, `.native_tokens` | 参数名称/形状/dtype 指纹一致，两份 RoPE buffer 的 dtype/SHA 完全一致；6 个样本齐全（中英 user/assistant、两个 >512 长输入），token IDs 相同、双头概率均有限且最大差 `<=0.005`。参考文件 SHA 要等于父报告 `reference_sha256`。不能只看总参数量或成功 load_state_dict。 |
| 包入口的 whole/文本追加/结束 | `V/bundle_audit.json`：`portable_child.jsonl.{pass,whole_cases,stream_cases,real_bpe_rollbacks,max_stream_probability_error,stream_probability_tolerance,cross_role_pass,session_release_pass,generated_tokens}`；`stream_cases[]` 的 `target_role,native_tokens,previous_native_tokens,net_new_tokens,forward_tokens,replay_tokens,rollback,threshold,target_binary_decision` | 子入口通过；真实回滚数 >=1，whole/stream 最大概率差有限且 `<=0.03`；跨角色及结束会话通过；每例生成 token 为 0，净增量等于新旧长度差，BPE 收缩允许负净增量。不能只用预先切好的 token-ID 流证明文本输入可用。 |
| Qwen3GuardTest 三片完整覆盖 | `V/official_student/metrics.json`、`V/official_a0/metrics.json`：`status`, `kind`, `checkpoint_sha256`, `splits[thinking/thinking_loc/response_loc].coverage.{requested,evaluated,excluded}`；各目录三个 `<split>_predictions.jsonl` 的 `sample_id,split,row_index,unique_id,label,status,reason`；真实 `<data-dir>/<split>.jsonl`、`<source-dir>/<split>.jsonl` | 学生 SHA 为 `H`；逐片记录数、身份、标签与冻结全量输入一致；三片请求数合计 2441，不能拿一个子集的 `evaluated=requested` 当全量。要声称全部评估，应逐片 `evaluated=requested` 且 `excluded={}`；否则明确缺口和原因。A0 的 metrics 中 checkpoint SHA 本来为 null，须用已冻结 A0 权重/运行来源补足身份，不能称该字段完成了 A0 SHA 核验。 |
| 完整基准能力数字与范围 | 同 metrics：各 split 的 `native_two_consecutive_strict`, `native_two_consecutive_loose`, `endpoint_argmax_unsafe` 下 `n,tp,fp,tn,fn,recall,fpr,precision,f1`；`location_metrics`, `selection_uses_this_benchmark`, `protocol`, `overlap_note`；逐例 `decision,endpoint_decision,endpoint_probabilities,input_tokens,eval_start_index` | 由逐例预测复算混淆矩阵、指标和分母；每片指标 `n` 等于实际评估数，概率有限。学生/参考使用各自模板和 token 边界，属于适配观测；`location_metrics=null`，不得宣称复现原定位/128-token 延迟。thinking 与 thinking_loc 重叠，不汇总成独立总分；两个 loc 无 safe 样本，FPR 为 null。 |
| 固定关键词诊断与阈值出处 | `V/keyword_candidate_metrics.json`、`V/keyword_a0_metrics.json`：`model`, `all_rows`, `by`, `probe_manifest_sha256`, `calibration_metrics_path`, `calibration_metrics_sha256`, `calibration_checkpoint_association_verified`, `existing_thresholds`, `thresholds_tuned_on_probe`；对应 `keyword_<kind>_predictions.jsonl` | 候选 `model.checkpoint_sha256=H`；1024 个 sample ID、256 个词族与冻结 probe 逐项对上，所有概率有限；probe manifest、实际校准文件 SHA 对上报告。**脚本明确写 `calibration_checkpoint_association_verified=false`**，必须另把候选校准文件联结到 `T/C/exported_dev_metrics.json` 的实际 SHA/strata，并核对 thresholds 与 MODEL_MANIFEST；A0 必须使用自己的既有校准来源。完成评估不自动证明阈值属于该权重。 |
| 缺失 en/user 校准不得造数 | 上述 keyword metrics：`missing_calibration_strata`, `by.stratum["en/user"].{n,calibrated_scored,calibrated_unscored,calibrated_recall,calibrated_fpr}`；逐例 `existing_calibration_threshold,calibrated_unsafe,calibration_status`；bundle 的 `portable_child.jsonl.{english_user_threshold_absent,english_user_missing_threshold_not_fabricated}` | 当前候选缺失 en/user 时，必须列出该缺口；该组 calibrated_scored=0、unscored=n，阈值/校准判决/校准召回为空，状态为 `missing_stratum_unscored`。包的英文 user whole/stream 样本也应为 null 判决和 `stratum_not_calibrated`。允许另列明确标注为未校准的 argmax 指标，不能借用 en/assistant 或声称完成独立英文用户公开测试。 |
| 下载包及权重完整性 | 实际 `B/MODEL_MANIFEST.json`、`B/BUNDLE_MANIFEST.json` 的 `checkpoint_sha256`, `files[relative_path].{sha256,bytes}`，特别是 `files["classifier.safetensors"].sha256`；实际 `B/classifier.safetensors` 和所有清单文件 | 本地重新计算每个文件的 SHA/字节数，完整权重 SHA 必须为 `H`；包内 MODEL_MANIFEST 与固定选择一致。若交付压缩包，还须记录并核对压缩文件本身的本地/源端 SHA。`bundle_audit.pass` 只验证测试时所在目录，不能证明随后下载、打包和解包无损。 |

## 只能证明流程状态，不能替代验收的字段

补充同场效率证据（新增验证阶段，待实际结果）：`V/benchmark_candidate.json` 与 `V/benchmark_a0.json` 必须各有六组唯一 `(context_tokens, chunk_tokens)`，两边 `protocol`、`protocol_version`、`trace.common_text_sha256`、`candidate_manifest_sha256` 和具体设备 UUID 一致；候选 `model.checkpoint_sha256=H`。逐组检查 `measured_calls=128`、128 条有限正延迟、`new_input_tokens=128×chunk`、`itps=new_input_tokens/elapsed_seconds`、P95 按 `ceil(128×0.95)-1` 复算。以完整 `prefill_cache/end_cache.storage_inventory` 去重字节数核对状态大小，不能漏掉 GDN 或根 memory。A0 的 `model.weights_sha256/config_sha256/modeling_code_sha256` 补足参考模型身份。实际 UUID 缺失、协议不一致、覆盖不全时只列单边结果，不计算提速比例；最长 token-ID 计时 12288 不作为文本 API 或长程风险能力验收。

- `T/final_summary.json.status=completed`、`V/stage_status.json.status=completed`：这里只证明流程标记。主训练步数、数据 SHA、分类 RL 256 次更新、奖励/错误率/导出开发集对照，仍须按 `COMPLETION_AUDIT.md` 检查三份 `T/{full,window,memory}/sft_summary.json`、更新轨迹、`T/classification_rl/summary.json` 和逐例保留测试。当前指定审查范围没有训练器字段定义，不能用渲染器读取的一个 `status` 补足这些门槛。
- `text_stream_audit.status=completed`：实现只检查已收集四个结果，**即使某个分支失败也会 completed**。最终候选必须单独查 `results[C].pass` 和 SHA；其他失败分支照实报告。
- `official_*/metrics.status=completed`：只代表三个 split 写完，**允许有排除记录，甚至没有有效样本**。`coverage.requested` 也只是本次传入数据行数，不自证全量。`neural_forward_calls` 因精确输入复用可小于评估条数，不能当覆盖数。
- `keyword_*_metrics.status=completed`：不证明风险能力提高，也不自动绑定校准与检查点；受控模板的结论不能扩大到真实自然对话分布。
- `bundle_audit.child_returncode=0` 或“包已生成”：不单独证明加载一致性；读取父/子 pass、逐例误差、指纹、阈值及源码来源。
- `render_final_report.py` 是展示器，缺文件时返回 `{}`；它不执行总体验收、不复算覆盖、不绑定所有 SHA，且部分说明文字无条件出现。`R/END_TO_END_RESULTS.md` 已生成、表格看起来完整，均不构成通过证据。只有上述输入门槛成立后，才可用报告作最终陈述。
- 相关 Kubernetes Job 终态、节点剩余 GPU 进程、本地包下载完成度不在这六个文件的 JSON 证明范围内；须另读实际 Job/Pod/节点与文件状态，不能由 `stage_status` 推断资源已释放。

## 最终口径

上述项目当前全部等待实际产物核对。最终结果允许训练完成但某项研究优化没有收益，应保留真实失败与能力/速度数字；不得据此改选测试更好的模型、提高容差或改变成功定义。最终候选的实际文本流或独立完整加载未通过、全量评估缺口未解释、下载包 SHA 未核对时，端到端交付仍不成立。

只把 safe/unsafe 当作本轮有硬标签训练与质量验收的类别。`controversial`、类别头、独立英文用户公开测试、生产批准均不可从三概率输出接口或研究候选状态推导出来。
