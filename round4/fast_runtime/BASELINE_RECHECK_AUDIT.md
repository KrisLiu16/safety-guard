# Baseline recheck：独立 CPU 审计

2026-09-24；只读取已有 JSON/JSONL 与测量源码，独立重算全部记录；未运行模型、访问集群、改动权重/阈值或原失败产物。以下是记录一致性审计，不是模型质量/交付总验收。

## v2 速度配对

输入是 `results/output/fast_runtime/baseline_recheck/speed_candidate_v2.json` 和 `speed_a0_v2.json`。**candidate 是 primary RL/full，不是 window graph，也不是 canonical32。** 权重 SHA 为 `29fc3e5663eaa333235b1c2ba84d4c0a964f6ece4fd146c9fc6ec01761eaf222`；A0 为原始 Qwen3Guard-Stream-0.6B，权重 SHA `e0a2eac6cc79cca5bf35bf8fb356f94c333dd9715f4f0dd58883b01f2fe33419`。

核对通过：

- 两份均为 `matched-classifier-token-id-v2`，协议字段完全一致：batch/session 各 1，user role，起始上下文 512/4096，chunk 1/8/32，六个不同组合各 128 次计时；每组独立缓存先完整预热同一条 128 次轨迹。
- 两份 CUDA 属性与单设备 `nvidia-smi` 记录均对应 `GPU-7e45a669-5e2c-ae71-e610-e91a4018b3b8` / NVIDIA L20，`identity_verified=true`，CPU threads=4。
- 两份 `candidate_manifest_sha256` 都等于固定 primary `round4/MODEL_MANIFEST.json` 的 `ac55fc33463f1bf1c0be87cbac2419ff07a4c40d431e02f06505ffcfc6e90bcd`；candidate 的 variant/权重 SHA 与该清单一致。测量源码 SHA 都等于本地 `round4/benchmark_pair.py` 的 `4a567e20a12bd1f454d849a6994a147419f600837b6154ed678b35fe1dac116c`。
- 按已保存 phrase/repetitions 重建同一 `USER:\n` 原文，确为 202,758 字符，SHA `358c998826b19ffaf57b50397c1ce0c2bf04ed915e9f584a1399686292db316c`。各自 tokenizer 得到的完整原生 token 数记录为 45,059 / 47,106；测量各取前 8,192 个原生 token。两模型的 token IDs/hash 不相同，不能把相同 token 数理解为完全相同文本跨度。速度产物保存了 IDs 的 hash，没有完整数组，本轮没有重跑 tokenizer 复算这些 IDs。
- 全部 12 组各 128 个原始 latency 都为有限正数；从原始数组重算 p50/p95 与产物逐值一致（nearest-rank，p95 为排序第 122 个）。`new_input_tokens=128×chunk`；`ITPS=new_input_tokens/elapsed_seconds`、每秒分类次数 `128/elapsed_seconds` 均逐值一致。延迟之和比外层 elapsed 少 0.883–1.085 ms/组，符合测量源码额外循环/同步开销；不能用 latency 之和替换已声明的总时间分母。
- 所有结束位置为起始位置加真实新输入数，范围 640–8,192；warmup 与 timed trace 都不超过 A0 配置上限 8,192。每个计时调用的三分类概率有限、归一，`generated_tokens=0`。

| 起始 tokens | chunk | RL/full ITPS | A0 ITPS | RL/A0 | RL p95 ms | A0 p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 54.972 | 64.600 | 0.851× | 18.349 | 15.643 |
| 512 | 8 | 401.679 | 431.568 | 0.931× | 20.059 | 19.211 |
| 512 | 32 | 1,577.194 | 1,698.005 | 0.929× | 20.533 | 18.992 |
| 4096 | 1 | 54.889 | 63.815 | 0.860× | 18.433 | 16.153 |
| 4096 | 8 | 403.495 | 432.161 | 0.934× | 19.967 | 19.044 |
| 4096 | 32 | 1,573.324 | 1,541.489 | 1.021× | 20.503 | 24.484 |

这不是全场景加速结论：RL/full 五组较慢，只有最后一组略快。计时包含 forward、现有分类 readout、user 风险概率复制到 CPU 和同步/循环，排除模型加载、prefill、tokenizer、文本回滚、HTTP、JSON。A0 执行双角色 risk/category 四个头，candidate 执行 user risk/category 两个头；Transformers/tokenizers 版本分别为 4.55.0/0.21.4 与 5.17.0/0.23.2。因此比较的是各自现有 runtime，不能隔离架构或分类头贡献；单次运行也没有置信区间。没有混用 v1 的超长失败轨迹、旧 24 次数字或 window graph 的 2.98× 数字。

## 关键词逐例重算

输入为 `baseline_recheck/keyword_primary_{predictions.jsonl,metrics.json}` 与同级 `keyword_window_{predictions.jsonl,metrics.json}`。两者分别对应上述 RL/full SHA 与 window SFT SHA `bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`。

每份 1,024 个唯一 sample_id 与冻结 `round4/data/keyword_probe_v1/probe.jsonl` 一一对应；模板、来源、词族、语言、角色、标签及原生 token 数均匹配，最长 100 tokens。四模板各 256 条，覆盖 29 个来源组。逐例用固定同语言/角色阈值执行 `p_unsafe >= threshold`，独立重算全部 `all_rows` 和 `by.template/stratum/source_group` 字段，与产物完全一致。

| 模板 | 样本 | primary 固定阈值结果 | window 固定阈值结果 | 未校准 argmax 诊断 |
|---|---:|---|---|---|
| zh_user_safe_script | 256 safe | FP 0/256，FPR 0% | FP 0/256，FPR 0% | primary FP 2/256；window 1/256 |
| zh_user_unsafe_fraud | 256 unsafe | TP 256/256，recall 100% | TP 256/256，recall 100% | 两者 TP 256/256 |
| en_user_unsafe_privacy | 256 unsafe | **未校准，recall=null** | **未校准，recall=null** | 两者 TP 256/256；不能代替阈值 recall |
| en_assistant_safe_refusal | 256 safe | FP 0/256，FPR 0% | FP 0/256，FPR 0% | 两者 FP 0/256 |

两份各 768 条被固定阈值评分，256 条 en/user 的 `existing_calibration_threshold` 和 `calibrated_unsafe` 全为 null，状态为 `missing_stratum_unscored`，没有借用语言或角色阈值。总表的 calibrated recall=100% 只覆盖 256 条 zh/user unsafe，不覆盖全部 512 条 unsafe。单标签模板不定义另一侧 FPR/recall；没有 ROC-AUC。没有概率恰好等于阈值的边界样本。

固定校准元数据核对与缺口：

- primary/window 的校准文件 SHA 分别为 `d22bea42da262e818892784618255f1ecd580c9f82ca2f160031dd5036add098` / `5bda62b0e7090b0def38faebf77d8235833d3a4aaaab9cd00750f63afe871847`，均匹配本地对应 `exported_dev_metrics.json`；逐个阈值同时匹配相应 MODEL_MANIFEST。`thresholds_tuned_on_probe=false`。
- **仍缺校准文件自身的 checkpoint/运行代码/序列化绑定。** 两份 `exported_dev_metrics.json` 没有这些身份字段；evaluator 接收外部校准文件，产物诚实标为 `calibration_checkpoint_association_verified=false`。本次只能证明结果使用了记录的校准文件、该文件阈值与固定清单一致，不能把它升级为校准报告自行证明来自同一权重的证据。
- 冻结 probe manifest SHA 为 `c1b58b122e09dc4d9e9a45c1d9ba8b02dff2741f3eae648672d15a64350f3df9`，其全部输出 hash 匹配；两份 tokenizer 证明均绑定 `keyword_tokenizer_compatibility_v1.json` SHA `bffaa24cc31237158b770b1a2781b007045353a09ab2d48dbd950eebc594e470`。逐行重算 1,024 个冻结 IDs hash，与证明中的 reference/runtime hash 相同；记录实际推理前又校验了全部 1,024 行。该证明仅覆盖这批确切输入，**不证明两个 tokenizer 全局等价**。

四个固定模板属于完整输入的受控诊断，不是自然语义安全金标、广泛敏感领域覆盖、前缀/思考文本评测，也没有 zh/assistant 或 en/user safe 模板。原 fast text audit 的 `pass=false` 以及后续 whole/cache 漂移反例保持不变；本次高分不能替代它们，第三风险类别和 category 仍未验证。

## 输入文件 SHA256

| 文件（位于 results/output/fast_runtime） | SHA256 |
|---|---|
| baseline_recheck/speed_candidate_v2.json | `2d2c468195ee807aa0c104f1c0a0e8320ea44c940eaa4d8d12a10add97cec817` |
| baseline_recheck/speed_a0_v2.json | `af22f74ab7b0e3b291ca72a55b5ae41b56f0bef85e1986f5ed0dcbb06d74c04b` |
| baseline_recheck/keyword_primary_metrics.json | `afa82df03d2b469254a5765c81d429b4840a557fa243390af78c0a44c4388a50` |
| baseline_recheck/keyword_primary_predictions.jsonl | `16ee2ca541cc94be6aab221ebb0466b5b7c7c2ed6d46ffcae4ed1414bf6dfbab` |
| keyword_window_metrics.json | `a39058c13459717ec7e8c48af2cff997fd95bc32aac8a0ab8875d75b79079813` |
| keyword_window_predictions.jsonl | `5f8bb3789228dd78100c003adcd53bc77ddc98db9de1ab667132027cb22db200` |
