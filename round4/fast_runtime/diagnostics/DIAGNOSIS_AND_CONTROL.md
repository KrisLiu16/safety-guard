# CUDA Graph 文本流失败：证据与独立控制方案

原始结果仍为 **`pass=false`**，未修改实现、容差、模型权重或原验收规范。这里只做了文件取回与 CPU 聚合，下面的控制脚本尚未运行 L20。

原始文件：`fast_text_audit.original.json`，940,606 bytes，SHA-256 `11635ba5abc48a466363aa75ef98b146e65f8dae0cf9fad19d07b98eb833234c`。取回来源、时间和远端 SHA 核验见 `retrieval.json`。完整 false JSON 路径、比较分组、逐类 dtype/最大绝对差/归一化差见 `failure_summary.json`，可由 `summarize_failed_audit.py` 重现。

## 已证实的失败分布

164 项检查，55 项通过，109 项失败：96 个精确长度案例、2 个 8192-token whole 对照、11 个文本会话案例。每个失败项的失败根都只有 whole 状态比较（`whole_state` 或 `whole_8192` 的 `state`）。没有发现非有限数、缓存拓扑或逻辑位置不一致。

| 比较 | 实际覆盖 | 状态/双头概率最大差 |
|---|---:|---:|
| 相同输入 cache 的 graph 对 eager | 96 | 全部 0 |
| 各自独立但同一分块轨迹的 graph 对 eager | 96 | 全部 0 |
| 文本会话 graph 对 eager | 22 | 全部 0 |
| 文本快照 graph 对 eager | 42 | 全部 0 |
| whole 对流式端点概率 | 108，加 2 个 8192-token 案例 | 全部通过；全局最大差 0.0001046378 |

whole 内部状态差并不小，不能直接称作可以忽略：

| 状态族 | dtype | 最大绝对差 | 最大归一化差 |
|---|---|---:|---:|
| conv_states | BF16 | 0.6171875 | 24.5342 |
| recurrent_states | FP32 | 0.05468178 | 19.1000 |
| attention keys | BF16 | 1.44921875 | 58.6098 |
| attention values | BF16 | 0.521484375 | 22.9381 |

归一化差沿用原检查的 `abs_delta / (atol + rtol * abs(reference))`，大于 1 即失败；未更改 BF16 的 0.02/0.01 或 FP32 的 0.002/0.001。

## 现有性能证据及其限制

同一权重、相同 128 次文本追加轨迹，净新增 384 tokens，两个引擎都实际前向 384 tokens、回放 0 tokens：

| 引擎 | 实际文本 ITPS | P50 ms | P95 ms | 总追加时间 s |
|---|---:|---:|---:|---:|
| eager | 116.5659 | 24.9308 | 27.3547 | 3.29427 |
| graph | 347.8196 | 8.47277 | 8.80424 | 1.10402 |

观察到 ITPS 比率 2.9838877，P95 比率 3.1069955。单次、固定顺序的已预热轨迹，不是并发服务压测；正确性总门槛失败仍未消除，不能据此宣布快包可以交付。

启动 15.4334 s，其中捕获/准备 8.73082 s；辅助输入 2624 tokens，独立于用户计数。32 个共享图 arena 的物理 cache storage 共 833,224,704 bytes（794.625 MiB）；引擎 allocated 增量 1,105,921,024 bytes，reserved 增量 2,489,319,424 bytes。原始报告还有逐图与峰值显存，不把共享引擎内存算成每个会话的大小。

## 根因假说，而非已经证明的结论

原 `test_fast_text_l20.py` 的 `pair_check`/`run_exact_lengths` 在验证“同一缓存、同一分块序列的 graph/eager 等价”之外，还把“一次处理整个前缀生成的全部内部状态≈多个 cached 调用生成的状态”作为同一个总 pass 条件。

冻结 `standalone_model.py` 的 GDN dispatch 明确区分两条计算路径：已存在 initial_state 且新段 <=32 时走 FLA fused recurrent；whole/prefill 走 chunk。输入/中间投影为 BF16；attention/投影的矩阵尺寸也随一次输入长度而变化。即使最终保存的 recurrent state 是 FP32，也不意味着产生该状态的低精度输入和归约顺序相同。

当前最有力的假说是：失败来自原有 eager 的 prefill 与 cached 数值/执行路径差异，CUDA Graph 在已测同轨迹中没有增加误差。这一假说不等于已经定位到某一个 GDN 内核，也不排除两条普通路径中存在共有的实现问题。单看当前概率很小的差，不能保证任意未来后缀都不受内部差影响。

## 独立、可证伪的最小 L20 控制

脚本：`test_prefill_stream_control_l20.py`，SHA-256 `09fac65fb3c9c9f48b8628598d3405c12d0689ace496f8d9a573095024f3a798`。有独立输出、独立进程组、**300 秒硬时限**；只读取冻结包/数据，不覆盖原失败结果，不运行训练，不增加接受阈值。

1. 固定相同 window/SFT 权重，核对原失败 audit、原比较 helper、实际加载代码的 SHA；所有神经调用由 bundle loader 限制在单张 L20。
2. 对 512、4096、7680 长度各做 whole 重算的确定性对照，再追加 1/8/32 tokens，分别计算 ordinary eager、graph 和 fresh whole。原状态/概率门槛全部保留，同时输出每层状态绝对差。关键预测是：ordinary eager 对 whole 也重现状态失败，而同轨迹 graph 对 eager 仍为数值 0。
3. 从已存在 dev 的中文 user、中文 assistant、英文 assistant × safe/unsafe 六组，每组最多评估 16 个按 sample ID 哈希确定的 8..192-token 样本，选择孤立 unsafe 概率最接近 0.5 的一个。仅挑诊断输入，不选模型、不调阈值；全部候选分数和所选真实样本保存在新报告中。
4. 每个长前缀在追加 8 tokens 后保留三套逻辑内容相同的历史状态：E=普通 eager 分块形成、G=graph 分块形成、W=一次 whole 形成。向三者追加完全相同的所选实际 dev 后缀，每段 <=32 tokens；E 与 W 的后续都用 ordinary eager，G 继续 graph。逐步记录双头概率、固定原校准阈值判决是否分歧，最后再与 fresh whole 比较。
5. 明确报告实际 continuation 中是否覆盖非饱和目标概率；没有覆盖就标缺口，不把“孤立样本接近 0.5”当作长上下文中也非饱和。没有 en/user 阈值时仍为空，不使用助手阈值。

运行方式（由主任务在新的 GPU Job 中串行执行；此处没有运行）：

```bash
/work/modern/bin/python -B /work/round4/fast_runtime/diagnostics/test_prefill_stream_control_l20.py \
  --bundle /work/output/fast_runtime/bundle \
  --original-audit /work/output/fast_runtime/fast_text_audit.json \
  --helpers /work/round4/fast_runtime/test_fast_text_l20.py \
  --dev /work/round4/data/dev.jsonl \
  --output /work/output/fast_runtime_diagnostic/prefill_stream_control.json
```

如果相同分块轨迹的 graph/eager 出现差异，或输入缓存/同前缀重算不稳定，就不能把问题仅归于 whole/cached 路径差异。如果 E/W 在非饱和语义后缀下出现超过原概率容差的差或固定阈值判决分歧，需要继续定位原路径的数值/实现行为；不能只提高 state 容差。如果 E/W 原状态失败得到重现，且 G/E 仍完全相同，只能根据这个新证据陈述“加速保持既有 cached 计算”，不能把原 audit 的 false 改写成 true。

即使有限控制后缀无输出分歧，也不能证明所有未来后缀均不敏感。新的质量训练和正式前缀误报问题独立存在，不由该数值诊断或吞吐提升替代。
