# T007 反馈

- 对应任务卡：v2（v1 的结果见 git 历史中本文件的上一版，以及 `round6/serving/results_v4/report.json`）
- 状态：**完成，验收通过**
- 执行时间：2026-09-24（北京时间）19:00 提交，19:00:26 写入 input_ready，19:07 写入 done。19:08 取回后写入 collected。实际用时约 7 分钟。
- 实际执行的命令：与任务卡相同。
  - 单测：`test_bench_v5_cpu.py`、`test_ring_attention_cpu.py`、`test_batched_engine_cpu.py`，共 7 项，全过。
  - Job 为 `safety-guard-serving-bench-v5-r1`，Pod 为 `safety-guard-serving-bench-v5-r1-psjxr`。提交前该节点的 GPU 分配为 0。
  - `round6/serving/*.py` 共 15 个文件，用 `sha256sum` 核对，与本地一致。
  - `serving_v5_exit_code.txt` 为 0，`status=completed`，`training=false`。

## 验收

- `sequences`=400，`positions`=374,953。
- 4 种配置的 `decision_drift` 中，cut 和 unsafe 两组的 `positions` 都等于 374,953。

## 1. `tiles_chosen_fp16`（会话数 → (bv, warps)）

1→(32,2) 2→(32,2) 4→(32,2) 8→(32,4) 16→(32,4) 32→(128,8) 64→(32,2) 128→(32,2) 256→(8,1) 512→(64,4)

## 2. `decision_drift`：逐位置漂移

参考是整段前向。`over_0.05` 和 `over_0.1` 是超过该偏差的位置数，总位置数为 374,953。

| 配置 | 分数 | max | p999 | over_0.05 | over_0.1 |
|---|---|---:|---:|---:|---:|
| v3 | cut | 0.2307 | 0.0363 | 119 | 11 |
| v3 | unsafe | 0.2307 | 0.0363 | 119 | 11 |
| v4_fp32_inplace | cut | 0.1859 | 0.0348 | 125 | 9 |
| v4_fp32_inplace | unsafe | 0.1859 | 0.0348 | 125 | 9 |
| v4_fp16_gather | cut | 0.1657 | 0.0379 | 149 | 13 |
| v4_fp16_gather | unsafe | 0.1658 | 0.0379 | 149 | 13 |
| v4_fp16_inplace_tiles | cut | 0.2911 | 0.0367 | 144 | 15 |
| v4_fp16_inplace_tiles | unsafe | 0.2912 | 0.0367 | 144 | 15 |

## 3. `decision_drift`：判决

- `fire_differs`：流式和参考中只有一方触发截断的序列数。
- `both_fire_other_index`：两边都触发，但触发位置不同的序列数。
- `reference_fires`：参考触发截断的序列数。

每项都是 400 条序列中的计数。

| 配置 | 分数 | 规则 | 阈值 | fire_differs | both_fire_other_index | reference_fires |
|---|---|---|---:|---:|---:|---:|
| v3 | cut | threshold | 0.5 | 0 | 2 | 386 |
| v3 | cut | threshold | 0.9 | 0 | 3 | 373 |
| v3 | cut | threshold | 0.99 | 2 | 4 | 321 |
| v3 | cut | consecutive_2 | 0.5 | 0 | 1 | 384 |
| v3 | cut | consecutive_2 | 0.9 | 0 | 2 | 367 |
| v3 | cut | consecutive_2 | 0.99 | 0 | 5 | 305 |
| v3 | unsafe | threshold | 0.5 | 0 | 3 | 386 |
| v3 | unsafe | threshold | 0.9 | 0 | 3 | 373 |
| v3 | unsafe | threshold | 0.99 | 1 | 6 | 321 |
| v3 | unsafe | consecutive_2 | 0.5 | 0 | 2 | 384 |
| v3 | unsafe | consecutive_2 | 0.9 | 0 | 2 | 367 |
| v3 | unsafe | consecutive_2 | 0.99 | 1 | 7 | 305 |
| v4_fp32_inplace | cut | threshold | 0.5 | 0 | 2 | 386 |
| v4_fp32_inplace | cut | threshold | 0.9 | 0 | 4 | 373 |
| v4_fp32_inplace | cut | threshold | 0.99 | 0 | 5 | 321 |
| v4_fp32_inplace | cut | consecutive_2 | 0.5 | 0 | 2 | 384 |
| v4_fp32_inplace | cut | consecutive_2 | 0.9 | 0 | 3 | 367 |
| v4_fp32_inplace | cut | consecutive_2 | 0.99 | 1 | 5 | 305 |
| v4_fp32_inplace | unsafe | threshold | 0.5 | 0 | 2 | 386 |
| v4_fp32_inplace | unsafe | threshold | 0.9 | 0 | 4 | 373 |
| v4_fp32_inplace | unsafe | threshold | 0.99 | 0 | 6 | 321 |
| v4_fp32_inplace | unsafe | consecutive_2 | 0.5 | 0 | 2 | 384 |
| v4_fp32_inplace | unsafe | consecutive_2 | 0.9 | 0 | 3 | 367 |
| v4_fp32_inplace | unsafe | consecutive_2 | 0.99 | 1 | 6 | 305 |
| v4_fp16_gather | cut | threshold | 0.5 | 0 | 4 | 386 |
| v4_fp16_gather | cut | threshold | 0.9 | 0 | 3 | 373 |
| v4_fp16_gather | cut | threshold | 0.99 | 0 | 4 | 321 |
| v4_fp16_gather | cut | consecutive_2 | 0.5 | 0 | 3 | 384 |
| v4_fp16_gather | cut | consecutive_2 | 0.9 | 0 | 1 | 367 |
| v4_fp16_gather | cut | consecutive_2 | 0.99 | 0 | 4 | 305 |
| v4_fp16_gather | unsafe | threshold | 0.5 | 0 | 3 | 386 |
| v4_fp16_gather | unsafe | threshold | 0.9 | 0 | 3 | 373 |
| v4_fp16_gather | unsafe | threshold | 0.99 | 1 | 5 | 321 |
| v4_fp16_gather | unsafe | consecutive_2 | 0.5 | 0 | 4 | 384 |
| v4_fp16_gather | unsafe | consecutive_2 | 0.9 | 0 | 1 | 367 |
| v4_fp16_gather | unsafe | consecutive_2 | 0.99 | 1 | 5 | 305 |
| v4_fp16_inplace_tiles | cut | threshold | 0.5 | 0 | 3 | 386 |
| v4_fp16_inplace_tiles | cut | threshold | 0.9 | 0 | 3 | 373 |
| v4_fp16_inplace_tiles | cut | threshold | 0.99 | 0 | 7 | 321 |
| v4_fp16_inplace_tiles | cut | consecutive_2 | 0.5 | 0 | 3 | 384 |
| v4_fp16_inplace_tiles | cut | consecutive_2 | 0.9 | 0 | 3 | 367 |
| v4_fp16_inplace_tiles | cut | consecutive_2 | 0.99 | 1 | 7 | 305 |
| v4_fp16_inplace_tiles | unsafe | threshold | 0.5 | 0 | 3 | 386 |
| v4_fp16_inplace_tiles | unsafe | threshold | 0.9 | 0 | 3 | 373 |
| v4_fp16_inplace_tiles | unsafe | threshold | 0.99 | 0 | 6 | 321 |
| v4_fp16_inplace_tiles | unsafe | consecutive_2 | 0.5 | 0 | 4 | 384 |
| v4_fp16_inplace_tiles | unsafe | consecutive_2 | 0.9 | 0 | 3 | 367 |
| v4_fp16_inplace_tiles | unsafe | consecutive_2 | 0.99 | 1 | 7 | 305 |

report 还给出了 0.8、0.95、0.98 三个阈值，任务卡没有要求。所有阈值里 `fire_differs` 不为 0 的只有以下几处：

| 配置 | cut | unsafe |
|---|---|---|
| v3 | threshold@0.99：2 | threshold@0.99：1；consecutive_2@0.99：1 |
| v4_fp32_inplace | consecutive_2@0.99：1 | consecutive_2@0.99：1 |
| v4_fp16_gather | 无 | threshold@0.99：1；consecutive_2@0.99：1 |
| v4_fp16_inplace_tiles | threshold@0.95：1；consecutive_2@0.99：1 | threshold@0.95：1；consecutive_2@0.99：1 |

## 4. `profile` 前 10 行

`aten::linear`、`aten::matmul`、`aten::mm` 是层层嵌套的调用，时间会重复计入，不能相加。

**128 个会话，每 tick 1 个 token（n128_c1）**

| # | 算子 | CUDA ms/tick | calls/tick |
|---:|---|---:|---:|
| 1 | `_gdn_slot_recurrent_kernel` | 2.987 | 18 |
| 2 | `aten::linear` | 2.744 | 198 |
| 3 | `aten::matmul` | 2.661 | 187 |
| 4 | `aten::mm` | 2.660 | 186 |
| 5 | `_ring_attention_kernel` | 1.481 | 6 |
| 6 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn` | 0.987 | 54 |
| 7 | `aten::copy_` | 0.981 | 405 |
| 8 | `aten::to` | 0.682 | 459 |
| 9 | `aten::_to_copy` | 0.682 | 318 |
| 10 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` | 0.667 | 48 |


**192 个会话，每 tick 1 个 token（n192_c1）**

| # | 算子 | CUDA ms/tick | calls/tick |
|---:|---|---:|---:|
| 1 | `_gdn_slot_recurrent_kernel` | 6.528 | 18 |
| 2 | `aten::linear` | 3.617 | 198 |
| 3 | `aten::matmul` | 3.524 | 187 |
| 4 | `aten::mm` | 3.522 | 186 |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16…` | 2.588 | 108 |
| 6 | `_ring_attention_kernel` | 2.059 | 6 |
| 7 | `aten::copy_` | 1.257 | 405 |
| 8 | `aten::to` | 0.809 | 459 |
| 9 | `aten::_to_copy` | 0.809 | 318 |
| 10 | `aten::conv1d` | 0.800 | 18 |

## 5. 两种分块方案的 `arrival_simulation`

| 分块方案 | 会话 | 实际 ITPS | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|
| v4_fp16_inplace_tiles | 160 | 6,614 | 10.7 | 14.1 | 14.5 |
| v4_fp16_inplace_tiles | 176 | 7,158 | 13.1 | 18.0 | 18.6 |
| v4_fp16_inplace_tiles | 192 | 8,094 | 14.5 | 19.1 | 19.5 |
| v4_fp16_inplace_tiles | 208 | 8,819 | 15.0 | 19.8 | 20.3 |
| v4_fp16_inplace_tiles | 224 | 9,473 | 27.2 | 43.6 | 222.5 |
| v4_fp16_inplace_tiles | 240 | 10,208 | 32.4 | 43.2 | 44.1 |
| v4_fp16_inplace_tiles | 256 | 10,829 | 33.5 | 47.8 | 230.3 |
| v4_fp16_inplace_tile32x2 | 160 | 6,616 | 10.8 | 14.1 | 14.5 |
| v4_fp16_inplace_tile32x2 | 176 | 7,161 | 12.6 | 17.8 | 18.6 |
| v4_fp16_inplace_tile32x2 | 192 | 8,099 | 14.4 | 19.1 | 19.5 |
| v4_fp16_inplace_tile32x2 | 208 | 8,821 | 15.1 | 19.9 | 20.3 |
| v4_fp16_inplace_tile32x2 | 224 | 9,465 | 26.9 | 41.8 | 224.5 |
| v4_fp16_inplace_tile32x2 | 240 | 10,208 | 30.3 | 51.7 | 242.3 |
| v4_fp16_inplace_tile32x2 | 256 | 10,826 | 32.9 | 55.2 | 246.1 |

`capacity_at_p95_20ms`：两种方案都是 **208**，`v4_fp16_inplace_tiles` 为 208，`v4_fp16_inplace_tile32x2` 为 208。

## 执行方观察（只陈述数字）

- **判决。** 阈值不超过 0.9 时，4 种配置在两种规则下的 `fire_differs` 都是 0。只有阈值在 0.95–0.99 时，才有 0–2 条序列不一致，v3 本身在 0.99 下也有 1–2 条。
- **逐位置最大漂移。** `v4_fp16_inplace_tiles` 的最大漂移是 0.291，高于 v3 的 0.231；`over_0.1` 为 15 个位置，v3 为 11 个。p999 两者接近：0.0367 对 0.0363。
- **样本里正常内容很少。** 400 条序列中，参考在阈值 0.5 时触发了 386 条，只有 14 条没触发。所以“漂移让本来不截的内容被截”这一方向，这次只在很少的序列上检验过。按红线口径，正常内容绝不能误判，是否要用以正常内容为主的序列再测一次，请设计方判断。
- **容量。** 按批大小选分块和固定 (32, 2)，容量都是 208，没有差别。
  - 208 个会话时，P95 为 19.8 ms 和 19.9 ms，已经贴近 20 ms 的上限。
  - 192 个会话时本轮 P95 为 19.1 ms，v1 为 18.5 ms，两次运行之间有约 0.6 ms 的差异。
  - 224 个会话起 P95 超过 40 ms，P99 超过 220 ms。
- **剖析。** 会话数从 128 增加到 192，是 1.5 倍：
  - GDN 内核从 2.99 ms/tick 增加到 6.53 ms/tick，是 2.2 倍，是占时最多的一项；
  - `_ring_attention_kernel` 从 1.48 ms/tick 增加到 2.06 ms/tick；
  - 192 个会话时，还有一个 cutlass 的 gemm+relu 占 2.59 ms/tick（108 次调用），128 个会话时它不在前 10 里；
  - `aten::copy_` 加上 `aten::to`，约占 2 ms/tick。

## 产物

| 文件 | SHA256 | 是否提交 |
|---|---|---|
| `round6/serving/results_v5/report.json` | `d96022153d4cfea8f5e98b1ae464101269e224f67daffce0d8d32e6635ce39fb` | 已提交 |
| `round6/serving/results_v5/serving_v5.log` | `4076d14cce2b4db441640d75b62e93835f99b1d725021c459525bac882d53f62` | 留在 Mac，不提交 |
