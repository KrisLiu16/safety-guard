# T007 反馈

- 对应任务单版本：v3（v1、v2 的反馈见本文件的 git 历史：cca4710、e4c6f97）
- 状态：**完成，验收通过**
- 执行时间：2026-09-24（北京时间）19:18 提交，19:18:38 写入 input_ready，19:25 写入 done，19:26 取回后写入 collected。实际用时约 7 分钟。
- 实际执行的命令：与任务单相同。
  - 单测：`test_bench_v5_cpu.py`、`test_ring_attention_cpu.py`、`test_batched_engine_cpu.py` 共 7 项，全部通过。
  - 作业：Job `safety-guard-serving-bench-v6-r1`，Pod `safety-guard-serving-bench-v6-r1-stbjg`。提交前节点的 GPU 分配为 0。
  - `round6/serving/*.py` 共 16 个文件，用 `sha256sum` 核对，与本地一致。
  - `serving_v6_exit_code.txt` 为 0，`status=completed`。

## 验收

- `status=completed`。
- `v6_coarse` 在 thinking 上，阈值 ≤ 0.9 时两种规则的 `fire_differs` 都是 0，与 v2 的 `v4_fp16_inplace_tiles` 同一量级。所有阈值中不为 0 的只有 `threshold@0.95` 和 `consecutive_2@0.99`，各 1 条，与 v2 相同。

## 1. `sets`

- thinking：400 条序列，374,953 个位置。
- prefix_v2_dev_assistant：800 条序列，safe 400 条、unsafe 400 条。

## 2. `decision_drift`：`cut` 分数的逐位置漂移

| 档位 | 数据集 | 位置数 | max | p999 | over_0.1 |
|---|---|---:|---:|---:|---:|
| v6_fine | thinking | 374,953 | 0.2615 | 0.0379 | 16 |
| v6_fine | prefix_v2_dev_safe | 32,741 | 0.2768 | 0.0707 | 16 |
| v6_fine | prefix_v2_dev_unsafe | 49,547 | 0.1092 | 0.0574 | 5 |
| v6_coarse | thinking | 374,953 | 0.1979 | 0.0383 | 10 |
| v6_coarse | prefix_v2_dev_safe | 32,741 | 0.2768 | 0.0686 | 14 |
| v6_coarse | prefix_v2_dev_unsafe | 49,547 | 0.1092 | 0.0578 | 5 |

## 3. `decision_drift`：判决（`cut` 分数）

| 档位 | 数据集 | 规则 | 阈值 | fire_differs | both_fire_other_index | reference_fires |
|---|---|---|---:|---:|---:|---:|
| v6_fine | thinking | threshold | 0.5 | 0 | 3 | 386 / 400 |
| v6_fine | thinking | threshold | 0.9 | 0 | 3 | 373 / 400 |
| v6_fine | thinking | threshold | 0.99 | 0 | 5 | 321 / 400 |
| v6_fine | thinking | consecutive_2 | 0.5 | 0 | 3 | 384 / 400 |
| v6_fine | thinking | consecutive_2 | 0.9 | 0 | 3 | 367 / 400 |
| v6_fine | thinking | consecutive_2 | 0.99 | 1 | 4 | 305 / 400 |
| v6_fine | prefix_v2_dev_safe | threshold | 0.5 | 1 | 1 | 178 / 400 |
| v6_fine | prefix_v2_dev_safe | threshold | 0.9 | 0 | 2 | 129 / 400 |
| v6_fine | prefix_v2_dev_safe | threshold | 0.99 | 0 | 0 | 46 / 400 |
| v6_fine | prefix_v2_dev_safe | consecutive_2 | 0.5 | 0 | 1 | 151 / 400 |
| v6_fine | prefix_v2_dev_safe | consecutive_2 | 0.9 | 0 | 3 | 102 / 400 |
| v6_fine | prefix_v2_dev_safe | consecutive_2 | 0.99 | 0 | 1 | 38 / 400 |
| v6_fine | prefix_v2_dev_unsafe | threshold | 0.5 | 0 | 1 | 377 / 400 |
| v6_fine | prefix_v2_dev_unsafe | threshold | 0.9 | 0 | 4 | 353 / 400 |
| v6_fine | prefix_v2_dev_unsafe | threshold | 0.99 | 1 | 2 | 305 / 400 |
| v6_fine | prefix_v2_dev_unsafe | consecutive_2 | 0.5 | 0 | 3 | 372 / 400 |
| v6_fine | prefix_v2_dev_unsafe | consecutive_2 | 0.9 | 0 | 3 | 348 / 400 |
| v6_fine | prefix_v2_dev_unsafe | consecutive_2 | 0.99 | 1 | 6 | 288 / 400 |
| v6_coarse | thinking | threshold | 0.5 | 0 | 3 | 386 / 400 |
| v6_coarse | thinking | threshold | 0.9 | 0 | 3 | 373 / 400 |
| v6_coarse | thinking | threshold | 0.99 | 0 | 5 | 321 / 400 |
| v6_coarse | thinking | consecutive_2 | 0.5 | 0 | 3 | 384 / 400 |
| v6_coarse | thinking | consecutive_2 | 0.9 | 0 | 2 | 367 / 400 |
| v6_coarse | thinking | consecutive_2 | 0.99 | 1 | 5 | 305 / 400 |
| v6_coarse | prefix_v2_dev_safe | threshold | 0.5 | 1 | 1 | 178 / 400 |
| v6_coarse | prefix_v2_dev_safe | threshold | 0.9 | 0 | 2 | 129 / 400 |
| v6_coarse | prefix_v2_dev_safe | threshold | 0.99 | 0 | 0 | 46 / 400 |
| v6_coarse | prefix_v2_dev_safe | consecutive_2 | 0.5 | 0 | 1 | 151 / 400 |
| v6_coarse | prefix_v2_dev_safe | consecutive_2 | 0.9 | 0 | 3 | 102 / 400 |
| v6_coarse | prefix_v2_dev_safe | consecutive_2 | 0.99 | 0 | 1 | 38 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | threshold | 0.5 | 0 | 1 | 377 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | threshold | 0.9 | 0 | 4 | 353 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | threshold | 0.99 | 1 | 2 | 305 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | consecutive_2 | 0.5 | 1 | 3 | 372 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | consecutive_2 | 0.9 | 0 | 3 | 348 / 400 |
| v6_coarse | prefix_v2_dev_unsafe | consecutive_2 | 0.99 | 1 | 6 | 288 / 400 |

report 里还有 0.8、0.95、0.98 三个阈值。所有阈值中 `fire_differs` 不为 0 的情况：

| 档位 | thinking | prefix_v2_dev_safe | prefix_v2_dev_unsafe |
|---|---|---|---|
| v6_fine | threshold@0.95：1<br>consecutive_2@0.99：1 | threshold@0.5：1 | threshold@0.95：1<br>consecutive_2@0.95：2<br>consecutive_2@0.98：1<br>threshold@0.99：1<br>consecutive_2@0.99：1 |
| v6_coarse | threshold@0.95：1<br>consecutive_2@0.99：1 | threshold@0.5：1 | consecutive_2@0.5：1<br>threshold@0.95：1<br>consecutive_2@0.95：2<br>consecutive_2@0.98：1<br>threshold@0.99：1<br>consecutive_2@0.99：1 |

## 4. `profile` 前 10 行

`aten::linear`、`aten::matmul`、`aten::mm` 是嵌套调用，时间会重复计入，不能相加。

**细档位，192 个会话，每 tick 1 个 token（fine_n192_c1）**

| # | 算子 | CUDA ms/tick | calls/tick |
|---:|---|---:|---:|
| 1 | `_gdn_slot_recurrent_kernel` | 5.065 | 18 |
| 2 | `aten::linear` | 3.091 | 198 |
| 3 | `aten::matmul` | 3.003 | 187 |
| 4 | `aten::mm` | 3.001 | 186 |
| 5 | `_ring_attention_kernel` | 2.153 | 6 |
| 6 | `aten::copy_` | 1.154 | 405 |
| 7 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16…` | 0.991 | 42 |
| 8 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn` | 0.854 | 48 |
| 9 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` | 0.809 | 48 |
| 10 | `aten::to` | 0.781 | 459 |


**细档位，256 个会话（fine_n256_c1）**

| # | 算子 | CUDA ms/tick | calls/tick |
|---:|---|---:|---:|
| 1 | `_gdn_slot_recurrent_kernel` | 7.034 | 18 |
| 2 | `aten::linear` | 3.618 | 198 |
| 3 | `aten::matmul` | 3.526 | 187 |
| 4 | `aten::mm` | 3.524 | 186 |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16…` | 2.589 | 108 |
| 6 | `_ring_attention_kernel` | 2.588 | 6 |
| 7 | `aten::copy_` | 1.360 | 405 |
| 8 | `aten::to` | 0.911 | 459 |
| 9 | `aten::_to_copy` | 0.911 | 318 |
| 10 | `aten::cat` | 0.871 | 558 |

## 5. `arrival_simulation`

| 档位 | 会话 | 实际 ITPS | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|
| v6_fine | 208 | 8,817 | 13.6 | 18.0 | 18.6 |
| v6_fine | 224 | 9,469 | 15.5 | 20.5 | 21.0 |
| v6_fine | 240 | 10,186 | 27.2 | 51.1 | 247.4 |
| v6_fine | 256 | 10,820 | 24.3 | 47.4 | 239.8 |
| v6_fine | 272 | 11,435 | 38.2 | 188.0 | 289.4 |
| v6_fine | 288 | 11,772 | 39.9 | 138.0 | 273.0 |
| v6_fine | 304 | 12,591 | 40.5 | 53.8 | 55.0 |
| v6_fine | 320 | 13,305 | 61.3 | 257.2 | 294.4 |
| v6_coarse | 208 | 8,813 | 14.7 | 19.4 | 19.9 |
| v6_coarse | 224 | 9,476 | 15.5 | 20.5 | 21.0 |
| v6_coarse | 240 | 10,188 | 29.8 | 47.3 | 239.3 |
| v6_coarse | 256 | 10,816 | 32.8 | 60.8 | 256.5 |
| v6_coarse | 272 | 11,357 | 1540.9 | 2215.7 | 2449.1 |
| v6_coarse | 288 | 11,866 | 679.1 | 908.2 | 929.5 |
| v6_coarse | 304 | 12,544 | 79.5 | 105.9 | 108.2 |
| v6_coarse | 320 | 11,693 | 93.8 | 1739.7 | 1915.7 |

- `capacity_at_p95_20ms`：`v6_fine` 208，`v6_coarse` 208。
- `graphs_captured`：`v6_fine` 105，`v6_coarse` 68。

## 执行方观察（只陈述数字）

- **正常内容上的判决**：prefix_v2 dev 的 400 条 safe 序列里，两种档位都只在 `threshold@0.5` 有 1 条判决不同。其余阈值和规则都是 0。
  - `fire_differs` 不记录方向。所以这 1 条是“流式截了、整段没截”，还是反过来，从 report 里看不出来。
- **正常内容上的逐位置漂移比 thinking 大**：
  - p999：safe 集约 0.069–0.071，thinking 约 0.038。
  - `over_0.1`：safe 集是 32,741 个位置里 14–16 个，thinking 是 374,953 个位置里 10–16 个。按位置比例算，safe 集约高 10 倍。
  - 两种档位在 safe 集上的 max 都是 0.2768。
- **参考模型本身在正常内容上也会截**：参考就是现在的头，按整段前向判。它在 prefix_v2 dev 的 safe 序列上，`threshold@0.5` 触发 178/400（44.5%），`threshold@0.99` 仍触发 46/400（11.5%）。
  - 这里用的是未校准的原始阈值，标签也是第五轮的旧口径，所以只能说明现在的头的状态，和引擎漂移无关。
- **容量**：两种档位的容量都停在 208。
  - 208 个会话时，P95 细档位 18.0 ms，原档位 19.4 ms。
  - 224 个会话时两者都是 20.5 ms，略超门槛。
  - 240 个会话以上结果不单调，P99 多在 239–294 ms。原档位在 272 和 320 个会话时 P95 超过 1.7 s。
- **剖析（与 v2 比）**：192 个会话时，GDN 内核从 v2 的 6.53 ms/tick 降到 5.07 ms/tick（−22%）；256 个会话时是 7.03 ms/tick。

## 产物

| 文件 | SHA256 | 是否提交 |
|---|---|---|
| `round6/serving/results_v6/report.json` | `73940fa3e799794832dcfd5984185d3b6bae9b5a6f7492381478a800c006e1cf` | 已提交 |
| `round6/serving/results_v6/serving_v6.log` | `ad977f0850c3a8e0440651497a47dec69923a7c3eb82aebf534cdedfed4239c0` | 留在 Mac，不提交 |
