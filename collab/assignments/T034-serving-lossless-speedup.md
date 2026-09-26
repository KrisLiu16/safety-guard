# T034 推理引擎提速（输出与 v6 逐位相同）

- 版本：v1（2026-09-26）
- 设计和执行：执行方（兼设计）。用户 2026-09-26：“端到端把这件事做好，不损失精度就行”。
- 依据：调研报告 [round6/serving/OPTIMIZATION_RESEARCH.md](../../round6/serving/OPTIMIZATION_RESEARCH.md)。
- 代码：`round6/serving/slot_engine_v7.py`、`slot_engine_v8.py`、`fused_kernels.py`；基准 `bench_v7_l20.py`、`bench_v8_l20.py`；
  单测 `test_serving_cpu.py`（CPU）、`test_fused_l20.py`（L20）；剖析 `profile_v7_l20.py`。

## 口径

“不损失精度”按最严的理解：新引擎的每个输出概率与 v6 **逐位相同**。这样阈值、评测结果、判决都不变，不需要重新标定。
改变算术的优化（换 GEMM 算法、拼接权重、改 ring 分块、重写归约）这一轮不做；以后要做，另走精度门槛（判决漂移不超过 v6、dev 指标不变）。

## 步骤与验收

1. v7：主机路径和图内的纯清理。验收：400 条思考序列（种子 7）和 prefix_v2 dev（种子 11）上与 v6 最大差 0。
2. 剖析 v7 在 208 个会话、77 行 × 1 token 时每个 kernel 的耗时，定下一步的对象。
3. v8：逐元素胶水融合成 Triton kernel，每个算子按 PyTorch 的舍入点和顺序写。验收：单测每个算子逐位相同；整机同 1。
4. 容量：P95 ≤ 20 ms 时的会话数，两个种子取较差的一个。

## 不在本任务内

- 把引擎接进测试台（`guard_demo/server.py` 现在用 HF 模型逐会话前向，没有用这套引擎）：需要调度线程、按 slot 的快照回退、
  末尾 token 的临时打分和类别头输出；恢复测试台要先问用户。
