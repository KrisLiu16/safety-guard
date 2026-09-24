# T007 服务引擎：v4 判决一致性、按批大小选分块、剖析

- 版本：v2（v1 已完成，结果见 [feedback/T007-serving-v4.md](../feedback/T007-serving-v4.md) 和 `round6/serving/RESULTS.md` 的 v4 一节）
- 目标：v4（fp16 状态 + 原地读注意力 + 分块 (32, 2)）把 P95 ≤ 20 ms 的容量从 96 个会话提到 192 个。这一轮确认三件事：
  1. 它的流式判决和整段前向是否一致。v1 的漂移最大值只由一个敏感位置决定，说明不了判决会不会变，这一轮改为按阈值统计判决不一致的序列数，用 400 条序列；
  2. 按批大小选 fp16 的 GDN 分块，容量还能不能再提；
  3. 剖析一步，看下一个瓶颈在哪里。

  只推理，不训练，不依赖标签和数据平台。
- 依赖：无。L20 空闲即可。
- 步骤：
  1. 单测（Mac，只用小张量，不跑模型）：
     ```bash
     cd round6/serving && ../../.venv/bin/python -m unittest test_bench_v5_cpu.py test_ring_attention_cpu.py test_batched_engine_cpu.py
     ```
     全过（`test_bench_v5_cpu.py` 3 项）。
  2. 提交作业：`kubectl apply -f round6/serving/bench_worker_v5.yaml`。Pod 进入 Running 后，把 `round6/serving/*.py` 拷到 `/work/round6/serving/`，用 `sha256sum` 核对，再 `touch /work/round6/serving_v5_input_ready`。
  3. 进度看 `/work/output/round6/serving_v5/report.json`，按顺序写入：
     - `tile_sweep_fp16`、`tiles_chosen_fp16`；
     - `decision_drift`（4 种配置）；
     - `profile`；
     - `arrival_simulation`、`capacity_at_p95_20ms`。
  4. 完成后 `serving_v5_exit_code.txt` 为 0、report 的 `status` 为 `completed`。取回 `report.json` 和日志到 `round6/serving/results_v5/`，核对 SHA 后 `touch /work/output/round6/serving_v5_collected`。
- 预期产物：提交 `results_v5/report.json`；日志留在 Mac。
- 验收：`status=completed`；`decision_drift` 每种配置的 `positions` 都等于 report 的 `positions`。
- 需要用户决定：无。
- 反馈里必须报告：
  - `tiles_chosen_fp16`；
  - 每种配置 `decision_drift` 的 `cut` 和 `unsafe` 两组的 max、p999、over_0.05、over_0.1；
  - 阈值 0.5、0.9、0.99 下 `threshold` 和 `consecutive_2` 两种规则的 `fire_differs`、`both_fire_other_index`、`reference_fires`；
  - `profile` 两个表的前 10 行；
  - 两种分块方案的 `arrival_simulation` 和 `capacity_at_p95_20ms`；
  - 产物 SHA256。
