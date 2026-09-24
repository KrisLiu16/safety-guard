# T007 服务引擎 v6：空行不读写状态、会话档位加密，并在正常内容上查判决

- 版本：v3（v1、v2 已完成，结果见 [feedback/T007-serving-v4.md](../feedback/T007-serving-v4.md) 的 git 历史和 `round6/serving/RESULTS.md` 的 v4、v5 两节）
- 目标：
  1. v2 的剖析显示，每步会话数补齐到档位（例如 192 补成 256）后，空行照样读写一整份 GDN 状态，GDN 内核耗时随会话数超线性增长。这一轮让空行不读写状态和环形缓冲，并加密会话档位。同时跑细档位和原档位，分开两项的收益。
  2. v2 的 400 条序列几乎全会被截断，“流式把本不该截的截了”这一方向没测到。这一轮加测 prefix_v2 dev 的助手侧记录，其中一半是 safe，按来源标签分开报告判决一致性。

  只推理，不训练，不依赖标签和数据平台。
- 依赖：无。L20 空闲即可。
- 步骤：
  1. 单测（Mac，只用小张量，不跑模型）：
     ```bash
     cd round6/serving && ../../.venv/bin/python -m unittest test_bench_v5_cpu.py test_ring_attention_cpu.py test_batched_engine_cpu.py
     ```
     共 7 项，全过。
  2. 提交作业：`kubectl apply -f round6/serving/bench_worker_v6.yaml`。Pod 进入 Running 后，把 `round6/serving/*.py` 拷到 `/work/round6/serving/`，用 `sha256sum` 核对，再 `touch /work/round6/serving_v6_input_ready`。
  3. 进度看 `/work/output/round6/serving_v6/report.json`，按顺序写入：`sets`、`decision_drift`（两种档位 × thinking 和 prefix_v2 dev 的 safe、unsafe）、`profile`、`arrival_simulation`、`capacity_at_p95_20ms`、`graphs_captured`。
  4. 完成后 `serving_v6_exit_code.txt` 为 0、`status` 为 `completed`。取回 `report.json` 和日志到 `round6/serving/results_v6/`，核对 SHA 后 `touch /work/output/round6/serving_v6_collected`。
- 预期产物：提交 `results_v6/report.json`；日志留在 Mac。
- 验收：
  - `status=completed`；
  - `v6_coarse` 在 thinking 上的判决一致性与 v2 的 `v4_fp16_inplace_tiles` 同一量级（阈值 ≤ 0.9 时 `fire_differs` 为 0 或接近 0）。空行不读写不应改变真实会话的结果。
- 需要用户决定：无。
- 反馈里必须报告：
  - `sets`；
  - 每种档位、每个数据集（thinking、prefix_v2_dev_safe、prefix_v2_dev_unsafe）的 `cut` 分数：max、p999、over_0.1，以及阈值 0.5、0.9、0.99 下两种规则的 `fire_differs`、`both_fire_other_index`、`reference_fires`；
  - `profile` 两个表的前 10 行；
  - 两种档位的 `arrival_simulation`、`capacity_at_p95_20ms`、`graphs_captured`；
  - 产物 SHA256。
