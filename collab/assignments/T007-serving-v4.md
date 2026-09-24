# T007 服务引擎 v4：GDN 状态降精度 + 注意力原地读环形缓冲

- 版本：v1
- 目标：在 v3 连续批处理引擎上测两项提速（`round6/serving/RESULTS.md` 的“下一步”第 1、2 项），看在 P95 ≤ 20 ms 的前提下，单卡能多承载多少个真实速度的流式会话（v3 约 64–100 个）：
  - GDN 状态池从 fp32 改为 bf16 或 fp16；
  - 注意力按槽位原地读环形缓冲（新 Triton 内核），不再 gather 复制。

  顺带扫 GDN 内核的分块参数。设计见 RESULTS.md 的 v4 一节。只推理，不训练，不依赖标签和数据平台。
- 依赖：无。L20 空闲即可。
- 步骤：
  1. 单测（Mac）：
     ```bash
     cd round6/serving && ../../.venv/bin/python -m unittest test_ring_attention_cpu.py test_batched_engine_cpu.py
     ```
     `test_ring_attention_cpu.py` 2 项，加上原有的 `test_batched_engine_cpu.py`，全过。
  2. 提交作业：`kubectl apply -f round6/serving/bench_worker_v4.yaml`。Pod 进入 Running 后，把 `round6/serving/*.py` 拷到 `/work/round6/serving/`，用 `sha256sum` 核对，再 `touch /work/round6/serving_v4_input_ready`。
  3. 进度看 `/work/output/round6/serving_v4/report.json`，按顺序写入：
     - `kernel_check`：`pass` 必须为 true，否则脚本会自己停下；
     - `drift`、`drift_gate`；
     - `gdn_tile_sweep`、`gdn_tile_chosen`；
     - `throughput`；
     - `arrival_simulation`、`capacity_at_p95_20ms`。

     预计 40–70 分钟。
  4. 完成后 `serving_v4_exit_code.txt` 为 0，且 report 的 `status` 为 `completed`。把 `report.json` 和 `serving_v4.log` 取回到 `round6/serving/results_v4/`，核对 SHA 后 `touch /work/output/round6/serving_v4_collected`。
- 预期产物：提交 `results_v4/report.json`；日志留在 Mac。
- 验收：
  - `kernel_check.pass` 为 true；
  - `v4_fp32_gather` 与 v3 的 `vs_v3.max` 为 0 或接近 0（这两个是同一套计算，用来确认 v4 没有改坏别的地方）；
  - `v4_bf16_inplace` 的 `graph_vs_eager.max` 接近 0。
- 需要用户决定：无。
- 反馈里必须报告：
  - `kernel_check` 的各个 case；
  - 每种组合的 `drift`：max、p999、argmax_flips、`cut_score`、`vs_v3`，以及 `drift_gate`；
  - `gdn_tile_sweep` 表和选中的分块；
  - 每种组合在 128×1、512×1、128×16 下的 tick P50/P95 和 ITPS，以及 `state_bytes_per_session`；
  - `arrival_simulation` 表和 `capacity_at_p95_20ms`；
  - 产物 SHA256。
