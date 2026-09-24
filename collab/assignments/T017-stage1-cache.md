# T017 第一阶段：特征缓存（推理，不训练）

- 版本：v1
- 目标：为只重训读出头准备训练数据：在 L20 上用固定的第五轮权重，对 prefix_v2 训练集和 Run A 训练样本走一遍前向，缓存头的输入和投影特征以及位置标签。设计见 [round6/stage1_head/README.md](../../round6/stage1_head/README.md)。
- 依赖：T001（`trainable.jsonl`）；PVC 上有 `/work/round5`、`/work/models/qwen35` 和第五轮输出（与 T004/T014 相同），以及 `/work/round6/probe/` 下 T014 用过的脚本。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/stage1_head/test_stage1_cpu.py`，5 项全过。
  2. 生成输入：`.venv/bin/python round6/stage1_head/make_stage1_input.py`，输出 `round6/stage1_head/input_v1/`。记下 manifest 的 records 和 words_by_split（train 8,000，calibration 1,264）。
  3. 提交作业：`kubectl apply -f round6/stage1_head/worker_cache_v1.yaml`。Pod 进入 Running 后，把 `round6/stage1_head/*.py` 和 `input_v1/` 拷到 Pod 的 `/work/round6/stage1_head/`，把 `round6/probe/*.py` 拷到 `/work/round6/probe/`（覆盖同名文件），在 Pod 里用 `sha256sum` 核对，再 `touch /work/round6/stage1_cache_v1_input_ready`。
  4. 开始后约 10 分钟查看 `/work/output/round6/stage1_cache_v1/report.json`：应当有 `head_keys`（包括 `projection.0.weight`、`risk.weight`、`category.weight`），`phase` 依次经过 prefix_v2 和 runA，`max_prob_diff` 小于 0.001。失败就取回 `report.json` 和 `/work/output/round6/stage1_cache_v1.log`，写反馈，然后 `touch /work/output/round6/stage1_cache_v1_collected` 释放 GPU。
  5. 完成后 `stage1_cache_v1_exit_code.txt` 应为 `cache=0`。只取回 `report.json` 到 Mac 的 `round6/stage1_head/results_cache_v1/`。**缓存文件（约 4 GB）留在 PVC 上**，T018 直接读取。然后 `touch /work/output/round6/stage1_cache_v1_collected`。
- 预期产物：提交 `round6/stage1_head/results_cache_v1/report.json`。
- 验收：`status=completed`、`integrity_pass=true`、`max_prob_diff < 0.001`；`stats` 中 prefix_v2 和 runA 的 train、calibration 都有记录和位置，runA train 的记录数约 32,000。
- 需要用户决定：无。这是单卡推理，预计 3–3.5 小时；节点被占用就等待，不要停别人的 Pod。
- 反馈里必须报告：单测结果；输入 manifest；作业名和起止时间；`report.json` 的 `stats`、`totals`、`max_prob_diff`、`elapsed_seconds`、`head_init_sha256`；PVC 上缓存文件的总大小。
