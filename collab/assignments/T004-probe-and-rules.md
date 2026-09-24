# T004 线性探针 + T005 决策规则重拟合（Run A）

- 版本：v1
- 目标：用固定第五轮权重在 Run A 上跑一次 L20 前向（不训练），回答两个问题：冻结特征能否区分“复述有害请求”和“真有害”（T004）；换成 Run A 校准集后，决策规则的误报和召回是多少（T005）。设计说明见 [round6/probe/README.md](../../round6/probe/README.md)。
- 依赖：T001 完成（`round6/response_v14/batch_50k/extracted/trainable.jsonl` 已在 Mac 上）。
- 输入：`round6/probe/` 下的代码；`trainable.jsonl`；PVC 上的第五轮代码 `/work/round5`、模型 `/work/models/qwen35` 和第五轮输出（与 decision_rule 作业相同）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/probe/test_probe_cpu.py`，8 项全过（需要 torch，Mac 的 .venv 里有）。
  2. 生成输入：`.venv/bin/python round6/probe/make_probe_input.py`，输出 `round6/probe/input_v1/`。记下 `manifest.json` 里的 records 和各 split 词数（预计约 16,000 条记录）。
  3. 提交作业：`kubectl apply -f round6/probe/worker_v1.yaml`。Pod 进入 Running 后，把 `round6/probe/` 下的 `*.py` 和 `input_v1/` 拷到 Pod 的 `/work/round6/probe/`，在 Pod 里用 `sha256sum` 核对和本地一致，再 `touch /work/round6/probe_v1_input_ready`。
  4. 开始后约 10 分钟查看 `/work/output/round6/probe_v1/report.json`：应当出现 `"smoke_passed": true`，并且 `max_prob_diff` 小于 0.001。如果 `status` 是 `failed`，取回 `report.json` 和 `/work/output/round6/probe_v1_dump.log`，写反馈，然后 `touch /work/output/round6/probe_v1_collected` 让作业退出、释放 GPU。
  5. 出现 `/work/output/round6/probe_v1_done` 后，看 `probe_v1_exit_code.txt`，应为 `dump=0 fit=0`。把下列文件取回 Mac 的 `round6/probe/results_v1/`：`report.json`、`probe_v1_results.json`、`runA_calibration.jsonl.gz`、`runA_dev.jsonl.gz`，以及两个 log。用 `report.json` 里的 `files` 核对 SHA256，然后 `touch /work/output/round6/probe_v1_collected`。`*.npz` 不取回。
  6. T005（Mac CPU）：
     ```bash
     .venv/bin/python round6/probe/analyze_runA_rules.py round6/probe/results_v1 --v1-dump round6/decision_rule/results_v1 --output round6/probe/results_v1/runA_rules_v1_results.json
     ```
     如果 Mac 上 `round6/decision_rule/results_v1/` 里没有 `prefix_v2_*.jsonl.gz` 和 `official_*.jsonl.gz`，就去掉 `--v1-dump` 再跑，并在反馈里注明。
- 预期产物：提交到仓库的只有 `round6/probe/results_v1/` 下的 `report.json`、`probe_v1_results.json`、`runA_rules_v1_results.json`；其他留在 Mac 或 PVC。
- 验收：`report.json` 中 `status=completed`、`integrity_pass=true`、`max_prob_diff < 0.001`；三个 split 的 records 与输入 manifest 基本一致（跳过的条数写在 `totals` 里）；`probe_v1_results.json` 和 `runA_rules_v1_results.json` 生成。
- 需要用户决定：无。这是单卡推理，预计 1 小时左右，与 decision_rule 作业同一节点、同样资源。如果节点的 GPU 被别人占用，就等待并在反馈里说明，不要停别人的 Pod。
- 反馈里必须报告：
  1. 单测结果；输入 manifest 的 records、words_by_split、records_by_slot_style；
  2. 作业名、开始和结束时间、`report.json` 中的 `totals`、`records`、`feature_positions`、`max_prob_diff`、`hf_tokenizer_cross_check`、`elapsed_seconds`；
  3. `fit_probe.py` 在终端打印的那段 JSON（三个打分器的 AUC 和 recall@5%）；
  4. `analyze_runA_rules.py` 打印的表格全文和温度；
  5. 产物文件的 SHA256。
