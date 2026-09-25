# T029 第二阶段 v1：训练主干（回答头和提问头一起），按红线逐 token 标签

- 版本：v1（2026-09-25）
- 设计和执行：执行方。设计方已不在，用户 2026-09-25 要求执行方一个人把训练端到端做完，这也就是批准了第二阶段。
- 设计见 [round6/stage2/README.md](../../round6/stage2/README.md)，代码在 `round6/stage2/`。
- 依赖：
  - T028 ✓：只重训读出头，calibration AUC 到顶约 0.87；
  - T026 v1.3 ✓：政治专项标签 `labels_political_v2/` 已拷到 PVC。

## 已定的取舍（设计方决定，用户可推翻）

1. **只用现有数据，不先补英文长思考**（BOARD“待用户决定”第 3 条）。
   - 用户要的是现在把训练做完。
   - T015（S5 安全一半，英文长思考）还要改生成指令、放量、复核，至少半天。
   - T027 指出的英文长思考误截，这一版先如实测出来，作为 v2 是否补数据的依据。
2. **标签用政治专项版**：`/work/round6/redline_v1/labels_political_v2/`（T026 v1.3，policy_digest `9900deabdbfdf07c`）。
3. **不改设计方写好的超参**：
   - 2 轮，主干学习率 8e-6，头 5e-5；
   - 每次更新 16 条，混合比例为 Run A 回答 1、prefix_v2 回答 1、Run A 提问 0.5、prefix_v2 提问 0.5；
   - 在 calibration 上按流式 AUC 均值选轮次，不看 dev。
4. **GPU 用 7 份时间片**（用户 2026-09-25 的指示），节点 172.19.1.144。

## 步骤

1. **单测**（Mac）：
   ```bash
   .venv/bin/python -m unittest round6/stage2/test_stage2_cpu.py
   ```
   应为 6 项，全过。
2. **代码上 PVC**：
   - 把 `round6/stage2/build_targets.py`、`train_stage2.py` 和 `round6/stage1_head/eval_head_l20.py`（含 `checkpoint` 变体）拷到 PVC 的同名路径；
   - 用 sha256 核对。
3. **预跑目标**（CPU 调试 pod，不占 GPU）：按作业里的命令把 `build_targets.py` 输出到 `/work/output/round6/stage2_targets_precheck`，看 `report.json` 里各组的条数、跳过数和类别计数。
4. **提交作业**：
   - `kubectl apply -f round6/stage2/worker_stage2_v1.yaml`，再 touch `/work/round6/stage2_backbone_v1_input_ready`；
   - 作业依次跑：目标 → 冒烟 4 步 → 训练 2 轮 → 回答侧评测 → 提问侧评测。
5. **收结果**：
   - 等 `/work/output/round6/stage2_backbone_v1_done` 出现，看 `stage2_backbone_v1_exit_code.txt`；
   - 拷回 `stage2_train_v1/`（不含 `epoch_*.safetensors`）、两份评测目录和日志；
   - 最后 touch `stage2_backbone_v1_collected`。
6. **分析**（Mac）：
   - 用 `analyze_redline.py` 按 `labels_political_v2` 的标签，把第二阶段模型和现在的头（init）、T018 full、T028 full（回答侧）、T025 full（提问侧）放在一起对比；
   - 各变体都用同一份标签重算，口径一致。
7. **官方 thinking 集**：
   - 按 T027 的本口径标签，看误截率，重点是“官方和我们都判 safe”的那 401 条；
   - 同时报按官方标签的 AUC。

## 通过标准

- **正常内容**：回答侧和提问侧 normal 层的流式误报，不高于第一阶段 full 头，目标接近 0。
- **召回**：Run A dev 有争议、风险两档的召回，都明显高于 T028 full（有争议 21–28%、风险 14–21%）。
- **流式 AUC**：Run A dev、prefix_v2 dev 都高于第一阶段的最好结果（T028 full 为 0.85 / 0.86）。
- **官方 thinking 集**：两边都判 safe 的 401 条，误截率要报出来。这一版不设门槛，但要和 T018、T028 比。
- **上线**：这一版只出候选模型，不替换线上模型。上线前要在服务引擎（T007）上做一致性检查，并由用户确认。

## 反馈里报告

- 单测结果；
- 预跑和正式跑的 `report.json` 条数；
- 冒烟 4 步的耗时和显存；
- 每轮的 loss 曲线概况、每轮在 calibration 上各组的 position AUC 和 stream AUC、选中的轮次、`best.safetensors` 的 SHA256；
- 评测的完整性检查（`integrity_pass`、`max_prob_diff`）；
- 各变体的对比表，按上面的通过标准逐条给结论；
- 官方集的误截率。
