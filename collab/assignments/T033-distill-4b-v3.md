# T033 第二阶段 v3：用 Qwen3Guard-Stream-4B 蒸馏一个通用头

- 版本：v1（2026-09-25）
- 设计和执行：执行方（兼设计）。用户 2026-09-25 同意“用更强的老师 4B”，并决定第二张 L20 不放推理测试，整卡训练。
- 代码：`round6/stage2/make_teacher_inputs.py`、`teacher_label_l20.py`、`build_targets.py --teacher`、`train_distill.py`、`export_heads.py`；作业 `worker_stage2_v3.yaml`、`worker_stage2_v3_eval.yaml`。

## 起因

只审红线以后，第二阶段模型不再判断红线以外的一般有害内容，Qwen3Guard 英文基准的分数随之大降（T029 反馈）。
用户希望尽量保留 Qwen3Guard-Stream 原有的这部分能力，甚至超过它。

## 做法

- **两个头，一个主干**：每个角色（提问、回答）保留红线头（projection / risk / category），另加通用头
  （general_projection / general / general_category），用 Round5 的头初始化。
  - 红线头的目标和 v2 完全一样（逐 token 三类 + 警觉软目标），决定截断的仍是它；
  - 通用头只学老师，用来对照 Qwen3Guard 和给产品方提供一般有害分数。
- **老师**：Qwen3Guard-Stream-4B，对 v2 全部训练对话（同样的 id）逐 token 输出风险分布和类别分布；
  按字符结束位置对齐到我们的分词。
- **损失**：红线 CE + 0.5 × KL(老师 ‖ 通用头) + 0.25 × 类别 CE（只在老师非安全概率 > 0.5 的位置）。
- **选轮次**：仍按红线头的 calibration 平均流式 AUC；另报通用头对老师的 KL 和流式 AUC。
- **精度**：同 v2（fp32 主权重，bf16 autocast）。

## 评测（与 v2 相同的评测器）

`export_heads.py` 把 checkpoint 拆成两个标准布局文件：
- `redline.safetensors`：红线视图，跑 eval_head（回答、提问）、正常问题扫描、领导人抽查、基准；
- `general.safetensors`：通用头放到标准槽位，跑正常问题扫描和 Qwen3Guard 基准。

## 验收

1. 红线视图不比 v2 差：calibration 平均流式 AUC 相差 ≤ 0.01；正常问题扫描同 T030 的标准。
2. 通用视图在 Qwen3Guard 英文基准上的平均 F1（提问、回答）明显高于 v1 的 21.5 / 31.4，目标是接近论文里 Stream-0.6B 的 84.7 / 78.3。
3. 两个视图共用主干，线上推理只多两个小头的计算。
