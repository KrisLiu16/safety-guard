# 第六轮第二阶段：训练主干（2026-09-25 用户批准，T029）

## 为什么要训练主干

第一阶段冻结主干，只重训读出头：
- T018：回答侧 full 头 calibration AUC 0.865；
- T025：提问侧 full 头 0.896；
- T028：训练 16 轮是 0.874，换成大一倍的全新读出是 0.873，调大学习率、去掉 L2 是 0.871。

三种做法停在同一个水平，说明瓶颈在主干最后一层的特征上。回答侧的召回也因此偏低：Run A 有争议 21–39%、风险 14–22%。

## 做法

| 项 | 内容 |
|---|---|
| 起点 | 现在的模型：Round5 的固定 checkpoint（`bb16a3a6…`，即 round4 window best），24 层，W512 窗口，与运行时相同 |
| 训练什么 | 整个主干全参训练，fp32 主权重，bf16 autocast，gradient checkpointing；两个角色的 projection 和 risk 层。category 头冻结，但主干变了，它的输出会漂移，这一轮不验证 |
| 目标 | 每个目标 token 都有三类目标（0 safe、1 unsafe、2 controversial），取该前缀在红线探针上的等级（`levels.position_level`）。起点分句内部、按词规则改标的词内部，等级不确定，不训练 |
| 数据 | 都是已有标签，不生成新数据：<br>- Run A 回答（第一阶段输入：train 8,000 词 + calibration），回答头；<br>- Run A 提问（每个不同的提问单独一条），提问头；<br>- prefix_v2 两个角色的记录（冻结的 ids，重新分词对不上的跳过），train 另加 augmentation 视图，权重减半 |
| 标签 | T021、T024 的红线标签。T026 通过后改用政治专项版（`labels_political_v2/`，作业里用环境变量 `LABELS_DIR` 指定） |
| 权重 | 每个（来源, 角色）组内，各类别的总权重相等（记录权重计入）。组之间按 `--mix` 分：Run A 回答 1、prefix_v2 回答 1、Run A 提问 0.5、prefix_v2 提问 0.5 |
| 优化 | 同 Round5 SFT：<br>- AdamW，主干 8e-6，头 5e-5，weight decay 0.01；<br>- 64 步 warmup，余弦衰减到 0.1，梯度裁剪 1.0；<br>- 每次更新 16 条记录，microbatch 不超过 4 条、16,384 个 padded token；<br>- 2 轮 |
| 选择 | 训练前和每轮结束后，在 calibration 上算截断分数 1 − p(safe)，按（来源, 角色）分组给出位置级 AUC 和流式 AUC（整条流的最大分数；正例为整段等级不是 safe）。流式 AUC 均值最好的那一轮导出为 `best.safetensors`。没有一轮超过起点，就用起点。dev 不读 |
| 评测 | `eval_head_l20.py` 的 `checkpoint` 变体：把整个新模型装进固定的运行时，完整性检查照旧。<br>- 回答侧：Run A、prefix_v2 和官方 thinking 集；<br>- 提问侧：Run A 提问和 prefix_v2。<br>在 Mac 上与 T018、T025 的 init、full 放在一起，用 `analyze_redline.py` 对比 |

每轮约 1,150 万个 token（按 T017、T023 的前向 token 数估算）。Round4 实测训练吞吐约 1,800 token/秒，所以 2 轮约 4–5 小时，加上两次评测约 1 小时。

## 文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `build_targets.py` | 作业第一步（CPU） | 生成逐 token 目标：`records_<split>.jsonl` 和 `report.json` |
| `train_stage2.py` | L20 | 训练、在 calibration 上选轮次、导出；`--smoke N` 只跑 N 步，先确认显存和速度 |
| `worker_stage2_v1.yaml` | 集群 | 一个作业：目标 → 冒烟 → 训练 → 回答侧评测 → 提问侧评测，7 份 GPU 时间片 |
| `test_stage2_cpu.py` | Mac | CPU 单测。训练一步和 calibration 打分需要 torch |

## 通过标准（草案）

- **正常内容**：回答侧和提问侧 `normal` 层的误报不高于第一阶段的 full 头，目标接近 0；
- **召回**：Run A 有争议和风险两档的召回，都明显高于 T028 的 full 头（有争议 21–28%、风险 14–21%）；
- **流式 AUC**：Run A dev 和 prefix_v2 dev 都高于第一阶段；
- **官方 thinking 集**：按 T027 的本口径标签看误报；
- **上线**：新模型要经服务引擎（T007）上的一致性检查以后才能上线。
