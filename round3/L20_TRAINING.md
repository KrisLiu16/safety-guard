# 第三轮训练只在 L20 执行

2026-09-23 用户指定：**训练、蒸馏、RL 与正式性能验收使用腾讯云 `cls-og1rjus2` 的 NVIDIA L20，不在 M5 上更新模型参数。** M5 第一轮及第三轮早期微基准仅为历史记录。数据下载、格式审计、打包与脚本编写可在本机完成。

训练分两个独立门槛：

1. **Stage0 开放语料语义/结构迁移**：仅在[20k 训练、2k 开发集清单](data/stage0_distillation_20k/manifest.json)的源文件哈希、去重、许可和任务边界核对后运行。教师和学生只看同一可见前缀，使用软分类分布及表征蒸馏；不使用源安全标签或项目政策标签。它可以先执行，产物只是结构适配权重，不代表已经获得政策安审能力。
2. **最终安全对齐**：须有冻结的政策/来源标签映射、实体族隔离的校准/测试、独立人工金标及流式风险起点；[候选池](data/candidates/manifest.json)和[语料审计](data/audit_report.json)现有 `training_ready` / `training_gate.ready` 为 false，故这个阶段仍不可开始。

每次提交前用 `python3 check_l20.py` 查看实际调度余量，它分别报告 `can_submit_stage0_now` 和 `can_submit_final_alignment_now`。Job 显式申请一张 `nvidia.com/gpu`，固定 L20 节点池；训练脚本核查 CUDA 与设备名。已有活跃 Pod 的 GPU 不当作空闲卡。若某个 Running Pod 仅占卡、经实际负载检查确实未运行 GPU 任务，则依用户授权停止该空闲 Pod，保留其卷与其他活跃 Pod。

[L20 C1 功能探针](l20/c1_l20_results.json)验证了加载、缓存与合成反传。[1k pilot](l20/stage0_output/summary.json)在 L20 完成 127 optimizer steps，dev teacher-matching loss 1.7076→1.3448；其输出仍非安全对齐模型。[20k/2k 续训](l20/STAGE0_20K_REPORT.md)从 pilot adapter 接续，上传哈希锁定的 checkpoint/语料/脚本包，Job 完成后下载并核对 adapter、损失曲线与 summary：2,472 步，dev loss 1.3504→1.0523。训练完成后再做独立能力与速度对照。

正式效率验收须在同一 L20、同一请求到达回放、同一政策阈值与质量要求下测 **ITPS（完成分类的新增输入 token/s）**、每次分类与首次分类 P50/P95/P99、分类结果/s、排队和状态显存；不得报告生成 token TPS，因为该模型不生成 token。先用[直接分类接口](guard-classification-v1.schema.json)测 Stage0 C1，正式策略动作须等独立校准。不能用过去的 M5 数值当加速比。H1/H2 固定状态网络见[架构实验规格](FAST_STREAM_ARCHITECTURE.md)，当前还没有实现或速度结论。
