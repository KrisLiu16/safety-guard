# 第三轮：专用流式分类器

最新[完整小模型基座调研](SMALL_BACKBONE_SELECTION.md)：Qwen3-0.6B-Base 优先于 Llama-3.2-1B 满足当前预算与中文任务，Qwen3.5-0.8B-Base 作为混合主干新候选。该路线可省略从零预训练；这是研究建议，尚未新训练这些 Base 或证明它们优于现有 A0 Guard。

[自然预训练语料接入](data/pretrain_sources/README.md)已实际读取中文/英文各 10,000 篇，确认两套公开源可直接获取。尚待预处理及 token 打包，未开始 R24 训练；此阶段不依赖最终安全对齐标签。

用户接受从零预训练后，新增[纯循环 R24 预训练候选](PURE_RECURRENT_PRETRAIN_PLAN.md)：零 softmax 注意力，约 0.619B，部分层 MoE 预算约 0.817B。当前仅为设计与预算；A0 是现有基线，H24 混合方案保留为对照，未启动新的从零预训练。

[架构 v2](HYBRID_MOE_ARCHITECTURE_V2.md)将主线改为完整 24 层混合主干，并独立研究有限窗口注意力与部分层 MoE。已按官方 tensor header 核算约 0.753B / 0.951B 参数；尚未实现或训练，不沿用删层实验的加速结论。

[当前架构与下一步](CURRENT_ARCHITECTURE_AND_NEXT_STEPS.md)明确区分继承的 Qwen 基座、已训练的裁剪实验和尚未实现的固定状态主干。[A0 HTTP 实测](l20/A0_HTTP_ITPS_REPORT.md)已完成，当前串行 GPU 服务在并发增加时主要增加排队，动态批处理仍待实现。

[执行 DAG](EXECUTION_DAG.md)区分了已在 L20 启动的开放语料 Stage0 蒸馏和仍待政策金标的最终安全对齐；完整损失、奖励和评测方案见[架构与训练计划](ARCHITECTURE_TRAINING_PLAN.md)。[H1/H2 固定状态架构](FAST_STREAM_ARCHITECTURE.md)是后续效率实验，不把理论复杂度当作 L20 实测收益。所有训练与正式性能验收只用 L20，见[L20 执行边界](L20_TRAINING.md)。

当前 C1 的可运行[分类接口](guard-classification-v1.schema.json)与[流式推理实现](classifier_runtime.py)在输入 chunk 到达时直接返回风险等级和类别分布，不产生输出 token。[本机 HTTP 适配层](serve_classifier.py)提供完整输入与会话追加接口；Stage0 权重未校准，因此不产生自动拦截动作。L20 的[分类/ITPS 评测代码](benchmark_classification_cuda.py)比较 A0 与 C1，指标是新增输入 token/s、每次分类延迟和分类结果/s，而不是生成 TPS。

[接口示例与 ITPS 验收定义](CLASSIFICATION_AND_ITPS.md)区分真实新增输入 token、重算前向 token、分类结果/s 和含队列的客户端延迟；当前 L20 评测包见 `l20/bundle_classification_eval_manifest.json`。正式实测只在有空闲 L20 时执行，不占用正在计算的其他任务。

[首轮 L20 分类与 ITPS 报告](l20/L20_CLASSIFICATION_ITPS_REPORT.md)已完成：14 层 C1 约 2 倍输入速率，但独立官方子集的误报和漏判明显高于 A0，因此暂不作为安全服务模型。下一步优先做质量恢复，不以吞吐单项替代能力门槛。

[21 层结构诊断](l20/C21_SOURCE_PROBE_REPORT.md)在独立来源 valid 集上也未过质量门槛：直接裁剪即使保留最终层仍严重漏判。当前可运行的质量基线是完整 A0；H1/H2 等新主干需要继承语义并重新训练/评测，不能只凭理论状态复杂度替换 A0。

当前实测：C1 是从 A0 保留 14 层的 376,878,080 参数直接分类模型。[L20 功能探针](l20/c1_l20_results.json)通过缓存与合成反传；[Stage0 1k pilot](l20/stage0_output/summary.json)的教师匹配 dev loss 从 1.7076 降到 1.3448。[20k/2k L20 续训](l20/STAGE0_20K_REPORT.md)已完成 2,472 步，dev loss 1.3504→1.0523。Stage0 不消费项目安全标签，不把 loss 下降当成独立安审能力。第二轮 Luna 已切到[一词一请求、每词一中一英的 v12 格式](../round2/oneword2_v12/flow/pipeline.py)，不再沿用 v8 全量 Run。

只读检查资源和两个数据门槛：

```sh
python3 check_l20.py
```

可复核的其他产物：`data/audit_report.json` 记载源 revision、许可、重复；`data/stage0_distillation_20k/manifest.json` 固定扩大蒸馏集哈希；`l20/` 有 L20 Job、脚本、包清单和结果。`data/candidates/` 是保留源标签但尚无项目政策金标的候选池。最终安全对齐仍需独立人工校验与评测隔离。
