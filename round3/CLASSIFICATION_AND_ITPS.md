# 输入即分类：接口与效率验收口径

当前 C1 是一个直接分类模型。完整输入 `POST /classify` 返回当前目标消息的分类；流式输入先 `POST /sessions`，每次 `POST /sessions/{id}/append` 到达文本 chunk 后，立即返回该**可见前缀**的 `safe / controversial / unsafe` 概率、最大概率等级和原版类别分布。最终 chunk 设置 `final=true`。服务不调用 `generate()`，无 LM 输出头，没有生成 token 或生成 TPS。[当前 JSON Schema](guard-classification-v1.schema.json)和[实现](classifier_runtime.py)是可执行契约；[HTTP 适配层](serve_classifier.py)默认只绑定 `127.0.0.1`。
可核对的[完整 JSON 示例](guard-classification-v1.example.json)是契约示例，不是模型实测输出。

当前类别头沿用 Qwen3Guard 原版：用户侧 9 个、助手侧 8 个互斥类别，以 softmax 给出主题/类别分布。`Political` 只是原版主题名，**不是不安全的证据或自动拦截理由**。首轮 L20 对照后，以完整 A0 作为当前可运行质量基线；其输出标为 `pretrained_baseline_unaligned`。Stage0 裁剪 adapter 只作结构实验，标为 `stage0_teacher_distilled_unaligned`；两者的概率均未用项目金标校准，`calibrated=false`，不含 `allow/hold/block`、证据充分或拒答分数。[未来策略输出 Schema](guard-decision-v1.schema.json)须等这些头真正训练并独立校准后才能启用。

L20 上启动本机服务后，完整请求与流式请求分别如下（会话 ID 取创建接口的返回值）：

```sh
python serve_classifier.py --model ../round1/export/base_model
curl -s http://127.0.0.1:8765/classify -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"请解释一份公开文件。"}]}'
curl -s http://127.0.0.1:8765/sessions -H 'Content-Type: application/json' -d '{"target_role":"user","history":[]}'
curl -s http://127.0.0.1:8765/sessions/SESSION_ID/append -H 'Content-Type: application/json' -d '{"chunk":"请解释","final":false}'
```

仅供实验对照的 14 层 C1 服务启动时另传 `--model checkpoints/c1_even14_init --adapter l20/stage0_20k_output/stage0_adapter.safetensors`；其质量未过线，不作为安全拦截服务。

## ITPS 与延迟

主吞吐指标 **ITPS = 测量窗口内完成分类的新增输入 token 数 / 墙钟秒数**。使用本分类器 tokenizer 计数：完整请求计其完整输入；流式固定 token-ID 回放计每次追加的 token，历史前缀不重复计。文本 HTTP 回放另核对最终消息的 token 增量，记录因 BPE 边界变化而重新前向的 `input_tokens_forwarded`，使缓存成本可见。不能用重新计算的 token 或生成端 TPS 抬高 ITPS。

推理内核有两条入口：同词表上游可调用 `append_token_ids(new_ids)`，只处理新增 token，时间复杂度不随既有前缀增长；普通 HTTP 文本流使用 `update_messages()`，会为正确处理跨 chunk 的 BPE 合并重新分词并回退改变的后缀。两条入口需分别测 ITPS，并在相同完整输入上验证末态分类一致；不能把精确 token-ID 快路的数字冒充任意文本 chunk 的服务吞吐。

同时报告：

| 指标 | 定义 |
|---|---|
| 首次分类延迟 | 第一个输入 chunk 到达到首个三档概率与类别返回，P50/P95/P99 |
| 每次追加分类延迟 | 后续 chunk 到达到新前缀分类返回，按 chunk 大小 1/8/16/32 与历史长度分层 |
| 完整输入分类延迟 | 单个完整请求到分类返回；完整输入 token 长度分桶 |
| 分类结果/s | 完成的分类响应数 / 墙钟秒数；与 ITPS 同时报 |
| 缓存重算率 | `sum(input_tokens_forwarded) / logical_new_input_tokens`，越接近 1 越好 |
| 状态成本 | GPU 权重、每会话 KV/递归状态、峰值显存与并发上限 |
| 质量 | 三档 macro-F1、Unsafe PR-AUC、固定误拦率的召回、校准误差；流式首次风险证据后的分类延迟 |

测量必须分层：L20 同卡的纯模型前向、含 tokenizer/JSON/本机 HTTP 的服务回放、再到真实网络和排队的外部请求。A0 与 C1 在相同输入、相同 chunk 切法、相同并发和质量阈值下对比；报告置信区间、热身与显存。纯模型基准用预先固定的 token 序列保证 ITPS 分母精确；HTTP 测试验证“输入后立刻有分类”的端到端路径。`benchmark_classification_cuda.py` 保存原始结果，不把小样单路峰值外推成生产吞吐。

当前实施边界：C1 的 KV 随历史增长，长会话/多并发成本仍需实测；[H1/H2 固定状态混合架构](FAST_STREAM_ARCHITECTURE.md)只是候选结构，只有安全质量维持且 L20 实测 ITPS 或决策延迟获益才替代 C1。最终安全对齐仍待政策金标和独立流式风险起点。
