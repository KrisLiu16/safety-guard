# 追加文本的流式分类会话

`text_stream_runtime.py` 在已加载的 `Classifier` 和原生 tokenizer 上增加会话状态；不改模型权重、训练数据或原有 CLI。真实模型适配器只允许一张可见 L20。CPU 测试使用假的分词器、缓存和分类函数，不运行神经网络。

```python
from text_stream_runtime import TextStreamRuntime, audit_cache_tensor_isolation

# model 已加载 full/window/memory 或 RL 导出权重，并配置相应结构；
# tokenizer 是该模型的原生 tokenizer。
model.eval().requires_grad_(False)
runtime = TextStreamRuntime(model, tokenizer, chunk_tokens=32, prefill_chunk_tokens=2048)
session = runtime.new_session()
session.begin_message("user")
result = session.append_text("请解释公开资料中的这个词")
result = session.append_text("，以及它的上下文。")
result = session.begin_message("assistant", "这个词在不同上下文中有不同含义。")
print(result["target_role"], result["risk_probabilities"])

# 仅在 L20 检查时调用，检查 current cache 与快照、不同会话之间
# 没有共享可变 tensor storage，不在每次推理的热路径调用。
print(audit_cache_tensor_isolation(session))
```

`begin_message(role, text="")` 只接受 `user`、`assistant`；`append_text(text)` 只追加当前消息。最后消息决定风险读出角色，不支持修改历史消息。每次成功调用只在全部必要 token 前向完成后返回一次分类结果。输入、分词、缓存复制、前向或概率检查出错时，不提交任何候选会话状态；超出 **8,192 tokens** 在复制缓存和前向之前拒绝，不截断。

每次分词前检查原生 tokenizer 及 HF 后端是否已启用截断，启用时直接拒绝，请调用方先关闭。HF `encode` 明确传入 `truncation=False`；原生 Tokenizers 不接受该参数，使用其原生签名。分词结果若含 `Encoding.overflowing` 也会拒绝，不能把首段当作完整输入。截断/overflow 错误不修改已有 messages、IDs、缓存或快照，也不执行模型前向。

每次追加都按训练的 `ROLE:\ncontent` 序列化整个会话并完整重分词。若新 token IDs 保留旧 IDs 前缀，则复制当前完整缓存后仅前向新 IDs。若 BPE 改变旧尾部，则计算最长公共前缀，从不超过该位置的最新快照恢复并重放；没有合适快照时从零重放。不会只裁剪注意力 KV，因为 GDN、记忆矩阵和归一化状态也必须恢复。

快照位置按 64 tokens 对齐，只保留当前序列末尾最近两个位置。事务预先计算这两个最终位置，已有且仍有效的快照可复用；仅在需要新快照的最终位置切开前向并深拷贝，不为之后必被淘汰的中间边界创建快照。剩余前向超过 128 tokens 时使用可配的 `prefill_chunk_tokens`（默认 2,048），较短追加/尾部使用 `chunk_tokens`（默认 32），两者都会准确停在需要保留的快照位置。长前缀与深回放不会因 64-token 对齐而退化成大量小步调用。

BPE 合并如果使序列跨过 64-token 边界回缩，新的两个目标位置可能包含已经淘汰的旧快照。例如 192→191 tokens 会把目标从 `[128,192]` 改成 `[64,128]`。运行时先检查缺失的目标位置，再选择足够早的恢复源；必要时从零重放来重建 64，已有且仍正确的 128 可保留。这个额外回放保证之后继续追加时快照仍完整，不能仅因为公共前缀达到 190 就直接从 128 恢复。

为保障异常事务性，纯追加也要复制当前缓存；因此缓存复制和全量重分词都是实际开销。极端重分词回滚会全量重放。窗口/记忆模型的单份状态随窗口固定，两个快照和事务工作副本也有固定数量；**完整注意力模型的单份 KV 仍随历史增长**。消息文本与 token IDs 随输入增长，受 8,192 上限限制。缓存复制期间还有临时副本，不能把单份缓存大小当作整个运行时峰值显存。

输出 `net_new_tokens` 是新旧序列长度差，BPE 合并时可为 0 或负数；`forward_tokens` 是实际重新前向的 token 数。`replay_tokens` 统计回滚后重新处理、位置仍落在旧长度内的 token，包含该区间的分词变化；`replayed_unchanged_prefix_tokens` 只统计不变前缀的重放。`native_tokens` 是当前完整会话 token 数，`input_characters` 是所有消息内容的 Unicode 码点数，不含角色分隔符；`serialized_characters` 包含分隔符。

`tokenization_seconds`、`cache_clone_seconds`、`model_call_seconds` 和 `wall_seconds` 保留各类开销。正式 ITPS 应用最终会话 `native_tokens` 除以包含全部 append、分词、恢复与复制的外部墙钟时间，另报告 `forward_tokens`/`replay_tokens` 的额外工作；不能把每步可能为负的净 token 数当作生成 TPS。会话累计时间只包含成功调用，正式压测的外部时间还应计入失败、排队和传输。

第三档风险及类别本轮没有独立监督/验收；这里只输出三档风险概率，不输出类别，不默认套用阈值。中间文本分类结果不证明风险证据已经可判，更不代表安全前缀可以立即对外放行。该层没有 HTTP、动态批处理、跨进程会话持久化；共享模型适配器用锁串行执行前向。

## 验证

CPU 状态机测试：

```sh
python3 -B /Users/liuzhihao/Work/safety-guard/round4/test_text_stream_runtime.py
```

L20 后置验证应使用每份实际导出检查点，固定 tokenizer 和模型结构，不同切分汇合到完全相同的训练序列化内容：

1. 中英、emoji、换行混合文本，逐 Unicode 码点追加，以及循环 7/1/19/3/64 字符切分；记录实际出现的 BPE 回滚，不预设每个测试都会发生回滚。
2. 长度超过 W512，建议在 513、1,025、4,097、8,192 附近核对；将每个最终会话结果与同一完整 token 序列的一次性分类比较，两个角色分别覆盖。阈值沿用数值审计误差门槛，误差和失败原样保存。
3. user→assistant→user，多轮消息头也参与重新分词；每步只比较最后消息对应的分类头。空追加和无 token 变化必须不增加 forward 数。
4. 两个会话交错追加，分别与独立运行及完整前向比较；调用 `audit_cache_tensor_isolation(session_a, session_b)` 验证 current/snapshot 的 PyTorch 底层 storage 没有交叉别名。
5. 用包装 runner 在真正更新候选 cache 后注入一次异常，确认既有 messages、IDs、snapshot positions 和下一次成功输出保持一致；8,192 上限拒绝也应不改变会话。
6. 检查窗口/记忆模型长序列的单份 cache 和快照空间上界，并分别报告总显存与复制时间。完整注意力只要求数值正确，不要求其 KV 空间与历史长度无关。

CPU 测试只能证明状态机规则；真实 `DynamicCache` 深拷贝、GDN/记忆状态恢复的数值正确性和运行性能必须由这些 L20 检查确认。
