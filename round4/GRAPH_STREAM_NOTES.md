# 有限 CUDA Graph 原型

该原型仅支持 **window、W512、batch=1、固定 chunk=8、已填满窗口、默认 RoPE**。full、memory、变长 chunk、padding、文本重分词和生产服务均不在范围内。没有改动训练、权重、已有 loader 或文本流入口。

`graph_stream.py` 为每个输入显式刷新固定地址的 token IDs 与 position IDs。捕获内把新注意力 K/V 写回原地址；GDN 的全部卷积及递归状态保持完整。Python `seen` 在每次 replay 后推进一次，不能依赖捕获时的 Python 赋值。预热和捕获会执行真实前向，因此接收正式调用前恢复种子缓存和逻辑位置。

外部会话状态通过加锁的 copy-in/copy-out 与图内 arena 隔离；返回缓存不与 arena 共用张量存储。接口返回两个角色头的 CPU 可见概率，没有自回归 token 输出。与 eager 比较时包含状态复制、输入/位置更新和 CPU 输出成本；不把仅 `replay()` 的内核时间冒充接口吞吐。

验证顺序是：捕获后完整种子恢复、连续不同输入块、回滚快照后分叉、交替两个会话及隔离、相同轨迹完整预热、eager/graph 同条件 ITPS 和 P95。概率绝对容差固定为 0.001；BF16 状态固定 `atol=0.02, rtol=0.01`，FP32 状态固定 `atol=0.002, rtol=0.001`。逐类报告实际误差；不因失败放宽阈值。

L20 运行：

```bash
/work/modern/bin/python /work/validation/test_graph_stream_l20.py
```

默认权重为 `/work/output/round4/window/best.safetensors`。外层进程强制限制独立 worker 最多 900 秒；捕获不支持、正确性失败或超时均写入 `/work/output/round4_validation/graph_stream_audit.json`，并保留旁边的 `.log`。退出码 0 只表示产出了实验报告，必须读取 `pass`。该报告不会替换已训练权重或自动修改推理入口；即使数值通过，也可能没有加速。

本地只执行以下非神经检查：

```bash
python3 -B safety-guard/round4/test_graph_stream_l20.py --cpu-check-only
```

CPU 检查覆盖语法、固定参数拒绝、ID 校验和生命周期。它不证明实际 FLA/CUDA Graph 捕获兼容性；这一点保留给 L20 实测。
