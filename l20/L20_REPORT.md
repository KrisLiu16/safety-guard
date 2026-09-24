# L20 单卡试跑结果

2026-09-23，集群 `cls-og1rjus2`。经用户明确授权，删除两个已申请整卡的 Pod：`naive-kg-guided-20260921` 和 `naive-structure-guided-20260921`。未删除其 PVC；两份 1000 GiB PVC 删除 Pod 后均仍为 Bound。之后在节点 `172.19.0.92` 创建 `safety-guard-l20-smoke-20260923` Job，单卡请求 1 张 L20，30 分钟截止，无重试，完成后 1 小时自动清理。

Job 于 `2026-09-23T03:08:23Z` 完成，状态 Succeeded。GPU 是 NVIDIA L20，PyTorch 2.7.1+cu128，物理显存约 44.4 GiB。运行包 SHA-256：`e40ef093ab0ea85fdee8b92ca8129774176bcc83a4f33242a47fef81fbc1d62c`。运行结果在 `results.json`。

| 检查 | 结果 |
| --- | ---: |
| M5 与 L20 强置信度参考样本判定 | 16/16 一致 |
| 最大分类概率差 | 0.001845 |
| 流式 KV 缓存相对完整计算最大概率差 | 0.0000269 |
| 策略梯度试跑 | 4 条训练样本、梯度有限、更新成功 |
| 1K 上下文 warm 单 token 缓存推理 P50 / P95 | 19.21 / 19.49 ms |
| 1K 上下文 warm 完整重算 P50 / P95 | 25.13 / 26.51 ms |
| CUDA 峰值显存分配 | 1.277 GiB |

该任务用于证明已有模型的 CUDA 推理、流式缓存和训练反向传播可运行。它没有开展新一轮完整 RL 训练，也没有测并发吞吐或真实端到端服务延迟。镜像首次拉取约 5 分 49 秒，Pod 内下载 PyPI wheel 约 90 KB/s，传输 1.2 GB 模型包也占了数分钟；这些准备成本不包含在 warm 单 token 延迟中。

收尾脚本最初试图在 Pod Succeeded 后使用 `kubectl exec` 取结果，被 Kubernetes 拒绝；Job 本身已经成功。已从完成 Pod 的日志保存 `results.json`，并修正 `launch.py` 为从日志读取结果。Job 完成后两台 L20 在 Kubernetes 的剩余 GPU 数均恢复为 1。
