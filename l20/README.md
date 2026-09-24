# L20 运行准备

集群 `cls-og1rjus2` 的两个独占 L20 节点分别是 `172.19.0.92`、`172.19.0.130`，每台 `nvidia.com/gpu=1`。2026-09-23 10:51（北京时间）检查时两卡分别被 `naive-kg-guided-20260921` 和 `naive-structure-guided-20260921` 的 Running Pod 申请，Kubernetes 可调度 GPU 为 0。两台 GPU 现场读取均为显存 0 MiB、利用率 0%、没有计算进程。两个内层 `sandbox` 容器分别于 10:49:14 和 10:49:09 退出，退出码 143，重启策略均为 `no`；`structure-guided` 当时仍有 `structure-web-relay` 在运行。此后用户明确授权清退这两个 Pod，现已删除；两份 PVC 均保留。

已完成本地准备：16 条按角色和安全标签各取 4 条的 M5 参考样本；`bundle.tar` 共 1,216,798,720 字节，SHA-256 为 `e40ef093ab0ea85fdee8b92ca8129774176bcc83a4f33242a47fef81fbc1d62c`；Python 编译检查、tar 校验和 Kubernetes 服务端 dry-run 均通过。此后的授权清退和 L20 试跑已完成，实际结果见 `L20_REPORT.md` 与 `results.json`。

这里提供了单卡、30 分钟上限的 CUDA 试跑。它加载上一轮选中的 LoRA 和风险头，在冻结的 16 个强置信度 dev 样本上核对 M5 概率与判定，检查流式 KV 缓存，测量 1K 上下文 warm 推理，并在训练集上执行一次仅用于验证的策略梯度更新；不会保存新的权重。这不是第二轮训练。输入 tar 只包含公开模型、本项目数据和代码，不包含云凭证。

本地准备：

```bash
/Users/liuzhihao/Work/safety-guard/.venv/bin/python /Users/liuzhihao/Work/safety-guard/l20/make_reference.py
/Users/liuzhihao/Work/safety-guard/.venv/bin/python /Users/liuzhihao/Work/safety-guard/l20/package.py
kubectl --kubeconfig=/Users/liuzhihao/.kube/cls-og1rjus2-private-latest -n default apply --dry-run=server -f /Users/liuzhihao/Work/safety-guard/l20/job.yaml
```

空闲 L20 出现后运行 `python launch.py`。它先核对 tar 的 SHA-256 和每台 L20 的剩余 GPU；若两张卡仍被占用，它会直接退出，不创建 Job。若有空卡，它创建并唤醒暂停的 Job，等 Pod 进入 Running，上传 tar 和校验文件，取得 `/work/output/results.json`。上传采用临时文件并在完成后原子改名，避免 Pod 读取半个 tar。Job 完成一小时后由 TTL 控制器清理；如试跑失败，查看其日志和事件后清理。不要把已有 L20 Pod 当作本任务资源。

镜像采用 PyTorch 官方的 `2.7.1-cuda12.8-cudnn9-runtime` 标签，容器内安装与上一轮相同的 Transformers、Tokenizer、Safetensors、psutil 版本。镜像能否从集群拉取、CUDA 训练算子及精确延迟必须由真正的 L20 运行验证；服务端 dry-run 只验证 Kubernetes 配置和创建权限。
