# T036 训练和评测提速

- 版本：v1（2026-09-26）
- 设计和执行：执行方（兼设计）。用户 2026-09-26：“训练效率能优化吗”“多打点日志看看怎么用满 L20 理论性能”“评测有没有把算力用满，第一种先做一下”。
- 代码：`round6/stage2/profile_train.py`、`batched_loss.py`、`train_distill.py` 的新参数；
  `round6/stage1_head/shard_eval.py`、`autotune_pin.py`、`autotune_l20_v1.json`、`eval_head_l20.py --shard / --autotune`。

## 训练

用剖析找出每步的时间去向（卡的上限、按批量扫描、真实训练组上的几种跑法、profiler 时间线、与现行训练步的梯度一致性），
把不改变训练结果的改动做进训练脚本：一组一次前向、不开梯度检查点、批量 loss（每步只同步一次）、fused AdamW。

## 评测

红线评测按线上流式方式一条一条跑，一个进程只用到约四分之一的卡。改成同一张卡上 N 个进程分片，合并回单进程的文件顺序。
验收：分片结果与单进程逐行相同。
