# M5 本地安审模型实验

本轮执行和验收已完成。结果见 `ROUND1_REPORT.md`，逐项证据见 `completion_audit.json`。阶段日志保留在 `local_stages.jsonl`，不自动启动第二轮。

环境：Python 3.12.14、PyTorch 2.14.0、Transformers 4.55.0，使用项目独立 `.venv` 和 MPS。依赖版本见 `requirements.lock.txt`。模型与官方数据均固定 revision，下载文件哈希见 `asset_hashes.json`。

## 推理演示

训练及导出完成后，运行以下命令。输入输出均为 JSONL；流式输入每次传入截至当前的完整 messages，runtime 只计算变化的 token 后缀。每个 session 有独立 KV；角色切换可以复用历史，末尾分词变化会正确回退。

```sh
/Users/liuzhihao/Work/safety-guard/.venv/bin/python /Users/liuzhihao/Work/safety-guard/round1/export/run_guard.py --mode batch < /Users/liuzhihao/Work/safety-guard/round1/export/demo_batch.jsonl

/Users/liuzhihao/Work/safety-guard/.venv/bin/python /Users/liuzhihao/Work/safety-guard/round1/export/run_guard.py --mode stream < /Users/liuzhihao/Work/safety-guard/round1/export/demo_stream.jsonl
```

默认使用未合并 adapter，以保留开发集选择的训练数值路径；加 `--merged` 可测试合并版。两者已分别校准和评测。批量输入：`{"conversations": [[{"role":"user","content":"文本"}]]}`。

流式输入：`{"session":"会话标识","messages":[...],"final":false}`。最后一次设 `final:true`。`reset:true` 开始新会话。结果包含概率、`release/hold/block`、已审查/已释放字符位置以及本次可释放文本。默认保留 32 个字符的输出缓冲；争议、异常及上下文超长进入 hold，风险命中后该消息保持 block。已释放文本不能撤回或改写。这个演示不是生产服务，尚无网络端点、排队器或跨进程 session 管理。

完整消息限 8,192 token，超长明确报错。模型语义上采用本轮有害行为政策；敏感词是生成主题，普通政治讨论不自动等于风险。

## 训练与验证实现

- `runtime.py`：保留原始分类头，批量/增量缓存、末 8 层 q/v LoRA rank 8、FP32 可训练参数和合并。
- `smoke.py`：16 条 CPU/MPS 对齐、缓存/消息追加、padding、两套风险头与 LoRA 的真实反向更新。
- `prepare_data.py`、`apply_review.py`：固定词族、防跨组关键词引用、近重复筛选、有限语义抽检和前缀修正。
- `train.py`：只向模型传入对话 token；奖励标签留在 loss 侧。每例 4 个分类动作、policy gradient、固定参考 KL，无生成式 SFT loss。训练文件已有时禁止不经检查重启覆盖。
- `evaluate.py`：中文独立校准及测试、官方三分片全量分类/位置指标。`check_official_metrics.py` 用真实预测逐项对齐保存的官方代码。
- `performance.py`：同步设备计时、热身、原包装器/增量缓存、chunk 与 batch 比较。缓存加速不能算作新注意力结构的收益。
- `export.py`：导出 adapter、风险头和合并模型；重新加载检查。通过本地 `runtime.load(model_path=...)` 明确恢复 FP32 风险头，避免载入全局 BF16 时丢失精度。
- `compare.py`：同一冻结测试集对比、词族 bootstrap 区间和最终报告。

## 数据和运行记录

`aster-dev-254` 是本轮 Luna Run，50 个词任务，并发 50、每词最多 1 次尝试；未自动补跑失败任务。原始请求和响应在 `generated/artifacts/`，提取器使用 Aster CLI 并验证 SHA256。模型凭证留在平台。

`data/manifest.json` 记录实际划分和哈希。完整中文数据为 Luna 初标；本次 Codex 抽检不是人工金标。旧 pilot 不参与本轮冻结测试。官方 benchmark 只用于评测，不进入训练或奖励。三份官方分片有重叠，不相加为独立样本数。

本轮一次训练，至多 300 次更新/一次数据遍历/2 小时活跃训练，先到者停止。超内存、swap 增长和可观测热限速会触发停止并保存最近检查点。开发集选择 checkpoint，校准集定阈值，测试集只作最终比较。任何续训或第二轮须另记实验，不覆盖本轮证据。
