# Round4 分类推理入口

[infer_classifier.py](infer_classifier.py) 加载实际导出的权重，直接输出风险概率，不生成文本 token。模型只能在 **一张可见的 CUDA L20** 上运行，CPU/MPS 没有降级路径。所有候选都使用 Qwen3.5-0.8B 的原生 tokenizer 和完整 24 层文本主干；`window`、`memory` 固定部署窗口 W512。

## 输入与运行

将输入保存为 UTF-8 JSON（也可直接使用消息数组）：

```json
{
  "messages": [
    {"role": "user", "content": "请解释公开资料中的这个词是什么意思。"},
    {"role": "assistant", "content": "可以，我会根据上下文说明它的常见含义。"}
  ]
}
```

最后一条消息必须是 `user` 或 `assistant`，它决定 `target_role`。输入按训练时的 `USER:\n内容\n\nASSISTANT:\n内容` 格式序列化，调用原生 tokenizer 的 `encode(..., add_special_tokens=False)`；不使用 chat template，不增加特殊 token，不删改内容，不静默截断。默认最多 8,192 个输入 token，超限报错。较长长度可显式设置 `--max-input-tokens`，但不因此获得更长输入的能力验证。

在现有 L20 容器中运行（Python 使用已安装依赖的 `/work/modern` 环境）：

```sh
/work/modern/bin/python /work/round4/infer_classifier.py \
  --variant memory \
  --checkpoint /work/output/round4/memory/best.safetensors \
  --base-root /work \
  --messages /work/request.json
```

精确 token-ID 分块，每新增 8 个输入 token 输出一次概率并复用缓存：

```sh
/work/modern/bin/python /work/round4/infer_classifier.py \
  --variant memory \
  --checkpoint /work/output/round4/memory/best.safetensors \
  --messages /work/request.json \
  --chunk-tokens 8
```

`--variant full` / `window` 对应各自目录的 `best.safetensors`。分类 RL 权重使用 `--variant classification_rl --checkpoint /work/output/round4/classification_rl/best.safetensors`，从同目录 `summary.json` 读取实际结构并核对该导出权重 SHA-256。没有 summary 时必须显式提供 `--rl-base-variant full|window|memory`。各候选不能复用对方的缓存。

## 概率与阈值

标准输出是 JSONL：一条 `metadata`、每个输入分块一条 `classification`、最后一条 `complete`。每次输出 `user` 和 `assistant` 两个读出的三档概率，顺序为 `safe, unsafe, controversial`。只有最后消息对应的 `target_role` 用于本次判决，另一角色读出作为诊断信息；模型不生成回复。输入原文不回显。

默认只输出概率，`target_binary_decision` 为 `null`。需要二分类判决时，可给出显式 `--threshold 0.5`，该阈值会标记为**未校准**；也可读取相应检查点的已有校准结果：

```sh
/work/modern/bin/python /work/round4/infer_classifier.py \
  --variant memory \
  --checkpoint /work/output/round4/memory/best.safetensors \
  --messages /work/request.json \
  --language zh \
  --calibration-metrics /work/output/round4/memory/exported_dev_metrics.json
```

入口只读取报告 `strata[语言/目标角色].threshold_from_calibration`，不会用开发或测试标签重算阈值。`sealed_test_metrics.json` 如果具备同一结构也可读取。**必须传入与检查点对应的报告**；报告本身没有检查点哈希，入口会明确标记无法独立核验两者的对应关系。缺目标分组会报错，尤其英文用户不能借用英文助手或中文用户阈值。中间前缀沿用该完整输入阈值不代表前缀放行策略已验证。

第三档 `controversial` 本轮没有独立硬标签监督，不能宣称已验收三档风险能力。类别头本轮没有类别监督/验证，入口不输出类别。类别头参数仍属于导出结构，严格加载可以防止无声缺失。系统消息可序列化，但不是独立验收过的部署输入分组。

## 本地依赖与支持边界

- `/work/models/qwen35`：本轮锁定基座的本地权重、配置和 tokenizer；`--base-root` 指其上两级运行根目录，入口不下载模型。
- `/work/input`：`train_base.py`、`experiment_common.py`、`speed_probe.py`，复用 `load` 和 `Classifier`，可用 `--base-code-dir` 改路径。
- `/work/window`：`window_attention.py`、`run_probe.py`，复用 W512 缓存与短 GDN 推理核，可用 `--window-code-dir` 改路径。
- `/work/round4/memory_attention.py`：记忆候选的新增结构，默认从入口同目录加载，也可用 `--memory-code-dir` 指定。
- 已验证环境版本：PyTorch 2.7.1+cu128、Transformers 5.17.0、FLA 0.5.2、safetensors 0.8.0；内核首次编译还依赖现有容器的编译工具链。入口不安装依赖。

加载不依赖训练数据、初始 H24 检查点、初始 H24 summary 或它的哈希。`--help` 和纯 Python 的输入/序列化函数可在 CPU 检查；任何模型执行均有 L20 硬件检查。

`--chunk-tokens` 先将完整消息一次分词，再切分这组**确定的 token ID**；它不是逐段接收任意文本的服务接口，尚不支持 BPE 后缀重分词、GDN 状态回滚、跨进程会话持久化、HTTP 或动态批处理。中间输出不证明安全前缀可以立即放行。完整输入与分块路径需要在实际导出权重上分别验证。

输出的 `observed_native_itps` 只计新输入 token，分母包含模型、两个角色分类读出和 CPU 可见概率，包含首个冷启动前向，排除加载、分词及 JSON 写出。它用于单次运行观察，不能替代预热后的正式 ITPS、P95 延迟与吞吐评测。
