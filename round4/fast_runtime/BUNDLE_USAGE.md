# Window CUDA Graph 文本流模型包

这是单独的低延迟交付，固定使用已训练的 window/W512 SFT 权重：

```text
bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2
```

不重新训练或改变权重，不替换质量优先的 full/RL 候选。包内 `MODEL_MANIFEST.json` 与 `BUNDLE_MANIFEST.json` 都必须声明 `inference_engine: window_cuda_graph`；此加载器拒绝其他架构、其他权重和不匹配的引擎身份。

包包含完整分类器权重、Qwen3.5 配置/分词器、依赖版本和所有运行代码。`window_attention.py`、`text_stream_runtime.py`、`graph_stream.py`、`graph_text_runtime.py` 都列入 SHA 清单，并从当前包的精确路径加载。运行时不读取原基座权重或旧 H24 检查点。

## 打包和运行

在已准备好的 L20 环境打包；打包命令本身不执行模型：

```bash
/work/modern/bin/python -B /work/round4/fast_runtime/package_model.py \
  --manifest /work/round4/fast_runtime/MODEL_MANIFEST.json \
  --model-assets /work/models/qwen35 \
  --window-code-dir /work/window \
  --output /work/output/fast_runtime/bundle
```

输出目录须不存在。默认在同一文件系统硬链接完整权重，否则复制；`--copy-weights` 始终复制。把包和源权重视为不可变文件。`--check-only` 只校验输入文件。

运行需恰好一个可见 L20、CUDA 12.8，以及包中精确的 `requirements.txt` 版本。参考镜像为 `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`，需 `gcc`、`libc6-dev`；`torch==2.7.1+cu128` wheel 可从 `https://download.pytorch.org/whl/cu128` 安装。CPU/MPS 模型执行关闭。

```bash
python -B /path/to/bundle/run_guard.py --verify-only
python -B /path/to/bundle/run_guard.py --input requests.jsonl --output predictions.jsonl

# 同一包、同一权重的显式 eager 对照；不会捕获 CUDA Graph。
python -B /path/to/bundle/run_guard.py --inference-engine eager \
  --input requests.jsonl --output eager_predictions.jsonl
```

默认在接收用户请求之前用独立合成输入准备长度 1..32 的 32 个精确 CUDA Graph。已满 512-token 窗口的短追加走对应图；未满窗口的早期前缀和大块 prefill 走原 eager 路径。图捕获或回放失败会报错，不静默切换引擎。`whole` 始终使用同一权重做完整 eager 分类；只需整段分类时可显式选 eager，省去图初始化。

## 输入和输出

`requests.jsonl` 每行一个 JSON 对象：

```jsonl
{"id":"whole","op":"whole","language":"zh","messages":[{"role":"user","content":"请解释公开资料中这个词的历史背景。"}]}
{"op":"begin_message","session":"demo","role":"user","language":"zh","text":"请解释"}
{"op":"append_text","session":"demo","text":"公开资料中这个词的历史背景。"}
{"op":"begin_message","session":"demo","role":"assistant","language":"zh","text":"可以从公开史料出发，"}
{"op":"append_text","session":"demo","text":"区分事实记录与个人评价。"}
{"op":"end_session","session":"demo"}
```

这段短示例只演示接口，未达到图路径所需的 512-token 历史。`begin_message` 在同一个会话内增加消息，不清除历史。每次追加重分词，BPE 尾部变化时恢复完整 GDN/注意力状态快照并重放；返回的会话状态不与图 arena 或其他会话共享。超出 8192 tokens 的输入拒绝且不截断。默认最多四个活跃会话，`end_session` 释放该会话缓存；引擎的 32 份共享图 arena 留到进程关闭，供其他会话复用。

输出是风险概率与目标角色，始终 `generated_tokens=0`。每次请求保留 `native_tokens`、`net_new_tokens`、`forward_tokens`、`replay_tokens`；`engine_execution` 另给出实际 `graph_calls`、`graph_forward_tokens`、`eager_calls`、`eager_forward_tokens`。后两组之和须分别等于本次真实前向次数和 token 数。BPE 合并可令净新增 token 为负或零；回放不算新输入。

启动信息单列在首个 metadata 事件的 `engine_initialization`：`startup_seconds`、`capture_seconds`、`auxiliary_input_tokens`、`captured_chunk_lengths`、`shared_graph_arena_storage_bytes`，以及分配/保留显存增量。合成 seed 和捕获准备只属于启动开销，不进入用户请求 ITPS、forward、replay 计数。会话延迟包含分词、缓存复制、回放及 CPU 可见结果；首次加载/捕获时间另外报告，不能把图的纯 token-ID 实验提速直接当成文本接口提速。

语言显式指定。新消息未给 `language` 会清除上一条的语言；追加默认延用当前消息语言。仅有匹配语言/角色的既有校准阈值时输出二分类判决。缺少 en/user 校准时仍给概率，判决为 null，不借用助手阈值。`--threshold` 是明确未校准的覆盖值。整段阈值用于中间前缀时标注尚未验证，不能据此自动放行内容。

## 验收边界

独立验证使用 `python -I -B` 新进程、HF 离线及旧路径访问审计，验证完整权重与旧 loader 的概率、参数 dtype、RoPE buffer，以及真实 BPE 回滚、跨角色、会话结束和实际图调用。所有新增 runtime 的源码来源必须在包内。文件校验与打包成功不代替 L20 数值、状态隔离和实际文本性能验收；扩展到全部 1..32 长度的结果必须以本快包的验证产物为准。

本轮有硬标签和质量验证的是 safe/unsafe。保留的 controversial 概率不等于第三档已经独立训练，类别头不输出；研究模型包不构成生产效果保证。
