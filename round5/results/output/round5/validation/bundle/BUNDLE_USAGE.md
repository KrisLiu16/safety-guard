# Round5 直接分类推理包

本包格式为 `round5-prefix-classifier-bundle-v1`，与 round4 发布包分开。完整保留 24 层 W512 backbone，读取自己的完整 checkpoint、配置和 tokenizer；不读取基座模型权重、旧初始 checkpoint、训练集或外部训练代码。神经执行只允许一张 CUDA L20。所有输出都是风险分类概率，生成 token 数为 0。

训练尚未完成时不会生成候选 manifest。完成固定 4164 次 SFT 和 256 次 RL 后，使用真实导出的权重 SHA 和已保存的 cal/dev 选择记录准备身份：

```sh
/work/modern/bin/python /work/round5/runtime/prepare_manifest.py \
  --summary /work/output/round5/prefix_v2/summary.json \
  --canonical-calibration /work/output/round5/canonical32_calibration/selected/calibration.json \
  --output /work/output/round5/validation/MODEL_MANIFEST.json
/work/modern/bin/python /work/round5/runtime/package_model.py \
  --manifest /work/output/round5/validation/MODEL_MANIFEST.json \
  --output /work/output/round5/validation/bundle
```

没有原训练 eligible 改进时保留初始 window 权重。原训练晋升状态单独保留；canonical dev 的各 stream 组 FPR≤5% 且 whole recall 不低于同 canonical 初始模型 2pp 才通过服务验收。服务门槛失败时保存 `research_unvalidated_runtime` 研究产物；服务门槛通过而原权重回退时为 `fallback_not_promoted`，不能视为训练晋升。manifest 与包均包含已完成训练的 summary 证据及哈希；不读取 fresh/test 分数选模型。无生产批准，第三风险档与 category 能力未获本轮验收。

唯一执行定义是 `canonical32-v1`，默认 eager。whole 与每次追加均从 token 0 的固定 32-token 块历史出发；未满块补有效 vocab pad，仅读真实位置，dummy 后的缓存绝不提交。服务阈值必须在固定权重选出之后，重新用 canonical 路径的 calibration 数据拟合；旧 bulk 的 `best_metrics` 只证明原权重选模，不能直接作为 serving 阈值。whole 与 begin/append 分别读取 canonical artifact 的 whole/stream 阈值，比较都为严格 **`p_unsafe > threshold`**。恰好等于阈值时返回 safe。缺失语言或缺失语言/角色校准（如 en/user）返回概率及空 binary decision，不借用其他组阈值。`--threshold` 是明确标注未校准的临时覆盖。

stream 阈值来自预声明的 native target token 与最多八个重新分词文本切点上的 episode 最大值校准。API 每次返回当前目标角色概率及当前判定；不承诺任意 BPE 到达顺序的 FPR，不执行跨请求的自动停止策略，也不保证预测尚未出现的风险证据。

```sh
python /path/to/bundle/run_guard.py --bundle /path/to/bundle --verify-only
/work/modern/bin/python /path/to/bundle/run_guard.py \
  --bundle /path/to/bundle --input requests.jsonl --output results.jsonl
```

`--verify-only` 可在 CPU 上检查所有文件、完整权重 SHA、版本锁和选择证据，不运行模型；实际加载还需匹配 `requirements.txt` 对应库版本。

JSONL 请求格式不改动 v12 训练数据 schema：

```json
{"op":"whole","language":"zh","messages":[{"role":"user","content":"请解释公开资料中证据与观点的区别。"}]}
{"op":"begin_message","session":"s1","role":"user","language":"zh","text":"请解释"}
{"op":"append_text","session":"s1","text":"公开资料。"}
{"op":"begin_message","session":"s1","role":"assistant","language":"zh","text":"可以区分事实与观点。"}
{"op":"end_session","session":"s1"}
```

序列化与训练一致：大写角色、`:\n`、原文本，消息间 `\n\n`，不额外添加 token。每次追加重新分词，使用最长公共 token 前缀恢复 cache，必要时回放；只提交 floor(n/32)×32 个完整 token 的 aligned cache，另留前两份 32 对齐快照；未满块在下次输入从其 anchor 重算。固定 8192 上限，超长及隐式截断直接拒绝，失败不提交会话状态。净新增 token 可因 BPE 合并为负；ITPS 分子是净新增输入，实际 forward/replay tokens 另报，不能当生成 TPS。

`--inference-engine window_cuda_graph` 显式选择同一 canonical32 定义的图实现：前缀 <512 时固定32 eager，≥512 时单个32图；启动 seed 也从 token0 逐32计算，捕获/预热与请求计数分开。两个角色全部32位置均计算分类头，partial 不拷出无效未来 cache。旧 bulk 与普通可变分块历史存在已测偏差，所以旧 whole-state 失败留存为诊断，不能扩容差当通过。本轮新图实现仍需真实 core audit 与独立包验收，CPU通过不代表L20通过。

独立 L20 包审计命令：

```sh
/work/modern/bin/python /work/round5/runtime/test_bundle_l20.py \
  --manifest /work/output/round5/validation/MODEL_MANIFEST.json \
  --bundle /work/output/round5/validation/bundle \
  --output /work/output/round5/validation/bundle_audit.json
```

它在父进程生成六组完整输入的旧加载器参考，释放模型后用 `python -I -B` 子进程只从 bundle 加载，核对完整结构、RoPE、概率、真实 BPE 收缩、双阈值来源、角色切换、会话隔离、越界失败事务及缺失阈值。子进程禁用网络并审计旧模型/训练路径访问；这是 Python 层审计而非 OS 沙箱。独立加载器同 canonical32 的概率容差 0.005；同运行时 canonical whole/append 采用 1e-5，核心算法审计为概率1e-6、BF16状态精确和FP32 atol/rtol1e-6，不据结果放宽。旧 bulk 概率单独记录，绝不作为新定义的通过依据。

`download_release.py` 保留 CPU artifact Pod 下载、部分下载续传及完整 bundle 哈希验证，并使用独立 `release_round5` 默认目的地。示例：`python download_release.py --pod <CPU artifact pod> --metadata <collected bundle metadata>`。运行下载器需要原有 kubectl 配置；它不会创建或占用 GPU。源码及 CPU 测试完成不等于本轮真实 L20 审计通过。

在校准之前先运行 `test_canonical_l20.py --checkpoint <fixed path> --checkpoint-sha256 <verified SHA> --training-summary <completed summary> --output <new audit.json>`；不依赖bundle或服务阈值。该脚本通过有界子进程检查all1..31真实长度的未来pad独立性、全prefix双头输出、eager/graph完整状态、8192、BPE/多会话/失败事务，以及128次实际文本追加的原始延迟与净输入/重放/padding计数。随后顺序执行canonical calibration、manifest、package、独立-I包审计及使用锁定阈值的最终测试。
