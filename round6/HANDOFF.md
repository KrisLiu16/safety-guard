# 第六轮交接文档

2026-09-24。写给接手第六轮的人。先读本文件，再读 [README.md](README.md)（第六轮计划与用户决定）和 [DATA_SPEC_v15.md](DATA_SPEC_v15.md)（数据规格）。更早的背景见仓库根目录的 `TRAINING_PLAN.md` 和 [第五轮复盘](../round5/DIAGNOSIS.md)。

## 1. 项目与硬规则

流式安全分类器：Qwen3.5-0.8B-Base 改造，24 层（18 层 GDN + 6 层窗口 512 注意力），约 0.753B，无 LM 头，每个角色（user / assistant）一个头：512 维投影 → 风险 3 类（safe / controversial / unsafe）+ 类别（提问侧 9 类、回答侧 8 类，与 Qwen3Guard 一致）。当前固定权重是第四轮 window/SFT 初始权重，SHA `bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2`，状态 research_unvalidated_runtime。

以下规则来自用户，一直有效：

- 只做分类器，指标看 ITPS 和延迟，模型不超过 1B。
- **训练和推理只在腾讯云 L20 上做**；Mac 只做 CPU 上的数据准备和分析。
- 按真实风险指标（误报、漏报）选模。官方 Qwen3GuardTest 和 fresh390 只做观察，不用于拟合或选模。
- 已冻结的格式在一次运行中途不改；新版本用新目录。
- **Aster 的采样次数、并发、尝试次数由用户决定**，提交前要问。
- GPU Pod 确认空闲后才能停，PVC 保留。
- 小规模 pilot 先行，再放量。
- 凭证只写存放位置，不写内容，也不在命令行和日志里出现。

## 2. 用户已定的决定

| 项 | 决定 |
|---|---|
| 逐位一致 | 放弃“整段与流式逐位一致”，只保留流式一条路径 |
| 速度目标 | 单次分类 P95 ≤ 20 ms 的前提下，单张 L20 的 ITPS 最大化 |
| RL | 停止分类 RL；改用代价敏感 CE、稠密前缀监督、安全流最大风险分损失、流式/端点双头 |
| 裁判模型 | DeepSeek（Aster 模型配置 `deepseek/deepseek-flash`，走 tokenhub）。生成和裁判不用同一个模型 |
| 造数模型 | 以后可以全量切到 DeepSeek，DeepSeek 失败或拦截的由 luna（`gpt-5.6-luna`）兜底 |
| 政治口径 | 按国内监管口径打标 |
| 类别体系 | 与 Qwen3Guard 一致，另加一个国内专用头（GB/T 45654—2025 附录 A 的 29 种风险，去掉 A.5） |
| 数据规格 | 同意 [DATA_SPEC_v15](DATA_SPEC_v15.md) 的标签体系、去掉 A.5、各切片规模 |
| Run B | 原 Run B（v14 格式其余约 29 万词）**暂不提交**；v15 各切片替代它 |

还没定的：思考过程（thinking）是否需要审核。这决定 S5 长流切片做多大。我的看法是：模型不需要会思考，但如果线上要审核模型输出的思考文本，思考文本就是审核对象，需要这类数据；如果线上只审最终回答，S5 可以很小。

## 3. 当前进度

### 已完成

| 工作 | 结果 | 文件 |
|---|---|---|
| 第五轮复盘 | 流式误报主要来自提问风险泄漏到回答判定；第五轮门槛差异大多是阈值落点噪声 | [../round5/DIAGNOSIS.md](../round5/DIAGNOSIS.md) |
| 第 1 步：决策规则校准（不训练） | 官方 thinking 误报从 89.8% 降到 23.9%（logit 滑动平均 α=0.1，温度 T=2.2，阈值只在校准集拟合），召回 72.1%；A0 为 22.2% / 81.5%。剩下的差距在模型本身：会把“复述有害请求的安全推理”判成高风险 | [decision_rule/RESULTS.md](decision_rule/RESULTS.md) |
| v13 → v14 造数格式 | 一词一请求，2×2（有害/正常提问 × 安全/不安全回答），不安全回答带逐字起点引用，程序转字符位置；拒收否定式起点，标记危害评论和“虚构”字样 | [response_v13/](response_v13/)、[response_v14/](response_v14/)、两份 PILOT_REVIEW.md |
| 裁判 pilot | mimo（aster-dev-277）和 DeepSeek（aster-dev-278）各判 438 条。DeepSeek 与预期一致率：安全回答 100%，起点前前缀 86.7%，含起点前缀 89.2%，整条不安全回答 100%，请求失败 24 条。选定 DeepSeek | [judge_v1/](judge_v1/)、[judge_v2/](judge_v2/) |
| DeepSeek 造数 pilot | aster-dev-279，60 词：53 词 4 条全合格，7 词因输出被截断（finish_reason=length）失败，之后 max_tokens 调到 16000 | [response_v14_ds/](response_v14_ds/) |
| 服务引擎 v3 | 槽位状态池 + 分桶 CUDA Graph + 自写 GDN Triton 内核 + 分组窗口注意力。模拟真实到达速度：64 个会话 P95 12.9 ms（约 2,600 ITPS），128 个会话 P95 31.4 ms | [serving/RESULTS.md](serving/RESULTS.md) |
| 数据规格 v15 | 标签体系、国内专用头、差距分析、切片 S1–S5 + R | [DATA_SPEC_v15.md](DATA_SPEC_v15.md) |

所有 pilot 用的测试数据集都已在 Aster 上删除（各目录下的 `dataset_rm.json` 是删除回执）。

### 进行中（2026-09-24 13:10 北京时间的状态）

| 工作 | 状态 | 接手要做的 |
|---|---|---|
| Run A `aster-dev-276`（v14，5 万词，luna，352 个 Task，平台上的并发设置为 400） | 307 个成功，45 个在跑，0 失败 | 跑完后抽取，见第 5 节命令 |
| Run B 草稿上传（数据集版本 `dsv_01M38NHHQF0DQZ1MESYZC4WBTR`，1,943 个压缩包） | 在用户的 Mac 上后台运行 `response_v14/resume_upload_only.sh`，已传到第 1,180 个，日志 `response_v14/batch_rest/upload.log` | **只上传，不提交**。是否提交或删除这个草稿由用户决定 |

## 4. 待办（按优先级）

1. **Run A 抽取**：跑完后用 `response_v14/extract_archives.py` 下载并在本地重新校验。训练只用 4 条回答全合格的词，并剔除“要求先铺垫但起点为 0”的回答（见 [response_v14/BATCH_PLAN.md](response_v14/BATCH_PLAN.md)）。
2. **R：Run A 按国内口径重新裁判**。现有 v14 提示词写的是“政治讨论、批评属正常内容”，与新口径冲突，政治类样本的标签要重新判。裁判用 DeepSeek，DeepSeek 拦截的政治类交给 luna。同时补上 `gbt_codes` 和 `qwen_category` 字段。可以从 [judge_v2/flow/](judge_v2/flow/) 改出新版本（新目录，不改旧的）。
3. **线性探针**：冻结主干，在 Run A 上检验现有特征能否区分“复述有害请求”和“真有害”（README 第 5 步）。
4. **决策规则重拟合**：用 Run A 的校准 split 重新拟合温度和阈值（README 第 6 步）。
5. **v15 造数切片**（见 DATA_SPEC_v15 第 4 节，每块先 pilot 再放量，并发由用户定）：
   - **S1 类别均衡 2×2：尚未开始，也没有生成指令。** 本轮的 AI 助手（Claude）不编写这部分“成批生成各类违规内容”的指令，需要由团队成员自己编写并发起生成。下游环节可以直接复用：v14 的程序校验（`response_v14/flow/pipeline.py` 里的 `align_quote`、否定式起点、拒答开头等检查）、DeepSeek 的 chat_completions 适配（`response_v14_ds/flow/chat_adapter.py`）、抽取脚本和按词族切分的做法。
   - S2 非拒答难负例（国标 B.1.3 所列的制度、民族、历史、英烈、性别、健康等方面，话题敏感但内容正当），约 2 万条。目前误报最多、数据最缺的就是这一块，建议最先做。
   - S3 争议档约 1 万条；S4 提问侧对抗形式约 2 万条；S5 长流规模待用户回答思考过程的问题后再定。
6. **服务引擎继续提速**（[serving/RESULTS.md](serving/RESULTS.md) 第“下一步”节）：GDN 状态改 BF16、注意力环形缓冲原地读取、权重 FP8、A0 放到同一引擎对比。
7. **第六轮训练**：稠密前缀 + 最大风险分损失 + 流式/端点双头 + 代价敏感 CE，加上国内专用头（29 维 sigmoid）。数据和探针结果出来后，训练前再请用户确认。

## 5. 操作手册

### Aster

- 先 `aster whoami`；未登录时在 Web「配置 → 访问令牌」签发，自己在终端运行 `aster login`。模型 API key 只从环境变量 `ASTER_MODEL_API_KEY` 读，不写命令行、不打印。
- 写 flow 前先读 `aster docs show flow-authoring`。flow 引用辅助模块时 `--flow` 要传**目录**，提交后用 `aster runs flow <run> --out <dir>` 下载平台快照，与本地逐字比对（`response_v14/submit_batch.py` 已经这样做）。
- 提交前先 `aster runs plan`。测试或小批量用完的数据集要 `aster datasets rm`。
- 出现 `_notice.update` 时运行 `aster update`。
- 模型配置：luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（responses 接口）；DeepSeek `mdl_01M38TDJK6CAG1XK5BAMY14N2X`（`deepseek/deepseek-flash`，chat_completions 接口）；mimo `mdl_01M33V8C39GGNX33TFS5ETDNR3`（只做过裁判对照）。放置位置 `sg-public`。
- 单个 Task 的 instruction 实测 196 KB 能传、208 KB 被拒，所以 v14 每个 Task 最多 150 个词。

Run A 抽取（`--expected-terms` 是 Run A 的词数）：

```bash
.venv/bin/python round6/response_v14/extract_archives.py aster-dev-276 --out round6/response_v14/batch_50k/extracted --expected-terms 50000
```

### L20

- 集群 kubeconfig：`~/.kube/cls-og1rjus2-private-latest`；kubectl 在 `~/.local/bin/kubectl`。
- 节点 `172.19.1.144`，PVC `safety-guard-base-compare-data`（挂在 `/work`）。
- 第五轮模型包：`/work/output/round5/validation/bundle`；官方 thinking 数据：`/work/validation/official_adapted_v1/thinking.jsonl`。第六轮输出在 `/work/output/round6/`。
- 作业写法参照 `round6/serving/bench_worker_v3.yaml`、`round6/decision_rule/worker_v1.yaml`：镜像 `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`，脚本和结果都经 PVC 交换，跑完写 `*_done` 和 `*_exit_code.txt`。
- 腾讯云凭证在 `~/.config/tencent-cloud/tencent-test.env`（权限 600）。不要读出、粘贴或提交它。

## 6. 仓库里没有的内容

仓库是公开的，下列内容没有提交，只在原处保存：

- 生成的样本原文：各 pilot 的 `extracted/` 下的复核文本（`review.txt`、`unsafe_review.txt`、`failures.txt`）、`selected_attempts.json`、`attempts.json`、`samples_*.json`。
- 上传回执里带样本内容的 `dataset_add.json`、`dataset_publish.json`，以及请求示例 `example_request.json`。
- `.gitignore` 原有规则排除的：词种子 `seeds.jsonl`、Task 压缩包 `archives/*.tar.gz`、所有 `*.jsonl` 和日志。

这些在用户 Mac 的 `/Users/liuzhihao/Work/safety-guard/` 下原位保存，也可以从 Aster 上对应的 Run 和数据集重新下载。
