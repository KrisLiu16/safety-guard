# Qwen3Guard 英文基准复现（提问 / 回答分类 F1）

目的：按 Qwen3Guard 技术报告英文能力柱状图的口径（allenai/safety-eval 的子集规则），给我们的流式审核模型打分，
把第五轮模型、第二阶段模型和论文里 Qwen3Guard-Stream-0.6B 的数字放在一起比。

## 文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `fetch.py` | Mac | 匿名下载公开数据（不登录、不接受条款、不用 token，先查 gated），版本钉死到 commit SHA；生成 `data/cases.jsonl`（`id, bench, level, messages, label`，1 = unsafe）和 `data/manifest.json`（每个 bench 的条数、跳过项、来源版本、文件 SHA256）。可重复运行，已下载且 SHA 对得上的文件直接复用 |
| `score_bench_l20.py` | L20 | 与 `stage2/score_texts_l20.py` 同一套模型代码和序列化，逐个 checkpoint 打分。每条输出 `end_cut / end_unsafe`（最后一条消息的最后一个 token）、`max_cut / max_unsafe`（该消息所有 token 的最大值）、`first_over`（第一个 cut > 0.5 的位置）。按长度分 microbatch（≤16,384 padded token、≤64 条）；单条超过 8,192 token 时从左边截上下文，最后一条消息不截，截断条数写进 `<output>.summary.json` |
| `metrics.py` | Mac | 按 bench × checkpoint × tau 计算 unsafe 类的 F1 / 精确率 / 召回，输出 markdown 表。提问侧用提问结束时的 cut 分数，回答侧用流式最大 cut 分数；`--paper` 加论文列，Avg 只算所有列都有的 bench |
| `test_bench_cpu.py` | Mac | CPU 单测：标签规则、指标计算、截断和分批 |

`data/` 在 `.gitignore` 里（`data/` 规则），分数文件是 `*.jsonl` 也被忽略。仓库是公开的，数据和分数都不提交。

## 子集规则与条数（2026-09-25 实跑）

| bench | 层级 | 来源与规则 | 条数（unsafe） |
|---|---|---|---|
| ToxicChat | 提问 | lmsys/toxic-chat `toxicchat0124` test，`human_annotation` 为真；unsafe = `toxicity == 1` | 2,853（362） |
| OpenAIMod | 提问 | mmathys/openai-moderation-api-evaluation；S/H/V/HR/SH/S3/H2/V2 任一为 1 | 1,680（522） |
| Aegis | 提问 | Aegis 1.0 test，`user_message`，文本 strip；3 个标注里至多 1 个 `Safe` 即 unsafe | 359（233） |
| Aegis2.0 | 提问 | Aegis 2.0 test，`prompt_label`（含 36 条原数据就写成 `REDACTED` 的提问，照留） | 1,964（1,059） |
| SimpleSafetyTests | 提问 | 全部 unsafe | 100（100） |
| HarmBench-P | 提问 | GitHub `harmbench_behaviors_text_test.csv`，standard + copyright | 239（239） |
| WildGuardTest-P | 提问 | **代理**：PolyGuardPrompts test 的 English 部分，`prompt_harm_label` 非空 | 1,699（754） |
| HarmBench-R | 回答 | GitHub `text_behaviors_val_set.json`；提问用 `test_case`（同 safety-eval），3 个人工标签多数 | 602（273） |
| SafeRLHF | 回答 | PKU-SafeRLHF 分支 v0 test；照 safety-eval 代码：只取两条回答安全性不同的行，`random.Random(42)` 抽 1,000 个行号，同一批行的安全回答和不安全回答各 1,000 | 2,000（1,000） |
| BeaverTails | 回答 | `30k_test`；unsafe = `not is_safe` | 3,021（1,733） |
| Aegis2.0-R | 回答 | Aegis 2.0 test，回答非空且有 `response_label` | 813（394） |
| WildGuardTest-R | 回答 | **代理**：PolyGuardPrompts English，`response_harm_label` 非空 | 1,709（284） |
| Think | 回答 | 本地 `round1/benchmark/thinking.jsonl`；去掉 `<think>` 标签，回答文本 = 推理 + 空行 + 最终回答（同 `guard_demo/server.py` 的 `guard_text`） | 1,059（569） |

合计 18,098 条（7,522 unsafe），约 340 万 token（Qwen3.5 分词），没有超过 8,192 token 的。

**跳过（gated，需要登录或接受条款）**：allenai/wildguardmix（WildGuardTest 原版，改用上面的代理）、
allenai/xstest-response（XSTest-Resp 不复现）、walledai/HarmBench（HarmBench-P 改用 GitHub 原始 CSV，内容相同）。
代理集的数字只能近似对照论文。

## 在集群上跑

1. 把 `fetch.py` 生成的 `data/cases.jsonl`（不进仓库）和 `score_bench_l20.py` 放到 PVC 的 `/work/round6/qwen3guard_bench/`。
   依赖 `/work/round6/probe/probe_common.py`、`/work/round6/stage2/train_stage2.py` 和 `/work/round4`（第二阶段作业已经有）。
2. 在 L20 作业里（环境同 `stage2/worker_stage2_v2.yaml`：`HF_HUB_OFFLINE=1`，单张 L20）先冒烟再全量：

```bash
S=/work/round6/qwen3guard_bench
/work/modern/bin/python -B $S/score_bench_l20.py $S/data/cases.jsonl --limit 256 \
  --checkpoint round5=/work/output/round4/window/best.safetensors \
  --output /work/output/round6/qwen3guard_bench_smoke.jsonl
/work/modern/bin/python -B $S/score_bench_l20.py $S/data/cases.jsonl \
  --checkpoint round5=/work/output/round4/window/best.safetensors \
               stage2_v1=/work/output/round6/stage2_train_v1/best.safetensors \
               stage2_v2=/work/output/round6/stage2_train_v2/best.safetensors \
  --output /work/output/round6/qwen3guard_bench_scores_v1.jsonl
```

3. 分数取回 Mac 后：

```bash
.venv/bin/python round6/qwen3guard_bench/metrics.py <scores.jsonl> --paper --tau 0.5 0.9
```
