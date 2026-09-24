# Luna 固定词条数据生产

本轮：50 个固定词条 Task，Luna 每词一次生成 10 条（安全 5、不安全 5），目标 500 条；并发 5、重复 1、最多尝试 1。Aster Run 为 `aster-dev-253`。本流程使用已保存模型配置，不读取或导出 API key。

选择 25 个政治公共事务词条、20 个贪腐/司法词条、5 个民生词条；这是有意挑选的流程试跑，不代表整个词库分布。每词 task_key 由原词 SHA-256 稳定派生。别名归并需在正式 train/dev 划分前补齐，同一含义的变体不能跨集合。

## 文件与使用

- `seeds.jsonl`：50 个词及固定来源 revision。
- `lexicon_manifest.json`、`lexicon_raw/`：原始词库文件、校验值与许可副本。
- `tasks/<word-id>/instruction.md`：一个词一个固定 Task，只含来源与主题。
- `flow/settings.json`：本次每词目标与批大小。
- `flow/pipeline.py`：生成提示、严格 JSON schema、字段及位置校验。
- `flow/flow.py`：调用平台 Luna，每批保留 request/response、usage、校验记录和汇总；不自动补采样。
- `extract.py`：下载产物、核对 SHA-256、重新校验、去重并输出 JSONL；可重复执行，不重复请求模型。

```bash
python3 /Users/liuzhihao/Work/safety-guard/pilot/extract.py aster-dev-253
```

输出：

- `output/examples.jsonl`：格式和位置关系校验通过的完整样本。
- `output/rl_tasks.jsonl`：未触发已知质量疑点的合成 RL 候选任务，`observation` 与 `reward_spec` 分离。模型仅接收 observation，不把标签、解释、证据或生成 slots 拼入输入。没有触发规则不等于语义已获独立验证。
- `output/review_required.jsonl`：虚构上下文、词条实体被改称虚构、疑似输入标签冲突等需要复核的记录。此规则是保守筛选，不自动推翻标签。
- `output/rejected.jsonl`：不符合格式/前缀/词条约束及重复样本的记录。
- `output/summary.json`：任务数、样本数、标签/角色分布、token 用量和每词完成度。
- `output/artifacts/`：保存下来的原始请求、响应和校验产物。

JSON 示例结构：

```json
{
  "id": "word-id-0000",
  "origin_group_id": "word-id",
  "split": "unassigned",
  "observation": {
    "messages": [{"role": "user", "content": "正常的公共事务询问"}],
    "target_role": "user",
    "target_message_index": 0
  },
  "reward_spec": {
    "expected_label": "safe",
    "input_label": "safe",
    "policy_id": "benign",
    "policy_version": "guard-pilot-v1",
    "evidence_spans": [],
    "decidable_at_char": null,
    "label_origin": "luna_synthetic_unverified"
  }
}
```

这是可以直接被数据加载器解析的合成候选集，不是人工金标。结构 reward 仅表示格式合格；语义正确性与首次可判定点还需抽检。政治词命中、一般批评与合法讨论不自动等于违规。

## 扩展到每词 100 条以上

保持词条 Task 不变，在新 flow 快照中将 `examples_per_word` 设为 100 或所需的 10 的倍数，`examples_per_request` 保持 10。每词顺序发 10 次请求，索引 0–99，每批独立保存；增加每 task wall_seconds 以覆盖全部批次，再 `aster flows check`、`aster runs plan`。

每次完整运行仍应明确重复次数、并发和最大尝试次数。词条数量 × 每词样本数是目标样本数；词条数量 × ceil(每词样本数/批大小) 是生成请求次数，需据 pilot token 用量估算预算。失败批次不静默重试或自动更换模型。大规模运行不因修改本地 settings 而自动提交。

批间需要增加场景/长度/语言/表达方式配额与近重复过滤；仅增加索引不会自然带来更多多样性。未来可给每批提供已经接受例句的短摘要，避免重复，但生成摘要同样不能涉及评测集。

## 推理后端选择

首版采用 PyTorch + Transformers 自定义分类后端，直接使用模型 forward 和真正增量 KV Cache，先用 SDPA/兼容 FlashAttention 建立正确性与性能基线；随后加形状分桶、编译/CUDA Graph、动态 batching 和量化。固定分类头/流式读出需要专项适配，不默认普通聊天引擎即插即用。更换 vLLM/SGLang/TensorRT 后端的条件是实际支持该分类路径并在 L20 上测出质量约束下的净收益。

已经准备但未提交 `profiles/100-per-word/`，对应每词 100 条、50 词合计 5,000 条、500 次模型请求。该配置使用经过首批问题修订的 v2 提示和批次场景配额。可用 `prepare_profile.py --examples-per-word 100 --out 新目录` 再生成独立配置；脚本只写本地文件，不调用模型。
