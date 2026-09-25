# T031 领导人敏感度数据 v1：中英文对话，以思考过程结尾

- 版本：v1（2026-09-25）
- 设计和执行：执行方（兼设计）。
- 起因：用户在测试台上看到一段英文思考过程，复述了媒体报道过的“把领导人比作卡通形象”的对比，线上模型和第二阶段 v1 都没有截。
  - 线上模型最高 0.079，v1 第 1 轮最高 0.035。
  - 训练数据里提到领导人的英文回答，Run A 只有 24 条，所以模型对这类英文内容不敏感。
- 口径：见 [POLICY 第 9 节](../feedback/POLICY-redline-scope.md)。
- 代码在 `round6/leader_v1/`：`flow/pipeline.py`、`flow/flow.py`、`make_batch.py`、`extract.py`、`test_leader_cpu.py`。

## 内容

- 每个种子由程序指定：类型、领导人、语言（英文约 70%）、组织方式、长度。
- 模型为每个种子写 1–3 轮上下文，最后是用户的提问，再写助手回复前的思考过程。

| 类型 | 预期 | 内容 |
|---|---|---|
| satire_restated | 截 | 思考里点出媒体广泛报道过的嘲讽对比或绰号 |
| criticism_restated | 截 | 思考里概括批评者、学者对领导人的负面评价 |
| rumor_restated | 截 | 思考里复述媒体报道过的未经证实的传言 |
| harm_politics | 截 | 辱骂、骚扰、威胁的请求和政治事件或领导人绑在一起 |
| leader_news | 放行 | 正常新闻：出访、讲话、会见、政策 |
| cartoon_only | 放行 | 只谈卡通角色 |
| cartoon_china | 放行 | 卡通在中国的文化和商业情况，不谈政治 |
| insult_assistant | 放行（警觉） | 用户骂助手，思考保持冷静 |

- **只能引用媒体广泛报道过的内容**，不许编造关于真实人物的新指控、引语、事件或数字。
- **标签只看红线裁判**，“预期”一栏只用来抽查。
- 生成用 luna：DeepSeek 在政治类上多半不返回。每个 Task 只调一次，高优先级。
- 裁判也直接用 luna，走 `redline_v1` 的 flow；apply_policy 用政治专项词表和排除表。

## 步骤

1. **单测**：`round6/leader_v1/test_leader_cpu.py`，5 项。
2. **试点 200 个种子**：`make_batch.py --count 200 --output round6/leader_v1/pilot`。
   - 结果（aster-dev-370）：完成 176 个，跳过 24 个，失败 0 个。
   - 跳过的集中在 satire、rumor 两类，都是没有公开嘲讽或传言可说的领导人，luna 选择跳过，这是对的。
3. **试点判标签**：`redline_v1/make_tasks.py --source .../pilot/extracted/examples.jsonl`，在 luna 上跑，run aster-dev-371。
   - 核对“应截”的几类被判成有争议或风险的比例，“应放行”的几类被判成安全的比例。
4. **放量约 3,000 个种子**：`make_batch.py --count 3000 --offset 200 --leader-pools`。
   - satire、rumor 只从有公开案例的领导人里抽，其中卡通对比的那位权重最大。
   - 然后判标签，方法同第 3 步。
5. **重出标签**：用 apply_policy（v1.4：警觉档、政治组合）出 `labels_political_v2/leader_v1/`，交给 T030 训练。
6. **删除数据集**：生成和判标签的数据集，抽取完就删。

## 反馈里报告

- 各轮的 run_no 和 Task 数；
- `summary.json` 的 states 和按类型、语言的统计；
- 裁判结果按类型 × 标签交叉；
- 抽查的计数（不贴原文）。
