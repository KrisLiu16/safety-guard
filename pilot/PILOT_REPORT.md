# Luna 词库例句试跑报告

日期：2026-09-22。Aster Run：aster-dev-253（run_01M353MXQ938K0SG0GXPRW4Z33）。

**结果**

| 项目 | 实测 |
|---|---:|
| 固定词条 task | 50 |
| 成功 / 失败 task | 50 / 0 |
| 模型生成响应 | 50 |
| 原始例句 | 500 |
| 结构校验通过 | 459（91.8%） |
| 结构拒收 | 41 |
| 触发语义复核规则 | 107 |
| 未触发规则的 RL 候选 | 352 |
| 输入 token | 70,017 |
| 输出 token | 76,141 |
| 合计 token | 146,158 |
| 单次请求耗时中位数 / 最大值 | 65.55s / 85.53s |

单次请求时间包括上游等待与网络，不是 Guard 模型推理延迟。并发上限配置为 5，每词重复 1、最多尝试 1；未重跑或自动补齐失败样本。

**产物**

- [全部结构合格例句](output/examples.jsonl)
- [RL 候选任务](output/rl_tasks.jsonl)
- [语义待复核](output/review_required.jsonl)
- [结构拒收记录](output/rejected.jsonl)
- [完整统计](output/summary.json)
- [提取脚本](extract.py)
- [生成及使用说明](README.md)

RL 候选标签分布：{"safe": 228, "unsafe": 124}。筛选后不再严格五五开，训练采样需重新平衡，不自动补造记录。每条 observation 与 reward_spec 分离，标签、解释和证据不能泄漏进模型输入。

**质量发现**

结构问题计数（同一例可能有多项）：{"message_roles": 17, "missing_seed_word": 1, "evidence_after_prefix": 13, "prefix": 6, "evidence_span": 4, "slot:input_label": 1, "unsafe_policy": 2}。

语义复核触发计数：{"fiction_context_requires_semantic_review": 102, "seed_entity_recast_as_fictional": 17, "safe_input_label_has_risk_cues": 1}。主要问题是虚构上下文与现实意图混合、把真实姓名加“虚构人物”前缀，以及少数安全输入标签与文本意图不一致。规则仅触发复核，不证明所有这些标签都是错的。

本轮做了有限的助手抽看，未做独立人工全面标注；352 条也是合成初标候选，不能直接称为人工金标。可直接加载到 RL 开发数据管线，但奖励可靠性仍需抽样确认。模板化的“请求伪造/威胁/泄露隐私”占比明显，尚未覆盖完整的政治公共事务政策边界或高级多轮绕过。

**扩量准备**

已经准备 [每词 100 条的 flow 配置](profiles/100-per-word/settings.json)，未提交。50 个词扩到每词 100 条将是 5,000 条目标、500 次生成请求。按本次体量线性估计约 1,461,580 token（约 150 万），实际随长度、推理用量、场景变化而变；没有已核实单价，因此不提供虚构费用。

下一版提示已增加禁止把真实姓名伪装成虚构人物的规则、输入标签约束和批次场景配额；需要下一次实际生成才能知道合格率是否提高。保留一词一个稳定 task，不把前缀、同义词和模板变体跨 train/dev/test 拆分。

训练计划已更新为 [v0.3](../TRAINING_PLAN.md)。推理后端首选 PyTorch + Transformers 自定义增量分类服务，后续优化按 L20 实测决定；本轮没有训练 Guard，也没有证明 GPU 可调度。
