# T015 S5 安全一半放量（约 5,000 词）

- 版本：v1
- 目标：按 S5 安全一半 v2 提示词，为约 5,000 个词各写两段 500–1,500 token 的安全思考（一段针对原有的有害提问，一段针对正常提问），用 luna 上的 judge v3.2 复核，只保留复核全部判为 safe 的样本。设计见 [round6/s5_safe_v15/README.md](../../round6/s5_safe_v15/README.md)。
- 依赖：T012 全量预筛完成；T013 第 2 步已生成 `round6/s2_v15/batch_v2/seeds.jsonl`（S5 与 S2 的词不重叠）。
- 步骤：
  1. 单测：分别运行 `.venv/bin/python -m unittest round6/s5_safe_v15/test_s5_safe_cpu.py`（7 项）和 `.venv/bin/python -m unittest round6/test_shared_cpu.py`（6 项），全部通过。
  2. 生成任务：
     ```bash
     .venv/bin/python round6/s5_safe_v15/make_pilot.py --words 5000 --screen round6/word_screen_v1/full/extracted/screen.jsonl --exclude round6/s2_v15/pilot/seeds.jsonl --exclude round6/s5_safe_v15/pilot/seeds.jsonl --exclude round6/s2_v15/batch_v2/seeds.jsonl --words-per-task 10 --name-prefix s5safe-v2 --output round6/s5_safe_v15/batch_v2
     ```
     manifest 里记下 words、tasks、archives、excluded_by_screen、unscreened_words、lengths、shapes。
  3. 生成 Run：DeepSeek，`--flow round6/s5_safe_v15/flow`，尝试 1 次，并发 512（或 Task 数）；按份冻结后抽取：
     ```bash
     .venv/bin/python round6/s5_safe_v15/extract_s5.py <run_no> --out round6/s5_safe_v15/batch_v2/extracted --attempts-json round6/s5_safe_v15/batch_v2/attempts.json
     ```
  4. 复核 Run（长文本单条就大，打包器会按 180 KB 上限自动少装几个词）：
     ```bash
     .venv/bin/python round6/judge_v3/make_pilot.py --source round6/s5_safe_v15/batch_v2/extracted/examples.jsonl --words 0 --words-per-task 10 --name-prefix s5safe-v2-judge --output round6/s5_safe_v15/batch_v2/judge
     ```
     luna，`--flow round6/judge_v3/flow`；冻结后用 `judge_v3/extract_judgments.py` 抽取到 `batch_v2/judge/extracted`。
  5. 过滤：
     ```bash
     .venv/bin/python round6/verified_filter.py --examples round6/s5_safe_v15/batch_v2/extracted/examples.jsonl --items round6/s5_safe_v15/batch_v2/judge/items.jsonl --judgments round6/s5_safe_v15/batch_v2/judge/extracted/judgments.jsonl --out round6/s5_safe_v15/batch_v2/verified
     ```
  6. 审计：从 `verified.jsonl` 按固定种子随机抽 20 个词，读两段思考，重点看风险推理是否只概括请求、有没有写出有害细节或“先写后否定”，记下有问题的词数和类型（不写原文）。
  7. 正式放量的数据集**不删除**。
- 预期产物：同 T013 的对应文件（路径换成 `round6/s5_safe_v15/batch_v2/`）。
- 验收：生成 `states.complete` ≥ 80%；长度 p50 在各档目标附近（短约 800 字 / 450 词，中约 1,300 字 / 750 词，长约 1,900 字 / 1,100 词）；复核失败 ≤ 3%；kept 不少于完整样本的 90%；审计 20 个词里有问题的不超过 2 个。
- 需要用户决定：无（放量规则已定）。规模：生成约 5,000 次 DeepSeek 请求（长输出），复核约 2 万条条目。
- 反馈里必须报告：同 T013，另加 `length_units_by_language_style` 和 `quality_flags`。
