# T012 词表预筛 pilot：侮辱国家领导人的词

- 版本：v1
- 目标：验证 luna 能不能可靠地从词表里挑出对国家领导人的贬损说法，为 S2 和 S5 放量、Run A 训练集清理和 R 切片重判提供词级依据。设计说明见 [round6/word_screen_v1/README.md](../../round6/word_screen_v1/README.md)。
- 依赖：无。
- 输入：`round6/word_screen_v1/`；Mac 上的 Run A 和 Run B 种子（`round6/response_v14/batch_50k/seeds.jsonl`、`round6/response_v14/batch_rest/seeds.jsonl`）。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/word_screen_v1/test_word_screen_cpu.py`，6 项全过。
  2. 在 Mac 上写本地已知阳性清单 `round6/word_screen_v1/known_positives.txt`，一行一个词：T006 反馈第 4 节第 1、2 条的两个词，以及 T002 反馈第 5 节里那个因为侮辱国家领导人被改判的词。确认 `git status` 里看不到这个文件（已在 `.gitignore` 里），**不要提交**。
  3. 生成 pilot：
     ```bash
     .venv/bin/python round6/word_screen_v1/make_batch.py --pilot 1000 --include round6/word_screen_v1/known_positives.txt --output round6/word_screen_v1/pilot
     ```
     记下 manifest 里的 words、known_positives、tasks（应为 6 个 Task 左右）。
  4. Run：上传 `pilot/screen.tar.gz`，模型 luna `mdl_01M2W0QMTH0KFNJQFTBYY2QTVG`（responses），`--flow round6/word_screen_v1/flow`，尝试 1 次、重复 1 次，并发拉满。提交前 `aster runs plan`，提交后下载平台快照与本地 `flow/` 逐字比对。run_no 存到 `pilot/run_no.txt`。
  5. 抽取：`.venv/bin/python round6/word_screen_v1/extract_screen.py <run_no> --words round6/word_screen_v1/pilot/words.jsonl --out round6/word_screen_v1/pilot/extracted`。
  6. 人工核对：
     - 所有 yes 和 unsure 的词逐个看，记下误判（其实不是贬损领导人的）有几个；
     - 从政治类来源组（citizenlab 下的各组、`permissive/GFW补充词库.txt`、`permissive/网易前端过滤敏感词库.txt`）里判为 no 的词中随机看 100 个，记下漏判几个。

     词本身不写进反馈，只写数量和一句结论。
  7. 抽取完成后删除数据集（已定规则），回执存为 `pilot/dataset_rm.json`。
- 预期产物：可提交 `pilot/manifest.json`、`pilot/run_no.txt`、`pilot/extracted/summary.json`、`pilot/dataset_rm.json`。`words.jsonl`、`screen.jsonl`、`known_positives.txt`、归档都含词本身，不提交。
- 验收：
  - 已知阳性全部判为 yes 或 unsure；
  - 没拿到结论的词（missing）不超过 1%；
  - 人工核对：yes/unsure 中误判不超过 20%（宁可多筛，因为筛掉的词只是不进安全数据）；随机 100 个 no 中漏判不超过 2 个。
- 需要用户决定：无（并发拉满、尝试 1 次、pilot 数据集用完即删，均为已定规则）。
- 反馈里必须报告：
  1. 单测结果；manifest 的 words、known_positives、tasks；
  2. run_no；`extracted/summary.json` 里除 `by_source_group` 以外的全文，`by_source_group` 只列 yes 或 unsure 不为 0 的组；
  3. 第 6 步两项人工核对的数量和结论。
