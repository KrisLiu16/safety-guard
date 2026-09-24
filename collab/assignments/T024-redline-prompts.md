# T024 提问侧红线标注：试跑，再放量

- 版本：v2（试跑通过；裁判提问模式升到 user-v2，补了试跑核对里发现的三类边界；领导人称呼的判定改为以词表预筛为准）
- 目标：按用户 2026-09-24 定的提问侧分档，给提问打红线标签，供 T023 缓存上的用户头重训使用。
  - 分档：
    - 提问本身写出违规内容，或请求生成红线内容：风险；
    - 引用违规原话，或问违规词、违规口号的意思：有争议；
    - 只问红线话题，或请求辟谣（不引原话）：安全。
  - 裁判：`round6/redline_v1` 的提问模式（`make_tasks.py --target user`，judge `guard-judge-redline-user-v2`）。user-v2 只比 v1 多记三类边界：
    - B5：不露骨的针对未成年人的性诱导；
    - B6：枪支、爆炸物、管制刀具的买卖渠道；
    - B7：针对种族（不是国内民族）的仇恨。

    归不归红线由规则表开关决定，现在是设计方的建议值，等用户定。红线清单、边界代码、变体写法规则、领导人严格规则和词表预筛，都与回答侧相同。回答侧的裁判提示词逐字不变（单测检查），所以 T021 的兜底和补跑不受影响。
  - Run A 的提问编号是 `<task_key>:prompt:<safe|unsafe>`，与 T023 缓存的记录一致。
- 依赖：T021 的 Run 可以同时在跑（这里只是排队），建议高优先级。T012 全量抽取完成后，定标签时带上 `--screen`。
- 步骤：
  1. 单测：`.venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py`，全过。
  2. 导出 prefix_v2 的 user 角色记录（Mac，本地冻结副本，同 T020 第 2 步）：
     ```bash
     .venv/bin/python round6/redline_v1/export_prefix_v2.py --data-root round5/data/prefix_v2 --role user --output round6/redline_v1/input/prefix_v2_user.jsonl
     ```
     提交 `prefix_v2_user.manifest.json`（只有计数和 SHA），`.jsonl` 不提交。
  3. **试跑**：
     ```bash
     .venv/bin/python round6/redline_v1/make_tasks.py --target user --source round6/response_v14/batch_50k/extracted/trainable.jsonl --splits dev --count 200 --name-prefix redline-user-pilot-runA --output round6/redline_v1/user_pilot_runA
     .venv/bin/python round6/redline_v1/make_tasks.py --target user --source round6/redline_v1/input/prefix_v2_user.jsonl --splits dev --count 100 --name-prefix redline-user-pilot-pv2 --output round6/redline_v1/user_pilot_pv2
     ```
     - DeepSeek，`--flow round6/redline_v1/flow`，尝试 1 次，高优先级；
     - `extract_probes.py` 抽取；`failed_ids.txt` 交给 luna 兜底（`make_tasks.py --target user --sample-ids ...`）；
     - 定标签：`apply_policy.py <主> <兜底> --target user --out .../labels`。T012 已抽取完就加上 `--screen round6/word_screen_v1/full/extracted/screen.jsonl --source <同上 source>`（只对 Run A 有意义）；
     - 人工核对：`review_sample.py --target user --labels ... --source ... --count 40`（Run A）和 `--count 20`（prefix_v2），问题同 T020。
  4. 试跑数据集抽取完即删除（已定规则）。
  5. **放量**（用户确认规模后；用 user-v2，不用重做试跑）：
     - Run A 第一阶段输入的提问（`stage1_runA_v1.jsonl`，约 1.85 万条）；
     - Run A dev 的提问（`trainable.jsonl --splits dev`，约 2,600 条）；
     - prefix_v2 user 角色的全部记录（第 2 步导出）。

     每个 Task 放 40 条（提问短）。流程同 T021 第 3–8 步，标签输出到 `labels/user_runA_stage1`、`labels/user_runA_dev`、`labels/user_prefix_v2`，再拷到 PVC 的 `/work/round6/redline_v1/labels/` 下同名目录。
- 放量后的附加检查：Run A dev 和 prefix_v2 dev 的放量标签与试跑标签（v1）比较：
  ```bash
  compare_judges.py <试跑 labels.jsonl> <放量 labels.jsonl> --out ...
  ```
  报告 `label_agreement` 和 transitions。重叠部分只相差三类新边界和预筛降档。
- 预期产物：可提交各目录的 `manifest.json`、`run_no.txt`、`extracted*/summary.json`、`labels*/summary.json`、`dataset_rm.json`，以及 `prefix_v2_user.manifest.json`。
- 验收（试跑）：
  - 加上 luna 兜底后 `usable` ≥ 98%；
  - `nonmonotonic` ≤ 5%；
  - `normal` 分层里的非 safe 逐条列出（口径要求正常内容绝不能判错）；
  - 人工核对：剔除口径存疑后，标签对 ≥ 90%。
- 需要用户决定：
  - **放量规模**（第 5 步）：按试跑实测，39,444 条提问、988 个 Task，约 DeepSeek 4.8 万次、luna 1,600 次；
  - B5–B7 的归属（任务总表“待用户决定”）。只影响规则表开关，不影响放量本身。
- 反馈里必须报告：
  - 单测；
  - 两份 manifest 和 `prefix_v2_user.manifest.json` 的计数；
  - run_no；
  - 各 `extracted*/summary.json` 的 statuses 和 `judge_calls_per_response`；
  - `labels/summary.json` 的 `labels_by_split`、`old_to_new`、`stratum_by_label`、`rules_non_safe`、`boundary_hits_by_label`、`screen_overrides`；
  - 人工核对的计数和不写原文的理由；
  - 按实测重估的放量调用次数。
