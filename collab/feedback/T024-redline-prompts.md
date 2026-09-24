# T024 反馈：提问侧红线标注试跑

- 卡片版本：v1（a2e8407）做了试跑；v2（6481dc4，2cbf8fa）做放量
- 状态：试跑完成，验收通过。**放量已提交（00:05），正在跑。**
- 执行时间：2026-09-24 23:40 – 00:05（北京时间）

## 单测

`.venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py`：28 项全过。

## 第 2 步：prefix_v2 user 角色导出

`round6/redline_v1/input/prefix_v2_user.manifest.json`（已提交，`.jsonl` 不提交）：

- records 18,322，output_sha256 `a5d9fecc51bf7f27…`
- train 17,622（safe 9,848 / unsafe 7,774），calibration 300（150 / 150），dev 400（200 / 200）

## 第 3 步：试跑

| 目录 | 来源 | 条数 | Task | Run |
|---|---|---|---|---|
| `user_pilot_runA` | Run A dev | 200（safe 100 / unsafe 100；中文 100 / 英文 100） | 20 | aster-dev-339 |
| `user_pilot_runA_fb` | 上面失败的 11 条 | 11 | 2 | aster-dev-342（luna） |
| `user_pilot_pv2` | prefix_v2 dev | 100（safe 50 / unsafe 50） | 10 | aster-dev-340 |
| `user_pilot_pv2_fb` | 上面失败的 2 条 | 2 | 1 | aster-dev-341（luna） |

- judge `guard-judge-redline-user-v1`，policy_digest `9b2667295fadc6f2`。
- 主跑用 DeepSeek，兜底用 luna；`--flow round6/redline_v1/flow`，尝试 1 次，并发 = Task 数，高优先级，SDK 4.3。
- 4 个 Run 的 flow 快照都与本地逐字一致，全部 Task 完成，0 失败。
- 4 个数据集抽取后都已删除，回执在各目录的 `dataset_rm.json`。

### 抽取（`extracted/summary.json`）

| | responses | safe | located | nonmonotonic | judge_error | 每条调用（均值 / 最大） |
|---|---|---|---|---|---|---|
| Run A 主跑 | 200 | 158 | 31 | 0 | 11 | 1.12 / 3 |
| Run A 兜底 | 11 | 8 | 3 | 0 | 0 | 1.09 / 2 |
| prefix_v2 主跑 | 100 | 86 | 12 | 2 | 0 | 1.34 / 7 |
| prefix_v2 兜底 | 2 | 2 | 0 | 0 | 0 | 1.00 / 1 |

Run A 主跑的 11 条 judge_error 都是 DeepSeek 的临时错误：输出被截断（`stop_reason:length`），或者返回空内容（`JSONDecodeError`）。

### 标签（`labels/summary.json`）

命令：`apply_policy.py <主> <兜底> --target user --out .../labels`。Run A 加了 `--screen round6/word_screen_v1/full/extracted/screen.jsonl --source round6/response_v14/batch_50k/extracted/trainable.jsonl`，T012 此时已经抽取完。

| | usable | safe | controversial | unsafe | nonmonotonic |
|---|---|---|---|---|---|
| Run A | 200 / 200（100%） | 166 | 12 | 22 | 0 |
| prefix_v2 | 100 / 100（100%） | 88 | 1 | 11 | 0（主跑的 2 条由兜底补上） |

- `old_to_new`：
  - Run A：safe→safe 83、safe→controversial 12、safe→unsafe 5、unsafe→unsafe 17、unsafe→safe 83。
  - prefix_v2：safe→safe 50、unsafe→unsafe 11、unsafe→controversial 1、unsafe→safe 38。
- `stratum_by_label`：
  - Run A：controversial 全部是 redline（12），unsafe 全部是 redline（22）；safe 里 normal 51、non_redline_harm 90、redline_topic 18、other_sensitive 7。
  - prefix_v2：controversial 是 redline 1，unsafe 是 redline 11；safe 里 normal 44、non_redline_harm 39、other_sensitive 3、redline_topic 2。
  - **两份的 `normal` 分层都没有非 safe**，没有要逐条列出的。
- `rules_non_safe`：
  - Run A：leader_epithet_written 5，screen:insult 2，screen:rumor 1。其余是提问侧的动作：`requests` 共 12 条（R2、R6、R7、R10、R12、B2），`term_explained` 共 11 条（R3、R5、R6、R8、R9、R11、B1），`writes` 共 7 条（R2、R8、R10、B1），`quotes` 1 条（R4）。
  - prefix_v2：`requests` 共 9 条（R2、R7、R12、R13、B1、B2），`writes` 2 条（R8、R13），`term_explained` 1 条（R11）。
- `boundary_hits_by_label`：
  - Run A：B4:requests 在 safe 里 10 条、unsafe 里 1 条；B1:term_explained 在 controversial 里 2 条；B2:requests 在 unsafe 里 2 条；B1:writes 在 unsafe 里 1 条。
  - prefix_v2：B2:requests 2 条、B1:requests 1 条，都在 unsafe；B1:topic_mention 1 条在 safe。
- `switch_sensitivity`：
  - Run A：B4 改为 redline 时，safe→unsafe 11；term_explained 改为 safe 或 unsafe 时，各有 11 条变。
  - prefix_v2：B2 改为 non_redline 时，unsafe→safe 2。
- `screen_overrides`：Run A 有 insult:unsafe→unsafe 2、rumor:unsafe→unsafe 1。这 3 条裁判本来就判了 unsafe，预筛没有改变标签。

### 人工核对

用 `review_sample.py --target user` 抽样，Run A 40 条、prefix_v2 20 条，按 POLICY 和已定的提问侧分档逐条判断。核对表只留在本地（`review/` 已被 gitignore 忽略）。

| | 对 | 误判红线 | 漏判红线 | 档位错 | 口径存疑 | 剔除存疑后的正确率 |
|---|---|---|---|---|---|---|
| Run A | 34 | 1 | 0 | 0 | 5 | 34 / 35 = **97.1%** |
| prefix_v2 | 17 | 0 | 0 | 0 | 3 | 17 / 17 = **100%** |

起点分句：
- Run A：17 条非 safe 且标签正确的，起点都对。
- prefix_v2：6 条非 safe 里 5 条对，1 条偏晚。那条是请求为煽动暴力的角色写对话，要看完整段才判出来。

不写原文，只说理由：

- **误判红线 1 条（Run A）**：
  - 裁判判了 `leader_epithet_written`，但这个词在 T012 的三遍结果里都是 `no`。
  - 按 POLICY 8e，领导人名字的变体算不算侮辱，由词表预筛决定，所以这条应该是 safe。
- **口径存疑，Run A 5 条：**
  - 预筛为 `unsure` 的领导人名字变体 1 条：裁判判 unsafe。
  - 演员名加“种子”的色情资源词 1 条：判 R11 term_explained。词本身不露骨，算不算红线词不确定。
  - 网络评论里对评论员的贬称 1 条：判 R5 term_explained。算不算有损国家荣誉不确定。
  - 针对亚裔同学组织骚扰 1 条：判 R8 unsafe。种族（不是国内民族）算不算 R8 不确定。
  - 讨论气枪出售广告措辞 1 条：判 safe。枪支买卖渠道没有像 B1、B3 那样单独列出。
- **口径存疑，prefix_v2 3 条：**
  - 问一个露骨性俚语的意思 1 条：判 R11 term_explained，是否算露骨不确定。
  - 隐晦地索要种族蔑称 1 条：判 safe。种族算不算 R8 不确定，与上面亚裔那条口径相反。
  - 诱导未成年人的对话请求（提到交换有伤风化的物品、安排见面，但不露骨）1 条：判 safe。按“红线以外的有害内容标 safe”是对的，但这是未成年人性诱导，建议请用户明确一下。

## 给设计方的问题

1. **领导人规则与词表预筛怎么配合。** 现在 apply_policy 只会按预筛**往上**改成 unsafe；裁判给出的 `leader_epithet_written`，即使预筛判 `no` 也照样保留。按 POLICY 8e，名字变体应由预筛决定。
   - T021 的 Run A dev 回答里有 70 条带 `leader_epithet_written`。按样本主词的预筛结果分：insult 27、rumor 12、unsure 16、evasion 6、no 9。
   - 其中除 `leader_epithet_written` 外没有别的红线规则的：no 4、unsure 10。
   - 回答里被点名的称呼不一定就是样本的主词，所以这些只是上限。要不要让预筛为 `no` 时降档，请设计方决定。执行方没有改任何标签。
2. **问领导人侮辱称呼是什么意思，该标哪一档。** 提问侧已定的分档是“问违规词的意思：有争议”。但是 `leader_epithet_written` 和预筛改档都会把它标成 unsafe。Run A 里有 3 条是这种情况，两档都截断，只影响训练时的类别。
3. **种族（非国内民族）仇恨算不算 R8。** 试跑里一条判了 unsafe，一条判了 safe，口径不一致。
4. **未成年人性诱导（不露骨）** 目前按“红线外”标 safe，是否要请用户确认。

## 放量估算（第 5 步，需要用户确认）

按 `--per-task 40` 在本地试生成（没有上传）：

| 部分 | 条数 | Task |
|---|---|---|
| Run A 第一阶段提问（`stage1_runA_v1.jsonl`） | 18,528（train 16,000 / calibration 2,528） | 464 |
| Run A dev 提问 | 2,594 | 65 |
| prefix_v2 user 全部 | 18,322 | 459 |
| **合计** | **39,444** | **988** |

按试跑实测的每条调用次数估算：

- **DeepSeek：** Run A 21,122 × 1.12 ≈ 2.37 万次，prefix_v2 18,322 × 1.34 ≈ 2.46 万次，**合计约 4.8 万次**。
- **luna 兜底：** Run A 的失败率约 5.5%，prefix_v2 约 2%，**约 1,600 次**。
- **时间：** 每个 Run 都不超过 2,000 个 Task。参照 T021 高优先级时 3.7 万条回答约 45 分钟跑完，提问更短，预计三部分 1 小时内跑完，再加兜底和抽取。

## 第 5 步：放量（v2，00:05 提交）

- 用户已确认规模（2cbf8fa 记录；用户在执行方对话里说“来新的活了”）。B5–B7 先按设计方的建议执行：B5、B6 算红线，B7 不算。
- 单测：28 项全过（6481dc4 之后重跑）。
- `make_tasks.py --target user --per-task 40`：

| 目录 | 来源 | 条数 | Task | judge | policy_digest | Run |
|---|---|---|---|---|---|---|
| `user_full_runA_stage1` | `stage1_runA_v1.jsonl` | 18,528 | 464 | guard-judge-redline-user-v2 | bd5c444c00dd4b7a | aster-dev-344 |
| `user_full_runA_dev` | `trainable.jsonl --splits dev` | 2,594 | 65 | 同上 | 同上 | aster-dev-345 |
| `user_full_pv2` | `prefix_v2_user.jsonl` | 18,322 | 459 | 同上 | 同上 | aster-dev-346 |

- DeepSeek，`--flow round6/redline_v1/flow`，尝试 1 次，并发 = Task 数（都不超过 512），高优先级，SDK 4.3。
- `runs plan` 没有 blocking，三个 Run 的平台快照都与本地逐字一致。
- 接下来：冻结 attempt 列表 → 抽取 → luna 兜底 → 定标签（Run A 两份带 `--screen`）→ 拷到 PVC → 删除数据集 → 与试跑标签比较（`compare_judges.py`）。

### 放量抽取（01:08 完成）

先冻结 attempt 列表，再用 `extract_probes.py --attempts-json` 抽取到各目录的 `extracted_ds/`。三个 Run 都是 completed、0 失败。

| 份 | Run | responses | safe | located | nonmonotonic | judge_error | 每条调用（均值 / 最大） | 交给 luna |
|---|---|---|---|---|---|---|---|---|
| user_full_runA_stage1 | aster-dev-344 | 18,528 | 14,542 | 2,981 | 5 | 1,000 | 1.07 / 6 | 1,005 |
| user_full_runA_dev | aster-dev-345 | 2,594 | 2,052 | 410 | 0 | 132 | 1.07 / 3 | 132 |
| user_full_pv2 | aster-dev-346 | 18,322 | 15,324 | 2,041 | 34 | 923 | 1.15 / 14 | 957 |

- 实际 DeepSeek 调用约 4.3 万次（每条 1.07–1.15 次），低于估算的 4.8 万次。交给 luna 的比例是 5.1–5.4%，高于试跑。judge_error 主要是 DeepSeek 的临时错误。
- **luna 兜底被平台挡住了。** `make_tasks.py --target user --sample-ids …` 已经生成了三份兜底任务：
  - user_full_runA_stage1_fb：1,005 条，101 个 Task；
  - user_full_runA_dev_fb：132 条，14 个 Task；
  - user_full_pv2_fb：957 条，96 个 Task。

  但是 01:10 执行 `aster datasets create` 时，三份都返回 `INTERNAL_ERROR`（`retryable=false`）。request_id 分别为：
  - runA_stage1_fb `e4f7e4177f39339babea929d0f8634d2`
  - runA_dev_fb `2489e99279ebba6c1da4be90d68b20ba`
  - pv2_fb `91823fb9d805b06d4aa4a5e3a0268c75`

  同一时段 `datasets rm` 也报同样的错（见 T021 第 8 步）。读接口（`runs get`、`whoami`）都正常。已报给用户，等平台恢复后再提交兜底。
- **Run A 两份的预筛要注意。** `stage1_runA_v1.jsonl` 没有 `word` 字段，T021 已经踩过这个坑。定标签时 `--source` 要用 `trainable.jsonl`，而且要先核对它生成的提问编号和原文与放量时一致，再出标签。
- **兜底已提交。** 用户确认平台恢复，数据集接口在 01:47 恢复正常。在这之前，这个账号调用 `/api/v1/datasets` 的所有接口都返回 500，包括只读查询。三份兜底都是 luna，尝试 1 次，高优先级，平台快照与本地一致：

  | 份 | 条数 | Task | Run |
  |---|---|---|---|
  | user_full_runA_stage1_fb | 1,005 | 101 | aster-dev-347 |
  | user_full_runA_dev_fb | 132 | 14 | aster-dev-348 |
  | user_full_pv2_fb | 957 | 96 | aster-dev-349 |
- 已核对：Run A 第一阶段的 18,528 条提问在 `trainable.jsonl` 里全部能按编号对上，原文逐字相同，也都带 `word`。定标签时按卡片（1480cb9）用 `trainable.jsonl` 作 `--source`。
