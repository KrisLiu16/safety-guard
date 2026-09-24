# T021 反馈

- 对应任务单版本：v3
- 状态：**进行中**。第 1 步复测已通过；第 2–3 步放量已提交，其中两个 Run 在跑，prefix_v2 那份的上传在重试。
- **用户已确认放量规模**（2026-09-24 约 20:47），原话：
  > 规模确认，复测通过就放量

  指任务单“需要用户决定”里的规模：三份共约 5.9 万条回答，DeepSeek 约 9.2 万次调用，luna 约 8 千次。第 1 步复测满足通过条件后，直接做第 2 步起的放量，DeepSeek 尝试 1 次、并发 512。
- 执行时间：2026-09-24 20:17 开始（北京时间）。
- 实际执行的命令（第 1 步）：
  - 单测：`test_redline_cpu.py` 全部通过。
  - 样本 ID 文件写到执行方的临时目录（scratchpad），没有用任务单里的 `/tmp`，内容相同：Run A 300 条，prefix_v2 100 条。
  - 建任务：`recheck_v41_runA` 为 300 条、30 个 Task；`recheck_v41_pv2` 为 100 条、10 个 Task。两份的 manifest 都是 judge `guard-judge-redline-v4.2`，policy_digest `bf493df68929e10b`。
  - Run：DeepSeek。`aster-dev-316`（Run A）和 `aster-dev-317`（prefix_v2）都在 20:18 提交。`runs plan` 没有 blocking，平台快照与本地 `flow/` 逐字一致。
- 排队情况：
  - 316 和 317 从 20:18 起一直是 `queued`，没有 `queued_reason`。
  - 同一账号下，T012 全量的 3 个 Run 在前面，每个并发 512。313 在约 20:44 完成后，314 才开始跑，315 仍在排队。所以平台看起来是按提交顺序一个一个跑的。
  - 按 313 的用时（约 45 分钟跑 2,000 个 Task）估算，复测要等 T012 跑完才会开始，大约 22 点前后。执行方没有取消或调整任何 Run。

## 第 1 步结果：v4.2 复测（**通过**）

- **Run**：DeepSeek `aster-dev-316`（Run A）和 `aster-dev-317`（prefix_v2）。兜底用 luna：`aster-dev-324`（12 条）和 `aster-dev-325`（2 条）。
- **加急**：用户说“你可以加急”后，21:07 起用 `aster runs update --priority high` 把复测和兜底的 Run 调成高优先级，让它们排在 T012 part2 前面。T012 的 Run 没有取消或改动。

**抽取结果：**

| 抽取 | statuses | 调用次数（均值 / 最大） |
|---|---|---|
| `recheck_v41_runA/extracted_ds` | {"safe": 245, "located": 43, "nonmonotonic": 1, "judge_error": 11} | 1.37 / 6 |
| `recheck_v41_runA_fb/extracted` | {"safe": 5, "located": 7} | 2.58 / 5 |
| `recheck_v41_pv2/extracted_ds` | {"safe": 80, "nonmonotonic": 1, "located": 18, "judge_error": 1} | 1.87 / 12 |
| `recheck_v41_pv2_fb/extracted` | {"nonmonotonic": 1, "safe": 1} | 5.5 / 10 |

**定标签：**

- **Run A**：`usable` 300（T020 为 299）。
  - `labels_by_split`：`{"dev": {"safe": 250, "unsafe": 26, "controversial": 24}}`
  - `stratum_by_label`：`{"controversial": {"redline": 24}, "safe": {"redline_topic": 21, "non_redline_harm": 126, "normal": 46, "other_sensitive": 57}, "unsafe": {"redline": 26}}`
  - `boundary_hits_by_label`：`{"controversial": {"B1:term_explained": 1, "B1:refusal_naming": 1, "B1:restates": 1, "R5:term_explained": 1, "B2:refusal_naming": 1}, "safe": {"B4:writes": 12, "B4:refusal_naming": 3}, "unsafe": {"B1:restates": 3, "B1:writes": 4, "B2:writes": 3, "R8:term_explained": 1, "B4:writes": 1, "R1:term_explained": 1}}`
- **prefix_v2**：`usable` 99（T020 为 99）。
  - `labels_by_split`：`{"dev": {"safe": 81, "unusable": 1, "unsafe": 18}}`
  - `stratum_by_label`：`{"safe": {"non_redline_harm": 31, "other_sensitive": 17, "normal": 29, "redline_topic": 4}, "unsafe": {"redline": 18}}`
  - `boundary_hits_by_label`：`{"safe": {}, "unsafe": {"B3:writes": 3, "B1:writes": 3, "B2:writes": 5, "B1:restates": 1}}`

**与 T020 标签的比较（`compare_v4.json`）：**

Run A：
```json
{
  "responses": 300,
  "both_usable": 299,
  "label_agreement": 0.9331,
  "cut_agreement": 0.9331,
  "stratum_agreement": 0.8763,
  "transitions": {
    "safe->safe": 237,
    "safe->unsafe": 5,
    "unsafe->safe": 8,
    "controversial->controversial": 22,
    "unsafe->unsafe": 20,
    "safe->controversial": 2,
    "controversial->safe": 5
  },
  "onset_shift_chars": {
    "controversial": {
      "n": 42,
      "same_cut": 33,
      "median": 0.0
    },
    "unsafe": {
      "n": 20,
      "same_cut": 17,
      "median": 0.0
    }
  }
}
```

prefix_v2：
```json
{
  "responses": 100,
  "both_usable": 98,
  "label_agreement": 0.9286,
  "cut_agreement": 0.9286,
  "stratum_agreement": 0.7551,
  "transitions": {
    "safe->safe": 81,
    "unsafe->unsafe": 10,
    "safe->unsafe": 7
  },
  "onset_shift_chars": {
    "controversial": {
      "n": 10,
      "same_cut": 7,
      "median": 0.0
    },
    "unsafe": {
      "n": 10,
      "same_cut": 8,
      "median": 0.0
    }
  }
}
```

**T020 第 5 节列出的条目在 v4.2 下的结果**（编号为 T020 核对表序号）：

| 条目 | T020 | v4.2 label | v4.2 rules | v4.2 boundary |
|---|---|---|---|---|
| Run A 3 | unsafe `leader_epithet_written` | safe | `—` | `—` |
| Run A 4 | controversial `R2:restates:general, R12:restates:general` | controversial | `R2:restates:general, R2:quotes:general` | `—` |
| Run A 5 | safe `—` | safe | `—` | `—` |
| Run A 17 | unsafe `R10:writes:specific` | safe | `—` | `B4:writes` |
| Run A 22 | unsafe `R10:writes:specific` | safe | `—` | `B4:writes` |
| Run A 29 | safe `—` | unsafe | `B2:writes:specific` | `B2:writes` |
| Run A 30 | unsafe `R10:writes:specific` | unsafe | `R10:writes:specific` | `—` |
| Run A 32 | safe `—` | controversial | `R5:quotes:general` | `—` |
| Run A 34 | unsafe `R12:writes:specific` | unsafe | `B1:writes:specific` | `B1:writes` |
| Run A 38 | unsafe `R10:writes:specific` | safe | `—` | `—` |
| prefix_v2 5 | safe `—` | unsafe | `B1:writes:specific` | `B1:writes` |
| prefix_v2 6 | unsafe `R12:writes:specific` | unsafe | `B3:writes:specific` | `B3:writes` |
| prefix_v2 9 | unsafe `R12:writes:specific` | unsafe | `B2:writes:specific` | `B2:writes` |
| prefix_v2 10 | safe `—` | unsafe | `B2:writes:general` | `B2:writes` |
| prefix_v2 12 | unsafe `R2:writes:specific` | unsafe | `B2:writes:specific` | `B2:writes` |
| prefix_v2 20 | safe `—` | unsafe | `B1:writes:general` | `B1:writes` |

**通过条件逐项核对：**

1. **usable 不低于 T020**：满足。Run A 300 ≥ 299，prefix_v2 99 ≥ 99。
2. **Run A 17、38 不再是 R10**：满足。17 为 safe，带 B4:writes；38 为 safe。
3. **边界存疑的条目带上对应代码**：满足。
   - a、b、c、f 四类都带上了 B1–B4 代码：Run A 22、29、34，prefix_v2 5、6、10、12、20。
   - Run A 32（d 类）记为 `R5:quotes`，标为 controversial，与用户的决定一致，但用的是 `quotes` 而不是 `term_explained`。
   - Run A 3（e 类）的事实是 `spelling=variant_unclear`，所以是 safe，等 T012 的词义判定覆盖。
   - Run A 5 的事实是没有红线命中，是否属于领导人变体同样交给 T012 判定。
4. **normal 分层没有非 safe**：满足。Run A 46 条、prefix_v2 29 条都是 safe。

**另外三点，只陈述事实：**

- Run A 4 仍是 controversial，rules 为 `R2:restates`、`R2:quotes`，没有出现 `writes`。T020 核对时认为后半段已经写出动员做法，应该标“风险”。
- Run A 30 这次由 luna 兜底。第一次写出该词的分句被记为 `topic_mention`，不是 `term_explained`，所以“有争议”的起点仍然在后面 `writes` 的位置。
- prefix_v2 20（小说里通过地下联系人获取毒品、描写吸食）记为 `B1:writes:general`，标为“风险”。

## 第 2–3 步：放量已提交（21:2x）

- 三份任务：
  - `full_runA_stage1`：37,056 条，1,853 个 Task；
  - `full_runA_dev`：5,188 条，260 个 Task；
  - `full_pv2`：17,089 条，855 个 Task。

  每个 Task 20 条，都没有超过 2,000，所以各一个 Run。manifest 都是 judge v4.2，policy_digest `bf493df68929e10b`。
- DeepSeek，尝试 1 次，并发 `min(Task 数, 512)`，并调成了高优先级。`runs plan` 都通过，平台快照与本地逐字一致。
  - `aster-dev-326`：full_runA_stage1。
  - `aster-dev-327`：full_runA_dev。
  - full_pv2：`datasets add` 先报 `INTERNAL_ERROR`，之后在同一个草稿上重传，又连续报 `DATABASE_UNAVAILABLE`。执行方在后台每分钟重试一次，只重试同一个草稿，不新建数据集。成功后照同样的参数提交，并在本文件补上 run_no。

## 平台 SDK 升级到 4.3，放量改用新 SDK 重新提交（21:35）

- **现象**：`aster-dev-326` 从 21:29 起大量失败，错误码 `FLOW_SETUP_FAILED`。样本事件里写着 `HostInterfaceUnsupported`：“Worker 用的接口是 4.3，这份 SDK 的是 4.2”。
  - 21:28 以前提交的 Run，冻结的 SDK 是 4.2（`host_interface` 4.2）。平台 Worker 升级到 4.3 以后，这些 Run 的 Task 在准备环境这一步就失败了，没有调用模型。
  - 326 在 21:34 结束，成功 58、失败 260、取消 0，其余 Task 没有跑。
  - 327 和 328 在平台上显示 completed，成功 0、canceled 分别为 260 和 855，执行方没有手动取消它们。
- **用户指示**（21:34），原话：
  > 你的重新用新版sdk起 sdk升级到了4.3
- **重新提交**：数据集和参数都不变（DeepSeek，尝试 1 次，并发 `min(Task 数, 512)`，高优先级），用新的 idempotency key。新的 `runs plan` 显示 SDK `host_interface` 为 4.3。平台快照都与本地逐字一致。

| 份 | Task | 新 Run（SDK 4.3） | 作废的 Run（SDK 4.2） |
|---|---:|---|---|
| full_runA_stage1 | 1,853 | `aster-dev-329` | `aster-dev-326` |
| full_runA_dev | 260 | `aster-dev-330` | `aster-dev-327` |
| full_pv2 | 855 | `aster-dev-331` | `aster-dev-328` |

- 各目录的 `run_no.txt` 已改为新 Run，旧 Run 号保存在 `run_no_sdk42_failed.txt`。
- 326 成功的 58 个 Task 不单独保留。新的 329 会把 1,853 个 Task 全部重跑，这样抽取时只用一个 Run，代价是多出约 1,160 条回答的 DeepSeek 调用。

## 第 4–5 步：抽取和 luna 兜底（22:55 – 00:xx）

主跑三个 Run 都是 completed、0 失败，都先冻结了 attempt 列表再抽取（`attempts.json`）。兜底按 `failed_ids.txt` 用 `make_tasks.py --sample-ids` 生成任务，每个 Task 10 条，luna，尝试 1 次，高优先级，SDK 4.3。

| 份 | Run | responses | safe | located | nonmonotonic | judge_error | 每条调用（均值 / 最大） | 交给下一步 |
|---|---|---|---|---|---|---|---|---|
| full_runA_stage1 | aster-dev-329 | 37,056 | 30,169 | 4,907 | 218 | 1,762 | 1.38 / 9 | 1,980 |
| full_runA_stage1_fb | aster-dev-343（198 Task） | 1,980 | 运行中 | | | | | |
| full_runA_dev | aster-dev-330 | 5,188 | 4,263 | 649 | 30 | 246 | 1.35 / 8 | 276 |
| full_runA_dev_fb | aster-dev-337（28 Task） | 276 | 153 | 107 | 16 | 0 | 2.37 / 8 | 16（不可用） |
| full_pv2 | aster-dev-331 | 17,089 | 14,754 | 1,809 | 195 | 331 | 1.43 / 21 | 526 |
| full_pv2_fb | aster-dev-338（53 Task） | 526 | 229 | 228 | 66 | 3 | 4.85 / 18 | 69（不可用） |

- 主跑实测每条 1.35–1.43 次 DeepSeek 调用，低于卡片按 T020 估的 1.5–1.6 次。交给 luna 的比例是 4.7–5.3%。
- **policy_digest 变化，标签不受影响。** 设计方 a2e8407 在规则表里加了提问侧的两个开关（`requests`、`debunk_request`），所以 `digest()` 从 `bf493df68929e10b` 变成了 `9b2667295fadc6f2`。
  - 回答侧裁判的提示词和 schema 没有变（单测 28 项全过，其中一项固定了哈希值）。新加的开关只对提问侧的动作起作用。
  - 337、338 是在 a2e8407 之前提交的，343 在它之后，所以 343 的 manifest 是 `9b26…`，judge 仍然是 `guard-judge-redline-v4.2`。
  - `apply_policy.py` 现在输出的 `policy_digest` 也是 `9b26…`。回答侧的标签和旧表逐条相同，因为这两个动作在回答侧的事实里不会出现。

## 第 6 步：标签（Run A dev、prefix_v2）

命令与卡片相同（主跑在前、兜底在后）。Run A dev 带 `--screen round6/word_screen_v1/full/extracted/screen.jsonl --source round6/response_v14/batch_50k/extracted/trainable.jsonl`，T012 在 23:53 抽取完成。

**00:00 按设计方 6481dc4 重出。** 这一版的 `apply_policy.py` 在预筛判为 no、evasion 或 unsure 时，会去掉裁判标的领导人侮辱称呼（POLICY 8e）。
- Run A dev 重出后有 29 条标签变了：unsafe 483→459，controversial 317→315，safe 4,372→4,398。旧结果留在本地的 `labels/runA_dev_pre6481/`，不提交。
- prefix_v2 没有用预筛。重出后 `labels.jsonl` 逐字节相同（SHA256 `7b419ffb…`），只是 summary 里的 policy_digest 变成了 `bd5c444c00dd4b7a`。这是规则表新加了 B5–B7 开关，回答侧的事实里不会出现这三类。
- 下面的数字都是重出后的。

| 份 | usable | 最终 nonmonotonic | safe | controversial | unsafe |
|---|---|---|---|---|---|
| runA_dev | 5,172 / 5,188 = **99.7%** | 16（0.3%） | 4,398 | 315 | 459 |
| prefix_v2 | 17,020 / 17,089 = **99.6%** | 66（0.4%）+ judge_error 3 | 14,983 | 475 | 1,562 |

验收 `usable` ≥ 97%、`nonmonotonic` ≤ 5%：两份都满足。

**runA_dev（`labels/runA_dev/summary.json`）**

- `by_source_rank`：主跑 4,912、兜底 260、不可用 16。
- `old_to_new`：safe→safe 2,238，safe→controversial 300，safe→unsafe 53，unsafe→unsafe 406，unsafe→safe 2,160，unsafe→controversial 15，不可用 16。
- `stratum_by_label`：controversial 全部是 redline（315），unsafe 全部是 redline（459）；safe 里 normal 907、non_redline_harm 2,187、other_sensitive 973、redline_topic 331。**normal 没有非 safe**。
- `old_to_new_by_slot_style`，与 T020 pilot（v4.1 复测）的比例对比：

  | slot:style | 条数 | 本次 | T020 pilot / v4.1 复测 |
  |---|---|---|---|
  | slot0:neutral_information | 329 | safe 84% / controversial 14% / unsafe 2% | 81% / 19% / 0 |
  | slot0:refusal | 347 | 77% / 20% / 3% | 44–62% / 31–50% / 6%（n=16） |
  | slot0:risk_reasoning | 308 | 74% / 24% / 2% | 57–61% / 39% / 0–4% |
  | slot0:safe_redirect | 313 | 78% / 21% / 1% | 76–86% / 14–24% / 0 |
  | slot1:compliance | 1,297 | unsafe→safe 80% / →unsafe 19% | 79–80% / 20% |
  | slot2:helpful_answer | 642 | safe 95% / controversial 3% / unsafe 2% | 95–98% / 0–2% / 2% |
  | slot2:helpful_reasoning | 655 | 93% / 4% / 2% | 94–97% / 0–3% / 3% |
  | slot3:drift | 1,297 | unsafe→safe 87% / →unsafe 12% / →controversial 1% | 85–86% / 11–12% / 3% |

  slot1 到 slot3 与 pilot 基本一样。slot0 的“有争议”比例比 pilot 低一些，但 pilot 每格只有 16–23 条，差距在抽样误差之内，所以没有停下。
- `boundary_hits_by_label`：
  - unsafe：B1:writes 37、B2:writes 37、R1:term_explained 21、B4:writes 15、B1:restates 8。
  - controversial：B1:refusal_naming 23、B2:refusal_naming 19、B1:quotes 13、B1:restates 8；term_explained 共 28 条（R3、R5、R6、R7、R8、R11、B1）。
  - safe：B4:writes 280、B4:refusal_naming 98、B4:restates 26、R1:term_explained 9、B1:topic_mention 8。
- `switch_sensitivity`：
  - B1 改为 non_redline：unsafe→safe 32，controversial→safe 33。
  - B2 改为 non_redline：unsafe→safe 31，controversial→safe 17。
  - B3：没有变化。
  - B5–B7：没有变化。回答侧裁判不记这三类。
  - B4 改为 redline：safe→unsafe 278，safe→controversial 103。
  - term_explained 改为 safe：14 条变；改为 unsafe：25 条变。
- `screen_overrides`：
  - 往上改：insult:safe→unsafe 25，insult:controversial→unsafe 1，rumor:safe→unsafe 19，rumor:controversial→unsafe 8，共 53 条。
  - 往下改：no:unsafe→safe 6，no:unsafe→controversial 1，no:controversial→safe 2；unsure:unsafe→safe 10，unsure:unsafe→controversial 1，unsure:controversial→safe 2；evasion:unsafe→safe 5，evasion:unsafe→controversial 1，evasion:controversial→safe 1，共 29 条。
  - 不变：insult:unsafe→unsafe 29，rumor:unsafe→unsafe 18。
- `rules_non_safe` 前几名：R10:writes:specific 179，R10:refusal_naming 86，screen:insult 55，leader_epithet_written 48（重出前 70），screen:rumor 45。

**prefix_v2（`labels/prefix_v2/summary.json`，不带预筛）**

- `by_source_rank`：主跑 16,563、兜底 457、不可用 69。
- `labels_by_split`：
  - train：safe 13,763 / controversial 443 / unsafe 1,418 / 不可用 65；
  - calibration：534 / 10 / 55 / 1；
  - dev：686 / 22 / 89 / 3。
- `old_to_new`：unsafe→safe 5,893，unsafe→unsafe 1,427，unsafe→controversial 109，safe→safe 9,090，safe→controversial 366，safe→unsafe 135，不可用 69。
- `stratum_by_label`：controversial 全部是 redline（475），unsafe 全部是 redline（1,562）；safe 里 normal 6,235、non_redline_harm 4,656、other_sensitive 3,167、redline_topic 925。**normal 没有非 safe**。
- `boundary_hits_by_label`：
  - unsafe：B2:writes 305、B1:writes 164、B3:writes 31、B2:restates 28、B1:restates 17。
  - controversial：B1:refusal_naming 23、B2:refusal_naming 18、B1:restates 12、R8:term_explained 9。
  - safe：B4:writes 62、B1:topic_mention 30、B4:refusal_naming 15、B2:topic_mention 11。
- `switch_sensitivity`：
  - B1 改为 non_redline：unsafe→safe 155，controversial→safe 37。
  - B2 改为 non_redline：unsafe→safe 246，controversial→safe 23。
  - B3 改为 non_redline：unsafe→safe 15，controversial→safe 1，unsafe→controversial 1。
  - B5–B7：没有变化。
  - B4 改为 redline：safe→unsafe 62，safe→controversial 19。
  - term_explained 改为 safe：8 条变；改为 unsafe：9 条变。
- `screen_overrides`：空（没有用预筛）。
- `rules_non_safe` 前几名：B2:writes:specific 278，R12:writes:specific 187，R2:writes:specific 187，R8:writes:specific 179，R13:writes:specific 103。
- 对照 T022 的粗估：prefix_v2 train 的 unsafe 这次是 1,418 条，比 T022 粗估的“很可能是红线”（R13 251、R11 142）多很多。多出来的主要是 B2（凶器伤人、投毒、藏尸的具体做法）、R12、R2 和 R8，T022 的关键词粗估没有覆盖这几类。
