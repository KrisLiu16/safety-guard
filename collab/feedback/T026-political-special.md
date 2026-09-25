# T026 反馈：政治专项（v1.3：完成，标签已拷到 PVC）

- 对应任务单版本：v1.3（df9781f）
- 状态（v1.3）：完成。第 7b 步人工核对了 539 个命中过的词，本地排除 41 个。第 9 步四张表都过了门槛：runA 5%、prefix_v2 0%、user_prefix_v2 3.3%、领导人词 10%。usable 的减少全部来自 `general_scope`，最多 0.52%，没有超过 0.6%。六份标签已拷到 PVC `/work/round6/redline_v1/labels_political_v2/`，SHA256 核对一致。执行方从 2026-09-25 起兼任设计方（设计方已不在），见文末“v1.3”一节和 BOARD。
- 更早的状态：v1.2（e6dff53），规则表代码在 716ef02（含 `general_scope`）
- 状态（v1.2）：第 5c 步（run aster-dev-367）、第 6 步（过门槛，leader_negative 正好 10%）、第 7、8 步完成。**第 9 步没过门槛**：runA 15%（6/40），领导人词 25%（5/20），prefix_v2 9.1%、user_prefix_v2 0% 过了。**usable 也没过**：runA_stage1 少 0.52%（194 条），全部来自 716ef02 的 `general_scope=all_acts`，换回 `written_only` 后与 v1.1 完全一样。第 10 步（拷到 PVC）没做，旧标签没动。详见文末“v1.2”一节；以下先是 v1、v1.1 的记录。
- 历史版本：v1（4ea0271）
- 状态：v1.1 第 5b、6、7、8 步完成，**第 9 步没过门槛，已停下**：prefix_v2 为 37%，领导人词规则为 70%，门槛是 10%。第 10 步（拷到 PVC）没做，旧标签没动。原因主要是短词在普通词里被子串匹配上，见文末。以下先是 v1 的记录（v1 停在第 6 步）：leader_unsure 类的“普通词”加“不是政治敏感词”占 15%（6/40），超过 10%。第 1–5 步已完成，第 7 步起没有做，没有出任何新标签，也没有拷到 PVC。

## 第 1 步：单测

| 文件 | 结果 |
|---|---|
| `round6/political_screen_v1/test_political_screen_cpu.py` | 7 项全过 |
| `round6/redline_v1/test_redline_cpu.py` | 35 项全过 |
| `round6/word_screen_v1/test_word_screen_cpu.py` | 12 项全过 |

## 第 2 步：第一遍（全部词）

- `make_batch.py --output round6/political_screen_v1/full_p1`：manifest 为 `words` 340,883、`tasks` 1,705、`batch_words` 200，只有一个 `screen.tar.gz`，与卡片一致。
- 上传为一个数据集，提交 **aster-dev-359**：luna，`--flow round6/political_screen_v1/flow`，尝试 1 次，并发 512，高优先级。`runs plan` 没有 blocking，平台快照（`flow.py`、`pipeline.py`）与本地逐字一致。
- 完成后按卡片冻结 attempt 列表，再抽取。

## 第 3 步：抽取第一遍（04:07）

- 冻结 attempt 列表：1,705 个样本，1,705 个 attempt。
- `extract.py aster-dev-359 --attempts-json … --words … --leader-screen … --out full_p1/extracted`。执行方另外按 attempt_id 并行预取了一部分归档，文件名和脚本用的一样，脚本遇到已存在的归档直接跳过，结果不受影响。
- `summary.json`（只报计数）：
  - `words` 340,638，`archives` 1,705；
  - **`words_without_verdict` 245（0.07%）**，≤ 1%，可以继续；
  - `batch_errors`：word_changed 43、missing 43、bad_index 2、request 1；
  - `verdicts`：no 289,583、other_political 20,233、leader 10,787、event 8,621、unsure 6,079、figure 3,518、org 1,817；
  - `ambiguous_by_verdict`：no 132,069、other_political 6,674、unsure 5,327、leader 1,639、event 1,475、figure 1,109、org 182；
  - `matched_as`：event 7,146（另有 ambiguous 1,475）、figure 2,409（1,109）、org 1,635（182）、leader_insult 1,606（226）、leader_rumor 2,080（121）、leader_evasion 1,074（271）、leader_unsure 2,688（700）；
  - `confirm_words` 37,983。
- 245 个缺的词按卡片补筛一次：`make_batch.py --only missing_words.txt --tag r1 --output full_r1`，245 个词、2 个 Task。

## 第 4 步：复查

- `make_batch.py --only full_p1/extracted/confirm_words.txt --tag c1 --output full_c1`：37,983 个词，190 个 Task。
- 两个 Run 都用 luna，尝试 1 次，高优先级，平台快照与本地一致：
  - r1：**aster-dev-363**（2 个 Task）；
  - c1：**aster-dev-364**（190 个 Task，超过 100，完成后先冻结 attempt 列表）。

## 第 5 步：合并各遍（04:14）

- r1、c1 都跑完了，0 失败，都已冻结 attempt 列表：r1 有 2 个样本，c1 有 190 个。
- 合并命令：`extract.py aster-dev-359 aster-dev-363 aster-dev-364 --attempts-json <p1> <r1> <c1> … --review-sample 40 --out full/extracted`。
  - 第一遍的 1,705 个归档是用硬链接从 `full_p1/extracted/archives/` 放进来的，文件名相同，所以不用重新下载。
- `summary.json` 全文（只有计数），SHA256 `c97f407b…`：
  - `runs`：359、363、364，都是 completed；`archives` 1,897；
  - `words` 340,876；**`words_without_verdict` 7**；
  - `passes`：p1 340,638，c1 37,978，r1 238；
  - `batch_errors`：word_changed 55，missing 49，bad_index 2，request 1；
  - `verdicts`：no 283,590，other_political 20,037，leader 14,295，event 9,860，unsure 6,158，figure 4,810，org 2,126；
  - `ambiguous_by_verdict`：no 128,323，other_political 6,684，unsure 5,870，leader 5,384，event 3,176，figure 2,606，org 586；
  - `leader_screen_by_verdict`：
    - leader：unsure 5,229，insult 2,416，rumor 2,328，evasion 2,174，no 2,148；
    - event：no 9,297，unsure 358，rumor 165，evasion 23，insult 17；
    - figure：no 3,839，unsure 399，rumor 387，evasion 132，insult 53；
    - org：no 2,047，unsure 55，rumor 12，evasion 8，insult 4；
    - other_political：no 19,597，unsure 291，rumor 99，insult 27，evasion 23；
    - unsure：no 3,117，unsure 2,090，evasion 899，insult 33，rumor 19；
    - no：no 283,271，unsure 209，evasion 87，insult 21，rumor 2；
  - `matched_as`（括号里是带 `:ambiguous` 的）：
    - event 6,684（3,176）
    - figure 2,204（2,605）
    - org 1,540（585）
    - leader_insult 1,666（749）
    - leader_rumor 2,012（316）
    - leader_evasion 1,069（1,103）
    - leader_unsure 2,704（2,518）
  - `confirm_words` 38,008。
- 合并以后删除了 p1、r1、c1 三个数据集，回执在各目录的 `dataset_rm.json`。`political_terms.jsonl`（SHA256 `d142c460…`）和 `review/` 都被 gitignore 忽略，留在本地。

## 第 6 步：核对词表（本地，只报计数和理由）

执行方逐个核对了 `review/terms_sample.txt`：共 14 类，每类 40 个词。“无法判断”是执行方看不出这个词指什么的情况，单独列出来。

| 类 | 对 | 普通词 | 类别错 | 不是政治敏感词 | 无法判断 | 门槛：普通词 + 不是 ≤ 10% |
|---|---|---|---|---|---|---|
| event | 38 | 0 | 2 | 0 | 0 | 0% ✓ |
| figure | 33 | 0 | 3 | 3 | 1 | 7.5% ✓（无法判断也算进去是 10%，仍然 ✓） |
| leader_evasion | 37 | 3 | 0 | 0 | 0 | 7.5% ✓ |
| leader_insult | 35 | 2 | 3 | 0 | 0 | 5% ✓ |
| leader_rumor | 36 | 3 | 1 | 0 | 0 | 7.5% ✓ |
| **leader_unsure** | 33 | 1 | 0 | **5** | 1 | **15% ✗** |
| org | 36 | 1 | 3 | 0 | 0 | 2.5% ✓ |
| event:ambiguous | 26 | 12 | 1 | 1 | 0 | 不设门槛 |
| figure:ambiguous | 15 | 21 | 0 | 2 | 2 | 不设门槛 |
| leader_evasion:ambiguous | 32 | 1 | 0 | 0 | 7 | 不设门槛 |
| leader_insult:ambiguous | 34 | 4 | 0 | 2 | 0 | 不设门槛 |
| leader_rumor:ambiguous | 34 | 6 | 0 | 0 | 0 | 不设门槛 |
| leader_unsure:ambiguous | 22 | 10 | 0 | 3 | 5 | 不设门槛 |
| org:ambiguous | 25 | 9 | 4 | 0 | 2 | 不设门槛 |

理由（不写词）：

- **leader_unsure 没过门槛，主要原因是一类拼接出来的词。** 5 个“不是政治敏感词”都是“领导人姓氏的单字 + 一个普通机构或群体名”，比如这个字加某个会议、场馆或团体的名称。拼在一起没有任何负面或敏感含义，看起来是词表来源拼出来的。
  - 同一类里还有 1 个是正常的对台政策术语，按“普通词”算。
  - 按新规则，leader_unsure 的词只要写出就截，所以这几个词会直接造成误截。
  - leader_unsure:ambiguous 里还有 2 个词，其中那个字是“毛发”的意思，和领导人无关。leader_insult:ambiguous 里有 2 个粗俗词也是这样，跟领导人无关。
- **领导人名字的“变体”里混进了正常写法**（leader_evasion 和它的 ambiguous 类）：
  - 1 个是繁体字写的普通新闻短语（领导人出访），繁体字不是变体；
  - 2 个是标准拼音，一个连写，一个带空格，英文文本里的正常写法就是这样；
  - 1 个是和领导人名字谐音的日常用语。

  按“变体写法提到领导人标有争议”的规则，繁体中文新闻和英文新闻里正常出现的领导人名字也会被截。
- **leader_rumor**：3 个“普通词”都是普通的历史或新闻叙述，没有负面评价或传言的成分；1 个“类别错”是颠覆类口号，应该归到其他政治类。
- **figure**：3 个“不是政治敏感词”分别是一位正常履职的前官员、一位革命烈士和一位公众人物，一提就截会误伤正常内容；3 个“类别错”分别是 1 个群体称呼、1 个组织，还有 1 位前国家领导人，他应该按领导人规则处理。
- **event**：2 个“类别错”，一个是运动或组织，一个是常规的换届话题。
- **org**：1 个“普通词”是藏传佛教的常见教法名，只是和某功法的书名同名。
- **ambiguous 类**里普通词多，符合预期，因为这些词只在裁判已经找到政治红线时才生效。
  - figure:ambiguous 里一大半（21 个）是普通官员、学者和常见人名。
  - org:ambiguous 里有武侠小说里的门派、一般宗教用语、泛指的“邪教”拼音。
- **无法判断**：有两组看不出意思的变体字，分别在 figure、leader_evasion、leader_unsure 这几类里反复出现，不写原文。建议设计方对照 T012 的原始词表确认一下它们指什么。

## 执行方建议（请设计方定）

1. 对 leader_unsure，只截 T012 预筛本身不是 `no` 的词；或者 leader_unsure 改成只在裁判找到政治红线时才生效，和 ambiguous 一样处理。
2. 过滤掉“领导人单字 + 普通名词”这类拼接出来的词，比如要求词里有完整的名字或者明确的变体特征。
3. 繁体字的正常写法和标准拼音，不应该算变体写法。
4. 调整以后，第 6 步重新抽样核对，再做第 7 步。第 1–5 步的产物可以直接用，不用重新跑 Aster。

## v1.1（5a21af9）

- 单测：`test_political_screen_cpu.py` 8 项、`test_redline_cpu.py` 36 项、`test_word_screen_cpu.py` 12 项，全过。
- 第 5b 步：
  - 用 v1.1 代码在同一批 Run（359、363、364）上重跑第 5 步，输出到同一目录。归档都在本地，没有重新下载。多出了 `leader_words.txt`，共 14,295 个词。v1 的核对表另存在本地的 `review_v1/`。
  - `make_batch.py --only leader_words.txt --tag l1 --mode leader --output full_l1`：manifest 的 `mode` 为 leader，prompt_version `guard-political-screen-v1.1`，14,295 个词，72 个 Task。
  - 提交 **aster-dev-366**：luna，尝试 1 次，高优先级，平台快照与本地一致。完成后冻结 attempt 列表，四个 Run 一起合并，并用 `--review-seed r2` 重新抽样核对。

### 第 5b 步结果（04:47）

- l1 Run **aster-dev-366**：72 个 Task，0 失败，已冻结 attempt 列表。四个 Run（359、363、364、366）用 `--review-seed r2` 一起合并，之后删除了 l1 的数据集。
- 合并后的 `summary.json`（只有计数）：
  - `archives` 1,969，`words` 340,876，`words_without_verdict` 7；
  - `batch_errors`：word_changed 58，missing 54，bad_index 5，request 1；
  - `leader_words` 14,295；
  - `leader_form`：variant 6,009，negative 4,888，formal 3,063，not_leader 190，unsure 139；
  - `leader_form_by_leader_screen`：
    - T012 判 insult 的：negative 2,146，variant 226，formal 34，not_leader 6，unsure 4；
    - rumor：negative 1,662，formal 607，variant 50；
    - evasion：variant 1,807，formal 269，negative 73；
    - unsure：variant 3,593，negative 902，formal 537，not_leader 112，unsure 82；
    - no：formal 1,616，variant 333，negative 105，not_leader 61，unsure 32；
  - `matched_as`（括号里是带 `:ambiguous` 的）：
    - event 6,684（3,176）
    - figure 2,204（2,605）
    - org 1,540（585）
    - leader_negative 3,317（1,465）
    - leader_variant 2,869（2,803）

### 第 6 步（v1.1，用新种子 r2 抽样重新核对）

| 类 | 对 | 普通词 | 类别错 | 不是政治敏感词 | 无法判断 | 门槛 |
|---|---|---|---|---|---|---|
| event | 38 | 0 | 1 | 0 | 1 | 0% ✓ |
| figure | 36 | 3 | 0 | 1 | 0 | 10% ✓（刚好在门槛上） |
| org | 37 | 1 | 2 | 0 | 0 | 2.5% ✓ |
| leader_negative | 38 | 0 | 0 | 2 | 0 | 5% ✓ |
| leader_variant | 36 | 1 | 1 | 0 | 2 | 2.5% ✓ |
| event:ambiguous | 27 | 9 | 2 | 0 | 2 | — |
| figure:ambiguous | 18 | 18 | 3 | 0 | 1 | — |
| leader_negative:ambiguous | 29 | 3 | 0 | 7 | 1 | — |
| leader_variant:ambiguous | 26 | 7 | 0 | 1 | 6 | — |
| org:ambiguous | 24 | 7 | 8 | 0 | 1 | — |

理由（不写词）：

- **event**：1 个“类别错”是枪支型号，和事件同名，属于武器买卖用语。
- **figure**：3 个“普通词”是历史人物和宗教称号，1 个“不是”是正常履职的军官。
- **org**：1 个“普通词”是常见短语，2 个“类别错”分别是宪章的变体和宗教人物。
- **leader_negative**：2 个“不是”又是“领导人单字 + 无关词”，而且那个字的意思是毛发，跟领导人无关。leader_negative:ambiguous 里还有 7 个同样的情况。
- **leader_variant**：1 个“普通词”是新闻里常见的“影射××”说法。
- **无法判断**：主要是上次提到的两组看不出意思的变体字，这次都被归到 leader_variant。

所以第 6 步过了门槛，接着做了第 7 步。

### 第 7 步：重出六份标签（`labels_political_v1/`，policy_digest `53c771f3b583e841`）

| 份 | usable 旧 → 新（减少） | safe / controversial / unsafe | political_rows | 从 normal 提档的条数 | `texts.length_mismatch` |
|---|---|---|---|---|---|
| runA_stage1 | 36,971 → 36,952（19，0.05%） | 29,216 / 4,392 / 3,344 | 1,545 | 35 | 0 |
| runA_dev | 5,172 → 5,172（0） | 4,089 / 610 / 473 | 241 | 3 | 0 |
| prefix_v2 | 17,020 → 17,019（1） | 14,845 / 612 / 1,562 | 107 | 0 | 0 |
| user_runA_stage1 | 18,527 → 18,527（0） | 14,271 / 2,103 / 2,153 | 661 | 11 | 0 |
| user_runA_dev | 2,594 → 2,594（0） | 2,004 / 282 / 308 | 102 | 0 | 0 |
| user_prefix_v2 | 18,304 → 18,304（0） | 15,374 / 1,193 / 1,737 | 169 | 1 | 0 |

- usable 最多减少 0.05%，没有超过 0.5%。
- `political_raised_from_stratum`（被专项规则提档的条目原来属于哪个分层）：
  - runA_stage1：non_redline_harm 394，redline_topic 283，other_sensitive 189，redline 56，**normal 35**；
  - user_runA_stage1：redline_topic 200，non_redline_harm 156，other_sensitive 80，redline 32，**normal 11**；
  - runA_dev：non_redline_harm 67，redline_topic 53，other_sensitive 33，redline 9，normal 3。
- `switch_sensitivity` 里 `leader_variant=safe` 的条数：
  - runA_stage1：controversial→safe 634；
  - runA_dev：74；
  - prefix_v2：93；
  - user_runA_stage1：626；
  - user_runA_dev：82；
  - user_prefix_v2：520。
- `screen_words_formal_by_leader_form`：Run A 四份都是 927，prefix_v2 两份没有用预筛。
- 各份 `labels_by_split`、`old_to_new`、`screen_overrides`、`political_table`、`political_overrides` 的全文见各自的 `summary.json`，已提交。

### 第 8 步：新旧对比（`compare_old.json`）

| 份 | label_agreement | 主要变化 |
|---|---|---|
| runA_stage1 | 0.942 | safe→controversial 2,085，safe→unsafe 59，controversial→unsafe 4 |
| runA_dev | 0.940 | safe→controversial 295，safe→unsafe 14 |
| prefix_v2 | 0.992 | safe→controversial 137，safe→unsafe 1 |
| user_runA_stage1 | 0.941 | safe→controversial 1,070，safe→unsafe 31，controversial→unsafe 2 |
| user_runA_dev | 0.943 | safe→controversial 139，safe→unsafe 6，controversial→unsafe 2 |
| user_prefix_v2 | 0.967 | safe→controversial 593，safe→unsafe 3，controversial→unsafe 2 |

### 第 9 步：人工核对专项规则（本地核对，只报计数）

问题 1：命中的词在这里确实指政治敏感的事件、人物、组织或领导人吗？
问题 2：新标签对不对？

| 核对表 | 条数 | 问题 1 答“不是” | 门槛 ≤ 10% | 问题 2 判错（都是误判红线） |
|---|---|---|---|---|
| runA_stage1，`--only-rule political` | 40 | 3（7.5%） | ✓ | 3 |
| prefix_v2，political | 30 | **11（36.7%）** | **✗** | 8 |
| user_prefix_v2，political | 30 | 2（6.7%） | ✓ | 0 |
| runA_stage1，`--only-rule leader_word` | 20 | **14（70%）** | **✗** | 7 |

按原标签和原分层分开（每格依次是条数、问题 1 答“不是”、问题 2 判错）：

- runA political：
  - **safe:normal 7 / 2 / 2**
  - safe:non_redline_harm 7 / 1 / 1
  - safe:other_sensitive 7 / 0 / 0
  - safe:redline_topic 6 / 0 / 0
  - controversial:redline 7 / 0 / 0
  - unsafe:redline 6 / 0 / 0
- prefix_v2 political：
  - safe:redline_topic 13 / 4 / 6
  - safe:non_redline_harm 2 / 2 / 2
  - safe:other_sensitive 3 / 0 / 0
  - controversial:redline 7 / 1 / 0
  - unsafe:redline 5 / 4 / 0
- user_prefix_v2 political：
  - safe:normal 1 / 0 / 0
  - safe:non_redline_harm 7 / 0 / 0
  - safe:other_sensitive 7 / 0 / 0
  - safe:redline_topic 7 / 0 / 0
  - controversial:redline 6 / 1 / 0
  - unsafe:redline 2 / 1 / 0
- runA leader_word：
  - safe:non_redline_harm 8 / 6 / 2
  - safe:other_sensitive 5 / 3 / 4
  - safe:redline_topic 3 / 2 / 0
  - controversial:redline 2 / 1 / 1
  - unsafe:redline 2 / 2 / 0

有些条目命中错了，但标签没错，因为别的规则本来就会把它判到同一档。问题 2 只统计因为命中错而判错的条目。

**原因（不写词，只说模式）：**

1. **拉丁字母的短词在英文单词里被子串匹配。**
   - 领导人词表里有两个字母的拼音，在另一个词的拼音里被匹配上，在 maximize、dehumanizing 这类英文单词里也被匹配上。leader_word 那 20 条里有 9 条是这样。
   - 一个四字母的组织缩写，在一个普通英文单词里被匹配上。
2. **两个汉字的短词在常用词里被子串匹配。**
   - 一位人物名字的后两个字，在“传播开来”“公开来源”“隔离开来”里被匹配上。prefix_v2 里 3 条，user_prefix_v2 里 1 条。
   - 另一位人物名字的后两个字，在“意志强加”里被匹配上。
   - 一个社运口号的中间两个字，在“配送中心”里被匹配上。
   - 领导人名字的一个变体，在“实习记录”“学习记录”里被匹配上。这 2 条都被判成了“有争议”。
3. **泛指的历史事件和普通年份也被当成政治事件。**
   - 泛指“屠杀”的词，用在二战或抗战时期的史实里。其中 1 条是正常的课堂教学设计，被判成了“有争议”。
   - 清末的一场革命，出现在正常的人物介绍里。
   - 建国后公开讨论过的两次历史运动，出现在一篇中性的人物简介里，这是 prefix_v2 里的一条 normal 内容。
   - 普通年份，比如 19 世纪的一个年份、一位领导人去世的年份。
4. **非中国的恐怖组织也算作了政治组织。** 2 条对外国恐怖组织的正常说明（其中一条是人物传记里提到）被判成了“有争议”，这超出了“中国政治专项”的范围。
5. **有一句不指向任何人的粗俗网络用语**（自称“朕”的性暗示说法）被领导人复查判成 negative。结果解释这个词、以及拿它骚扰别人的 4 条都被判成了 unsafe，其实都不是红线。
6. **带 `:gated` 的词门槛太低。** 在种族、历史、恐怖主义这类文本里，只要裁判记了任何政治类命中，门槛就打开了。所以上面这些普通词也跟着命中了。

**执行方建议（请设计方定）：**

- 拉丁字母的词只按完整单词匹配（加词边界），太短的拼音（比如 3 个字母以下）不参与匹配。
- 两个汉字的词不单独参与匹配，或者只在裁判找到中国政治红线的文本里生效；领导人词也照此处理。
- 纯年份、泛指的历史事件（泛指的屠杀、朝代更替时的革命、公开讨论过的历史运动），以及非中国的组织，从专项词表里去掉。
- 领导人复查的 negative 类里，去掉不指向具体人的词。
- 改完后重出六份标签，重做第 9 步。第 1–5b 步的产物不用重跑，除非设计方要求重筛。

- 本地核对表在 `labels_political_v1/review_*/`，但仓库的 `.gitignore` 只忽略名为 `review/` 的目录，所以这几张表**没有被忽略**。执行方已经加到本地的 `.git/info/exclude`，没有提交。建议设计方在 `.gitignore` 里加上 `**/review_*/`。

## v1.2（e6dff53；规则表代码 716ef02）

### 单测

三组都通过：`round6/political_screen_v1` 9 个，`round6/redline_v1` 37 个，`round6/word_screen_v1` 12 个。先在 e6dff53 上跑了一次，rebase 到 716ef02 之后又跑了一次，结果相同。

### 第 5c 步：最终核对（run aster-dev-367）

- 用 v1.2 代码按第 5b 步的命令重新合并，得到 `check_words.txt`，共 **27,255** 个词。
- 核对批次 k1：
  - manifest 里 `mode` 为 `check`，共 137 个 Task，每批 200 个词；
  - luna，并发 137，每个 Task 只跑一次，高优先级；
  - 137/137 成功，失败 0；
  - attempt 列表冻结了 137 条。
- 把 k1 加在合并命令末尾，用 `--review-seed r3` 合并（exit 0）。k1 的数据集已删除（`full_k1/dataset_rm.json`，ok）。
- 合并后 `summary.json` 的主要字段（全文已提交）：
  - `check_words` 27,255。其中 27,249 个拿到了核对结论。另有 6 个词因为 k1 批次出错没有核对结论，按代码仍用核对前的类别。
  - `check_by_candidate`（行是核对前的类别，列是核对结论）：

    | 核对前 \ 核对后 | event | figure | org | leader_negative | leader_variant | no | unsure |
    |---|---|---|---|---|---|---|---|
    | event（9,858） | 8,502 | 122 | 176 | 74 | 112 | 784 | 88 |
    | figure（4,810） | 77 | 2,767 | 69 | 221 | 167 | 1,434 | 75 |
    | org（2,126） | 67 | 11 | 1,778 | 34 | 11 | 218 | 7 |
    | leader_negative（4,782） | 35 | 256 | 9 | 3,863 | 397 | 213 | 9 |
    | leader_variant（5,673） | 34 | 97 | 11 | 570 | 4,648 | 277 | 36 |

  - `matched_as` 由 v1.1 的 26,748 个降到 **22,509** 个：

    | 类别 | v1.1 | v1.2 |
    |---|---|---|
    | event | 6,684 | 5,654 |
    | event:ambiguous | 3,176 | 3,063 |
    | figure | 2,204 | 1,512 |
    | figure:ambiguous | 2,605 | 1,741 |
    | org | 1,540 | 1,386 |
    | org:ambiguous | 585 | 656 |
    | leader_negative | 3,317 | 2,975 |
    | leader_negative:ambiguous | 1,465 | 1,787 |
    | leader_variant | 2,869 | 2,370 |
    | leader_variant:ambiguous | 2,803 | 2,965 |

  - `verdicts` 与 v1.1 相同：no 283,590、other_political 20,037、leader 14,295、event 9,860、unsure 6,158、figure 4,810、org 2,126。
  - `ambiguous_by_verdict` 变多了，因为只要 k1 判为常作他义，就记为 ambiguous：org 586→793，leader 5,945→6,699，figure 2,606→3,272，event 3,176→3,910。
  - `leader_form` 没有变：formal 3,063、variant 6,009、negative 4,888、not_leader 190、unsure 139。
  - `words_without_verdict` 7。
  - `batch_errors` 为 word_changed 62、missing 59、bad_index 7、request 1。v1.1 时是 58、54、5、1，所以 k1 新增了 4、5、2。

### 第 6 步：核对词表（新抽样 r3，每类 40 个，只报计数）

| 类别 | 对 | 普通词 | 不是政治敏感词 | 类别错 | 普通词 + 不是 | 门槛 |
|---|---|---|---|---|---|---|
| event | 39 | 1 | 0 | 0 | 2.5% | ✓ |
| figure | 38 | 1 | 0 | 1 | 2.5% | ✓ |
| org | 37 | 0 | 0 | 2 | 0%（算上 1 个正则式字面量为 2.5%） | ✓ |
| leader_negative | 34 | 2 | 2 | 2 | **10.0%** | ✓（正好在线上） |
| leader_variant | 37 | 0 | 3 | 0 | 7.5% | ✓ |
| event:ambiguous | 28 | 8 | 3 | 1 | — | 不设门槛 |
| figure:ambiguous | 25 | 6 | 7 | 2 | — | 不设门槛 |
| org:ambiguous | 24 | 6 | 7 | 3 | — | 不设门槛 |
| leader_negative:ambiguous | 21 | 5 | 9 | 5 | — | 不设门槛 |
| leader_variant:ambiguous | 30 | 4 | 5 | 1 | — | 不设门槛 |

各类里判为“普通词”“不是”或“类别错”的，主要是这几种（只说模式，不写词）：

- **event**：1 个只是普通日期，所以算“普通词”。
- **event:ambiguous**：
  - 3 个是某种手枪型号的买卖用语，与政治无关；
  - 8 个是泛泛的年份或月份，或普通描写；
  - 1 个是对执政党的蔑称，不是事件，算类别错。
- **figure**：
  - 1 个是前部长的名字，算普通人物；
  - 1 个是领导人的谣言，应归到领导人类，算类别错。
- **figure:ambiguous**：
  - 6 个是常见人名或普通官员、历史人物；
  - 7 个名字查不到是谁；
  - 2 个其实是领导人名字的变体，算类别错。
- **org**：
  - 2 个是反党口号或对党的蔑称，不是组织，算类别错；
  - 1 个是正则式字面量，普通文本里匹配不到。
- **org:ambiguous**：
  - 2 个是泛指的外国“圣战组织”；
  - 1 个是外语词；
  - 若干个是常见宗教或日常用语；
  - 3 个类别错，其中 2 个是宪章类文件，1 个是独立口号。
- **leader_negative**：
  - 2 个是不指向任何人的粗口；
  - 2 个是一个姓氏字加一句泛泛的骂人话，看不出在骂谁；
  - 2 个只是变体写法，并不负面，算类别错。
- **leader_negative:ambiguous**：9 个是通用俚语或其他词，比如某个成语、别国名称的变体、“温馨”一类的词。
- **leader_variant**：3 个是乱码串，看不出指谁。
- **leader_variant:ambiguous**：
  - 4 个是领导人名字的正常写法、常见名，或普通官员；
  - 1 个是源词表里的正则式字面量；
  - 1 个是领导人亲属的名字，算类别错。

v1.2 要求特别看的几种：
- 普通年份还有 1 个，在 event 类；
- 泛指的外国组织还有 2 个，在 org:ambiguous 类；
- 不指向人的粗口还有 2 个，在 leader_negative 类。

### 第 7 步：重出六份标签（`labels_political_v2/`，policy_digest `9900deabdbfdf07c`）

| 份 | usable 旧 → 新（减少） | safe / controversial / unsafe | political_rows（v1.1） | 从 normal 提档的条数（v1.1） | `texts.length_mismatch` |
|---|---|---|---|---|---|
| runA_stage1 | 36,971 → 36,777（**194，0.52%** ✗） | 29,825 / 3,673 / 3,279 | 1,319（1,545） | 24（35） | 0 |
| runA_dev | 5,172 → 5,148（24，0.46%） | 4,159 / 526 / 463 | 221（241） | 3（3） | 0 |
| prefix_v2 | 17,020 → 17,010（10，0.06%） | 14,996 / 455 / 1,559 | 37（107） | 0（0） | 0 |
| user_runA_stage1 | 18,527 → 18,515（12，0.06%） | 14,659 / 1,786 / 2,070 | 586（661） | 7（11） | 0 |
| user_runA_dev | 2,594 → 2,594（0） | 2,048 / 250 / 296 | 101（102） | 0（0） | 0 |
| user_prefix_v2 | 18,304 → 18,278（26，0.14%） | 15,660 / 996 / 1,622 | 111（169） | 0（1） | 0 |

**usable 减少的原因**：全部来自 716ef02 新加的开关 `general_scope=all_acts`，和 v1.2 的词表无关。在本地 scratchpad 里做了两次对照，没有提交：

- 词表照旧，只把开关换回 `written_only`：
  - runA_stage1 为 36,952，runA_dev 为 5,172，prefix_v2 为 17,019，user_runA_stage1 为 18,527，user_prefix_v2 为 18,304，都与 v1.1 一模一样；
  - user_runA_dev 本来就没有减少，所以没有重跑。
- 规则表照新，但去掉按词规则：runA_stage1 仍为 36,777。

所以可以推断，这个开关让一部分回答的探针不再单调，这些回答就变成了不可用。按卡的规定，这里停下，请设计方决定是否接受。

其他要报的字段：

- `political_raised_from_stratum`：
  - runA_stage1：non_redline_harm 346，redline_topic 285，other_sensitive 159，redline 69，**normal 24**；
  - runA_dev：non_redline_harm 64，redline_topic 56，other_sensitive 32，redline 9，normal 3；
  - user_runA_stage1：redline_topic 186，non_redline_harm 135，other_sensitive 68，redline 38，**normal 7**；
  - user_runA_dev：redline_topic 28，non_redline_harm 27，other_sensitive 9，redline 6；
  - prefix_v2：redline_topic 9，other_sensitive 3，non_redline_harm 1；
  - user_prefix_v2：redline_topic 37，non_redline_harm 8，other_sensitive 5，redline 2。
- `switch_sensitivity` 里 `leader_variant=safe` 的条数（都是 controversial→safe）：

  | 份 | 条数 |
  |---|---|
  | runA_stage1 | 634 |
  | runA_dev | 74 |
  | prefix_v2 | 93 |
  | user_runA_stage1 | 626 |
  | user_runA_dev | 82 |
  | user_prefix_v2 | 520 |

- `switch_sensitivity` 里 `general_scope=written_only` 的条数：

  | 份 | safe→controversial | safe→unsafe |
  |---|---|---|
  | runA_stage1 | 372 | — |
  | runA_dev | 49 | — |
  | prefix_v2 | 119 | — |
  | user_runA_stage1 | 212 | 52 |
  | user_runA_dev | 27 | 8 |
  | user_prefix_v2 | 163 | 97 |

- `screen_words_formal_by_leader_form`：Run A 四份都是 3,085（v1.1 为 927），prefix_v2 两份没有用预筛。
- `political_table`（六份相同）：

  | 规则 | 词数 |
  |---|---|
  | political:event | 5,648 |
  | political:event:gated | 3,067 |
  | political:figure:gated | 3,253 |
  | political:org | 1,369 |
  | political:org:gated | 673 |
  | leader_word:negative | 2,898 |
  | leader_word:negative:gated | 1,863 |
  | leader_word:variant | 2,356 |
  | leader_word:variant:gated | 2,975 |

- `labels_by_split`、`old_to_new`、`screen_overrides`、`political_overrides` 的全文见各份 `summary.json`，已提交。

### 第 8 步：新旧对比（`compare_old.json`）

| 份 | label_agreement（v1.1） | 主要变化 |
|---|---|---|
| runA_stage1 | 0.937（0.942） | safe→controversial 1,834，controversial→safe 344，safe→unsafe 70，unsafe→controversial 42，unsafe→safe 25，controversial→unsafe 3 |
| runA_dev | 0.934（0.940） | safe→controversial 274，controversial→safe 44，safe→unsafe 13，unsafe→controversial 5，unsafe→safe 4 |
| prefix_v2 | 0.987（0.992） | controversial→safe 119，safe→controversial 106 |
| user_runA_stage1 | 0.931（0.941） | safe→controversial 939，controversial→safe 206，unsafe→safe 58，safe→unsafe 38，unsafe→controversial 29，controversial→unsafe 1 |
| user_runA_dev | 0.932（0.943） | safe→controversial 131，controversial→safe 26，unsafe→safe 10，safe→unsafe 6，controversial→unsafe 2，unsafe→controversial 2 |
| user_prefix_v2 | 0.955（0.967） | safe→controversial 568，controversial→safe 163，unsafe→safe 97，controversial→unsafe 2，safe→unsafe 2 |

v1.1 没有任何往下调的变化。v1.2 新出现的 controversial→safe 和 unsafe→safe，条数和上面 `general_scope=written_only` 的敏感度对得上，所以应当来自 `general_scope`。

### 第 9 步：人工核对专项规则（本地核对，只报计数）

| 核对表 | 条数 | 问题 1 答“不是” | 门槛 ≤ 10% | 问题 2 判错 | v1.1 |
|---|---|---|---|---|---|
| runA_stage1，political | 40 | **6（15%）** | **✗** | 6（都是误判红线） | 3（7.5%） |
| prefix_v2，political | 22（只有这么多可抽） | 2（9.1%） | ✓ | 1 | 11（36.7%） |
| user_prefix_v2，political | 30 | 0 | ✓ | 0 | 2（6.7%） |
| runA_stage1，leader_word | 20 | **5（25%）** | **✗** | 5 | 14（70%） |

按原分层和原标签分开（每格依次是条数、问题 1 答“不是”、问题 2 判错）：

- runA political：
  - **safe:normal 7 / 4 / 4**
  - safe:non_redline_harm 7 / 2 / 2
  - safe:other_sensitive 7 / 0 / 0
  - safe:redline_topic 6 / 0 / 0
  - controversial:redline 7 / 0 / 0
  - unsafe:redline 6 / 0 / 0
- prefix_v2 political：
  - safe:non_redline_harm 1 / 1 / 1
  - safe:other_sensitive 3 / 0 / 0
  - safe:redline_topic 9 / 0 / 0
  - controversial:redline 7 / 0 / 0
  - unsafe:redline 2 / 1 / 0（另一条红线本来就判 unsafe，所以标签没错）
- user_prefix_v2 political：
  - safe:non_redline_harm 6 / 0 / 0
  - safe:other_sensitive 4 / 0 / 0
  - safe:redline_topic 9 / 0 / 0
  - controversial:redline 9 / 0 / 0
  - unsafe:redline 2 / 0 / 0
- runA leader_word：
  - safe:non_redline_harm → unsafe 2 / 2 / 2
  - safe:other_sensitive → unsafe 2 / 2 / 2
  - controversial:redline → controversial 6 / 1 / 0
  - controversial:redline → unsafe 5 / 0 / 1（档位错：只是变体写法，却判成 unsafe）
  - unsafe:redline 5 / 0 / 0

**runA 答“不是”的那 6 条**：词表里都不带 ambiguous，而且 p1、c1、k1 三遍给的结论都一致，所以最终核对没有把它们拦下来。具体是：

1. 一个带拉丁字母和数字、后缀像化学名的四字串，回答里也说查不到它的含义（normal）。
2. 一个泛指基督徒团契的名称，被当成组织（non_redline_harm）。
3. 人名加一个数字的组合，k1 从 figure 改成了 event（normal）。
4. 2010 年一句网络流行语，出自一起社会新闻，不是政治专项的事件，k1 同样从 figure 改成了 event（normal）。
5. 一个五字词，在文中被当作骂人的标签用（non_redline_harm）。
6. 一个泛指的新闻用语（描述军事车辆调动），回答也按字面解释（normal）。

第 3、5 两条我拿不准。如果把它们算作“是”，就是 4/40 = 10%，正好卡在门槛上。另有 3 条我算作“是”，但也有争议：两个隐语（回答没认出来）和一个说法本身有两种用法的词。

**领导人词答“不是”的那 5 条**：

- 4 条是同一个四字组合词，来自同一个题目家族的 4 条回答：
  - 三遍的结论分别是 leader、org、leader_negative；
  - 回答里也说不清它指谁。
  - 按题目家族算，14 个家族里只有 2 个错。
- 1 条是一个三字谐音变体，被当作子串在一个普通四字短语里匹配上了：
  - 这个词带 ambiguous，所以只在政治文本里生效；
  - 另一条规则本来就判“有争议”，所以标签没有因此判错。

另外有两处命中错了，但没有影响标签：

- 第 9 号：l1 判的是变体写法，k1 改成了 negative，于是判成 unsafe，而按规则应为 controversial。全表里有 570 个 leader_variant 候选被 k1 改成了 leader_negative。
- 第 20 号：领导人全名里的三个字被当成 negative 词，匹配在正式的中性用法里。这个词带 ambiguous，而且裁判本来就判“有争议”，所以标签没错。

**prefix_v2 答“不是”的那 2 条**：两个汉字的短词在政治文本里，仍然作为子串匹配到了普通词上：

- 一个人物词，匹配在“意志强加”里（第 1 号）；
- 一个事件词，匹配在“配送中心”里（第 4 号），这个问题 v1.1 已经报过。

这两个词都带 ambiguous，但文本被判为政治文本，所以还是生效了。

### 执行方建议（请设计方定）

1. **usable 门槛**：要么接受 `general_scope=all_acts` 带来的减少（runA_stage1 为 0.52%，其余都在 0.5% 以内），要么先修这个开关造成的探针不单调。词表本身不影响 usable。
2. **runA 的误命中不是模型再核对一遍能解决的**：三遍结论一致，错在词本身冷僻或太泛。建议加一个本地排除表：
   - 执行方先在本地列出所有让 normal 和 non_redline_harm 分层被提档的词（runA_stage1 有 24 条 normal、346 条 non_redline_harm）；
   - 由执行方人工逐个核对，结果只写进本地的排除文件（不提交）；
   - 再让 `apply_policy.py` 读取这个排除文件，比如加一个 `--exclude-words` 参数；
   - 仓库里只报排除了多少个词、各属于哪一类。
3. **两个汉字的短词**：event 和 figure 里带 ambiguous 的两字词，建议干脆不参与匹配，只留下人工确认过的少数几个。
4. **领导人词**：
   - 变体还是负面，以 l1 还是 k1 的结论为准？现在是 k1 覆盖 l1，结果有 570 个变体被改成了 negative，会判成 unsafe 而不是 controversial。请设计方定。
   - 前后几遍结论不一致的词（比如一遍判 org、一遍判 leader），建议不要算作领导人词。
5. 改完以后，重出标签、重做第 9 步即可。第 1–5c 步都不用重跑，除非要重新核对。

### 产物

- 已提交：
  - `full_k1/` 下的 `manifest.json`、`run_no.txt`、`dataset_rm.json`；
  - `full/extracted/summary.json`；
  - 六份 `labels_political_v2/*/summary.json` 和 `compare_old.json`。
- 没有提交，因为含词或原文：`political_terms.jsonl`、`check_words.txt`、`labels.jsonl`、`review_*/`、attempts、归档。
- SHA256（前 16 位）：

  | 文件 | SHA256 |
  |---|---|
  | political_terms.jsonl | `d996817835144dec` |
  | check_words.txt | `4e4d5fa875b501df` |
  | runA_stage1/labels.jsonl | `2557e54fd75c135a` |
  | runA_dev/labels.jsonl | `2ba3808b7b68994c` |
  | prefix_v2/labels.jsonl | `4180a6f4d5d6042d` |
  | user_runA_stage1/labels.jsonl | `2321b163aef0a292` |
  | user_runA_dev/labels.jsonl | `d54783a0268d6ebd` |
  | user_prefix_v2/labels.jsonl | `633a3535e499d6a4` |

- PVC 上没有新增任何文件。

## v1.3（df9781f）

从这一版起，设计方已不在，执行方同时负责设计和执行（用户 2026-09-25 的指示）。下面凡是超出任务卡的地方，都按设计方的身份单独注明。

### 单测

四组都通过：`political_screen_v1` 9 个，`redline_v1` 39 个，`word_screen_v1` 12 个，`stage2` 6 个。

### 第 7、7b 步：列词、人工核对、本地排除

- 先用 v1.3 的代码重出六份标签，再用 `list_fired_words.py` 列出命中过的词：共 **539 个**，其中 **314 个**曾把 normal、non_redline_harm 或 other_sensitive 的文本提档。
- 这 314 个都逐个核对了，拿不准的对照了例子原文。排除 **22 个**，拿不准的一律保留。
- **超出任务卡的部分**（设计方决定）：另外 225 个词只在 redline 或 redline_topic 的文本里命中过，也都扫了一遍。第 9 步又抽到几条误命中，都属于下面这几类，所以这 225 个里凡是这几类的，也加进排除表，一共 **19 个**：
  - 纯数字或纯年数；
  - 领导人全名里的片段；
  - 和常用短语撞车的谐音；
  - 小说里的门派；
  - 含领导人姓氏用字的普通俚语或粗口。

  排除只会让命中变少，不会新增命中。
- 排除表共 **41 个词**，放在本地的 `full/extracted/exclude_words.txt`，已 gitignore，没有提交。按规则分：

  | 规则 | 个数 |
  |---|---|
  | political:event | 15 |
  | political:event:gated | 5 |
  | political:org | 8 |
  | political:org:gated | 3 |
  | political:figure:gated | 2 |
  | leader_word:negative | 3 |
  | leader_word:negative:gated | 1 |
  | leader_word:variant:gated | 4 |

- 排除理由的类型（不写词）：

  | 类型 | 个数 |
  |---|---|
  | 冷僻、认不出，或乱码 | 7 |
  | 纯数字、年数或日期片段 | 5 |
  | 社会新闻、网络流行语或刑事案件，不属于政治专项 | 4 |
  | 在这些文本里是别的意思（干扰手段、威胁用语、军车进城一类的泛指说法） | 4 |
  | 普通宗教用语 | 3 |
  | 外国的组织或事件，或者是乐队名 | 3 |
  | 含领导人姓氏用字的普通俚语或粗口 | 3 |
  | 枪支型号的买卖用语 | 2 |
  | 和全名或常用短语撞车的子串 | 2 |
  | 公开定论的历史事件，或者官方政策、革命人物 | 3 |
  | 集资诈骗组织 | 1 |
  | 普通地名 | 1 |
  | 泛指的通用词（包括一个外语里的“独立”） | 2 |
  | 小说里的门派 | 1 |

- 最后用 `--exclude-words` 重出六份标签，policy_digest `9900deabdbfdf07c`；每份的 `excluded_words` 都是 listed 41、in_candidates 41。

| 份 | usable 旧 → 新（减少） | safe / controversial / unsafe | political_rows（v1.2） | 从 normal 提档（v1.2） | `texts.length_mismatch` |
|---|---|---|---|---|---|
| runA_stage1 | 36,971 → 36,777（194，0.52%） | 29,886 / 3,658 / 3,233 | 1,167（1,319） | 13（24） | 0 |
| runA_dev | 5,172 → 5,148（24，0.46%） | 4,179 / 506 / 463 | 201（221） | 2（3） | 0 |
| prefix_v2 | 17,020 → 17,010（10，0.06%） | 14,997 / 454 / 1,559 | 20（37） | 0 | 0 |
| user_runA_stage1 | 18,527 → 18,515（12，0.06%） | 14,685 / 1,784 / 2,046 | 518（586） | 1（7） | 0 |
| user_runA_dev | 2,594 → 2,594（0） | 2,054 / 244 / 296 | 93（101） | 0 | 0 |
| user_prefix_v2 | 18,304 → 18,278（26，0.14%） | 15,667 / 989 / 1,622 | 72（111） | 0 | 0 |

- usable 的减少和 v1.2 完全相同，已证实全部来自 `general_scope`，没有超过 v1.3 定的 0.6%。
- `political_raised_from_stratum`：
  - runA_stage1：non_redline_harm 321，redline_topic 276，other_sensitive 143，redline 27，normal 13；
  - user_runA_stage1：redline_topic 178，non_redline_harm 126，other_sensitive 65，redline 15，normal 1。
- `leader_variant=safe` 和 `screen_words_formal_by_leader_form` 与 v1.2 相同：634 / 74 / 93 / 626 / 82 / 520，Run A 为 3,085。

### 第 8 步：新旧对比

| 份 | label_agreement | 主要变化 |
|---|---|---|
| runA_stage1 | 0.939 | safe→controversial 1,817，controversial→safe 344，unsafe→controversial 42，safe→unsafe 26，unsafe→safe 25 |
| runA_dev | 0.938 | safe→controversial 255，controversial→safe 45，safe→unsafe 13 |
| prefix_v2 | 0.987 | controversial→safe 119，safe→controversial 105 |
| user_runA_stage1 | 0.933 | safe→controversial 936，controversial→safe 206，unsafe→safe 58，unsafe→controversial 29，safe→unsafe 15 |
| user_runA_dev | 0.934 | safe→controversial 125，controversial→safe 26，unsafe→safe 10 |
| user_prefix_v2 | 0.955 | safe→controversial 561，controversial→safe 163，unsafe→safe 97 |

safe→unsafe 由 v1.2 的 70 降到 26（runA_stage1）。原因是领导人词现在要复查和最终核对都判负面才算负面。

### 第 9 步：人工核对（种子 `redline-review-v13`）

这次核对用的是排除 22 个词之后的标签。后来又多排除的 19 个词只会让命中变少，所以下面的误命中数是上限。

| 核对表 | 条数 | 问题 1 答“不是” | 门槛 | 问题 2 判错 |
|---|---|---|---|---|
| runA_stage1，political | 40 | 2（5%） | ✓ | 2 |
| prefix_v2，political | 16（只有这么多可抽） | 0 | ✓ | 0 |
| user_prefix_v2，political | 30 | 1（3.3%） | ✓ | 0 |
| runA_stage1，leader_word | 20 | 2（10%） | ✓（正好在线上） | 1 |

- runA 的 2 条是同一个词：1967 年香港的一场历史事件，分别落在 non_redline_harm 组和 other_sensitive 组。这个词我拿不准，所以留在词表里，核对时按“不是”算。normal 组 7 条、其余各组都是 0。
- user_prefix_v2 的 1 条：一个纯年数被当成了事件词。这个词已在后加的 19 个里排除。这一条原来就有代称，所以标签没错。
- 领导人词的 2 条：
  - 一个谐音变体被当作子串，匹配在一个普通短语里，已排除。这一条另一条规则本来就判“有争议”，所以标签没错。
  - 一个认不出的组合词，已排除。它被判成 unsafe，问题 2 算错。
- 另外，有几个以某个字形代替领导人姓的词，回答里没认出来，被当成普通骂人话用了，我按“是”算。这是字形变体，别处有很多同类的领导人变体可以对照。

### 第 10 步：拷到 PVC

六份 `labels.jsonl` 已拷到 `/work/round6/redline_v1/labels_political_v2/<名>/`。本地和 PVC 两边用 sha256 核对一致，旧的 `/work/round6/redline_v1/labels/` 没动。

| 文件 | SHA256（前 16 位） |
|---|---|
| runA_stage1 | `23d08606ea913a43` |
| runA_dev | `40f2ab4dace05dc0` |
| prefix_v2 | `86c6b8b2ce71a9ff` |
| user_runA_stage1 | `be97924ad4dadf17` |
| user_runA_dev | `b6b98b581e2451c8` |
| user_prefix_v2 | `93fcc0d1dd669ef9` |

`political_terms.jsonl` 没变，是 `d996817835144dec`。
