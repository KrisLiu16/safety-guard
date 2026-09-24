# T026 反馈：政治专项（v1.1：第 9 步没过门槛，已停下）

- 对应任务单版本：v1（4ea0271）
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
