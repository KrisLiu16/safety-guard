# T026 反馈：政治专项（按第 6 步停下）

- 对应任务单版本：v1（4ea0271）
- 状态：**第 6 步没过门槛，已停下**。leader_unsure 类的“普通词”加“不是政治敏感词”占 15%（6/40），超过 10%。第 1–5 步已完成，第 7 步起没有做，没有出任何新标签，也没有拷到 PVC。

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
