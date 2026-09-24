# 词表预筛：侮辱国家领导人的词（T012）

用户定了严格口径（2026-09-24）：对国家领导人（现任或历任）的侮辱性称呼、蔑称、贬损性谐音，只要原样写出就判 unsafe（A.1.b）。S2 pilot（[T006 反馈](../../collab/feedback/T006-s2-pilot.md)）发现：

- 生成模型识别不了这类词：人工抽看的 20 个词里有 2 个本该 skip，模型没有 skip；
- 其中一个是谐音，模型给它编了一个无害的解释，复核裁判也没有认出来。

所以改为在生成和重判之前，直接对词本身判一次。一次请求判 200 个词，只判词，不生成任何内容。用 luna 判，因为 DeepSeek 会拦截政治类词。

## 下游用法

- **S2、S5 安全半边**：生成前排除 yes 和 unsure 的词。
- **Run A**：yes 和 unsure 的词，它的“安全”回答按严格口径其实不安全。训练时整词先移出安全标签数据，交给 R 切片按国内口径重判。
- **R 切片重判**：把预筛结果作为额外证据，裁判漏掉的谐音以词表结果为准。

## 文件

| 文件 | 作用 |
|---|---|
| `flow/pipeline.py` | 提示词、schema、请求体（responses 和 chat_completions 两种接口）、逐条校验（index 和词必须原样返回） |
| `flow/flow.py` | 一个 Task 是一批 200 个词，只请求一次，不在 flow 内重试 |
| `make_batch.py` | 全量模式：Run A 和 Run B 的全部不同词各判一次。pilot 模式：按来源组抽 N 个词，再加上本地已知阳性清单 |
| `extract_screen.py` | 与本地词索引对齐，输出每个词的结论、各来源组计数、已知阳性的召回；没拿到结论的词单独列出，后续补判 |
| `compare_manual.py` | 与本地人工标注对照，只输出计数 |
| `test_word_screen_cpu.py` | 7 项 CPU 单测 |

`known_positives.txt` 是本地文件，一行一个已知阳性词，只在执行方的 Mac 上。它已写进 `.gitignore`，**不能提交**：仓库是公开的。

## v2（pilot aster-dev-289 之后）

v1 只问“是不是贬损”，回答 yes / no / unsure。pilot 结果（[T012 反馈](../../collab/feedback/T012-word-screen-pilot.md)）：
- yes 很准：19 个里 15 个是明确贬损，另外 4 个是领导人姓名的规避写法，没有无关词；
- unsure 大多是噪声：13 个里只有 2 个是真的；
- 随机抽 100 个政治类的 no，漏了 1–2 个，都是“领导人姓名加贬损性传言”这类短语。

v2 改为让模型给出类别：`insult`（侮辱、蔑称、贬损性谐音）/ `rumor`（把领导人和负面传言绑在一起的短语）/ `evasion`（姓名的规避写法，本身不贬损）/ `unsure` / `no`。下游：
- **排除出安全数据**：所有非 `no` 的词（S2、S5 生成前排除；Run A 训练时移出安全标签数据）。多排除的代价很小，每 1,000 个词约 30 个。
- **改标为 unsafe**：`insult` 按用户已定的严格口径改标；`rumor` 和 `evasion` 是否也改标，由用户决定（见 [BOARD](../../collab/BOARD.md)）。

`compare_manual.py` 把执行方本地的人工标注（`manual_labels.tsv`，已写进 `.gitignore`）和 v2 的结论逐词对照，只输出计数。
