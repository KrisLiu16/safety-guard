# 内部候选词表

`sources/sensitive-lexicon/` 是 [konsheng/Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon) 在 `d967c30b053fa40b06c5a0dddf0be493f2dfae46` 的完整词表快照：17 个 `Vocabulary/` 文件、`Organized/`、TrChat JSON、README 和 LICENSE。`source_lock.json` 记录每个原始文件的 SHA-256。原仓库声明 MIT 许可，原文和版权声明保存在 `sources/sensitive-lexicon/LICENSE`；部分文件名暗示第三方来源，未独立核实这些上游词条的出处。

`build.py` 核验源文件后生成不可覆盖的 `derived/d967c30b/`。它对每条非空记录做首尾空白清理、NFKC 与 Unicode casefold 去重，仍保留每个原写法、来源文件和行号或 JSON 数组位置：

- `master.jsonl`：每个规范化候选项一条，含所有来源、别名、类型与审核状态。
- `source_rows.jsonl`：每一条原始来源记录，便于独立重算和排查。
- `text_candidates.txt`：普通文本候选词。
- `url_indicators.txt`：原“非法网址”文件的候选项。
- `combination_candidates.jsonl`：含 `+` 的候选记录，尚未解释加号的匹配语义。
- `summary.json`：数量、各文件来源和输出校验值。
- `task_seeds.jsonl`：每个普通文本候选项一个稳定 task_key，供将来的例句生成任务直接读取；尚未调用模型或分配安全标签。

同一项可能有多个原始来源；为了生成任务，这三个用途文件按优先级互斥分类：含 `+` 的组合候选、原网址表中的网址指标、其余普通文本。`verify.py` 独立重算源文件中的规范化词集合和出处数量，检查三个用途文件恰好覆盖主表且互不重叠，并核对所有产物哈希。

这是项目内部整理的候选词表，不是我们独占拥有的原始词汇，也不是官方敏感词清单。所有记录目前都是 `unreviewed_candidate`，`auto_block=false`。词条命中只能作为数据生成、复核或语义分析的线索，不能直接当作违规标签；普通政治讨论、批评、历史、文学与动画语境必须保留安全样本。

第一版范围是用户当时查看的 Sensitive-lexicon 仓库全部词表。随后在用户明确非商业用途后，Citizen Lab 数据已进入单独的非商业研究层，见 `research_nc/README.md`。`jasonqng/chinese-keywords` 的当前仓库树中未找到许可文件；中国数字时代研究页面未确认可供再分发的批量数据许可，仍未混入可发布的候选数据。

进一步扩展见 `RESOURCE_CATALOG.md`：已补入 houbb 的 Apache-2.0 候选词表，fwwdn 作为零增量镜像留档，COLDataset 作为独立句子级评测语料。新构建在 `derived/expanded-20260923/`，通过 `build_expanded.py`、`verify_expanded.py` 复现与校验。

构建命令：

```bash
python3 /Users/liuzhihao/Work/safety-guard/lexicon/build.py
python3 /Users/liuzhihao/Work/safety-guard/lexicon/verify.py
python3 /Users/liuzhihao/Work/safety-guard/lexicon/export_tasks.py
```
