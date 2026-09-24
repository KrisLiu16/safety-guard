# 中文安审资源目录（2026-09-23）

## 已落地、已校验

| 来源 | 本地用途 | 实测增量 | 许可与边界 |
| --- | --- | ---: | --- |
| [Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon) | 主候选词表，17 个 Vocabulary 文件与附带 TrChat JSON | 51,085 个规范化候选项 | 仓库声明 MIT；保留原 LICENSE。文件名中的第三方出处未独立验证。 |
| [houbb/sensitive-word-data](https://github.com/houbb/sensitive-word-data) | 补充词表、原始标签代码、白名单对照 | 相对主表新增 56,189 项（含 9 个英文辅助词） | 仓库声明 Apache-2.0；标签代码原样保留，不解释为我们的违规类别。 |
| [fwwdn/sensitive-stop-words](https://github.com/fwwdn/sensitive-stop-words) | 镜像词表与停用词对照 | 15,745 个去重词条对主表增量为 0 | 仓库声明 Apache-2.0；不重复并入主表。停用词不是天然安全标签。 |
| [SpaceGather/worldwide-sensitive-word-collection](https://github.com/SpaceGather/worldwide-sensitive-word-collection) | 简繁中文的色情、粗俗与暴力小词表 | 相对前两源新增 8 个繁体写法 | 仓库声明 MIT；简体部分重叠，空文件保留但不计词条。 |
| [COLDataset](https://github.com/thu-coai/COLDataset) | 独立中文冒犯语言评测：训练集 25,726、开发集 6,431、测试集 5,323 | 37,480 条评论；不计入词表 | 仓库声明 Apache-2.0；保持原始划分和标签，不用于本次词表生成。 |

原始文件及许可分别在 `sources/`、`corpora/`；各源的 Git commit 和 SHA-256 在 `source_lock.json`、`additional_source_lock.json`。两个不可覆盖构建分别在 `derived/d967c30b/` 与 `derived/expanded-20260923/`，后一版有 107,282 个规范化候选项：92,492 个普通文本项、14,592 个网址指标、198 个加号组合候选。所有候选项均为 `auto_block=false`，没有自动获得违规标签。`verify.py` 与 `verify_expanded.py` 分别校验两版的覆盖、来源数量、分区和哈希。

用户明确项目为非商业开源研究后，已另建 [Citizen Lab](https://github.com/citizenlab/chat-censorship) 的 CC BY-NC-SA 4.0 研究层：477,165 个其来源的去重候选项，与 107,282 项宽松许可基础表合并后为 492,640 项。原始归档、来源文件目录、非商业候选索引和 357,083 个新增普通文本任务种子见 `research_sources/citizenlab/`、`research_nc/`。该层仅以非商业研究数据身份使用，不改变基础表原有许可。

## 已发现，未混入可复用主表

| 来源 | 价值 | 当前边界 |
| --- | --- | --- |
| [jasonqng/chinese-keywords](https://github.com/jasonqng/chinese-keywords) | 汇合 13 份有来源说明的历史列表；其敏感词 CSV 实测 8,988 个去重键，492,640 项非商业索引中已覆盖 8,832 个 | 仓库当前树未找到 LICENSE；仅剩 156 个新增键，暂不混入可发布数据。源数据主要来自 2004–2014 年。 |
| [中国数字时代敏感词开源研究项目](https://chinadigitaltimes.net/chinese/sensitive-words-project) | 带发现时间、平台与出处的动态政治词线索 | 页面未确认可供我们批量再分发的统一许可；不同平台、时期的观察不能直接充当安全标签。 |
| [Qwen3GuardTest](https://huggingface.co/datasets/Qwen/Qwen3GuardTest) | 现有模型的官方验收基准，上一轮已使用 | CC BY-NC 4.0；保持独立测试，不导入词表或训练池。 |
| [CHiSafetyBench](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CHiSafetyBench) | 中文安全分类与对话历史评测，5 个领域、31 个类别 | 作为补充 benchmark 候选；先核对数据许可与标签定义，不和官方 test 混用。 |
| [Chinese Sensitive Topics QA](https://huggingface.co/datasets/CharlesBon/chinese-sensitive-topics-qa) | 100 条政治/历史问题及分析式回答，可观察不必要的拒答 | 虽标 Apache-2.0，但内容为英文且由模型生成，不能视作独立中文人工金标。 |
| [THUOCL](https://github.com/thunlp/THUOCL) | 大量普通中文词汇，可帮助设计误拦和歧义对照 | 不是敏感词表，不并入风险词库。 |

## 政策和持续更新

[TC260-003《生成式人工智能服务安全基本要求》](https://www.tc260.org.cn/upload/2024-03-01/1709282398070082466.pdf)和[国家网信办《生成式人工智能服务管理暂行办法》](https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm)提供分类与标注规则设计的官方依据，均不是可直接下载的“官方全量敏感词库”。新词覆盖应由来源追踪、真实误判/漏判案例、人工复核和版本化更新形成；平台曾经屏蔽某词不等于我们的模型应拒绝所有含该词的正常讨论。
