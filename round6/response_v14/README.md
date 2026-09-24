# 第六轮造数 v14

格式与 [v13](../response_v13/README.md) 相同：一词一请求，2×2 配对，unsafe 回答带逐字起点引用。改动全部来自 [v13 pilot 复核](../response_v13/PILOT_REVIEW.md)：

1. 提示词要求：有害部分必须是回答在鼓励、建议或直接实施，不能写成“不要/避免这样做”；unsafe 回答任何位置都不评价自身的危害；编造的细节不用“虚构/fictional”标注。跑偏格的提示也同步收紧。
2. 程序拒收 `negated_onset`：起点片段前 12 个字符内紧挨着否定词（不要、避免、不应、do not、avoid 等）。
3. 程序记录 `quality_flags`（`harm_commentary`、`fiction_marker`），只打标记、不拒收，供训练前筛选和统计。

`test_pipeline.py` 含 v13 真实数据回放：新规则恰好拦下 v13 的 5 条误标。`pilot/` 是 50 个新词，排除了 round4/round5 用过的词和 v13 pilot 的词。
