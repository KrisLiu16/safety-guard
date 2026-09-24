# 21 层候选的质量恢复实验

首轮 L20 对照显示 14 层 C1 的 ITPS 约 2 倍于 A0，却在 Qwen3GuardTest 子集上严重误报与漏判。因此先测试一个**保留 21/28 层且保留最后一层**的直接分类候选 C21，而不是继续把 14 层吞吐当成可用收益。[权重映射脚本](build_c21_student.py)每四层去掉第三层，保留原分类头和 tokenizer，无 LM 输出头；[初始化清单](checkpoints/c21_keep_final_init/build_manifest.json)记录具体层索引与权重哈希。

诊断集来自固定版本的 Nemotron 中文 `valid`，与 Stage0 使用的 `train` 来源 split 分离。[冻结清单](data/source_valid_probe_manifest.json)包含 400 条人标用户样本（safe/unsafe 各 200）、200 条 LLM jury 回答样本（各 100）。这些是**来源政策标签**，不等于本项目的政治公共事务政策金标，也不用于训练。先在同一 L20 上量 A0、C14 Stage0、未训练 C21 的 Unsafe AUC、argmax 误报/召回与精确 token-ID 追加 ITPS；[执行脚本](probe_c21_cuda.py)固定了完全相同的样本和负载。

推进条件：C21 的用户与回答 Unsafe AUC 均不能比 A0 低超过 0.05，且 argmax 安全样本误报率不能比 A0 高超过 0.05；同时至少一个目标流式负载的 ITPS 须稳定优于 A0。条件通过才用开放语料对 C21 做新的同前缀蒸馏，再用未参与选择的独立基准验证。即使通过来源标签条件，也不能直接进入政策安全服务；最终仍需金标、校准和真实网络/排队测试。
