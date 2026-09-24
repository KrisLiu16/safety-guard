# T026 政治专项：政治敏感词预筛，按专项规则重出全部标签

- 版本：v1.3
  - v1.2 停在第 9 步：Run A 的专项命中有 15% 不是政治敏感词，领导人词是 25%。这些误命中，三遍模型结论都一致，都是词本身冷僻或太泛。另外 runA_stage1 的 usable 少了 0.52%。
  - v1.3 的改动：
    - **usable**：这 0.52% 全部来自 716ef02 的 `general_scope`（执行方已用对照证实），设计方接受；这个开关带来的减少，上限放宽到 0.6%。
    - **本地排除表**（执行方的建议）：新增第 7b 步，列出所有在普通文本里命中过的词，由执行方逐个核对，把不是特指政治敏感内容的写进本地排除文件，再用 `apply_policy.py --exclude-words` 重出标签。排除表不提交。
    - **两个汉字、带 ambiguous 的词**：一律不参与匹配。
    - **领导人词的负面和变体**：只有复查（l1）和最终核对（k1）都判负面时才算负面（风险），不一致时按变体（有争议）。
    - **前后两段结论跨组的词**（初筛判为事件、人物或组织，最终核对判为领导人词，或者反过来）：只在政治文本里生效，最高到有争议。
  - 第 1–5c 步不重跑。
- 更早的版本：v1.2
  - v1.1 停在第 9 步：放到原文里核对时，prefix_v2 有 37% 命中的词不是政治敏感词，Run A 的领导人词是 70%。原因：
    - 短词被子串匹配：两个字母的拼音匹配到英文单词里，两个汉字的人名片段匹配到常用词里；
    - 泛指的历史事件、公开讨论过的历史运动、普通年份被当成事件；
    - 外国恐怖组织被当成组织；
    - 不指向任何人的粗俗用语被当成领导人蔑称；
    - 政治文本的门槛太低，种族、恐怖主义文本也会打开门槛。
  - v1.2 的改动：
    - 新增第 5c 步：对所有候选词做一遍“最终核对”，只有特指中国政治敏感内容的才匹配，类别以这一遍为准；
    - 拉丁字母的词至少 3 个字母，只按完整单词匹配；
    - 两个汉字的词和纯数字，只在政治文本里生效；
    - 政治文本只看中国政治类红线（R1、R3–R6、R9），不再包括恐怖主义（R7）和民族仇恨（R8）；
    - 最终核对判为非领导人的词，T012 预筛的侮辱或传言结论也不再用于种子词。
  - v1.1 的六份标签留在本地的 `labels_political_v1/`，已提交的 summary 作为记录。v1.2 输出到 `labels_political_v2/`。第 1–5b 步不重跑。
- 更早的版本：v1.1
  - v1 停在第 6 步：leader_unsure 一类里 15% 是普通词或不是政治敏感词，门槛是 10%。原因有三：
    - 领导人姓氏的单字和普通名词拼出来的词；
    - 繁体字、标准拼音的正常写法被当成变体；
    - figure 里混进了正常履职的官员、烈士和公众人物。
  - v1.1 新增第 5b 步：对 leader 词单独做一遍“怎样提到领导人”的复查，只有 negative、variant 两类才匹配。
  - figure 改为只在裁判找到政治红线的文本里生效。
  - “政治红线”不再包括正式写法的领导人新闻报道。
  - 第 1–5 步的产物不用重跑。
- 背景：用户 2026-09-25 决定，最后针对中国政治敏感词和话题做专项加强（见任务总表“已定”）：
  - 特殊事件一提就截（有争议）；
  - 领导人：正常新闻照常放行；负面、嘲讽、谣言照旧是风险；用变体写法提到领导人，标有争议；
  - 其他政治话题按现行红线规则判，再补专门数据、收紧阈值（放在后续任务）。

  设计说明见 [round6/political_screen_v1/README.md](../../round6/political_screen_v1/README.md)，标签改动见 [round6/redline_v1/README.md](../../round6/redline_v1/README.md) 的“政治专项”一节。
- 目标：用 luna 对 T012 的全部 340,883 个词做一遍政治预筛，再复查一遍，得到本地词表 `political_terms.jsonl`。然后在 Mac 上按新规则表和这份词表，重出 T021、T024 的六份标签。新标签放到新目录 `labels_political_v2/`，旧标签不动。**不重判，不用 L20，不训练。** 用新标签重训放在后续任务，等 T018、T025 的结果出来再派。
- 依赖：T012 全量 ✓（Mac 上的 `word_screen_v1/full/words.jsonl` 和 `full/extracted/screen.jsonl`）；T021 ✓、T024 ✓（Mac 上各目录的 `probes.jsonl`、`trainable.jsonl`、`input/prefix_v2_*.jsonl`）。
  - Aster 部分可以和 T018、T025 的 L20 作业同时跑。同一账号的 Run 会排队，建议高优先级。
- 步骤：
  1. **单测**（Mac）。三个文件分开跑，因为每个目录都有 `flow/pipeline.py`：
     ```bash
     .venv/bin/python -m unittest round6/political_screen_v1/test_political_screen_cpu.py
     .venv/bin/python -m unittest round6/redline_v1/test_redline_cpu.py
     .venv/bin/python -m unittest round6/word_screen_v1/test_word_screen_cpu.py
     ```
     应分别是 9、39、12 项，全过（v1.3）。
  2. **第一遍**（全部词）：
     ```bash
     .venv/bin/python round6/political_screen_v1/make_batch.py --output round6/political_screen_v1/full_p1
     ```
     manifest 应为 `words` 340,883，`tasks` 1,705，只有一个 `screen.tar.gz`。
     - 上传为一个数据集，提交一个 Run：luna，`--flow round6/political_screen_v1/flow`，尝试 1 次，并发 512，高优先级。
     - run_no 存到 `full_p1/run_no.txt`。
     - 完成后冻结 attempt 列表：
       ```bash
       .venv/bin/python round6/response_v14/freeze_attempts.py <run_no_p1> --out round6/political_screen_v1/full_p1/attempts.json --expected-samples 1705
       ```
  3. **抽取第一遍**：
     ```bash
     .venv/bin/python round6/political_screen_v1/extract.py <run_no_p1> --attempts-json round6/political_screen_v1/full_p1/attempts.json --words round6/word_screen_v1/full/words.jsonl --leader-screen round6/word_screen_v1/full/extracted/screen.jsonl --out round6/political_screen_v1/full_p1/extracted
     ```
     - `words_without_verdict` 不超过 1%（3,409 个）才能继续，否则停下写反馈。
     - 大于 0 时，用 `--only round6/political_screen_v1/full_p1/extracted/missing_words.txt --tag r1 --output round6/political_screen_v1/full_r1` 补筛一次，提交方式同第 2 步。
  4. **复查**（换一种分批顺序）：
     ```bash
     .venv/bin/python round6/political_screen_v1/make_batch.py --only round6/political_screen_v1/full_p1/extracted/confirm_words.txt --tag c1 --output round6/political_screen_v1/full_c1
     ```
     提交方式同第 2 步，run_no 存到 `full_c1/run_no.txt`。Task 数超过 100 时同样先冻结 attempt 列表。
  5. **合并各遍**：
     ```bash
     .venv/bin/python round6/political_screen_v1/extract.py <run_no_p1> [<run_no_r1>] <run_no_c1> --attempts-json <对应的 attempts.json，顺序相同> --words round6/word_screen_v1/full/words.jsonl --leader-screen round6/word_screen_v1/full/extracted/screen.jsonl --review-sample 40 --out round6/political_screen_v1/full/extracted
     ```
     合并完成后，删除 p1、r1、c1 三个数据集。它们只是筛词用的，标签用的是本地的合并结果。
  5b. **（v1.1）复查 leader 词**：
     - 先用 v1.1 的代码在同一批 Run 上重跑第 5 步，输出到同一目录。归档已在本地，不会重新下载。这一遍会多出 `leader_words.txt`，约 14,295 个词。然后：
       ```bash
       .venv/bin/python round6/political_screen_v1/make_batch.py --only round6/political_screen_v1/full/extracted/leader_words.txt --tag l1 --mode leader --output round6/political_screen_v1/full_l1
       ```
       manifest 的 `mode` 应为 `leader`，约 72 个 Task。
     - 提交方式同第 2 步：luna，`--flow round6/political_screen_v1/flow`，尝试 1 次，高优先级。
     - 冻结 attempt 列表后，把 l1 的 run_no 和 attempts 加在第 5 步命令的末尾，**换一个核对种子**再合并一次：
       ```bash
       .venv/bin/python round6/political_screen_v1/extract.py <p1> <r1> <c1> <l1> --attempts-json <四个，顺序相同> --words round6/word_screen_v1/full/words.jsonl --leader-screen round6/word_screen_v1/full/extracted/screen.jsonl --review-sample 40 --review-seed r2 --out round6/political_screen_v1/full/extracted
       ```
     - 删除 l1 的数据集。
  5c. **（v1.2）最终核对**：
     - 用 v1.2 的代码按第 5b 步最后那条合并命令再合并一次，得到 `check_words.txt`，即全部候选词：event、figure、org，以及形式为 negative 或 variant 的领导人词，约 2.8 万个。然后：
       ```bash
       .venv/bin/python round6/political_screen_v1/make_batch.py --only round6/political_screen_v1/full/extracted/check_words.txt --tag k1 --mode check --output round6/political_screen_v1/full_k1
       ```
       manifest 的 `mode` 应为 `check`。
     - 提交方式同第 2 步。
     - 冻结 attempt 列表后，把 k1 加在合并命令的末尾，改用 `--review-seed r3` 再合并一次。
     - 删除 k1 的数据集。
  6. **核对词表**（Mac，本地）。v1.2 用第 5c 步新抽的核对表，全部类别重新核对：打开 `full/extracted/review/terms_sample.txt`。每一类最多 40 个词，逐个回答：对 / 普通词（单独出现不该截）/ 类别错 / 不是政治敏感词。
     - 以下各类里，“普通词”加“不是政治敏感词”不超过 10%，才能继续：
       - 不带 `:ambiguous` 的 event、org、leader_negative、leader_variant；
       - v1.2 起类别由最终核对决定，核对时特别看泛指的历史、普通年份、外国组织、不指向具体人的词还有没有漏过；
       - figure 也照这个门槛看，但它现在只在政治文本里生效。

       超过就停下写反馈，只报计数和理由，不写词本身。
  7. **重出标签**（Mac CPU，每条命令约一分钟）。新标签输出到 `round6/redline_v1/labels_political_v2/`：
     ```bash
     T=round6/political_screen_v1/full/extracted/political_terms.jsonl
     S=round6/word_screen_v1/full/extracted/screen.jsonl
     A=round6/response_v14/batch_50k/extracted/trainable.jsonl
     R=round6/redline_v1
     .venv/bin/python $R/apply_policy.py $R/full_runA_stage1/extracted_ds/probes.jsonl $R/full_runA_stage1_fb/extracted/probes.jsonl --screen $S --political-terms $T --source $A --out $R/labels_political_v2/runA_stage1
     .venv/bin/python $R/apply_policy.py $R/full_runA_dev/extracted_ds/probes.jsonl $R/full_runA_dev_fb/extracted/probes.jsonl --screen $S --political-terms $T --source $A --out $R/labels_political_v2/runA_dev
     .venv/bin/python $R/apply_policy.py $R/full_pv2/extracted_ds/probes.jsonl $R/full_pv2_fb/extracted/probes.jsonl --political-terms $T --source $R/input/prefix_v2_assistant.jsonl --out $R/labels_political_v2/prefix_v2
     .venv/bin/python $R/apply_policy.py $R/user_full_runA_stage1/extracted_ds/probes.jsonl $R/user_full_runA_stage1_fb/extracted/probes.jsonl --target user --screen $S --political-terms $T --source $A --out $R/labels_political_v2/user_runA_stage1
     .venv/bin/python $R/apply_policy.py $R/user_full_runA_dev/extracted_ds/probes.jsonl $R/user_full_runA_dev_fb/extracted/probes.jsonl --target user --screen $S --political-terms $T --source $A --out $R/labels_political_v2/user_runA_dev
     .venv/bin/python $R/apply_policy.py $R/user_full_pv2/extracted_ds/probes.jsonl $R/user_full_pv2_fb/extracted/probes.jsonl --target user --political-terms $T --source $R/input/prefix_v2_user.jsonl --out $R/labels_political_v2/user_prefix_v2
     ```
     - 兜底目录以本地实际路径为准，和 T021、T024 定标签时用的相同，见各份旧 `labels/*/summary.json` 的 `inputs`。
     - `--source` 覆盖不到 99% 的编号时，脚本会报错。
     - 每份的 `texts.length_mismatch` 应为 0。不为 0 时报告条数；这些条目不套用按词规则，其余照常。
  7b. **（v1.3）本地排除表**：
     - 先用 v1.3 的代码按第 7 步重出六份标签（输出到同一目录）。
     - 然后列出命中过的词：
       ```bash
       R=round6/redline_v1; A=round6/response_v14/batch_50k/extracted/trainable.jsonl
       .venv/bin/python $R/list_fired_words.py --labels $R/labels_political_v2/{runA_stage1,runA_dev,prefix_v2,user_runA_stage1,user_runA_dev,user_prefix_v2}/labels.jsonl --source $A $A $R/input/prefix_v2_assistant.jsonl $A $A $R/input/prefix_v2_user.jsonl --target assistant assistant assistant user user user --out $R/labels_political_v2/review_words/fired_words.tsv
       ```
     - 逐个核对 normal、non_redline_harm、other_sensitive 三列有非零计数的词，也就是把普通文本提档的词。必要时对照例子原文。满足下面任一条，就把词写进 `round6/political_screen_v1/full/extracted/exclude_words.txt`（一行一个，已被 gitignore）：
       - 不是特指中国政治敏感事件、人物、组织或领导人；
       - 冷僻得认不出；
       - 在这些文本里明显是别的意思。

       拿不准的不排除。
     - 再用 `--exclude-words round6/political_screen_v1/full/extracted/exclude_words.txt` 把六份标签重出一次。
     - 反馈里只报：核对了多少个词、排除了多少个（按 rule 分）、排除理由的类型。不写词本身。
  8. **新旧对比**（Mac CPU），六份都做：
     ```bash
     .venv/bin/python round6/redline_v1/compare_judges.py round6/redline_v1/labels/<名>/labels.jsonl round6/redline_v1/labels_political_v2/<名>/labels.jsonl --out round6/redline_v1/labels_political_v2/<名>/compare_old.json
     ```
  9. **人工核对专项规则**（Mac，本地，表格不提交）：
     ```bash
     .venv/bin/python round6/redline_v1/review_sample.py --only-rule political --labels round6/redline_v1/labels_political_v2/runA_stage1/labels.jsonl --source round6/response_v14/batch_50k/extracted/trainable.jsonl --count 40 --out round6/redline_v1/labels_political_v2/review_runA
     .venv/bin/python round6/redline_v1/review_sample.py --only-rule political --labels round6/redline_v1/labels_political_v2/prefix_v2/labels.jsonl --source round6/redline_v1/input/prefix_v2_assistant.jsonl --count 30 --out round6/redline_v1/labels_political_v2/review_pv2
     .venv/bin/python round6/redline_v1/review_sample.py --only-rule political --target user --labels round6/redline_v1/labels_political_v2/user_prefix_v2/labels.jsonl --source round6/redline_v1/input/prefix_v2_user.jsonl --count 30 --out round6/redline_v1/labels_political_v2/review_user_pv2
     ```
     再用 `--only-rule leader_word` 对 runA_stage1 抽 20 条。v1.3 在这几条命令后面都加上 `--seed redline-review-v13`，换一批样本。
     - v1.1 的规则名：`political:<类>`、`political:<类>:gated`（只在政治文本里生效的词）、`leader_word:negative`、`leader_word:variant`。
     - 表里 ⟦P⟧…⟦P|⟧ 是命中的词。每条回答两个问题：
       1. 这个词在这里确实指政治敏感事件、人物、组织或领导人吗（是 / 不是，是普通用法）？
       2. 新标签对不对？
     - 按“原分层”分开计数，`normal:` 开头的组是重点。
     - 门槛：问题 1 答“不是”的不超过 10%。
  10. **拷到 PVC**：第 6 步和第 9 步都过了门槛，才把六份 `labels_political_v2/<名>/labels.jsonl` 拷到 PVC 的 `/work/round6/redline_v1/labels_political_v2/<名>/`，用 `sha256sum` 核对。旧的 `/work/round6/redline_v1/labels/` 不动，T018、T025 仍然用旧标签。
- 预期产物：可提交的有：
  - `full_p1/`、`full_c1/`（以及 `full_r1/`）的 `manifest.json`、`run_no.txt`、`dataset_rm.json`；
  - `full/extracted/summary.json`；
  - 六份 `labels_political_v2/*/summary.json` 和 `compare_old.json`。

  以下内容都含词或原文，**不提交**（.gitignore 已覆盖，提交前用 `git status` 再看一眼）：`political_terms.jsonl`、`confirm_words.txt`、`missing_words.txt`、`review/`、`labels.jsonl`、attempts、归档。
- 验收：
  - 第 3 步没有词的比例 ≤ 1%；
  - 第 6、9 步的门槛；
  - 六份标签的 `usable` 比旧标签少不超过 0.5%（v1.3：其中来自 `general_scope` 的部分放宽到 0.6%）。
    - 按词规则只改等级，不改可用性。
    - 但规则表把变体写法的 R1 改成了“有争议”。如果裁判只在部分探针上记了这一条，那条回答会变成不单调、不可用。
    - 超过 0.5% 就停下，报告减少的条数。
- 需要用户决定：无。口径已定（2026-09-25）。人物和组织也按“一提就截”处理，这是设计方按用户选项的理解；如果核对发现误伤太多，会在反馈后单独请用户确认。
- 反馈里必须报告：
  - 单测结果；
  - 各遍的 run_no、Task 数；
  - 最终 `summary.json` 全文，特别是 `verdicts`、`ambiguous_by_verdict`、`leader_screen_by_verdict`、`matched_as`、`words_without_verdict`、`batch_errors`；
  - 第 6 步每一类的计数；
  - 六份标签的 `labels_by_split`、`old_to_new`、`screen_overrides`、`political_table`、`political_rows`、`political_overrides`、`political_raised_from_stratum`、`texts`、`switch_sensitivity` 里的 `leader_variant=safe`；
  - 六份 `compare_old.json` 的 `label_agreement` 和 transitions；
  - （v1.1）第 5b 步的 run_no、合并后 `summary.json` 里的 `leader_form`、`leader_form_by_leader_screen`、`matched_as`；每份标签的 `screen_words_formal_by_leader_form`；
  - （v1.2）第 5c 步的 run_no、`check_words`、`check_by_candidate`、新的 `matched_as`；第 9 步按同样的分组计数；
  - 第 9 步的计数和理由（不写原文、不写词）；
  - 产物 SHA256。
