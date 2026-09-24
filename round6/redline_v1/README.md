# 红线标注 v1：事实型裁判 v4 + 规则表 + 两级起点

口径以 [collab/feedback/POLICY-redline-scope.md](../../collab/feedback/POLICY-redline-scope.md) 为准（用户 2026-09-24 17:40 定稿）：只截 13 类红线；红线内复述、引用（含引用后反驳）、拒绝时点名标“有争议”，写出违规内容标“风险”，两档都截断；红线以外的有害内容标 safe；正常内容绝不能误判。

本目录取代 judge v3.2（R 切片，T016）和 onset_v1（起点重定，T010/T019）：一次运行同时给出整段标签和逐位置标签。

## 做法

1. **裁判只记事实**（`flow/pipeline.py`，`guard-judge-redline-v4`）。对一段文字（整段或前缀，裁判分不出来）记录：
   - `hits`：每处涉及红线的地方一条，含类别 R1–R13、做法 `act`、详略 `detail`、写法 `spelling`；
   - `leader_epithet_written`、`non_redline_harm`、`normal_content`。
2. **规则表定标签**（`flow/policy.py`）。默认值就是用户的定稿口径：

   | 事实 | 等级 |
   |---|---|
   | `writes`、`fiction` | unsafe |
   | `restates`、`quotes`、`refusal_naming` | controversial |
   | `topic_mention`、`debunk_no_quote` | safe |
   | R2 血腥、R11 色情只有笼统写法（一般打斗和战争叙述、擦边） | safe |
   | 变体写法意思不明确 | 忽略（safe）；意思明确的按正常写法算 |
   | 原样写出领导人侮辱称呼或传言短语 | unsafe（v3.2 的严格规则保留） |
   | 红线以外的有害内容 | safe（专门教模型放行） |

   口径以后再改，只改开关，用 `apply_policy.py --switches` 对已存的事实重算，不用重判。
3. **两级起点**（`flow/levels.py`）。红线内容一旦写出，更长的前缀等级只升不降。先判整段：safe 就结束（1 次调用）；否则在分句边界上二分，找到第一个达到“有争议”的分句，整段是 unsafe 时再找第一个达到“风险”的分句。每个起点后多问一个分句做确认。每级约 log₂(分句数) + 2 次调用。所有探针的事实都保存下来。
4. **逐位置标签**：`position_level(char_end, probes)`。每个探针都给前缀定了上下界：比它短的前缀等级不高于它，比它长的不低于它。上下界相等的位置有确定标签，起点分句内部的位置不训练。

## v4.1（T020 pilot 之后）

T020 的人工核对：剔除口径待定的样本后，Run A 91%、prefix_v2 100%；`normal` 分层没有误判；每条回答平均 1.5 次调用。改动如下：
- R10 限于公共事务（公共事件、灾害疫情、公共安全、政策、金融市场）。编造的冷知识、词源、网站来历不算，pilot 里有两条被错判成 R10。
- 用户还没定的边界单独记为 B1–B4：
  - B1：毒品买卖、招揽、吸食（不含制毒）；
  - B2：没有血腥细节的针对个人的暴力伤害（凶器袭击、投毒、藏尸、具体威胁）；
  - B3：管制刀具等枪支、爆炸物以外凶器的制作；
  - B4：针对具体企业、网站、产品的不实信息或伪造举报。

  规则表的建议值：B1–B3 算红线，B4 不算（归为商业诽谤）。用户定了以后只改开关。
- 新做法 `term_explained`：原样写出本身就违规的词或口号（邪教口号、对国家机构的侮辱性谐音）并加以解释，建议标“有争议”。
- 无法判断的领导人名字变体记为 variant_unclear，按口径放行。
- 先复述后写出的两种做法各记一条。
- `apply_policy.py` 新增两项输出：`boundary_hits_by_label`（各边界代码出现在哪些标签里），以及 `switch_sensitivity`（把每个待定开关翻过来，有多少条标签会变），供用户做决定时参考。

## 文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `flow/pipeline.py` | Aster | 裁判 v4 提示词、请求体、解析和一致性校验（例如写出领导人侮辱称呼必须同时有 R1 条目） |
| `flow/policy.py` | Aster + 本地 | 规则表，纯函数 |
| `flow/levels.py` | Aster + 本地 | 分句切点、两级二分、逐位置标签，纯函数 |
| `flow/flow.py` | Aster | 一个 Task 放多条回答；任务单带裁判版本和规则表摘要，不一致就报错 |
| `make_tasks.py` | Mac | 从 v14 格式的行（Run A、第一阶段输入、S2/S5）建任务；只发送提问和回答，其余留在本地答案表。`--sample-ids` 用于兜底裁判重跑失败条目 |
| `extract_probes.py` | Mac | 下载归档，与答案表对齐，输出 `probes.jsonl` 和 `failed_ids.txt` |
| `apply_policy.py` | Mac | 按规则表从事实算标签；多个 `probes.jsonl` 按优先级合并（DeepSeek 在前，luna 兜底）；报告旧标签到新标签的转移、分层和规则命中 |
| `compare_judges.py` | Mac | 两个裁判在同一批回答上的标签一致率、截断一致率和起点偏移（各自单独 `apply_policy.py` 后比较） |
| `export_prefix_v2.py` | 集群或 Mac | 把第五轮 prefix_v2 的助手侧记录导出成 v14 格式的行，原样保留 messages；输出含数据集原文，不提交 |
| `review_sample.py` | Mac | 按“旧标签 → 新标签”分层抽样，生成本地人工核对表（在原文里标出起点分句）；输出在 `review/` 下，已加入 .gitignore，不提交 |
| `test_redline_cpu.py` | Mac | 23 项 CPU 单测，含一个用假模型跑通 flow、抽取和重算的端到端测试 |

## 还没做的

- **提问侧**：口径要求提问也按同一份红线清单判。第一阶段只重训助手侧读出头，提问侧的裁判模式放到后面。
- **prefix_v2**：第五轮数据要先用 `export_prefix_v2.py` 导出成 v14 格式的行，才能走本流程。
- **官方 Qwen3GuardTest**：标签口径和本口径不同，第一阶段只观察；要按本口径评测，需要单独标注。
