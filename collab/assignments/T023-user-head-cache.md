# T023 提问侧（用户头）特征缓存

- 版本：v1
- 目标：按定稿口径，提问也要用同一份红线清单来判（POLICY 第 1 节）。第一阶段目前只重训回答侧的头（T018），用户头还是第五轮的旧口径。这一步先把用户头的输入特征缓存下来，做法和 T017 相同，只推理、不训练。特征不依赖标签，等提问侧的红线标签出来后（另出任务单），直接在缓存上重训用户头。
- 内容（`cache_features_l20.py --role user`）：
  - prefix_v2 训练集和校准集的 user 角色记录，沿用它们自己的锚点；
  - Run A 第一阶段输入中每个不同的提问（每个词一个 unsafe 提问、一个 safe 提问，train 8,000 词加 calibration 1,264 词，约 1.85 万条），只放用户消息本身，每条最多 24 个位置。位置标签暂时用旧的提问标签占位，以后按红线标签重写，与 T018 的做法相同。
  - 同时保存现在的用户头权重 `head_user_init.pt`。
- 依赖：无。L20 空闲即可。不用数据平台，和 T021、T012 不冲突。
- 步骤：
  1. 单测（Mac，小张量，不跑模型）：`.venv/bin/python -m unittest round6/stage1_head/test_stage1_cpu.py`，7 项全过（新增用户角色一项；有 5 项需要 torch）。
  2. 提交作业：`kubectl apply -f round6/stage1_head/worker_cache_user_v1.yaml`。Pod 进入 Running 后：
     - `round6/stage1_head/*.py` 拷到 `/work/round6/stage1_head/`；
     - `round6/probe/*.py` 拷到 `/work/round6/probe/`；
     - `input_v1/` 已经在 PVC 上（T017 用过），核对 SHA 与 manifest 一致即可。

     用 `sha256sum` 核对后 `touch /work/round6/stage1_cache_user_v1_input_ready`。
  3. 完成后 `stage1_cache_user_v1_exit_code.txt` 为 `cache=0`。取回 `report.json` 到 `round6/stage1_head/results_cache_user_v1/`，核对后 `touch /work/output/round6/stage1_cache_user_v1_collected`。缓存本身留在 PVC 的 `/work/output/round6/stage1_cache_user_v1/`。
- 预期产物：提交 `results_cache_user_v1/report.json`。
- 验收：
  - `status=completed`、`integrity_pass=true`；
  - `role` 为 `user`；
  - `max_prob_diff` < 0.001；
  - `stats` 里有 `prefix_v2` 和 `runA_prompts` 两项，`runA_prompts` 的 train 记录数接近 16,000，calibration 接近 2,528（每个词 2 条）。
- 需要用户决定：无。提问侧的档位（见任务总表“待用户决定”）只影响之后的标签，不影响这一步。
- 反馈里必须报告：单测结果；report 的 status、integrity_pass、max_prob_diff、elapsed_seconds、stats、head_init_sha256；产物 SHA256。
