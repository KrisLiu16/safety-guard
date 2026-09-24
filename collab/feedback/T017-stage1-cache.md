# T017 反馈

- 对应任务单版本：v1
- 状态：**完成，验收通过**
- 执行时间：2026-09-24 15:52 至 17:47（北京时间）。之后 17:58 取回结果、写入 collected，Pod 17:58 以 Succeeded 退出，GPU 已释放。
- 实际执行的命令：与任务单相同。

## 结果

1. **单测**：`round6/stage1_head/test_stage1_cpu.py` 5 项全过。
2. **输入 manifest**：`round6/stage1_head/input_v1/manifest.json`，已提交。
   - records 37,056；words_by_split 为 train 8,000、calibration 1,264。
   - records_by_split_label：train 的 safe、unsafe 各 16,000，calibration 的 safe、unsafe 各 2,528。
   - `stage1_runA_v1.jsonl` 的 SHA256 为 `4e4afa5deefadaea694183b8988e919000b1c995cb62b0ca292f89499b30c8f9`，与 manifest 和 report 的 `inputs.runA` 一致。
3. **作业**：
   - Job `safety-guard-stage1-cache-20260924-r1`，Pod `safety-guard-stage1-cache-20260924-r1-wp4dn`。
   - 15:52:33 提交，15:53:10 写入 input_ready，17:47 写入 done。
   - `stage1_cache_v1_exit_code.txt` 为 `cache=0`。
4. **report.json**：
   - `status=completed`，`integrity_pass=true`，`phase=finished`，`training_performed=false`，`generated_tokens=0`。
   - `max_prob_diff` = 6.85e-7。
   - `elapsed_seconds` = 6,850（约 1 小时 54 分，短于预计的 3–3.5 小时）。
   - `head_init_sha256` = `166f6432a25bba3100ae75f0b2ce7ffaa1c63d86340a5e08b450c589e11043e2`。
   - `head_keys` 共 8 个，包括 `projection.0.weight`、`risk.weight`、`category.weight`。
   - 设备为 NVIDIA L20，torch 2.7.1+cu128，transformers 5.17.0，fla 0.5.2。
   - `totals`：sequences 62,228，forward_tokens 8,614,976。
   - `stats`：

     | 来源 | split | records | positions | shards |
     |---|---|---|---|---|
     | prefix_v2 | train | 24,582 | 326,460 | 2 |
     | prefix_v2 | calibration | 600 | 2,973 | 1 |
     | runA | train | 31,994 | 686,727 | 4 |
     | runA | calibration | 5,052 | 108,233 | 1 |

     runA 跳过了 10 条（train 6 条、calibration 4 条）。跳过有三种可能：token 数不在 1–8192 之内、两个分词器结果不一致、没有可用位置。report 没有区分是哪一种。加回跳过的条数后，与输入完全对上：train 32,000，calibration 5,056。
5. **PVC 上的缓存**：`/work/output/round6/stage1_cache_v1/`，共 12 个文件，3,482,546,133 字节（3.3 GB），留在 PVC 上。

## 产物

- `round6/stage1_head/results_cache_v1/report.json`：SHA256 `34a7527d111d1e6d672516044f76e2ceaff709cb0a734eb560cc4dda8fc4a65f`，与 Pod 上的文件一致，已提交。

## 偏差与问题

- **T018 没有启动。** 用户 17:05 起更正了口径，并在 17:40 定稿为只审红线，见 [POLICY-redline-scope.md](POLICY-redline-scope.md)。T018 等设计方按新口径出任务单。
- **缓存里的标签是旧口径。** 缓存里的位置标签是 runA 的旧标签：S、P 为 0，U 为 1。可以直接复用的是：特征（1024 维输入、512 维投影），以及每条记录的 `sample_id`、`slot`、`response_style`、`classes`。按新口径需要重新给位置打标签。
  - 另外，缓存的位置是按旧标签选的：起点片段内部不取，每条最多 24 个位置。新口径下起点变了的样本，新边界附近可能没有缓存的位置。
- **执行侧的一个小问题。** 自动取回的守护脚本没有检测到 done 标记。手动检查时，同一条 `kubectl exec test -f` 返回 0，原因没查。结果改为手动取回，GPU 因此多空占了约 11 分钟。
