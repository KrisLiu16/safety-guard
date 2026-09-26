# Sharded red-line evaluation on one L20 (T036)

eval_head_l20.py scores one record at a time through the canonical32 path; one process keeps the L20 at about 23%
(97 W, one CPU core busy). shard_eval.py runs N processes on the card and merges their files back in record order.
All runs below score the v4 checkpoint (redline view) and compare line by line with the single-process v4 run.

| run | shards | autotune | wall | lines equal to the single run |
|---|---|---|---|---|
| user | 4 | per process | 111 s (single: 253 s) | 80% (prefix_v2, Run A prompts) |
| user, again | 4 | per process | 111 s | 100% |
| user | 1 | per process | 256 s | 100% |
| user | 4 | pinned (warm-up table) | 124 s | 100% |
| assistant | 4 | per process | 640 s (single: 2,023 s) | 31% |
| assistant | 8 | per process | 618 s | 7% |
| assistant | 4 | pinned (warm-up table) | 692 s | 7%, and equal to the 8-shard run on every line |
| assistant | 4 | pinned (autotune_l20_v1.json) | 616 s | 7%, and equal to the two runs above on every line |

Where the lines differ: FLA's Triton kernels are autotuned in each process, and near-equal configs win by timing
noise, so a process lands on one of a few discrete variants (max |d logprob| 0.1 on Run A). Two warm-ups alone on
the card already chose differently for l2norm (BT 8 or 32) and the cumsum kernel. Prompts fit in one chunk and do
not depend on the multi-chunk kernels; answers do. The unpinned single-process v4 run landed on a variant no pinned
table reproduces, but the metrics do not move: v4 under autotune_l20_v1.json has the same stream AUCs to three
decimals on every set and the same threshold-rule rates within 0.1 point.

From v5 on, every red-line eval runs as 4 shards pinned to `round6/stage1_head/autotune_l20_v1.json`; any shard
count and any rerun then score alike. v4 under that table is `stage2_eval_assistant_v4_table4` (assistant) and the
unchanged v4 run (user). The `*.report.json` files hold counts, timings, autotune states and the line comparisons.
