# Stage-2 training step on one L20 (T036)

`profile_train.py` on the v4 targets, second L20, 2026-09-26. Raw outputs stay local (jsonl and logs are not committed): `profile.jsonl` is run r1 (peak and sweep; its check
part stopped on a head without gradient, fixed for r2), `profile_v2.jsonl` is run r2 (check, steps, trace).
`smi*.log` sample utilisation, memory, clock and power every 5 s.

## Ceilings

| | measured |
|---|---|
| bf16 GEMM 8192³ | 116.8 TFLOPS (datasheet 119.5) |
| model GEMMs fwd+bwd, 128 / 512 / 2048 / 8192 rows | 48 / 86 / 104 / 109 TFLOPS |
| device copy | 653 GB/s (datasheet 864) |
| one small eager op on the CPU | 4.7 µs |
| whole model fwd+bwd, best case (no checkpointing, 2048 tokens per pass) | 36% MFU |

A pass has a fixed cost of about 0.16 s with gradient checkpointing and 0.09 s without, whatever its size up to
~1k tokens. Without checkpointing a pass of 8192 tokens peaks at 26.9 GB and 16384 tokens does not fit.

## Settings on the same 152 training groups

| setting | s per group | tokens/s | MFU | vs current |
|---|---|---|---|---|
| current (4 rows per pass, checkpointing, per-row loss) | 0.854 | 2,335 | 5.9% | 1.0x |
| rows16 (one pass per group) | 0.468 | 4,256 | 10.8% | 1.8x |
| rows16_nockpt | 0.359 | 5,561 | 14.1% | 2.4x |
| batched (+ batched loss, one host sync per step, fused AdamW) | 0.242 | 8,240 | 20.8% | 3.5x |
| batched_ckpt | 0.326 | 6,114 | 15.5% | 2.6x |
| batch64 (4 groups per step; changes the optimisation) | 0.348 | 5,970 | 15.1% | 2.5x |

batch64 loses to batched because merging groups of different lengths pads 21% of its tokens. At 0.242 s per group
the v4 run (17,180 steps, 3 h 56 min) would take about 70 minutes.

## Where the time goes (profiler, 4 steps each)

| setting | kernels per step | host syncs per step | ms blocked on syncs |
|---|---|---|---|
| current | 33,333 | 177 | 229 |
| batched | 9,426 | 2 | 39 |

In `batched` the GPU is busy most of the step, but only about 31% of its kernel time is matmul; about 41% is dtype
casts and fp32 elementwise work (fp32 master weights under autocast, fp32 norms and gates). The next gains are
fusing those (torch.compile) and keeping bf16 compute weights, not larger batches.

## Agreement with the current training step

Every setting against `train_distill.train_step` on the same three groups from the same weights: loss relative
difference up to 2e-4 (red-line part up to 1e-3), gradient cosine 0.9993 or higher. Running the current step twice
gives cosine 0.99993 or higher, so a different batch shape changes bf16 rounding a little more than run-to-run
noise; a run with the new step is evaluated like any new run.
