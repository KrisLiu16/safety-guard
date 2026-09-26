"""Where a stage-2 training step spends its time on one L20, and how far that is from what the card can do (T036).

The v4 log shows every step with up to ~8k input tokens taking ~0.8 s: 84% of the run is a fixed cost per step, not
compute. This script measures where that cost goes and what each change buys. Parts (JSON lines in
--output/profile.jsonl, a readable summary on stdout):
  peak    bf16 GEMM throughput at the model's own weight shapes for 128..16384 rows (forward, input-gradient and
          weight-gradient GEMMs), a large square GEMM, device copy bandwidth, and the CPU cost of one small eager op:
          the ceilings the other parts are compared to
  sweep   forward+backward of the model on synthetic unpadded batches (rows x length), with and without gradient
          checkpointing: tokens/s, matmul TFLOPS, MFU and peak memory, i.e. how many tokens one pass needs to fill
          the card
  steps   real v4 training groups under each setting (SETTINGS): per step, CPU time and GPU time per phase (prepare,
          forward, loss, backward, clip, optimizer), real and padded tokens, passes, memory, tokens/s and MFU
  trace   torch.profiler over a few steps per setting: GPU busy share, kernels and host syncs per step, GPU time by
          kernel family, the top kernels
  check   loss, gradient norm and gradient cosine of every setting against train_distill.train_step itself on the
          same groups from the same weights (settings that only change how the step runs must agree to rounding;
          current_again gives the run-to-run noise floor)
MFU counts matmul FLOPs of the linear layers and the window attention (6 x params x tokens for training, no
recompute); the Gated DeltaNet recurrence (about 4% of the linear FLOPs per token) is left out, so MFU is slightly low.
Nothing is trained to completion and no checkpoint is written.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_stage2 as ts  # noqa: E402
import train_distill as td  # noqa: E402
from batched_loss import alert_table, batched_loss, loss_plan, plan_tensors  # noqa: E402,F401

L20_BF16_TFLOPS = 119.5          # datasheet, dense FP16/BF16 tensor
L20_HBM_GBS = 864                # datasheet

# rows: rows per forward pass; micro: padded-token cap per pass (a longer row goes alone); ckpt: gradient
# checkpointing; loss: "loop" is train_distill's per-row loss, "batched" gathers every position of a pass per role
# and builds all indices and targets on the host first, so a step syncs with the GPU once; fused: fused AdamW;
# merge: training groups per optimizer step (merge > 1 changes the optimisation, it is here for speed only).
# Without checkpointing a pass of 16384 tokens does not fit in 46 GB (profile_train_v1 sweep: 8192 tokens peak at
# 26.9 GB before optimizer state), so those settings cap a pass at 8192 tokens.
SETTINGS = {
    "current": dict(rows=4, micro=16384, ckpt=True, loss="loop", fused=False, merge=1),
    "rows16": dict(rows=16, micro=16384, ckpt=True, loss="loop", fused=False, merge=1),
    "rows16_nockpt": dict(rows=16, micro=8192, ckpt=False, loss="loop", fused=False, merge=1),
    "batched": dict(rows=16, micro=8192, ckpt=False, loss="batched", fused=True, merge=1),
    "batched_ckpt": dict(rows=16, micro=16384, ckpt=True, loss="batched", fused=True, merge=1),
    "batch64": dict(rows=64, micro=8192, ckpt=False, loss="batched", fused=True, merge=4),
}
PHASES = ("prepare", "forward", "loss", "backward", "clip", "optimizer")
SWEEP = [(4, 64), (16, 64), (4, 256), (16, 128), (64, 64), (16, 256), (16, 512), (32, 512), (8, 2048), (16, 1024),
         (4, 4096), (64, 512), (32, 1024)]
GEMM_ROWS = (128, 512, 2048, 8192, 16384)
FAMILIES = (   # first match wins
    ("short convolution", ("conv", "fprop", "dgrad", "wgrad")),
    ("gated deltanet (fla)", ("chunk", "fused_recurrent", "l2norm", "recompute_w_u", "solve_tril", "merge_16x16",
                              "kkt", "wy_", "cumsum", "gated_delta", "prepare_wy")),
    ("attention", ("fmha", "flash", "attention", "sdpa", "efficient")),
    ("matmul", ("gemm", "gemv", "cutlass", "xmma", "cublas")),
    ("copy", ("memcpy",)),
    ("memset", ("memset",)),
    ("multi-tensor (optimizer, clip)", ("multi_tensor", "adam", "foreach")),
    ("reduction / norm / softmax", ("reduce", "norm", "softmax", "logsumexp", "cross_entropy", "nll")),
    ("elementwise / indexing", ("elementwise", "vectorized", "unrolled", "index", "gather", "scatter", "cat",
                                "copy", "fill", "where", "masked", "arange", "embedding", "cast")),
    ("other triton", ("triton",)),
)
SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


# ----------------------------------------------------------------------------------------------- pure helpers
def kernel_family(name):
    low = name.lower()
    for family, keys in FAMILIES:
        if any(k in low for k in keys):
            return family
    return "other"


def busy_ns(intervals):
    """Pure: total length of the union of (start, end) intervals."""
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def attention_flops(lengths, window, heads, head_dim, layers):
    """Pure: forward matmul FLOPs of causal window attention (QK^T and PV) for unpadded rows of these lengths."""
    per_key = 4 * heads * head_dim * layers
    keys = 0
    for n in lengths:
        full = min(n, window)
        keys += full * (full + 1) // 2 + max(0, n - window) * window
    return per_key * keys


def merge_groups(groups, k):
    return [sum(groups[i:i + k], []) for i in range(0, len(groups) - k + 1, k)]


# ----------------------------------------------------------------------------------------------- loss paths
def loop_loss(torch, model, hidden, rows, *, alert_cut, distill, category, device):
    """train_distill.train_step's per-row loss, unchanged (one GPU sync per term through float())."""
    F = torch.nn.functional
    losses, parts = [], collections.Counter()
    for index, row in enumerate(rows):
        logits = ts.risk_logits(model, hidden, index, row)
        target = ts.soft_targets(torch, row["classes"], alert_cut, device)
        weight = torch.tensor(row["token_weights"], device=device, dtype=torch.float32)
        red = (-(target * F.log_softmax(logits, dim=-1)).sum(-1) * weight).sum()
        losses.append(red)
        parts["redline"] += float(red.detach())
        if row.get("t_positions") and row["distill_weight"] > 0:
            g_logits, g_cat = td.general_readout(model, hidden[index, row["t_positions"]], row["role"])
            q = torch.tensor(row["t_risk"], device=device, dtype=torch.float32).clamp_min(1e-6)
            q = q / q.sum(-1, keepdim=True)
            kl = (q * (q.log() - F.log_softmax(g_logits, dim=-1))).sum(-1).mean()
            term = distill * row["distill_weight"] * kl
            cats = torch.tensor(row["t_cat"], device=device)
            if category > 0 and bool((cats >= 0).any()):
                ce = F.cross_entropy(g_cat[cats >= 0], cats[cats >= 0])
                term = term + category * row["distill_weight"] * ce
            losses.append(term)
            parts["distill"] += float(term.detach())
    return losses, parts


# ----------------------------------------------------------------------------------------------- one step
class Laps:
    """CPU and GPU time per phase: a CUDA event per boundary, read once after the step's final synchronize."""

    def __init__(self, torch, on=True):
        self.torch, self.on = torch, on
        self.cpu, self.gpu = collections.Counter(), collections.Counter()

    def start(self):
        self.events, self.t = [], time.perf_counter()
        self._record(None)

    def _record(self, phase):
        if self.on:
            ev = self.torch.cuda.Event(enable_timing=True)
            ev.record()
            self.events.append((phase, ev))

    def lap(self, phase):
        now = time.perf_counter()
        self.cpu[phase] += now - self.t
        self.t = now
        self._record(phase)

    def finish(self):
        self.torch.cuda.synchronize()
        if self.on:
            for (_, a), (phase, b) in zip(self.events, self.events[1:]):
                self.gpu[phase] += a.elapsed_time(b) / 1000


class NoStep:
    """Optimizer stand-in for the check: zero_grad only, weights never change."""

    def __init__(self, model):
        self.model = model

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        pass


def run_step(torch, model, optimizer, group, s, *, pad, scale, alert_cut, distill, category, table, laps):
    model.train()
    laps.start()
    optimizer.zero_grad(set_to_none=True)
    stats = collections.Counter()
    total, parts = 0.0, collections.Counter()
    acc = []
    for rows in ts.microbatches(group, s["micro"], size=s["rows"]):
        if s["loss"] == "loop":
            ids, mask = ts.pad_batch(torch, rows, pad, "cuda")
        else:
            length = max(len(r["ids"]) for r in rows)
            host = torch.full((len(rows), length), pad, dtype=torch.long)
            hmask = torch.zeros_like(host)
            for i, r in enumerate(rows):
                host[i, :len(r["ids"])] = torch.tensor(r["ids"])
                hmask[i, :len(r["ids"])] = 1
            ids = host.pin_memory().to("cuda", non_blocking=True)
            mask = hmask.pin_memory().to("cuda", non_blocking=True)
            tensors = plan_tensors(torch, loss_plan(rows, distill, category), "cuda", pin=True)
        laps.lap("prepare")
        hidden = model(ids, mask, use_cache=False).last_hidden_state
        laps.lap("forward")
        if s["loss"] == "loop":
            losses, p = loop_loss(torch, model, hidden, rows, alert_cut=alert_cut, distill=distill,
                                  category=category, device="cuda")
            parts.update(p)
            loss = torch.stack(losses).sum() * scale
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss")
        else:
            red, dist = batched_loss(torch, model, hidden, tensors, table)
            loss = (red + dist) * scale
            acc.append(torch.stack([loss.detach(), red.detach(), dist.detach()]))
        laps.lap("loss")
        loss.backward()
        laps.lap("backward")
        if s["loss"] == "loop":
            total += float(loss.detach())
            stats["tokens"] += int(mask.sum())
        else:
            stats["tokens"] += sum(len(r["ids"]) for r in rows)
        stats["padded"] += ids.numel()
        stats["passes"] += 1
        del hidden, loss
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if s["loss"] == "loop":
        if not bool(torch.isfinite(grad)):
            raise RuntimeError("nonfinite gradient")
        grad_norm = float(grad)
    laps.lap("clip")
    if s["loss"] != "loop":
        values = torch.cat([torch.stack(acc).sum(0), grad.detach().reshape(1)]).tolist()   # the step's one sync
        if not all(v == v and abs(v) != float("inf") for v in values):
            raise RuntimeError("nonfinite loss or gradient")
        total, parts["redline"], parts["distill"], grad_norm = values[0], values[1], values[2], values[3]
    optimizer.step()
    laps.lap("optimizer")
    laps.finish()
    return {"loss": total, "redline": parts["redline"] * scale, "distill": parts["distill"] * scale,
            "gradient_norm": grad_norm, **stats}


# ----------------------------------------------------------------------------------------------- parts
def set_checkpointing(model, on):
    if on:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.backbone.gradient_checkpointing_disable()


def make_optimizer(torch, model, fused):
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if n.startswith("heads.") and p.requires_grad]
    return torch.optim.AdamW([{"params": backbone, "lr": 8e-6}, {"params": heads, "lr": 5e-5}],
                             weight_decay=0.01, eps=1e-6, **({"fused": True} if fused else {}))


def model_facts(torch, model):
    cfg = model.backbone.config
    linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    shapes = collections.Counter(tuple(m.weight.shape) for m in model.backbone.modules()
                                 if isinstance(m, torch.nn.Linear))
    kinds = list(getattr(cfg, "layer_types", []))
    return {"linear_params": sum(m.weight.numel() for m in linear),
            "backbone_linear_shapes": {f"{o}x{i}": n for (o, i), n in sorted(shapes.items())},
            "attention_layers": sum(1 for k in kinds if k == "full_attention"),
            "heads": cfg.num_attention_heads, "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            "params": sum(p.numel() for p in model.parameters()), "vocab": cfg.vocab_size, "window": 512}


def train_flops(facts, lengths):
    """Model FLOPs of one training pass over unpadded rows of these lengths (forward + backward = 3x forward)."""
    fwd = 2 * facts["linear_params"] * sum(lengths)
    fwd += attention_flops(lengths, facts["window"], facts["heads"], facts["head_dim"], facts["attention_layers"])
    return 3 * fwd


def cuda_time(torch, fn, reps, warm=3):
    for _ in range(warm):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 1000 / reps


def part_peak(torch, facts, emit):
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    sec = cuda_time(torch, lambda: big @ big, 10)
    emit({"part": "peak", "what": "square_gemm_8192_bf16", "tflops": 2 * 8192 ** 3 / sec / 1e12})
    del big
    shapes = [tuple(int(v) for v in k.split("x")) + (n,) for k, n in facts["backbone_linear_shapes"].items()]
    for m in GEMM_ROWS:
        flops = secs = 0.0
        per = {}
        for out_f, in_f, count in shapes:
            x = torch.randn(m, in_f, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
            dy = torch.randn(m, out_f, device="cuda", dtype=torch.bfloat16)
            t = (cuda_time(torch, lambda: x @ w.t(), 20) + cuda_time(torch, lambda: dy @ w, 20)
                 + cuda_time(torch, lambda: dy.t() @ x, 20))
            f = 3 * 2 * m * in_f * out_f
            per[f"{out_f}x{in_f}"] = round(f / t / 1e12, 1)
            flops += f * count
            secs += t * count
        emit({"part": "peak", "what": "model_gemms_fwd_bwd", "rows": m, "tflops": flops / secs / 1e12,
              "by_shape_tflops": per, "seconds_all_layers": secs})
    a = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    sec = cuda_time(torch, lambda: b.copy_(a), 10)
    emit({"part": "peak", "what": "device_copy_1GiB", "gbs": 2 * a.numel() * 4 / sec / 1e9})
    del a, b
    t = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    c0 = time.perf_counter()
    for _ in range(5000):
        t.add_(1)
    c1 = time.perf_counter()
    torch.cuda.synchronize()
    c2 = time.perf_counter()
    emit({"part": "peak", "what": "small_eager_op", "cpu_us_per_op": (c1 - c0) / 5000 * 1e6,
          "wall_us_per_op": (c2 - c0) / 5000 * 1e6})
    torch.cuda.empty_cache()


def part_sweep(torch, model, facts, emit, peak):
    for ckpt in (True, False):
        set_checkpointing(model, ckpt)
        for rows, length in SWEEP:
            ids = torch.randint(0, facts["vocab"], (rows, length), device="cuda")
            mask = torch.ones_like(ids)

            def once():
                model.zero_grad(set_to_none=True)
                hidden = model(ids, mask, use_cache=False).last_hidden_state
                model.readout(hidden[:, -8:], "assistant")[0].float().logsumexp(-1).mean().backward()
            rec = {"part": "sweep", "checkpointing": ckpt, "rows": rows, "length": length, "tokens": rows * length}
            try:
                torch.cuda.reset_peak_memory_stats()
                sec = cuda_time(torch, once, 5, warm=2)
                flops = train_flops(facts, [length] * rows)
                rec.update(seconds=sec, tokens_per_s=rows * length / sec, tflops=flops / sec / 1e12,
                           mfu=flops / sec / 1e12 / L20_BF16_TFLOPS, mfu_vs_measured=flops / sec / 1e12 / peak,
                           peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
            except torch.cuda.OutOfMemoryError:
                rec["oom"] = True
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            emit(rec)


def part_steps(torch, model, groups, facts, emit, opts, names, warmup, peak):
    out = {}
    for name in names:
        s = dict(SETTINGS[name])
        for attempt in range(3):
            set_checkpointing(model, s["ckpt"])
            optimizer = make_optimizer(torch, model, s["fused"])
            steps = merge_groups(groups, s["merge"])
            laps = Laps(torch)
            records = []
            try:
                torch.cuda.reset_peak_memory_stats()
                for i, group in enumerate(steps):
                    laps.cpu.clear()
                    laps.gpu.clear()
                    t0 = time.perf_counter()
                    rec = run_step(torch, model, optimizer, group, s, laps=laps, **opts)
                    rec["seconds"] = time.perf_counter() - t0
                    rec["cpu"] = dict(laps.cpu)
                    rec["gpu"] = dict(laps.gpu)
                    rec["flops"] = train_flops(facts, [len(r["ids"]) for r in group])
                    if i >= warmup:
                        records.append(rec)
                break
            except torch.cuda.OutOfMemoryError:
                del optimizer
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                emit({"part": "steps", "setting": name, "oom_at_micro": s["micro"]})
                s["micro"] //= 2
        else:
            continue
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        n = len(records)
        sec = sum(r["seconds"] for r in records)
        tokens = sum(r["tokens"] for r in records)
        flops = sum(r["flops"] for r in records)
        buckets = collections.defaultdict(list)
        for r in records:
            buckets[next(b for b in (500, 1000, 2000, 4000, 8000, 10 ** 9) if r["tokens"] // s["merge"] < b)].append(
                r["seconds"])
        summary = {"part": "steps", "setting": name, "config": s, "steps": n, "groups_per_step": s["merge"],
                   "seconds_per_group": sec / n / s["merge"], "median_step_s": statistics.median(r["seconds"] for r in records),
                   "tokens_per_s": tokens / sec, "tflops": flops / sec / 1e12, "mfu": flops / sec / 1e12 / L20_BF16_TFLOPS,
                   "mfu_vs_measured": flops / sec / 1e12 / peak,
                   "padding_share": 1 - tokens / sum(r["padded"] for r in records),
                   "passes_per_group": sum(r["passes"] for r in records) / n / s["merge"],
                   "cpu_ms_per_step": {p: 1000 * sum(r["cpu"].get(p, 0) for r in records) / n for p in PHASES},
                   "gpu_ms_per_step": {p: 1000 * sum(r["gpu"].get(p, 0) for r in records) / n for p in PHASES},
                   "median_s_by_group_tokens": {f"<{b}": statistics.median(v) for b, v in sorted(buckets.items())},
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        out[name] = summary
        emit(summary)
    return out


def kineto(prof):
    events = []
    for e in prof.profiler.kineto_results.events():
        start = e.start_ns() if hasattr(e, "start_ns") else int(e.start_us() * 1000)
        dur = e.duration_ns() if hasattr(e, "duration_ns") else int(e.duration_us() * 1000)
        events.append((e.name(), str(e.device_type()), start, dur))
    return events


def part_trace(torch, model, groups, emit, opts, names, n_steps, chrome, out_dir):
    from torch.profiler import ProfilerActivity, profile
    for name in names:
        s = SETTINGS[name]
        set_checkpointing(model, s["ckpt"])
        optimizer = make_optimizer(torch, model, s["fused"])
        steps = merge_groups(groups, s["merge"])[:n_steps + 2]
        laps = Laps(torch, on=False)
        for group in steps[:2]:
            run_step(torch, model, optimizer, group, s, laps=laps, **opts)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for group in steps[2:]:
                run_step(torch, model, optimizer, group, s, laps=laps, **opts)
            wall = time.perf_counter() - t0
        n = len(steps) - 2
        events = kineto(prof)
        # device-side record_function ranges ("Optimizer.step#AdamW.step", "ProfilerStep#3") are spans, not kernels
        device = [(nm, st, d) for nm, dev, st, d in events if "CUDA" in dev and "#" not in nm]
        host = [(nm, d) for nm, dev, st, d in events if "CUDA" not in dev]
        fam, top = collections.Counter(), collections.defaultdict(lambda: [0, 0])
        for nm, _, d in device:
            fam[kernel_family(nm)] += d
            top[nm[:120]][0] += 1
            top[nm[:120]][1] += d
        calls = collections.Counter(nm for nm, _ in host if nm.startswith("cu"))
        sync_ns = sum(d for nm, d in host if nm in SYNC_CALLS)
        busy = busy_ns([(st, st + d) for _, st, d in device])
        launches = sum(v for k, v in calls.items() if "LaunchKernel" in k)
        emit({"part": "trace", "setting": name, "steps": n, "wall_s_per_step": wall / n,
              "gpu_busy_share": busy / 1e9 / wall, "gpu_busy_ms_per_step": busy / 1e6 / n,
              "kernels_per_step": sum(1 for nm, _, _ in device if not nm.lower().startswith("mem")) / n,
              "launch_calls_per_step": launches / n,
              "host_syncs_per_step": sum(calls[c] for c in SYNC_CALLS) / n,
              "host_sync_ms_per_step": sync_ns / 1e6 / n,
              "memcpy_calls_per_step": sum(v for k, v in calls.items() if "Memcpy" in k) / n,
              "gpu_ms_per_step_by_family": {k: v / 1e6 / n for k, v in fam.most_common()},
              "top_kernels": [{"name": k, "calls_per_step": c / n, "ms_per_step": d / 1e6 / n,
                               "us_per_call": d / 1e3 / c}
                              for k, (c, d) in sorted(top.items(), key=lambda kv: -kv[1][1])[:25]],
              "runtime_calls_per_step": {k: v / n for k, v in calls.most_common(12)}})
        if name in chrome:
            raw = out_dir / f"trace_{name}.json"
            prof.export_chrome_trace(str(raw))
            with raw.open("rb") as f, gzip.open(str(raw) + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)
            raw.unlink()
        del optimizer, prof
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def part_check(torch, model, groups, emit, opts, names):
    """Gradients of each setting against train_distill.train_step on the same groups from the same weights."""
    params = [p for p in model.parameters() if p.requires_grad]
    start = [p.detach().to("cpu", copy=True) for p in params]
    by_len = sorted(groups, key=lambda g: max(len(r["ids"]) for r in g))
    picks = [by_len[len(by_len) // 2], by_len[-1], groups[0]]
    dummy = NoStep(model)
    ref_opts = {k: opts[k] for k in ("pad", "scale", "alert_cut", "distill", "category")}

    def restore():
        with torch.no_grad():
            for p, v in zip(params, start):
                p.copy_(v)

    for gi, group in enumerate(picks):
        restore()
        set_checkpointing(model, True)
        ref = td.train_step(torch, model, dummy, group, device="cuda", micro_tokens=16384, **ref_opts)
        ref_grads = [None if p.grad is None else p.grad.detach().to("cpu", copy=True) for p in params]
        for name in ["current_again"] + list(names):
            s = SETTINGS["current" if name == "current_again" else name]
            restore()
            set_checkpointing(model, s["ckpt"])
            rec = run_step(torch, model, dummy, group, s, laps=Laps(torch, on=False), **opts)
            dot = na = nb = 0.0
            for p, r in zip(params, ref_grads):     # a head no row of the group reaches has no gradient
                if p.grad is None and r is None:
                    continue
                g = p.grad.detach().float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
                r = r.to("cuda").float() if r is not None else torch.zeros_like(p, dtype=torch.float32)
                dot += float((g * r).sum())
                na += float((g * g).sum())
                nb += float((r * r).sum())
            emit({"part": "check", "setting": name, "group": gi, "tokens": rec["tokens"],
                  "loss_rel_diff": abs(rec["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-12),
                  "redline_rel_diff": abs(rec["redline"] - ref["redline"]) / max(abs(ref["redline"]), 1e-12),
                  "distill_rel_diff": abs(rec["distill"] - ref["distill"]) / max(abs(ref["distill"]), 1e-12),
                  "grad_norm_rel_diff": abs(rec["gradient_norm"] - ref["gradient_norm"]) / ref["gradient_norm"],
                  "grad_cosine": dot / (na * nb) ** 0.5})
        del ref_grads
    restore()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", type=Path, help="build_targets.py --teacher output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", default="peak,sweep,check,steps,trace")
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--groups", type=int, default=160, help="training groups timed per setting")
    parser.add_argument("--warmup", type=int, default=8, help="steps per setting left out of the timing")
    parser.add_argument("--trace-steps", type=int, default=4)
    parser.add_argument("--chrome", default="current,batched", help="settings whose profiler trace is kept (.json.gz)")
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    names = [n for n in args.settings.split(",") if n]
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / "profile.jsonl").open("w")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.monotonic()
    train = ts.read_jsonl(args.targets / "records_train.jsonl")
    mix = ts.parse_mix(ts.DEFAULT_MIX)
    ts.token_weights(train, mix)
    td.distill_weights(train, mix)
    torch, helper, load_file = ts.import_runtime(args.round4_code)
    model, pad = ts.make_model(helper, load_file, ts.START_PATH)
    helper.training_mode(model)
    td.add_general_heads(model)
    model.to("cuda")
    facts = model_facts(torch, model)
    emit({"part": "setup", "seconds": time.monotonic() - t0, "train_records": len(train),
          "device": torch.cuda.get_device_name(0), "torch": torch.__version__, **facts})
    groups = ts.epoch_groups(train, 0)[:args.groups]
    opts = {"pad": pad, "scale": len(train) / ts.EFFECTIVE_BATCH, "alert_cut": 0.4, "distill": 0.5, "category": 0.25,
            "table": alert_table(torch, 0.4, "cuda")}
    del train
    peak = L20_BF16_TFLOPS
    if "peak" in parts:
        part_peak(torch, facts, emit)
        with (args.output / "profile.jsonl").open() as f:
            peak = max(json.loads(l)["tflops"] for l in f if '"square_gemm' in l)
    if "sweep" in parts:
        part_sweep(torch, model, facts, emit, peak)
    if "check" in parts:
        part_check(torch, model, groups[:48], emit, opts, [n for n in names if SETTINGS[n]["merge"] == 1])
    if "steps" in parts:
        part_steps(torch, model, groups, facts, emit, opts, names, args.warmup, peak)
    if "trace" in parts:
        part_trace(torch, model, groups, emit, opts, names, args.trace_steps, set(args.chrome.split(",")), args.output)
    emit({"part": "done", "seconds": time.monotonic() - t0})


if __name__ == "__main__":
    main()
