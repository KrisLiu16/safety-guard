"""L20: T034 step A — v7 (bit-identical host path and graph cleanup) against v6, plus measurements that replace the
estimates in OPTIMIZATION_RESEARCH.md §2.
  1. identity: stream the 400 official thinking sequences (seed 7, as v6) and the prefix_v2 dev assistant records
     (seed 11) through v6 and v7; every probability must be equal (max |d| = 0). v7 with cuBLASLt preferred (v7_lt,
     not bit-identical by design) is reported as drift against v6 instead.
  2. tick: 208 primed sessions, 77 random rows x 1 token per tick (the v6 operating point): wall time per step (inputs
     to probabilities on the host) and the GPU time of the (96, 1) graph alone.
  3. arrival simulation (U(30, 60) tok/s after a 256-token context, as v6) at 208..288 sessions, 2 seeds: v6 with the v6
     simulator, then v6 / v7 / v7_lt with a vectorised simulator (same arrival model and random draws; the v6
     simulator rescans every session in Python on every tick, which delays the tick it measures).
  4. GEMM microbenchmark: every Linear shape of the backbone at M in {80, 96, 128, 160, 224}, cuBLAS vs cuBLASLt, and
     the concatenations proposed in §3 item 4; weights cycled through enough copies to defeat the L2.
Fixed Round5 weights; no training.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from bench_bucketed_l20 import compare, percentile

FINE = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)
TOKEN_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def due_tokens(now, offset, rate, delivered, cap=64):
    """Pure (numpy): tokens each session may send now, as bench_slot_l20.simulate: int((now - offset) * rate) have
    arrived once now > offset, minus those delivered, at most cap."""
    arrived = np.where(now > offset, ((now - offset) * rate).astype(np.int64), 0)
    return np.minimum(arrived - delivered, cap)


def arrival_times(t0, offset, rate, delivered, n):
    """Pure (numpy): arrival times of the tokens sent now, row by row as bench_slot_l20.simulate lists them:
    t0 + offset + (delivered + j + 1) / rate for j < n (offset, rate, delivered, n: arrays of the sending sessions)."""
    j = np.arange(int(n.max()) if len(n) else 0)
    return (t0 + offset[:, None] + (delivered[:, None] + j[None] + 1) / rate[:, None])[j[None] < n[:, None]]


def simulate_fast(engine, sources, sessions, seconds, rng, packed):
    """bench_slot_l20.simulate with the per-tick scan in numpy; same draws from rng, same latency definition."""
    import torch
    engine.reset()
    rate = np.array([rng.uniform(30, 60) for _ in range(sessions)])
    offset = np.array([rng.uniform(0, 1.0) for _ in range(sessions)])
    engine.step([(s, sources[s][:256]) for s in range(sessions)])
    torch.cuda.synchronize()
    delivered = np.zeros(sessions, dtype=np.int64)
    latencies, processed, ticks, sizes = [], 0, 0, []
    t0 = time.perf_counter() + 0.05
    while True:
        now = time.perf_counter() - t0
        if now > seconds:
            break
        n = due_tokens(now, offset, rate, delivered)
        active = np.flatnonzero(n > 0)
        if len(active) == 0:
            time.sleep(0.0002)
            continue
        na, da = n[active], delivered[active]
        requests = [(s, sources[s][256 + d:256 + d + k]) for s, d, k in zip(active.tolist(), da.tolist(), na.tolist())]
        arrivals = arrival_times(t0, offset[active], rate[active], da, na)
        delivered[active] += na
        if packed:
            engine.step_packed(requests)
        else:
            engine.step(requests)["assistant"][-1].cpu()
        done = time.perf_counter()
        latencies.append(done - arrivals)
        processed += int(na.sum())
        ticks += 1
        sizes.append(len(active))
    values = np.concatenate(latencies).tolist()
    return {"sessions": sessions, "offered_tokens_per_s": float(rate.sum()), "itps": processed / seconds,
            "latency_p50_ms": percentile(values, .5) * 1e3, "latency_p95_ms": percentile(values, .95) * 1e3,
            "latency_p99_ms": percentile(values, .99) * 1e3, "ticks_per_s": ticks / seconds,
            "mean_sessions_per_tick": sum(sizes) / max(1, len(sizes)), "mean_tokens_per_tick": processed / max(1, ticks)}


def fixed_tick(torch, engine, sources, sessions, rows, ticks, packed):
    rng = random.Random(5)
    engine.reset()
    engine.step([(s, sources[s][:256]) for s in range(sessions)])
    cursor = [256] * sessions
    walls = []
    for t in range(ticks + 20):
        pick = rng.sample(range(sessions), rows)
        requests = [(s, sources[s][cursor[s]:cursor[s] + 1]) for s in pick]
        for s in pick:
            cursor[s] += 1
        began = time.perf_counter()
        if packed:
            engine.step_packed(requests)
        else:
            engine.step(requests)["assistant"][-1].cpu()
        if t >= 20:
            walls.append(time.perf_counter() - began)
    nb = next(b for b in engine.session_buckets if b >= rows)
    graph, _ = engine.graphs[(nb, 1)]
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    gpu = start.elapsed_time(end) / 100
    p50 = percentile(walls, .5) * 1e3
    return {"sessions": sessions, "rows": rows, "bucket": nb, "wall_p50_ms": p50, "wall_p95_ms": percentile(walls, .95) * 1e3,
            "wall_mean_ms": sum(walls) / len(walls) * 1e3, "graph_gpu_ms": gpu, "host_ms_p50": p50 - gpu}


def time_linear(torch, m, k, n, backend, calls_bytes=256 << 20):
    import torch.nn.functional as F
    torch.backends.cuda.preferred_blas_library(backend)
    copies = max(4, min(512, math.ceil(calls_bytes / (2 * k * n))))
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for w in weights[:3]:
            F.linear(x, w)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in weights:
            F.linear(x, w)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1e3 / (5 * copies)
    del weights, graph
    torch.cuda.empty_cache()
    return {"us": us, "gb_per_s": 2 * (m * k + k * n + m * n) / (us * 1e-6) / 1e9}


def gemm_bench(torch, model, ms=(80, 96, 128, 160, 224)):
    import torch.nn as nn
    shapes = collections.Counter((mod.in_features, mod.out_features) for mod in model.backbone.modules()
                                 if isinstance(mod, nn.Linear))
    layer = {kind: next(l for l in model.backbone.layers if hasattr(l, attr))
             for kind, attr in (("gdn", "linear_attn"), ("attention", "self_attn"))}
    gdn, att, mlp = layer["gdn"].linear_attn, layer["attention"].self_attn, layer["gdn"].mlp
    groups = {"gdn_qkv_z_b_a": [gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a],
              "attention_q_k_v": [att.q_proj, att.k_proj, att.v_proj], "mlp_gate_up": [mlp.gate_proj, mlp.up_proj]}
    counts = {"gdn_qkv_z_b_a": sum(hasattr(l, "linear_attn") for l in model.backbone.layers),
              "attention_q_k_v": sum(hasattr(l, "self_attn") for l in model.backbone.layers),
              "mlp_gate_up": len(model.backbone.layers)}
    out = {"shapes": [{"in": k, "out": n, "count": c} for (k, n), c in sorted(shapes.items())], "per_m": {}}
    for m in ms:
        row = {}
        for backend in ("cublas", "cublaslt"):
            try:
                timed = {f"{k}x{n}": time_linear(torch, m, k, n, backend) for (k, n) in shapes}
                total = sum(timed[f"{k}x{n}"]["us"] * c for (k, n), c in shapes.items())
                concat = {}
                for name, mods in groups.items():
                    k = mods[0].in_features
                    sep = sum(timed[f"{k}x{mod.out_features}"]["us"] for mod in mods)
                    cat = time_linear(torch, m, k, sum(mod.out_features for mod in mods), backend)["us"]
                    concat[name] = {"separate_us": sep, "concatenated_us": cat, "layers": counts[name]}
                row[backend] = {"shapes": timed, "tick_total_ms": total / 1e3, "concat": concat,
                                "tick_total_after_concat_ms": (total - sum((v["separate_us"] - v["concatenated_us"]) * v["layers"]
                                                                          for v in concat.values())) / 1e3}
            except Exception as error:          # e.g. cuBLASLt capture problems: report, do not abort the run
                row[backend] = {"error": f"{type(error).__name__}: {error}"}
        out["per_m"][str(m)] = row
        print("gemm", m, {b: row[b].get("tick_total_ms") for b in row}, flush=True)
    torch.backends.cuda.preferred_blas_library("cublas")
    return out


def identity(got, ref, picks):
    diffs = [float((got[i] - ref[i]).abs().max()) if len(got[i]) else 0.0 for i in picks]
    unequal = sum(int((got[i] != ref[i]).any(-1).sum()) for i in picks)
    return {"sequences": len(picks), "positions": sum(len(ref[i]) for i in picks), "max_abs_diff": max(diffs),
            "positions_not_equal": unequal, "bit_identical": max(diffs) == 0.0 and unequal == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--prefix-dev", type=Path, default=Path("/work/round5/data/prefix_v2/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v7"))
    parser.add_argument("--max-seqs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sim-sessions", default="208,224,240,256,272,288")
    parser.add_argument("--sim-seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v7_step_a", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from bench_bucketed_l20 import stream
        from bench_v5_l20 import decision_drift
        from slot_engine_v4 import SlotStreamEngineV4
        from slot_engine_v7 import SlotStreamEngineV7
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], device=meta["device"],
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        kwargs = dict(state_dtype="float16", ring="inplace", gdn_bv=32, gdn_warps=2, session_buckets=FINE)

        def v6(n):
            return SlotStreamEngineV4(model, max_slots=n, **kwargs)

        def v7(n):
            return SlotStreamEngineV7(model, max_slots=n, **kwargs)

        def with_lt(factory):
            def make(n):
                torch.backends.cuda.preferred_blas_library("cublaslt")
                return factory(n)
            return make

        def stream_all(make, sequences, picks, seed):
            engine, out = make(args.batch), {}
            for start in range(0, len(picks), args.batch):
                part = picks[start:start + args.batch]
                engine.reset()
                out.update(stream(lambda reqs: engine.step(reqs)["assistant"], sequences, part, random.Random(seed + start)))
            del engine
            torch.cuda.empty_cache()
            return out

        thinking = [json.loads(line)["ids"] for line in open(args.data)][:args.max_seqs]
        prefix = [r["ids"] for r in map(json.loads, open(args.prefix_dev)) if r["target_role"] == "assistant"]
        sets = {"thinking": (thinking, 7), "prefix_v2_dev_assistant": (prefix, 11)}
        report["identity"], report["drift_v7_lt_vs_v6"] = {}, {}
        for name, (sequences, seed) in sets.items():
            picks = list(range(len(sequences)))
            got6 = stream_all(v6, sequences, picks, seed)
            got7 = stream_all(v7, sequences, picks, seed)
            report["identity"][name] = identity(got7, got6, picks)
            save()
            print(name, "identity", report["identity"][name], flush=True)
            try:
                got_lt = stream_all(with_lt(v7), sequences, picks, seed)
                report["drift_v7_lt_vs_v6"][name] = {"probs": {k: v for k, v in compare(got_lt, got6, picks).items() if k != "worst"},
                                                     "decisions": decision_drift(got_lt, got6, picks)}
            except Exception as error:
                report["drift_v7_lt_vs_v6"][name] = {"error": f"{type(error).__name__}: {error}"}
            finally:
                torch.backends.cuda.preferred_blas_library("cublas")
            save()

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in thinking]
        configs = {"v6": (v6, False), "v7": (v7, True), "v7_lt": (with_lt(v7), True)}
        report["tick"] = {}
        for name, (factory, packed) in configs.items():
            engine = factory(208)
            with torch.inference_mode():
                for lb in (1, 64):
                    for nb in FINE:
                        if nb <= 208:
                            engine._static(nb, lb)
            report["tick"][name] = fixed_tick(torch, engine, sources, 208, 77, 300, packed)
            save()
            print("tick", name, report["tick"][name], flush=True)
            del engine
            torch.backends.cuda.preferred_blas_library("cublas")
            torch.cuda.empty_cache()

        from bench_slot_l20 import simulate
        sim_sessions = list(map(int, args.sim_sessions.split(",")))
        runs = {"v6_v6sim": (v6, None), "v6": (v6, False), "v7": (v7, True), "v7_lt": (with_lt(v7), True)}
        report["arrival_simulation"], report["capacity_at_p95_20ms"] = {}, {}
        for name, (factory, packed) in runs.items():
            engine = factory(max(sim_sessions))
            with torch.inference_mode():
                for nb in engine.session_buckets:
                    if nb <= engine.max_slots:
                        for lb in TOKEN_BUCKETS:
                            engine._static(nb, lb)
            sims = []
            for n in sim_sessions:
                for seed in range(args.seeds):
                    rng = random.Random(n + 1000 * seed)
                    src = [sources[s % len(sources)] for s in range(n)]
                    row = (simulate(engine, src, n, args.sim_seconds, rng) if packed is None
                           else simulate_fast(engine, src, n, args.sim_seconds, rng, packed))
                    row["seed"] = seed
                    sims.append(row)
                    print(name, json.dumps(row), flush=True)
            report["arrival_simulation"][name] = sims
            worst = {n: max(r["latency_p95_ms"] for r in sims if r["sessions"] == n) for n in sim_sessions}
            ok = [n for n in sim_sessions if worst[n] <= 20.0]
            report["capacity_at_p95_20ms"][name] = {"sessions": max(ok) if ok else 0, "worst_seed_p95_ms": worst}
            save()
            del engine
            torch.backends.cuda.preferred_blas_library("cublas")
            torch.cuda.empty_cache()

        report["gemm"] = gemm_bench(torch, model)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


if __name__ == "__main__":
    main()
