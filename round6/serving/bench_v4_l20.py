"""L20: v4 engine — ring kernel check, drift per (state dtype, ring mode), GDN tile sweep, throughput and
arrival-rate latency, against v3 in the same run.

Fixed Round5 weights; no training, thresholds or labels. Drift reference = one whole-sequence forward, as in v2/v3.
Drift is reported for p(unsafe) (as before) and for the red-line cut score 1 - p(safe). A configuration passes
the drift gate when, against the whole-sequence reference, its p99.9 |dp| <= max(0.03, 1.5 x v3), its argmax
flips <= v3 + 4 and its max cut-score |d| <= 0.15; only passing configurations get the latency simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

from bench_bucketed_l20 import compare, percentile, stream
from bench_slot_l20 import simulate

CONFIGS = {                     # name: (state_dtype, ring)
    "v4_fp32_gather": ("float32", "gather"),
    "v4_fp32_inplace": ("float32", "inplace"),
    "v4_bf16_gather": ("bfloat16", "gather"),
    "v4_fp16_gather": ("float16", "gather"),
    "v4_bf16_inplace": ("bfloat16", "inplace"),
    "v4_fp16_inplace": ("float16", "inplace"),
}
TILES = ((8, 1), (16, 1), (16, 2), (32, 2), (32, 4), (64, 4))


def compare_cut(streamed, reference, picks):
    values = [float(v) for i in picks for v in ((1 - streamed[i][:, 0]) - (1 - reference[i][:, 0])).abs()]
    return {"max": max(values), "p999": percentile(values, .999), "p99": percentile(values, .99)}


def kernel_check(torch, seed=0):
    """ring_attention (Triton) vs the torch reference on random pools, several lens/starts/ring fills."""
    from ring_attention_kernel import ring_attention, ring_attention_reference
    gen = torch.Generator().manual_seed(seed)                  # built on CPU, then moved
    worst, cases = 0.0, []
    for n, lb, window in ((1, 1, 512), (8, 1, 512), (8, 16, 512), (32, 64, 512), (5, 7, 64)):
        slots_total, hkv, group, d = 40, 2, 4, 256
        ring_k = torch.randn(slots_total, hkv, window + 1, d, generator=gen).to(torch.bfloat16)
        ring_v = torch.randn(slots_total, hkv, window + 1, d, generator=gen).to(torch.bfloat16)
        ring_pos = torch.full((slots_total, window + 1), -1, dtype=torch.long)
        slots = torch.randperm(slots_total - 1, generator=gen)[:n]
        starts = torch.randint(0, 3 * window, (n,), generator=gen)
        lens = torch.randint(0, lb + 1, (n,), generator=gen)
        for row in range(n):              # the ring holds the previous min(start, W) positions at pos % W
            for p in range(max(0, int(starts[row]) - window), int(starts[row])):
                ring_pos[slots[row], p % window] = p
        q = torch.randn(n, hkv, group * lb, d, generator=gen).to(torch.bfloat16)
        new_k = torch.randn(n, hkv, lb, d, generator=gen).to(torch.bfloat16)
        new_v = torch.randn(n, hkv, lb, d, generator=gen).to(torch.bfloat16)
        args = [t.cuda() for t in (q, ring_k, ring_v, new_k, new_v, ring_pos, slots, starts, lens)]
        scale = d ** -0.5
        got = ring_attention(*args, scale, window)
        ref = ring_attention_reference(*args, scale, window)
        diff = float((got.float() - ref.float()).abs().max())
        worst = max(worst, diff)
        cases.append({"n": n, "lb": lb, "window": window, "max_abs_diff": diff})
    return {"cases": cases, "max_abs_diff": worst, "pass": worst < 0.05}


def tile_sweep(torch, sizes=(128, 512), repeats=50):
    """GDN slot kernel alone, one token per session: time per call and max |diff| against the v3 tile (8, 1)."""
    from gdn_slot_kernel import gdn_slot_recurrent
    out = []
    for n in sizes:
        for dtype in (torch.float32, torch.bfloat16):
            gen = torch.Generator(device="cuda").manual_seed(n)
            q = torch.randn(n, 1, 16, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            k, v = torch.randn_like(q), torch.randn_like(q)
            a = torch.randn(n, 1, 16, device="cuda", dtype=torch.bfloat16, generator=gen)
            b = torch.randn_like(a)
            A_log, dt_bias = torch.zeros(16, device="cuda"), torch.zeros(16, device="cuda")
            base_state = torch.randn(n + 1, 16, 128, 128, device="cuda", generator=gen).to(dtype) * 0.1
            slots = torch.arange(n, device="cuda")
            lens = torch.ones(n, dtype=torch.long, device="cuda")
            reference = None
            for bv, warps in TILES:
                state = base_state.clone()
                o = gdn_slot_recurrent(q, k, v, a, b, A_log, dt_bias, state, slots, lens, bv=bv, num_warps=warps)
                if reference is None:
                    reference = o.float()
                diff = float((o.float() - reference).abs().max())
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    gdn_slot_recurrent(q, k, v, a, b, A_log, dt_bias, state, slots, lens, bv=bv, num_warps=warps)
                end.record()
                torch.cuda.synchronize()
                out.append({"sessions": n, "state_dtype": str(dtype).split(".")[-1], "bv": bv, "warps": warps,
                            "ms_per_call": start.elapsed_time(end) / repeats, "max_abs_diff_vs_8_1": diff})
    return out


def best_tile(sweep):
    """Fastest bf16 tile at 128 sessions whose output matches the v3 tile."""
    rows = [r for r in sweep if r["sessions"] == 128 and r["state_dtype"] == "bfloat16" and r["max_abs_diff_vs_8_1"] < 1e-3]
    best = min(rows, key=lambda r: r["ms_per_call"])
    return best["bv"], best["warps"]


def throughput(torch, make, sources, sessions_list, chunks, ticks):
    rows = []
    for sessions in sessions_list:
        engine = make(sessions)
        source = [sources[s % len(sources)] for s in range(sessions)]
        engine.step([(s, source[s][:256]) for s in range(sessions)])
        offset = 256
        for chunk in chunks:
            latencies = []
            for _ in range(ticks + 5):
                requests = [(s, source[s][offset:offset + chunk]) for s in range(sessions)]
                torch.cuda.synchronize()
                began = time.perf_counter()
                engine.step(requests)["assistant"][-1].cpu()
                latencies.append(time.perf_counter() - began)
                offset += chunk
            timed = latencies[5:]
            rows.append({"sessions": sessions, "tokens_per_tick": chunk, "tick_p50_ms": percentile(timed, .5) * 1e3,
                         "tick_p95_ms": percentile(timed, .95) * 1e3, "itps": sessions * chunk / (sum(timed) / len(timed))})
        rows[-1]["state_bytes_per_session"] = engine.state_bytes_per_session()
        del engine
        torch.cuda.empty_cache()
    return rows


def capacity(sims, slo_ms=20.0):
    """Largest simulated session count whose P95 meets the SLO (no interpolation)."""
    ok = [r["sessions"] for r in sims if r["latency_p95_ms"] <= slo_ms]
    return max(ok) if ok else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v4"))
    parser.add_argument("--sessions", default="1,32,128,512")
    parser.add_argument("--chunks", default="1,16")
    parser.add_argument("--sim-sessions", default="64,96,128,160,192,256")
    parser.add_argument("--sim-seconds", type=float, default=8.0)
    parser.add_argument("--ticks", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v4", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from slot_engine import SlotStreamEngine
        from slot_engine_v4 import SlotStreamEngineV4
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], device=meta["device"])
        report["kernel_check"] = kernel_check(torch)
        save()
        if not report["kernel_check"]["pass"]:
            raise RuntimeError("ring attention kernel disagrees with the reference")

        sequences = [json.loads(line)["ids"] for line in open(args.data)][:400]
        rng = random.Random(0)
        picks = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))[:4] + rng.sample(range(len(sequences)), 12)
        picks = list(dict.fromkeys(picks))[:14]
        reference = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in picks:
                hidden = model.backbone(input_ids=torch.tensor(sequences[i], device="cuda")[None]).last_hidden_state[0]
                reference[i] = torch.softmax(model.readout(hidden, "assistant")[0].float(), -1).cpu()

        def drift(engine):
            got = stream(lambda reqs: engine.step(reqs)["assistant"], sequences, picks, random.Random(7))
            return got, {**compare(got, reference, picks), "cut_score": compare_cut(got, reference, picks)}

        engine = SlotStreamEngine(model, max_slots=16, use_graphs=True)
        got_v3, report_v3 = drift(engine)
        report["drift"] = {"v3": report_v3}
        del engine
        for name, (dtype, ring) in CONFIGS.items():
            engine = SlotStreamEngineV4(model, max_slots=16, use_graphs=True, state_dtype=dtype, ring=ring)
            got, row = drift(engine)
            row["vs_v3"] = {k: v for k, v in compare(got, got_v3, picks).items() if k != "worst"}
            if name == "v4_bf16_inplace":
                eager = SlotStreamEngineV4(model, max_slots=16, use_graphs=False, state_dtype=dtype, ring=ring)
                got_eager, _ = drift(eager)
                row["graph_vs_eager"] = {k: v for k, v in compare(got, got_eager, picks).items() if k != "worst"}
                del eager
            report["drift"][name] = row
            save()
            del engine
            torch.cuda.empty_cache()
        base = report["drift"]["v3"]
        gate = {name: (row["p999"] <= max(0.03, 1.5 * base["p999"]) and row["argmax_flips"] <= base["argmax_flips"] + 4
                       and row["cut_score"]["max"] <= 0.15) for name, row in report["drift"].items() if name != "v3"}
        report["drift_gate"] = gate
        save()

        report["gdn_tile_sweep"] = tile_sweep(torch)
        bv, warps = best_tile(report["gdn_tile_sweep"])
        report["gdn_tile_chosen"] = {"bv": bv, "warps": warps}
        save()

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in sequences]
        makers = {"v3": lambda n: SlotStreamEngine(model, max_slots=n, use_graphs=True)}
        for name in ("v4_bf16_inplace", "v4_fp16_inplace", "v4_fp32_inplace"):
            dtype, ring = CONFIGS[name]
            if gate[name]:
                makers[name] = (lambda d, r: lambda n: SlotStreamEngineV4(model, max_slots=n, use_graphs=True, state_dtype=d,
                                                                         ring=r))(dtype, ring)
                if (bv, warps) != (8, 1):
                    makers[name + "_tile"] = (lambda d, r: lambda n: SlotStreamEngineV4(
                        model, max_slots=n, use_graphs=True, state_dtype=d, ring=r, gdn_bv=bv, gdn_warps=warps))(dtype, ring)
        sessions_list = list(map(int, args.sessions.split(",")))
        chunks = list(map(int, args.chunks.split(",")))
        report["throughput"] = {}
        for name, make in makers.items():
            report["throughput"][name] = throughput(torch, make, sources, sessions_list, chunks, args.ticks)
            save()
            print(name, json.dumps(report["throughput"][name][-1]), flush=True)

        sim_sessions = list(map(int, args.sim_sessions.split(",")))
        report["arrival_simulation"], report["capacity_at_p95_20ms"] = {}, {}
        for name, make in makers.items():
            engine = make(max(sim_sessions))
            with torch.inference_mode():
                for nb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                    if nb <= engine.max_slots:
                        for lb in (1, 2, 4, 8, 16, 32, 64):
                            engine._static(nb, lb)
            sims = [simulate(engine, [sources[s % len(sources)] for s in range(n)], n, args.sim_seconds, random.Random(n))
                    for n in sim_sessions]
            report["arrival_simulation"][name] = sims
            report["capacity_at_p95_20ms"][name] = capacity(sims)
            save()
            print(name, "capacity", report["capacity_at_p95_20ms"][name], flush=True)
            del engine
            torch.cuda.empty_cache()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


if __name__ == "__main__":
    main()
