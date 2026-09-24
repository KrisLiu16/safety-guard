"""L20: v4 follow-up — decision-level drift on 400 sequences, fp16 GDN tiles per session bucket, a profile of the
candidate, and its P95 <= 20 ms capacity with per-bucket tiles.

Why: in T007 v4 the worst drift of every configuration came from one near-boundary position (reference p = 0.55),
so max |dp| over 14 sequences says little about decisions. Here every configuration streams the same 400
official thinking sequences on the same random schedule, and is compared with the whole-sequence forward on
  positions   max / p99.9 / p99 and how many positions move by more than 0.05 and 0.1,
  decisions   for each threshold on the score (cut score 1 - p(safe), and p(unsafe)) and rule (threshold,
              consecutive_2): streams whose fire / no-fire differs, and streams that both fire at different indices.
Fixed Round5 weights; no training, thresholds or labels (the thresholds are a grid, not calibrated values).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

from bench_bucketed_l20 import percentile, stream
from bench_slot_l20 import simulate

SESSION_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TILES = ((8, 1), (16, 1), (16, 2), (32, 2), (32, 4), (64, 4), (64, 8), (128, 8))
TAUS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)


def first_fire(scores, tau, k):
    run = 0
    for i, s in enumerate(scores):
        run = run + 1 if s > tau else 0
        if run >= k:
            return i
    return -1


def decision_drift(streamed, reference, picks):
    """Position and stream-decision agreement of streamed vs reference probabilities ([T, 3] per sequence)."""
    out = {}
    for score_name, score in (("cut", lambda p: 1 - p[:, 0]), ("unsafe", lambda p: p[:, 1])):
        diffs, fires = [], {}
        for i in picks:
            got, ref = score(streamed[i]), score(reference[i])
            diffs.extend(float(x) for x in (got - ref).abs())
            for tau in TAUS:
                for rule, k in (("threshold", 1), ("consecutive_2", 2)):
                    a, b = first_fire(ref.tolist(), tau, k), first_fire(got.tolist(), tau, k)
                    entry = fires.setdefault(f"{rule}@{tau}", {"streams": 0, "fire_differs": 0, "both_fire_other_index": 0,
                                                              "reference_fires": 0})
                    entry["streams"] += 1
                    entry["reference_fires"] += a >= 0
                    entry["fire_differs"] += (a >= 0) != (b >= 0)
                    entry["both_fire_other_index"] += a >= 0 and b >= 0 and a != b
        out[score_name] = {"positions": len(diffs), "max": max(diffs), "p999": percentile(diffs, .999),
                           "p99": percentile(diffs, .99), "over_0.05": sum(d > .05 for d in diffs),
                           "over_0.1": sum(d > .1 for d in diffs), "decisions": fires}
    return out


def stream_all(make, sequences, picks, batch, seed):
    """Stream `picks` through one engine, `batch` sequences at a time (slots 0..batch-1, reset in between)."""
    engine, out = make(batch), {}
    for start in range(0, len(picks), batch):
        part = picks[start:start + batch]
        engine.reset()
        out.update(stream(lambda reqs: engine.step(reqs)["assistant"], sequences, part, random.Random(seed + start)))
    return out


def tile_sweep(torch, dtype, repeats=50):
    from gdn_slot_kernel import gdn_slot_recurrent
    rows = []
    for n in SESSION_BUCKETS:
        gen = torch.Generator(device="cuda").manual_seed(n)
        q = torch.randn(n, 1, 16, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
        k, v = torch.randn_like(q), torch.randn_like(q)
        a = torch.randn(n, 1, 16, device="cuda", dtype=torch.bfloat16, generator=gen)
        b = torch.randn_like(a)
        A_log, dt_bias = torch.zeros(16, device="cuda"), torch.zeros(16, device="cuda")
        base = (torch.randn(n + 1, 16, 128, 128, device="cuda", generator=gen) * 0.1).to(dtype)
        slots, lens = torch.arange(n, device="cuda"), torch.ones(n, dtype=torch.long, device="cuda")
        reference = None
        for bv, warps in TILES:
            state = base.clone()
            o = gdn_slot_recurrent(q, k, v, a, b, A_log, dt_bias, state, slots, lens, bv=bv, num_warps=warps).float()
            reference = o if reference is None else reference
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                gdn_slot_recurrent(q, k, v, a, b, A_log, dt_bias, state, slots, lens, bv=bv, num_warps=warps)
            end.record()
            torch.cuda.synchronize()
            rows.append({"sessions": n, "bv": bv, "warps": warps, "ms_per_call": start.elapsed_time(end) / repeats,
                         "max_abs_diff_vs_8_1": float((o - reference).abs().max())})
    return rows


def tiles_per_bucket(rows):
    best = {}
    for row in rows:
        if row["max_abs_diff_vs_8_1"] < 1e-3 and (row["sessions"] not in best or row["ms_per_call"] < best[row["sessions"]]["ms_per_call"]):
            best[row["sessions"]] = row
    return {n: (r["bv"], r["warps"]) for n, r in best.items()}


def profile_tick(torch, engine, sources, sessions, chunk, ticks=3):
    from torch.profiler import ProfilerActivity, profile
    engine.step([(s, sources[s % len(sources)][:256]) for s in range(sessions)])
    offset = 256
    for _ in range(3):
        engine.step([(s, sources[s % len(sources)][offset:offset + chunk]) for s in range(sessions)])
        offset += chunk
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(ticks):
            engine.step([(s, sources[s % len(sources)][offset:offset + chunk]) for s in range(sessions)])
            offset += chunk
        torch.cuda.synchronize()
    events = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:30]
    return [{"name": e.key[:120], "cuda_ms_per_tick": e.device_time_total / (ticks * 1e3), "calls_per_tick": e.count / ticks}
            for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v5"))
    parser.add_argument("--max-seqs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sim-sessions", default="160,176,192,208,224,240,256")
    parser.add_argument("--sim-seconds", type=float, default=8.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v5_check", "training": False}

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

        report["tile_sweep_fp16"] = tile_sweep(torch, torch.float16)
        tiles = tiles_per_bucket(report["tile_sweep_fp16"])
        report["tiles_chosen_fp16"] = {str(n): list(t) for n, t in tiles.items()}
        save()

        configs = {
            "v3": lambda n: SlotStreamEngine(model, max_slots=n, use_graphs=True),
            "v4_fp32_inplace": lambda n: SlotStreamEngineV4(model, max_slots=n, state_dtype="float32", ring="inplace"),
            "v4_fp16_gather": lambda n: SlotStreamEngineV4(model, max_slots=n, state_dtype="float16", ring="gather"),
            "v4_fp16_inplace_tiles": lambda n: SlotStreamEngineV4(model, max_slots=n, state_dtype="float16", ring="inplace",
                                                                  gdn_tiles=tiles),
        }
        sequences = [json.loads(line)["ids"] for line in open(args.data)][:args.max_seqs]
        picks = list(range(len(sequences)))
        reference = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in picks:
                hidden = model.backbone(input_ids=torch.tensor(sequences[i], device="cuda")[None]).last_hidden_state[0]
                reference[i] = torch.softmax(model.readout(hidden, "assistant")[0].float(), -1).cpu()
        report["sequences"], report["positions"] = len(picks), sum(len(s) for s in sequences)
        report["decision_drift"] = {}
        for name, make in configs.items():
            began = time.perf_counter()
            got = stream_all(make, sequences, picks, args.batch, seed=7)
            report["decision_drift"][name] = {**decision_drift(got, reference, picks), "seconds": time.perf_counter() - began}
            save()
            torch.cuda.empty_cache()
            print(name, json.dumps({k: {m: v[m] for m in ("max", "p999", "over_0.1")} for k, v in report["decision_drift"][name].items()
                                    if isinstance(v, dict)}), flush=True)

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in sequences]
        report["profile"] = {}
        for sessions in (128, 192):
            engine = SlotStreamEngineV4(model, max_slots=sessions, use_graphs=False, state_dtype="float16", ring="inplace",
                                        gdn_tiles=tiles)
            report["profile"][f"n{sessions}_c1"] = profile_tick(torch, engine, sources, sessions, 1)
            save()
            del engine
            torch.cuda.empty_cache()

        sim_sessions = list(map(int, args.sim_sessions.split(",")))
        report["arrival_simulation"], report["capacity_at_p95_20ms"] = {}, {}
        for name in ("v4_fp16_inplace_tiles", "v4_fp16_inplace_tile32x2"):
            fixed = {n: (32, 2) for n in SESSION_BUCKETS}
            engine = SlotStreamEngineV4(model, max_slots=max(sim_sessions), state_dtype="float16", ring="inplace",
                                        gdn_tiles=tiles if name.endswith("tiles") else fixed)
            with torch.inference_mode():
                for nb in SESSION_BUCKETS:
                    if nb <= engine.max_slots:
                        for lb in (1, 2, 4, 8, 16, 32, 64):
                            engine._static(nb, lb)
            sims = [simulate(engine, [sources[s % len(sources)] for s in range(n)], n, args.sim_seconds, random.Random(n))
                    for n in sim_sessions]
            report["arrival_simulation"][name] = sims
            ok = [r["sessions"] for r in sims if r["latency_p95_ms"] <= 20.0]
            report["capacity_at_p95_20ms"][name] = max(ok) if ok else 0
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
