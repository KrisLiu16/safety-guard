"""L20: T034 step B — v8 (fused element-wise glue) against v6 and v7.
  1. identity: the 400 official thinking sequences (seed 7) and the prefix_v2 dev assistant records (seed 11) streamed
     through v6 and v8 with CUDA graphs; every probability must be equal (max |d| = 0);
  2. tick: 208 primed sessions, 77 rows x 1 token (bucket 96): wall time per step and graph GPU time, v6 / v7 / v8;
  3. arrival simulation (as bench_v7_l20, vectorised simulator) for v7 and v8 up to 352 sessions, 2 seeds each.
Fixed Round5 weights; no training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import traceback

from bench_v7_l20 import FINE, TOKEN_BUCKETS, fixed_tick, identity, simulate_fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--prefix-dev", type=Path, default=Path("/work/round5/data/prefix_v2/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v8"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sim-sessions", default="224,240,256,272,288,304,320,336,352")
    parser.add_argument("--sim-seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--skip-identity", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v8_step_b", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from bench_bucketed_l20 import stream
        from slot_engine_v4 import SlotStreamEngineV4
        from slot_engine_v7 import SlotStreamEngineV7
        from slot_engine_v8 import SlotStreamEngineV8
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], gpu=torch.cuda.get_device_name(0))
        kwargs = dict(state_dtype="float16", ring="inplace", gdn_bv=32, gdn_warps=2, session_buckets=FINE)
        engines = {"v6": SlotStreamEngineV4, "v7": SlotStreamEngineV7, "v8": SlotStreamEngineV8}

        def make(name):
            return lambda n: engines[name](model, max_slots=n, **kwargs)

        def stream_all(factory, sequences, picks, seed):
            engine, out = factory(args.batch), {}
            for start in range(0, len(picks), args.batch):
                part = picks[start:start + args.batch]
                engine.reset()
                out.update(stream(lambda reqs: engine.step(reqs)["assistant"], sequences, part, random.Random(seed + start)))
            del engine
            torch.cuda.empty_cache()
            return out

        thinking = [json.loads(line)["ids"] for line in open(args.data)][:400]
        prefix = [r["ids"] for r in map(json.loads, open(args.prefix_dev)) if r["target_role"] == "assistant"]
        if not args.skip_identity:
            report["identity"] = {}
            for name, (sequences, seed) in {"thinking": (thinking, 7), "prefix_v2_dev_assistant": (prefix, 11)}.items():
                picks = list(range(len(sequences)))
                got6 = stream_all(make("v6"), sequences, picks, seed)
                got8 = stream_all(make("v8"), sequences, picks, seed)
                report["identity"][name] = identity(got8, got6, picks)
                save()
                print(name, "identity v8 vs v6", report["identity"][name], flush=True)

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in thinking]
        report["tick"] = {}
        for name, packed in (("v6", False), ("v7", True), ("v8", True)):
            engine = make(name)(208)
            with torch.inference_mode():
                for nb in FINE:
                    if nb <= 208:
                        for lb in (1, 64):
                            engine._static(nb, lb)
            report["tick"][name] = fixed_tick(torch, engine, sources, 208, 77, 300, packed)
            save()
            print("tick", name, report["tick"][name], flush=True)
            del engine
            torch.cuda.empty_cache()

        sim_sessions = list(map(int, args.sim_sessions.split(",")))
        report["arrival_simulation"], report["capacity_at_p95_20ms"] = {}, {}
        for name in ("v7", "v8"):
            engine = make(name)(max(sim_sessions))
            with torch.inference_mode():
                for nb in engine.session_buckets:
                    if nb <= engine.max_slots:
                        for lb in TOKEN_BUCKETS:
                            engine._static(nb, lb)
            sims = []
            for n in sim_sessions:
                for seed in range(args.seeds):
                    row = simulate_fast(engine, [sources[s % len(sources)] for s in range(n)], n, args.sim_seconds,
                                        random.Random(n + 1000 * seed), True)
                    row["seed"] = seed
                    sims.append(row)
                    print(name, json.dumps(row), flush=True)
            report["arrival_simulation"][name] = sims
            worst = {n: max(r["latency_p95_ms"] for r in sims if r["sessions"] == n) for n in sim_sessions}
            ok = [n for n in sim_sessions if worst[n] <= 20.0]
            report["capacity_at_p95_20ms"][name] = {"sessions": max(ok) if ok else 0, "worst_seed_p95_ms": worst}
            save()
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
