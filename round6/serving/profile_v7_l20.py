"""L20 (dev pod): kernel-level profile of the v7 engine at the v6 operating point (208 primed sessions, 77 rows x 1
token per tick, session bucket 96), eager mode so every kernel is visible. Prints per-kernel and per-category device
time per tick and the kernel count; with --graph also the CUDA-graph time of the same bucket for reference.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import sys

CATEGORIES = (("gdn", ("_gdn_slot",)), ("ring", ("_ring_attention",)), ("gemm", ("gemm", "cutlass", "sm80_", "sm89_", "cublas", "gemv", "splitK")),
              ("conv", ("conv", "cudnn")), ("norm_reduce", ("reduce", "norm", "mean")),
              ("index", ("index", "gather", "scatter")), ("cat_copy", ("cat", "copy", "Memcpy", "Memset", "fill")),
              ("elementwise", ("elementwise", "vectorized", "unrolled", "pointwise")))


def category(name):
    for cat, keys in CATEGORIES:
        if any(k.lower() in name.lower() for k in keys):
            return cat
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--engine", default="slot_engine_v7:SlotStreamEngineV7")
    parser.add_argument("--sessions", type=int, default=208)
    parser.add_argument("--rows", type=int, default=77)
    parser.add_argument("--ticks", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle))
    import standalone_model
    import torch
    from torch.profiler import ProfilerActivity, profile
    module, name = args.engine.split(":")
    engine_cls = getattr(__import__(module), name)
    _, model, _, _ = standalone_model.load_bundle(args.bundle)
    fine = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)
    kwargs = dict(state_dtype="float16", ring="inplace", gdn_bv=32, gdn_warps=2, session_buckets=fine)
    thinking = [json.loads(line)["ids"] for line in open(args.data)][:400]
    sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in thinking]
    out = {}
    for mode in ("eager", "graph"):
        engine = engine_cls(model, max_slots=args.sessions, use_graphs=mode == "graph", **kwargs)
        rng = random.Random(5)
        engine.step([(s, sources[s][:256]) for s in range(args.sessions)])
        cursor = [256] * args.sessions

        def tick():
            pick = rng.sample(range(args.sessions), args.rows)
            requests = [(s, sources[s][cursor[s]:cursor[s] + 1]) for s in pick]
            for s in pick:
                cursor[s] += 1
            engine.step(requests)
        for _ in range(5):
            tick()
        torch.cuda.synchronize()
        if mode == "eager":
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(args.ticks):
                    tick()
                torch.cuda.synchronize()
            kernels = collections.defaultdict(lambda: [0.0, 0])
            for event in prof.events():
                if event.device_type.name == "CUDA":
                    kernels[event.name][0] += event.device_time_total if hasattr(event, "device_time_total") else event.cuda_time_total
                    kernels[event.name][1] += 1
            per_tick = {k: (t / args.ticks / 1e3, c / args.ticks) for k, (t, c) in kernels.items()}
            cats = collections.defaultdict(lambda: [0.0, 0.0])
            for k, (ms, count) in per_tick.items():
                cats[category(k)][0] += ms
                cats[category(k)][1] += count
            out["eager"] = {"total_ms": sum(v[0] for v in per_tick.values()), "kernels": sum(v[1] for v in per_tick.values()),
                            "categories": {k: {"ms": round(v[0], 4), "count": v[1]} for k, v in sorted(cats.items(), key=lambda x: -x[1][0])},
                            "top": [{"name": k[:140], "ms": round(v[0], 4), "count": v[1], "category": category(k)}
                                    for k, v in sorted(per_tick.items(), key=lambda x: -x[1][0])[:45]]}
            print(json.dumps({k: v for k, v in out["eager"].items() if k != "top"}, indent=1))
            for row in out["eager"]["top"]:
                print(f'{row["ms"]:8.4f} ms {row["count"]:6.1f}x  {row["category"]:12s} {row["name"]}')
        else:
            nb = next(b for b in fine if b >= args.rows)
            graph, _ = engine.graphs[(nb, 1)]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            out["graph_ms"] = start.elapsed_time(end) / 100
            print("graph ms per tick", out["graph_ms"])
        del engine
        torch.cuda.empty_cache()
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
