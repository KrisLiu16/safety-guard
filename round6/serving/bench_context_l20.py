"""L20: the v8 engine at long contexts, same protocol as bench_official_l20.py (the Qwen3Guard-Stream baselines).

bench_v8_l20 primes every session with 256 tokens. Thinking streams run to thousands of tokens, and Qwen3Guard's cost
per token grows with the stream, so the comparison is made at contexts of 512 to 8K: every session is primed with
`ctx` tokens (in 256-token steps), then receives tokens at U(30, 60)/s for 10 s; latency = verdict on the host minus
arrival. Also the single-stream service time (one session, one token per tick, back to back). Same thinking
conversations as the baselines (our tokenizer's ids, repeated to length). Fixed Round5 weights; no training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from bench_v7_l20 import FINE, TOKEN_BUCKETS, arrival_times, due_tokens, percentile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="512,1024,2048,4096,8000")
    parser.add_argument("--sessions", default="320,336,352,368")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "kind": "round6_slot_engine_v8_context", "training": False,
              "settings": {k: str(v) for k, v in vars(args).items()}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from slot_engine_v8 import SlotStreamEngineV8
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], gpu=torch.cuda.get_device_name(0))
        contexts = list(map(int, args.contexts.split(",")))
        grid = list(map(int, args.sessions.split(",")))
        need = max(contexts) + 1024
        thinking = [json.loads(line)["ids"] for line in open(args.data)][:400]
        sources = [(seq * (need // len(seq) + 2))[:need] for seq in thinking]
        engine = SlotStreamEngineV8(model, max_slots=max(grid), state_dtype="float16", ring="inplace", gdn_bv=32,
                                    gdn_warps=2, session_buckets=FINE)
        with torch.inference_mode():
            for nb in engine.session_buckets:
                if nb <= engine.max_slots:
                    for lb in TOKEN_BUCKETS:
                        engine._static(nb, lb)

        def prime(n, ctx):
            engine.reset()
            for c in range(0, ctx, 256):
                engine.step([(s, sources[s % len(sources)][c:min(c + 256, ctx)]) for s in range(n)])
            torch.cuda.synchronize()

        def service(ctx, tokens=200, warm=20):
            prime(1, ctx)
            src = sources[0]
            times = []
            for i in range(tokens + warm):
                began = time.perf_counter()
                engine.step_packed([(0, src[ctx + i:ctx + i + 1])])
                if i >= warm:
                    times.append(time.perf_counter() - began)
            return {"context": ctx, "p50_ms": percentile(times, .5) * 1e3, "p95_ms": percentile(times, .95) * 1e3}

        def simulate(n, ctx, seed):
            rng = random.Random(n + 1000 * seed)
            rate = np.array([rng.uniform(30, 60) for _ in range(n)])
            offset = np.array([rng.uniform(0, 1.0) for _ in range(n)])
            prime(n, ctx)
            delivered = np.zeros(n, dtype=np.int64)
            latencies, processed = [], 0
            t0 = time.perf_counter() + 0.05
            while True:
                now = time.perf_counter() - t0
                if now > args.seconds:
                    break
                due = due_tokens(now, offset, rate, delivered)
                active = np.flatnonzero(due > 0)
                if len(active) == 0:
                    time.sleep(0.0002)
                    continue
                na, da = due[active], delivered[active]
                requests = [(s, sources[s % len(sources)][ctx + d:ctx + d + k])
                            for s, d, k in zip(active.tolist(), da.tolist(), na.tolist())]
                arrivals = arrival_times(t0, offset[active], rate[active], da, na)
                delivered[active] += na
                engine.step_packed(requests)
                latencies.append(time.perf_counter() - arrivals)
                processed += int(na.sum())
            values = np.concatenate(latencies).tolist()
            return {"sessions": n, "context": ctx, "seed": seed, "offered_tokens_per_s": float(rate.sum()),
                    "itps": processed / args.seconds, "tokens_measured": len(values),
                    "latency_p50_ms": percentile(values, .5) * 1e3, "latency_p95_ms": percentile(values, .95) * 1e3,
                    "latency_p99_ms": percentile(values, .99) * 1e3}

        report["service"], report["arrival_simulation"], report["capacity_at_p95_20ms"] = [], [], {}
        with torch.inference_mode():
            for ctx in contexts:
                row = service(ctx)
                report["service"].append(row)
                save()
                print("service", json.dumps(row), flush=True)
            for ctx in contexts:
                best = 0
                for n in grid:
                    rows = [simulate(n, ctx, seed) for seed in range(args.seeds)]
                    report["arrival_simulation"].extend(rows)
                    save()
                    for row in rows:
                        print("sim", json.dumps(row), flush=True)
                    if max(r["latency_p95_ms"] for r in rows) > 20.0:
                        break
                    best = n
                report["capacity_at_p95_20ms"][str(ctx)] = best
                save()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


if __name__ == "__main__":
    main()
