"""L20: v3 slot engine — drift vs whole/v2, graph vs eager, fixed-shape throughput, and an
arrival-rate simulation that measures per-token latency (arrival -> probabilities on CPU).

Fixed Round5 weights; no training, thresholds or labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

from bench_bucketed_l20 import compare, percentile, stream


def simulate(engine, sources, sessions, seconds, rng):
    """Each session streams tokens at U(30, 60) tok/s after a 256-token context; the scheduler
    packs every token that has arrived into one tick. Latency = result time - token arrival time."""
    import torch
    engine.reset()
    rate = [rng.uniform(30, 60) for _ in range(sessions)]
    offset = [rng.uniform(0, 1.0) for _ in range(sessions)]
    engine.step([(s, sources[s][:256]) for s in range(sessions)])
    torch.cuda.synchronize()
    delivered = [0] * sessions
    latencies, processed, ticks, sizes = [], 0, 0, []
    t0 = time.perf_counter() + 0.05
    while True:
        now = time.perf_counter() - t0
        if now > seconds:
            break
        requests, arrivals = [], []
        for s in range(sessions):
            arrived = int((now - offset[s]) * rate[s]) if now > offset[s] else 0
            n = min(arrived - delivered[s], 64)
            if n > 0:
                start = 256 + delivered[s]
                requests.append((s, sources[s][start:start + n]))
                arrivals.append([t0 + offset[s] + (delivered[s] + j + 1) / rate[s] for j in range(n)])
                delivered[s] += n
        if not requests:
            time.sleep(0.0002)
            continue
        out = engine.step(requests)["assistant"]
        out[-1].cpu()
        done = time.perf_counter()
        for row in arrivals:
            latencies.extend(done - t for t in row)
        processed += sum(len(c) for _, c in requests)
        ticks += 1
        sizes.append(len(requests))
    return {"sessions": sessions, "offered_tokens_per_s": sum(rate), "itps": processed / seconds,
            "latency_p50_ms": percentile(latencies, .5) * 1e3, "latency_p95_ms": percentile(latencies, .95) * 1e3,
            "latency_p99_ms": percentile(latencies, .99) * 1e3, "ticks_per_s": ticks / seconds,
            "mean_sessions_per_tick": sum(sizes) / max(1, len(sizes)),
            "mean_tokens_per_tick": processed / max(1, ticks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v3"))
    parser.add_argument("--sessions", default="1,8,32,64,128,256,512")
    parser.add_argument("--chunks", default="1,2,4,8,16")
    parser.add_argument("--sim-sessions", default="32,64,128,192,256,320,384,512")
    parser.add_argument("--sim-seconds", type=float, default=8.0)
    parser.add_argument("--ticks", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v3", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from bucketed_engine import BucketedStreamEngine
        from slot_engine import SlotStreamEngine
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], device=meta["device"])
        sequences = [json.loads(line)["ids"] for line in open(args.data)][:400]
        rng = random.Random(0)
        picks = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))[:4] + rng.sample(range(len(sequences)), 12)
        picks = list(dict.fromkeys(picks))[:14]
        reference = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in picks:
                hidden = model.backbone(input_ids=torch.tensor(sequences[i], device="cuda")[None]).last_hidden_state[0]
                reference[i] = torch.softmax(model.readout(hidden, "assistant")[0].float(), -1).cpu()
        v2 = BucketedStreamEngine(model, max_slots=16, use_graphs=True)
        got_v2 = stream(lambda reqs: v2.step(reqs)["assistant"], sequences, picks, random.Random(7))
        del v2
        v3 = SlotStreamEngine(model, max_slots=16, use_graphs=True)
        got_v3 = stream(lambda reqs: v3.step(reqs)["assistant"], sequences, picks, random.Random(7))
        v3e = SlotStreamEngine(model, max_slots=16, use_graphs=False)
        got_v3e = stream(lambda reqs: v3e.step(reqs)["assistant"], sequences, picks, random.Random(7))
        report["drift"] = {"v3_graph_vs_whole": compare(got_v3, reference, picks), "v3_graph_vs_v2": compare(got_v3, got_v2, picks),
                           "v3_graph_vs_v3_eager": compare(got_v3, got_v3e, picks), "engine_stats": v3.stats}
        save()
        del v3, v3e
        torch.cuda.empty_cache()

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in sequences]
        results = []
        for sessions in map(int, args.sessions.split(",")):
            engine = SlotStreamEngine(model, max_slots=sessions, use_graphs=True)
            source = [sources[s % len(sources)] for s in range(sessions)]
            engine.step([(s, source[s][:256]) for s in range(sessions)])
            offset = 256
            for chunk in map(int, args.chunks.split(",")):
                latencies = []
                for _ in range(args.ticks + 5):
                    requests = [(s, source[s][offset:offset + chunk]) for s in range(sessions)]
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    engine.step(requests)["assistant"][-1].cpu()
                    latencies.append(time.perf_counter() - began)
                    offset += chunk
                timed = latencies[5:]
                row = {"sessions": sessions, "tokens_per_tick": chunk,
                       "tick_p50_ms": percentile(timed, .5) * 1e3, "tick_p95_ms": percentile(timed, .95) * 1e3,
                       "itps": sessions * chunk / (sum(timed) / len(timed))}
                results.append(row)
                report["throughput"] = results
                save()
                print(json.dumps(row), flush=True)
            del engine
            torch.cuda.empty_cache()

        engine = SlotStreamEngine(model, max_slots=max(map(int, args.sim_sessions.split(","))), use_graphs=True)
        # capture every bucket the simulation can hit before timing
        with torch.inference_mode():
            for nb in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                if nb <= engine.max_slots:
                    for lb in (1, 2, 4, 8, 16, 32, 64):
                        engine._static(nb, lb)
        sims = []
        for sessions in map(int, args.sim_sessions.split(",")):
            row = simulate(engine, [sources[s % len(sources)] for s in range(sessions)], sessions, args.sim_seconds,
                           random.Random(sessions))
            sims.append(row)
            report["arrival_simulation"] = sims
            save()
            print(json.dumps(row), flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


if __name__ == "__main__":
    main()
