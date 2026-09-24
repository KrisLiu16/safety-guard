"""L20: bucketed/graph engine (v2) drift, v1-vs-v2 outlier check, graph-vs-eager, and throughput.

Fixed Round5 weights; no training, thresholds or labels. Drift reference = one whole-sequence
forward of the same weights. Throughput: N sessions each append c tokens per tick after a
256-token context; tick latency covers host packing + graph replay + probabilities to CPU.
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


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def stream(engine_step, sequences, picks, rng, first_sizes=(700, 64), max_chunk=24):
    """Random interleaved chunked streaming; returns {seq: [T,3] assistant probs}."""
    cursor = {i: 0 for i in picks}
    out = {i: [] for i in picks}
    first = True
    while any(cursor[i] < len(sequences[i]) for i in picks):
        active = [i for i in picks if cursor[i] < len(sequences[i]) and (first or rng.random() < 0.7)]
        if not active:
            continue
        requests = []
        for i in active:
            size = rng.choice(first_sizes) if first else rng.randint(1, max_chunk)
            requests.append((picks.index(i), sequences[i][cursor[i]:cursor[i] + size]))
            cursor[i] += len(requests[-1][1])
        for i, probs in zip(active, engine_step(requests)):
            out[i].append(probs.cpu())
        first = False
    import torch
    return {i: torch.cat(v) for i, v in out.items()}


def compare(streamed, reference, picks, top=10):
    rows, flips = [], 0
    for i in picks:
        got, ref = streamed[i], reference[i]
        d = (got[:, 1] - ref[:, 1]).abs()
        flips += int((got.argmax(-1) != ref.argmax(-1)).sum())
        rows += [(float(d[p]), i, p, float(ref[p, 1]), float(got[p, 1])) for p in range(len(d))]
    values = [r[0] for r in rows]
    worst = sorted(rows, reverse=True)[:top]
    return {"positions": len(rows), "max": max(values), "p999": percentile(values, .999),
            "p99": percentile(values, .99), "mean": sum(values) / len(values), "argmax_flips": flips,
            "worst": [{"abs_dp": a, "seq": i, "pos": p, "ref_p_unsafe": r, "got_p_unsafe": g} for a, i, p, r, g in worst]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v2"))
    parser.add_argument("--sessions", default="1,8,32,64,128,256,384,512")
    parser.add_argument("--chunks", default="1,2,4,8,16")
    parser.add_argument("--ticks", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_bucketed_engine_v2", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import standalone_model
        import torch
        from batched_engine import BatchedStreamEngine
        from bucketed_engine import BucketedStreamEngine
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

        # drift: v1 varlen engine and v2 graph engine on the identical random schedule
        v1 = BatchedStreamEngine(model, max_slots=16)
        got_v1 = stream(lambda reqs: [p for p in _split(v1.step(reqs)["assistant"], reqs)], sequences, picks, random.Random(7))
        del v1
        v2 = BucketedStreamEngine(model, max_slots=16, use_graphs=True)
        got_v2 = stream(lambda reqs: v2.step(reqs)["assistant"], sequences, picks, random.Random(7))
        v2e = BucketedStreamEngine(model, max_slots=16, use_graphs=False)
        got_v2e = stream(lambda reqs: v2e.step(reqs)["assistant"], sequences, picks, random.Random(7))
        report["drift"] = {"v1_vs_whole": compare(got_v1, reference, picks), "v2_graph_vs_whole": compare(got_v2, reference, picks),
                           "v2_graph_vs_v2_eager": compare(got_v2, got_v2e, picks), "v2_graph_vs_v1": compare(got_v2, got_v1, picks),
                           "engine_stats": v2.stats}
        save()
        del v2, v2e
        torch.cuda.empty_cache()

        long_sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in sequences]
        results = []
        for kernel in ("recurrent", "chunk"):
            for sessions in map(int, args.sessions.split(",")):
                engine = BucketedStreamEngine(model, max_slots=sessions, gdn_kernel=kernel, use_graphs=True)
                source = [long_sources[s % len(long_sources)] for s in range(sessions)]
                engine.step([(s, source[s][:256]) for s in range(sessions)])      # untimed context
                offset = 256
                for chunk in map(int, args.chunks.split(",")):
                    latencies = []
                    for tick in range(args.ticks + 5):
                        requests = [(s, source[s][offset:offset + chunk]) for s in range(sessions)]
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        engine.step(requests)["assistant"][-1].cpu()
                        latencies.append(time.perf_counter() - began)
                        offset += chunk
                    timed = latencies[5:]
                    mean = sum(timed) / len(timed)
                    row = {"kernel": kernel, "sessions": sessions, "tokens_per_tick": chunk,
                           "tick_p50_ms": percentile(timed, .5) * 1e3, "tick_p95_ms": percentile(timed, .95) * 1e3,
                           "itps": sessions * chunk / mean, "graph": True,
                           "state_mib_per_session": engine.state_bytes_per_session() / 2 ** 20,
                           "max_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
                    results.append(row)
                    report["throughput"] = results
                    save()
                    print(json.dumps(row), flush=True)
                del engine
                torch.cuda.empty_cache()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


def _split(probs, requests):
    out, start = [], 0
    for _, chunk in requests:
        out.append(probs[start:start + len(chunk)])
        start += len(chunk)
    return out


if __name__ == "__main__":
    main()
