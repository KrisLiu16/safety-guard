"""L20: correctness drift and throughput of the continuous-batching engine (fixed Round5 weights).

1. Drift: real official-thinking token sequences are streamed through the engine in random
   interleaved chunks and compared per position with one whole-sequence forward of the same
   weights (the streaming-only contract accepts numeric drift; this measures it).
2. Throughput: N concurrent sessions each append c tokens per tick after a 256-token context;
   tick latency (packing + forward + probabilities to CPU) and ITPS = N*c / tick time.
No training, no thresholds, no official labels are read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def load_sequences(path, limit):
    rows = []
    with open(path) as handle:
        for line in handle:
            rows.append(json.loads(line)["ids"])
    return rows[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v1"))
    parser.add_argument("--sessions", default="1,8,32,64,128,256,384")
    parser.add_argument("--chunks", default="4,8,16")
    parser.add_argument("--kernels", default="chunk,recurrent")
    parser.add_argument("--ticks", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_batched_engine_v1", "training": False}
    out = args.output / "report.json"

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    save()
    sys.path.insert(0, str(args.bundle))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import standalone_model
    import torch
    from batched_engine import BatchedStreamEngine
    torch_, model, _, meta = standalone_model.load_bundle(args.bundle)
    report.update(checkpoint_sha256=meta["checkpoint_sha256"], device=meta["device"],
                  versions=meta["runtime_versions"])
    sequences = load_sequences(args.data, 400)
    rng = random.Random(0)

    # ---------- 1. drift versus whole-sequence forward ----------
    picks = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))[:4] + rng.sample(range(len(sequences)), 12)
    picks = list(dict.fromkeys(picks))[:14]
    reference = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in picks:
            ids = torch.tensor(sequences[i], device="cuda")[None]
            hidden = model.backbone(input_ids=ids).last_hidden_state[0]
            reference[i] = torch.softmax(model.readout(hidden, "assistant")[0].float(), -1).cpu()
    engine = BatchedStreamEngine(model, max_slots=64)
    streamed = {i: [] for i in picks}
    cursor = {i: 0 for i in picks}
    slot_of = {i: n for n, i in enumerate(picks)}
    first = True
    while any(cursor[i] < len(sequences[i]) for i in picks):
        active = [i for i in picks if cursor[i] < len(sequences[i]) and (first or rng.random() < 0.7)]
        if not active:
            continue
        requests = []
        for i in active:
            size = rng.choice([700, 64]) if first else rng.randint(1, 24)
            chunk = sequences[i][cursor[i]:cursor[i] + size]
            requests.append((slot_of[i], chunk))
            cursor[i] += len(chunk)
        probs = engine.step(requests)["assistant"].cpu()
        start = 0
        for (slot, chunk), i in zip(requests, active):
            streamed[i].append(probs[start:start + len(chunk)])
            start += len(chunk)
        first = False
    diffs, flips, n_pos = [], 0, 0
    for i in picks:
        got = torch.cat(streamed[i])
        ref = reference[i]
        d = (got[:, 1] - ref[:, 1]).abs()
        diffs.extend(d.tolist())
        flips += int((got.argmax(-1) != ref.argmax(-1)).sum())
        n_pos += len(d)
    report["drift"] = {"sequences": len(picks), "positions": n_pos,
                       "max_abs_dp_unsafe": max(diffs), "p999": percentile(diffs, .999),
                       "p99": percentile(diffs, .99), "mean": sum(diffs) / len(diffs),
                       "argmax_flips": flips,
                       "note": "engine: random interleaved chunks 1-24 after a 700/64-token first append; "
                               "reference: one whole-sequence forward of the same weights"}
    save()
    del engine
    torch.cuda.empty_cache()

    # ---------- 2. throughput ----------
    results = []
    for kernel in args.kernels.split(","):
        for sessions in map(int, args.sessions.split(",")):
            engine = BatchedStreamEngine(model, max_slots=sessions, gdn_kernel=kernel)
            source = [sequences[s % len(sequences)] * 8 for s in range(sessions)]
            for start in range(0, sessions, 32):                      # 256-token context, untimed
                engine.step([(s, source[s][:256]) for s in range(start, min(sessions, start + 32))])
            for chunk in map(int, args.chunks.split(",")):
                offset = 256
                latencies = []
                for tick in range(args.ticks + 5):
                    requests = [(s, source[s][offset:offset + chunk]) for s in range(sessions)]
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    probs = engine.step(requests)["assistant"]
                    probs.cpu()
                    latencies.append(time.perf_counter() - began)
                    offset += chunk
                timed = latencies[5:]
                mean = sum(timed) / len(timed)
                row = {"kernel": kernel, "sessions": sessions, "tokens_per_tick": chunk,
                       "tick_p50_ms": percentile(timed, .5) * 1e3, "tick_p95_ms": percentile(timed, .95) * 1e3,
                       "itps": sessions * chunk / mean,
                       "state_mib_per_session": engine.state_bytes_per_session() / 2 ** 20,
                       "max_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
                results.append(row)
                report["throughput"] = results
                save()
                print(json.dumps(row), flush=True)
            del engine
            torch.cuda.empty_cache()
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
