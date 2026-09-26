"""L20: Qwen3Guard-Stream-0.6B served the two ways its model card documents, under our arrival simulation.

  hf      transformers + trust_remote_code, model.stream_moderate_from_ids one token per call (the model card's first
          example). The released config has use_cache=false, so stream_generate re-runs the whole sequence for every
          new token; this is measured as released, not patched.
  sglang  the model card's SGLang example (branch support_qwen3_guard): Engine(context_length=10000, page_size=1,
          chunked_prefill_size=131072), one resumable request id per session; each append carries every token that
          has arrived (the card's example uses chunks of 8; sending on arrival is the latency-optimal choice).
          mem_fraction_static 0.85 instead of the card's 0.6 (more KV cache, i.e. more sessions for Qwen3Guard).

Same protocol as bench_v7_l20.simulate_fast / bench_context_l20.py for our engine: every session receives tokens at
U(30, 60)/s with a U(0, 1) s offset, after a context of `ctx` tokens; latency = time the verdict for a token is on the
host minus the token's arrival time; capacity = most sessions with P95 <= 20 ms. Token streams: the first 400
Qwen3GuardTest thinking conversations (official_adapted_v1), encoded with Qwen3Guard's chat template and tokenizer and
repeated to length; the start of every stream is rotated per measurement point so the radix cache never sees a
prefix twice. No training; nothing is written but report JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def percentile(values, q):
    """As bench_bucketed_l20.percentile (nearest rank); nan when nothing was measured."""
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))] if values else float("nan")


def due_tokens(now, offset, rate, delivered, cap=64):
    """Pure (numpy), as bench_v7_l20.due_tokens."""
    arrived = np.where(now > offset, ((now - offset) * rate).astype(np.int64), 0)
    return np.minimum(arrived - delivered, cap)


def draws(sessions, rng):
    """Arrival parameters in the order bench_v7_l20.simulate_fast draws them."""
    rate = np.array([rng.uniform(30, 60) for _ in range(sessions)])
    offset = np.array([rng.uniform(0, 1.0) for _ in range(sessions)])
    return rate, offset


def summary(sessions, ctx, rate, values, processed, seconds, extra=None):
    row = {"sessions": sessions, "context": ctx, "offered_tokens_per_s": float(rate.sum()), "itps": processed / seconds,
           "tokens_measured": len(values), "latency_p50_ms": percentile(values, .5) * 1e3,
           "latency_p95_ms": percentile(values, .95) * 1e3, "latency_p99_ms": percentile(values, .99) * 1e3}
    row.update(extra or {})
    return row


def guard_sources(tokenizer, data, need, count=400):
    """Qwen3Guard token ids of the first `count` thinking conversations, each repeated to `need` + 4096 tokens."""
    out = []
    for line in open(data):
        row = json.loads(line)
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True)
        out.append((ids * ((need + 4096) // len(ids) + 2))[:need + 4096])
        if len(out) == count:
            break
    return out


def stream_of(sources, s, point):
    """Session s's stream at measurement point `point`: rotated so no two points share a prefix."""
    src = sources[s % len(sources)]
    k = (97 * point + 13 * s) % 4096
    return src[k:]


# ----------------------------------------------------------------------------------------------------------- hf
def run_hf(args, report, save):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).eval()                   # as on the model card
    report.update(gpu=torch.cuda.get_device_name(0), use_cache_in_config=bool(model.config.use_cache))
    contexts = list(map(int, args.contexts.split(",")))
    sources = guard_sources(tokenizer, args.data, max(contexts) + 1024)
    point = 0

    def service(ctx, tokens=24, warm=4):
        """Back-to-back per-token time of one stream after `ctx` tokens (no queueing)."""
        stream = torch.tensor(stream_of(sources, 0, 10_000 + ctx))
        _, state = model.stream_moderate_from_ids(stream[:ctx], role="assistant", stream_state=None)
        times = []
        for i in range(tokens + warm):
            began = time.perf_counter()
            result, state = model.stream_moderate_from_ids(stream[ctx + i], role="assistant", stream_state=state)
            if i >= warm:
                times.append(time.perf_counter() - began)
        model.close_stream(state)
        return {"context": ctx, "p50_ms": percentile(times, .5) * 1e3, "p95_ms": percentile(times, .95) * 1e3,
                "max_streams_throughput": 1.0 / (percentile(times, .5) * 45.0)}

    def simulate(n, ctx, seed):
        nonlocal point
        point += 1
        rng = random.Random(n + 1000 * seed)
        rate, offset = draws(n, rng)
        streams = [torch.tensor(stream_of(sources, s, point)) for s in range(n)]
        states = []
        for s in range(n):
            _, state = model.stream_moderate_from_ids(streams[s][:ctx], role="assistant", stream_state=None)
            states.append(state)
        torch.cuda.synchronize()
        delivered = np.zeros(n, dtype=np.int64)
        values, processed = [], 0
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
            for s in active.tolist():                                                      # one model, served in turn
                for j in range(int(due[s])):
                    arrival = t0 + offset[s] + (delivered[s] + j + 1) / rate[s]
                    token = streams[s][ctx + delivered[s] + j]
                    _, states[s] = model.stream_moderate_from_ids(token, role="assistant", stream_state=states[s])
                    values.append(time.perf_counter() - arrival)
                    processed += 1
                delivered[s] += due[s]
        for state in states:
            model.close_stream(state)
        backlog = int(np.minimum(((args.seconds - offset) * rate).astype(np.int64), 10 ** 9).sum() - delivered.sum())
        return summary(n, ctx, rate, values, processed, args.seconds, {"seed": seed, "undelivered_at_end": backlog})

    report["service"], report["arrival_simulation"], report["capacity_at_p95_20ms"] = [], [], {}
    for ctx in contexts:
        row = service(ctx)
        report["service"].append(row)
        save()
        print("service", json.dumps(row), flush=True)
    for ctx in contexts:
        best = 0
        for n in (1, 2, 3, 4, 6, 8, 12, 16):
            rows = [simulate(n, ctx, seed) for seed in range(args.seeds)]
            report["arrival_simulation"].extend(rows)
            save()
            for row in rows:
                print("sim", json.dumps(row), flush=True)
            if max(r["latency_p95_ms"] for r in rows) > args.budget_ms:
                break
            best = n
        report["capacity_at_p95_20ms"][str(ctx)] = best
        save()


# ------------------------------------------------------------------------------------------------------- sglang
def run_sglang(args, report, save):
    import sglang
    import torch
    from sglang.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = Engine(model_path=str(args.model), context_length=10000, page_size=1, tp_size=1,
                    mem_fraction_static=args.mem_fraction, chunked_prefill_size=131072)          # as on the model card
    report.update(gpu=torch.cuda.get_device_name(0), sglang_version=sglang.__version__, mem_fraction_static=args.mem_fraction)
    contexts = list(map(int, args.contexts.split(",")))
    sources = guard_sources(tokenizer, args.data, max(contexts) + 1024)
    loop = asyncio.get_event_loop()                                                         # the engine's own loop
    point = 0
    counter = [0]

    async def request(ids, rid, resumable):
        return await engine.async_generate(input_ids=list(map(int, ids)), sampling_params={"max_new_tokens": 1},
                                           rid=rid, resumable=resumable)

    def check(out, n_tokens, prime=False):
        """Appends return one 3-way logit row per new token; a priming request may hit a few cached prefix tokens."""
        size = np.asarray(out["risk_level_logits"]).size
        assert (0 < size <= 3 * n_tokens) if prime else size == 3 * n_tokens, (size, n_tokens)

    async def simulate(n, ctx, seed):
        nonlocal point
        point += 1
        rng = random.Random(n + 1000 * seed)
        rate, offset = draws(n, rng)
        streams = [stream_of(sources, s, point) for s in range(n)]
        counter[0] += 1
        rids = [f"p{counter[0]}s{s}" for s in range(n)]
        outs = await asyncio.gather(*[request(streams[s][:ctx], rids[s], True) for s in range(n)])
        for out in outs:
            check(out, ctx, prime=True)
        values, processed = [], [0]
        t0 = time.perf_counter() + 0.05
        delivered = [0] * n

        async def session(s):
            while True:
                now = time.perf_counter() - t0
                if now > args.seconds:
                    return
                arrived = int((now - offset[s]) * rate[s]) if now > offset[s] else 0
                k = min(arrived - delivered[s], 64)
                if k <= 0:
                    wake = offset[s] + (delivered[s] + 1) / rate[s]
                    await asyncio.sleep(max(0.0, wake - now))
                    continue
                d = delivered[s]
                arrivals = [t0 + offset[s] + (d + j + 1) / rate[s] for j in range(k)]
                delivered[s] += k
                out = await request(streams[s][ctx + d:ctx + d + k], rids[s], True)
                done = time.perf_counter()
                check(out, k)
                values.extend(done - a for a in arrivals)
                processed[0] += k

        await asyncio.gather(*[session(s) for s in range(n)])
        await asyncio.gather(*[request(streams[s][ctx + delivered[s]:ctx + delivered[s] + 1], rids[s], False)
                               for s in range(n)])                                         # release the sessions
        return summary(n, ctx, rate, values, processed[0], args.seconds, {"seed": seed})

    def flush():
        try:
            engine.flush_cache()
        except Exception as error:                                                          # older engines: no flush
            print("flush_cache:", error, flush=True)

    # single-stream service time (back-to-back one-token appends)
    async def service(ctx, tokens=48, warm=8):
        stream = stream_of(sources, 0, 10_000 + ctx)
        counter[0] += 1
        rid = f"svc{counter[0]}"
        check(await request(stream[:ctx], rid, True), ctx, prime=True)
        times = []
        for i in range(tokens + warm):
            began = time.perf_counter()
            check(await request(stream[ctx + i:ctx + i + 1], rid, i < tokens + warm - 1), 1)
            if i >= warm:
                times.append(time.perf_counter() - began)
        return {"context": ctx, "p50_ms": percentile(times, .5) * 1e3, "p95_ms": percentile(times, .95) * 1e3}

    report["service"], report["arrival_simulation"], report["capacity_at_p95_20ms"] = [], [], {}
    for ctx in contexts:
        row = loop.run_until_complete(service(ctx))
        report["service"].append(row)
        save()
        print("service", json.dumps(row), flush=True)
        flush()
    grid = list(map(int, args.sessions.split(",")))
    for ctx in contexts:
        best = 0
        for n in grid:
            rows = []
            for seed in range(args.seeds):
                rows.append(loop.run_until_complete(simulate(n, ctx, seed)))
                flush()
            report["arrival_simulation"].extend(rows)
            save()
            for row in rows:
                print("sim", json.dumps(row), flush=True)
            if max(r["latency_p95_ms"] for r in rows) > args.budget_ms:
                break
            best = n
        report["capacity_at_p95_20ms"][str(ctx)] = best
        save()
    engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hf", "sglang"))
    parser.add_argument("--model", type=Path, default=Path("/tmp/guard06"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="512,1024,2048,4096,8000")
    parser.add_argument("--sessions", default="1,8,16,24,32,48,64,80,96,128,160,192,224,256,288,320,352")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget-ms", type=float, default=20.0)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "kind": f"qwen3guard_stream_0.6b_official_{args.mode}", "training": False,
              "settings": {k: str(v) for k, v in vars(args).items()}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        (run_hf if args.mode == "hf" else run_sglang)(args, report, save)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()


if __name__ == "__main__":
    main()
