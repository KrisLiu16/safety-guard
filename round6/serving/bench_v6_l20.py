"""L20: v6 — padded sessions skip all state traffic, finer session buckets; decision check on normal content too.

Changes against the T007 v2 run (v4 fp16 + in-place ring, tile (32, 2), capacity 208 sessions):
  - GDN and ring-attention kernels no longer read or write the state / ring of a padded session (lens = 0). The
    v2 profile showed the GDN kernel growing 2.2x from 128 to 192 sessions because 192 is padded to the 256 bucket;
  - finer session buckets (FINE) cut that padding; COARSE keeps v2's powers of two, to separate the two effects.
Decision drift is measured on the 400 official thinking sequences (as v2) and, because nearly all of those fire,
also on the prefix_v2 dev assistant records (the Round5 held-out set; half safe), reported per source label, so the
direction "streaming fires where the whole-sequence forward does not" is tested on content that should pass.
Fixed Round5 weights; no training; thresholds are a grid, not calibrated values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import traceback

from bench_slot_l20 import simulate
from bench_v5_l20 import decision_drift, profile_tick, stream_all

COARSE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
FINE = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)
TOKEN_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def reference_probs(torch, model, sequences):
    out = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, ids in enumerate(sequences):
            hidden = model.backbone(input_ids=torch.tensor(ids, device="cuda")[None]).last_hidden_state[0]
            out[i] = torch.softmax(model.readout(hidden, "assistant")[0].float(), -1).cpu()
    return out


def sliced(probs, starts):
    return {i: p[starts[i]:] for i, p in probs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data", type=Path, default=Path("/work/validation/official_adapted_v1/thinking.jsonl"))
    parser.add_argument("--prefix-dev", type=Path, default=Path("/work/round5/data/prefix_v2/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/serving_v6"))
    parser.add_argument("--max-seqs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sim-sessions", default="208,224,240,256,272,288,304,320")
    parser.add_argument("--sim-seconds", type=float, default=8.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "kind": "round6_slot_engine_v6", "training": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        sys.path.insert(0, str(args.bundle))
        import standalone_model
        import torch
        from slot_engine_v4 import SlotStreamEngineV4
        _, model, _, meta = standalone_model.load_bundle(args.bundle)
        report.update(checkpoint_sha256=meta["checkpoint_sha256"], device=meta["device"])

        def make(buckets):
            return lambda n: SlotStreamEngineV4(model, max_slots=n, state_dtype="float16", ring="inplace",
                                                gdn_bv=32, gdn_warps=2, session_buckets=buckets)

        configs = {"v6_fine": make(FINE), "v6_coarse": make(COARSE)}

        thinking = [json.loads(line)["ids"] for line in open(args.data)][:args.max_seqs]
        prefix_rows = [r for r in map(json.loads, open(args.prefix_dev)) if r["target_role"] == "assistant"]
        prefix = [r["ids"] for r in prefix_rows]
        prefix_start = {i: r["target_token_positions"][0] for i, r in enumerate(prefix_rows)}
        ref_thinking = reference_probs(torch, model, thinking)
        ref_prefix = sliced(reference_probs(torch, model, prefix), prefix_start)
        labels = {i: r["source_label"] for i, r in enumerate(prefix_rows)}
        report["sets"] = {"thinking": {"sequences": len(thinking), "positions": sum(map(len, thinking))},
                          "prefix_v2_dev_assistant": {"sequences": len(prefix), "by_label": {
                              lab: sum(v == lab for v in labels.values()) for lab in ("safe", "unsafe")}}}
        report["decision_drift"] = {}
        for name, factory in configs.items():
            got_t = stream_all(factory, thinking, list(range(len(thinking))), args.batch, seed=7)
            got_p = sliced(stream_all(factory, prefix, list(range(len(prefix))), args.batch, seed=11), prefix_start)
            report["decision_drift"][name] = {
                "thinking": decision_drift(got_t, ref_thinking, list(range(len(thinking)))),
                **{f"prefix_v2_dev_{lab}": decision_drift(got_p, ref_prefix, [i for i in labels if labels[i] == lab])
                   for lab in ("safe", "unsafe")}}
            save()
            torch.cuda.empty_cache()
            print(name, "decision drift done", flush=True)

        sources = [(seq * (4096 // len(seq) + 2))[:4096] for seq in thinking]
        report["profile"] = {}
        for sessions in (192, 256):
            engine = SlotStreamEngineV4(model, max_slots=sessions, use_graphs=False, state_dtype="float16", ring="inplace",
                                        gdn_bv=32, gdn_warps=2, session_buckets=FINE)
            report["profile"][f"fine_n{sessions}_c1"] = profile_tick(torch, engine, sources, sessions, 1)
            save()
            del engine
            torch.cuda.empty_cache()

        sim_sessions = list(map(int, args.sim_sessions.split(",")))
        report["arrival_simulation"], report["capacity_at_p95_20ms"] = {}, {}
        for name, factory in configs.items():
            engine = factory(max(sim_sessions))
            with torch.inference_mode():
                for nb in engine.session_buckets:
                    if nb <= engine.max_slots:
                        for lb in TOKEN_BUCKETS:
                            engine._static(nb, lb)
            sims = [simulate(engine, [sources[s % len(sources)] for s in range(n)], n, args.sim_seconds, random.Random(n))
                    for n in sim_sessions]
            report["arrival_simulation"][name] = sims
            ok = [r["sessions"] for r in sims if r["latency_p95_ms"] <= 20.0]
            report["capacity_at_p95_20ms"][name] = max(ok) if ok else 0
            report.setdefault("graphs_captured", {})[name] = engine.stats["captures"]
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
