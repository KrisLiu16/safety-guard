"""L20: torch.profiler breakdown of one bucketed-engine tick (eager, no graph) to find the
per-session cost that dominates large-N / small-c ticks. No training, no labels."""
from __future__ import annotations

import json
from pathlib import Path
import sys

BUNDLE = Path("/work/output/round5/validation/bundle")
OUT = Path("/work/output/round6/serving_profile_v1")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(BUNDLE))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import standalone_model
    import torch
    from torch.profiler import ProfilerActivity, profile
    from bucketed_engine import BucketedStreamEngine
    _, model, _, _ = standalone_model.load_bundle(BUNDLE)
    sequences = [json.loads(line)["ids"] for line in open("/work/validation/official_adapted_v1/thinking.jsonl")][:200]
    source = [(s * 20)[:4096] for s in sequences]
    report = {}
    for sessions, chunk in ((128, 1), (128, 8), (32, 1)):
        engine = BucketedStreamEngine(model, max_slots=sessions, use_graphs=False)
        engine.step([(s, source[s % len(source)][:256]) for s in range(sessions)])
        offset = 256
        for _ in range(3):                                          # warm up kernels/autotune
            engine.step([(s, source[s % len(source)][offset:offset + chunk]) for s in range(sessions)])
            offset += chunk
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                engine.step([(s, source[s % len(source)][offset:offset + chunk]) for s in range(sessions)])
                offset += chunk
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70)
        name = f"n{sessions}_c{chunk}"
        (OUT / f"{name}.txt").write_text(table)
        events = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:30]
        report[name] = [{"name": e.key[:120], "cuda_ms_per_tick": e.device_time_total / 3e3, "calls_per_tick": e.count / 3}
                        for e in events]
        print(name, flush=True)
        print(table, flush=True)
        del engine
        torch.cuda.empty_cache()
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
