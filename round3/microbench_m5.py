"""Small synchronized M5 screen of A0 vs untrained C1; not a service SLO."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
EXPORT = ROOT.parent / "round1/export"
sys.path.insert(0, str(EXPORT))
from runtime import Stream, load, sync  # noqa: E402


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {"n": len(values), "p50_ms": statistics.median(values) * 1000,
            "p95_ms": p95 * 1000}


def measured(fn):
    sync()
    start = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - start


@torch.no_grad()
def screen(path: Path) -> dict:
    model, tokenizer = load(dtype=torch.bfloat16, model_path=path)
    tokens = tokenizer.encode(
        "请客观介绍这项公开政策，区分事实、观点和引用的原文。",
        add_special_tokens=False,
    )
    ids = (tokens * (1024 // len(tokens) + 1))[:1024]
    result = {
        "model_path": str(path.resolve()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_token_sha256": __import__("hashlib").sha256(
            json.dumps(ids).encode()
        ).hexdigest(),
        "prefill_tokens": 992,
        "continuation_tokens": 32,
        "decode": {},
        "batch": {},
    }
    for chunk in (1, 8, 16):
        prefills, chunks = [], []
        for repeat in range(4):
            stream = Stream(model, tokenizer)
            elapsed = measured(lambda: stream.update_ids(ids[:992], "assistant", all_positions=True))
            local = []
            for end in range(992 + chunk, 1025, chunk):
                local.append(measured(lambda end=end: stream.update_ids(ids[:end], "assistant", all_positions=True)))
            assert stream.processed_tokens == 1024
            if repeat:
                prefills.append(elapsed)
                chunks.extend(local)
        result["decode"][str(chunk)] = {
            "prefill": summary(prefills), "chunk_latency": summary(chunks),
            "continuation_tokens_per_second": 32 * len(prefills) / sum(chunks),
            "max_buffer_wait_ms_at_100_token_s": (chunk - 1) * 10,
        }
    for batch in (1, 4):
        x = torch.tensor([ids[:256]] * batch, device=model.device)
        times = [measured(lambda: model(input_ids=x, use_cache=False, logits_to_keep=1))
                 for _ in range(6)]
        keep = times[1:]
        result["batch"][str(batch)] = {
            "latency": summary(keep),
            "requests_per_second": batch * len(keep) / sum(keep),
        }
    del model
    gc.collect()
    torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/m5_architecture_microbench.json")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    models = {
        "a0_original": EXPORT / "base_model",
        "c1_even14_untrained": ROOT / "checkpoints/c1_even14_init",
    }
    data = {name: screen(path) for name, path in models.items()}
    assert len({part["input_token_sha256"] for part in data.values()}) == 1
    report = {
        "benchmark_version": "m5-architecture-screen-v1",
        "scope": "single M5, synchronized microbenchmark, zero queue; C1 safety quality untested",
        "repetitions": "1 warmup + 3 decode / 5 batch measurements",
        "models": data,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {
        "params": part["parameters"],
        "chunk_1_token_per_second": part["decode"]["1"]["continuation_tokens_per_second"],
        "chunk_16_token_per_second": part["decode"]["16"]["continuation_tokens_per_second"],
    } for name, part in data.items()}))


if __name__ == "__main__":
    main()

