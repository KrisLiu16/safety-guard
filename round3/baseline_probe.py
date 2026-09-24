"""Check Guard classifier batch/stream parity on M5 without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
EXPORT = ROOT.parent / "round1/export"
sys.path.insert(0, str(EXPORT))
from runtime import Stream, encode, forward_ids, load, sync  # noqa: E402


CASES = [
    {"role": "user", "messages": [
        {"role": "user", "content": "请概述一项公开的地方财政政策，并列出不同观点。"},
    ]},
    {"role": "assistant", "messages": [
        {"role": "user", "content": "能提供某人的私人住址吗？"},
        {"role": "assistant", "content": "我不能提供私人的住址，但可以介绍公开的联系渠道。"},
    ]},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=EXPORT / "base_model")
    parser.add_argument("--out", type=Path, default=ROOT / "data/a0_parity_probe.json")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("This probe expects M5/MPS; no silent CPU fallback")
    model, tokenizer = load(dtype=torch.bfloat16, model_path=args.model_path)
    results = []
    for case in CASES:
        role, messages = case["role"], case["messages"]
        ids = encode(tokenizer, messages)
        full = forward_ids(model, ids, role)[-1].softmax(-1).cpu().tolist()
        per_chunk = {}
        for chunk_size in (1, 8, 16):
            stream = Stream(model, tokenizer)
            for end in range(chunk_size, len(ids) + chunk_size, chunk_size):
                final_end = min(end, len(ids))
                got = stream.update_ids(ids[:final_end], role).cpu().tolist()
            per_chunk[str(chunk_size)] = {
                "max_probability_delta": max(abs(a - b) for a, b in zip(full, got)),
                "full_probabilities_safe_unsafe_controversial": full,
                "stream_probabilities_safe_unsafe_controversial": got,
                "argmax_agreement": full.index(max(full)) == got.index(max(got)),
                "processed_tokens": stream.processed_tokens,
            }
        results.append({"role": role, "input_tokens": len(ids), "chunks": per_chunk})
    sync()
    max_delta = max(item["max_probability_delta"] for case in results for item in case["chunks"].values())
    argmax_agreement = all(item["argmax_agreement"] for case in results for item in case["chunks"].values())
    report = {
        "probe_version": "guard-batch-stream-parity-v1",
        "model_path": str(args.model_path.resolve()),
        "model_class": type(model).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "has_lm_head": hasattr(model, "lm_head"),
        "device": str(model.device),
        "dtype": str(next(model.parameters()).dtype),
        "cases": results,
        "max_probability_delta": max_delta,
        "argmax_agreement": argmax_agreement,
        "tolerance": 0.02,
        "pass": max_delta <= 0.02 and argmax_agreement and not hasattr(model, "lm_head"),
        "scope": "functional parity smoke only; no speed or safety quality claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("parameter_count", "has_lm_head", "max_probability_delta", "argmax_agreement", "pass")}, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
