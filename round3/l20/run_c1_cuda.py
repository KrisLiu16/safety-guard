"""L20-only C1 load, cache, latency, and backward compatibility probe."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runtime import Stream, encode, forward_ids, load  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def ms(values):
    ordered = sorted(values)
    return {"n": len(values), "p50": statistics.median(values) * 1000,
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    gpu_name = torch.cuda.get_device_name(0)
    if "L20" not in gpu_name.upper():
        raise RuntimeError(f"Expected L20, got {gpu_name}")
    model, tokenizer = load(dtype=torch.bfloat16, device="cuda", model_path=ROOT / "c1_model")
    manifest = json.loads((ROOT / "c1_model/build_manifest.json").read_text())
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != manifest["parameter_count"] or hasattr(model, "lm_head"):
        raise AssertionError("C1 checkpoint architecture mismatch")
    torch.cuda.reset_peak_memory_stats()
    messages = [{"role": "user", "content": "请概述这项公开政策并区分事实与观点。"}]
    ids = encode(tokenizer, messages)
    reference = forward_ids(model, ids, "user")[-1].softmax(-1).cpu().tolist()
    parity = {}
    for size in (1, 8, 16):
        stream = Stream(model, tokenizer)
        for end in range(size, len(ids) + size, size):
            got = stream.update_ids(ids[:min(end, len(ids))], "user").cpu().tolist()
        parity[str(size)] = max(abs(a - b) for a, b in zip(reference, got))
    if max(parity.values()) > 0.02:
        raise AssertionError(f"C1 CUDA cache parity error: {parity}")

    phrase = tokenizer.encode("客观介绍公开资料并区分事实与观点。", add_special_tokens=False)
    sequence = (phrase * (1024 // len(phrase) + 1))[:1024]
    speed = {}
    for chunk in (1, 16):
        prefill, continuation = [], []
        for repeat in range(4):
            stream = Stream(model, tokenizer)
            _, first = timed(lambda: stream.update_ids(sequence[:992], "assistant", all_positions=True))
            local = []
            for end in range(992 + chunk, 1025, chunk):
                _, elapsed = timed(lambda end=end: stream.update_ids(
                    sequence[:end], "assistant", all_positions=True
                ))
                local.append(elapsed)
            if stream.processed_tokens != 1024:
                raise AssertionError("Incremental cache processed the wrong token count")
            if repeat:
                prefill.append(first)
                continuation.extend(local)
        speed[str(chunk)] = {
            "prefill_ms": ms(prefill),
            "chunk_ms": ms(continuation),
            "continuation_token_per_second": 32 * len(prefill) / sum(continuation),
            "scope": "single ready stream, no queue or network time",
        }

    # This only checks that the new architecture can backpropagate on L20.
    # The artificial target is not a safety annotation; no weights are saved.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.query_risk_level_head.parameters():
        parameter.requires_grad_(True)
    model.train()
    optimizer = torch.optim.AdamW(model.query_risk_level_head.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    x = torch.tensor([ids], device="cuda")
    output = model(input_ids=x, use_cache=False, logits_to_keep=1)
    loss = F.cross_entropy(output.query_risk_level_logits[:, -1, :].float(),
                           torch.tensor([0], device="cuda"))
    loss.backward()
    grad_finite = all(torch.isfinite(p.grad).all().item()
                      for p in model.query_risk_level_head.parameters() if p.grad is not None)
    if not grad_finite:
        raise AssertionError("Nonfinite classification-head gradient")
    optimizer.step()
    result = {
        "status": "passed",
        "gpu": gpu_name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "model_version": manifest["architecture_version"],
        "model_sha256": manifest["model_sha256"],
        "parameters": parameters,
        "has_lm_head": False,
        "cache_probability_max_error": max(parity.values()),
        "cache_probability_errors": parity,
        "single_stream_speed": speed,
        "synthetic_backward_gradient_finite": grad_finite,
        "synthetic_backward_only": True,
        "saved_training_weights": False,
        "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

