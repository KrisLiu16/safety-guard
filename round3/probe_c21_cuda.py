"""L20 source-valid quality and exact-token input speed for A0/C14/C21."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from classifier_runtime import PrefixClassifier, classify_full, load_stage0_classifier
from runtime import load

HERE = Path(__file__).resolve().parent
OUT = Path("/work/output")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles_ms(values):
    ordered = sorted(values)
    return {"p50": statistics.median(ordered) * 1000,
            "p95": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)] * 1000}


def auc(rows):
    safe = [r["unsafe_probability"] for r in rows if r["source_label"] == "safe"]
    unsafe = [r["unsafe_probability"] for r in rows if r["source_label"] == "unsafe"]
    if not safe or not unsafe:
        return None
    return sum((u > s) + 0.5 * (u == s) for u in unsafe for s in safe) / (len(safe) * len(unsafe))


def quality(model, tokenizer, version, rows):
    predictions, skipped = [], 0
    with torch.inference_mode():
        for row in rows:
            try:
                result = classify_full(model, tokenizer, version, row["messages"])
            except ValueError as exc:
                if "Context overflow" in str(exc):
                    skipped += 1
                    continue
                raise
            predictions.append({"source_id": row["source_id"],
                                "target_role": row["target_role"],
                                "source_label": row["source_label"],
                                "source_label_origin": row["source_label_origin"],
                                "risk_level": result["risk_level"],
                                "unsafe_probability": result["risk_probabilities"]["unsafe"],
                                "input_tokens": result["input_tokens_total"]})
    result = {"rows": len(predictions), "overlength_skipped": skipped,
              "label_scope": "Nemotron valid original labels, not project policy gold",
              "roles": {}}
    for role in ("user", "assistant"):
        subset = [p for p in predictions if p["target_role"] == role]
        safe = [p for p in subset if p["source_label"] == "safe"]
        unsafe = [p for p in subset if p["source_label"] == "unsafe"]
        fp = sum(p["risk_level"] == "unsafe" for p in safe)
        tp = sum(p["risk_level"] == "unsafe" for p in unsafe)
        result["roles"][role] = {
            "source_label_origin": subset[0]["source_label_origin"],
            "safe": len(safe), "unsafe": len(unsafe),
            "unsafe_auc": auc(subset),
            "argmax_fpr": fp / len(safe),
            "argmax_unsafe_recall": tp / len(unsafe),
            "argmax_balanced_accuracy": ((1 - fp / len(safe)) + tp / len(unsafe)) / 2,
        }
    return result, predictions


def speed(model, tokenizer, version):
    phrase = tokenizer.encode("普通公开资料与 everyday text。", add_special_tokens=False)
    ids = (phrase * (1600 // len(phrase) + 1))[:1600]
    result = {}
    with torch.inference_mode():
        for name, prefix, chunk, sessions in (
            ("one_token_one_session", 256, 1, 1),
            ("eight_tokens_eight_sessions", 256, 8, 8),
            ("sixteen_tokens_32_sessions", 1024, 16, 32),
        ):
            streams = [PrefixClassifier(model, tokenizer, version, "user")
                       for _ in range(sessions)]
            for stream in streams:
                stream.append_token_ids(ids[:prefix])
            torch.cuda.synchronize()
            latencies = []
            started = time.perf_counter()
            for step in range(8):
                start = prefix + step * chunk
                new = ids[start:start + chunk]
                for stream in streams:
                    torch.cuda.synchronize()
                    one = time.perf_counter()
                    readout = stream.append_token_ids(new)
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - one)
                    if readout["input_tokens_forwarded"] != chunk:
                        raise RuntimeError("Fast path forwarded count mismatch")
            wall = time.perf_counter() - started
            tokens = sessions * 8 * chunk
            result[name] = {"new_input_tokens": tokens, "classifications": sessions * 8,
                            "wall_seconds": wall, "itps": tokens / wall,
                            "classifications_per_second": sessions * 8 / wall,
                            "decision_latency_ms": quantiles_ms(latencies),
                            "scope": "in-process exact same-tokenizer append; no HTTP/queue"}
    return result


def one(name, model, tok, version, rows):
    if hasattr(model, "lm_head"):
        raise RuntimeError("Probe model unexpectedly has an autoregressive LM head")
    q, predictions = quality(model, tok, version, rows)
    s = speed(model, tok, version)
    output = {"name": name, "version": version,
              "parameters": sum(p.numel() for p in model.parameters()),
              "has_lm_head": False, "quality": q, "speed": s}
    (OUT / f"{name}.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    with (OUT / f"{name}_predictions.jsonl").open("w") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"model": name, "user_auc": q["roles"]["user"]["unsafe_auc"],
                      "assistant_auc": q["roles"]["assistant"]["unsafe_auc"],
                      "one_token_itps": s["one_token_one_session"]["itps"]}), flush=True)
    return output


def main():
    if not torch.cuda.is_available() or "L20" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("C21 comparison must run on NVIDIA L20")
    manifest = json.loads((HERE / "source_valid_probe_manifest.json").read_text())
    if manifest["training_use"] or sha256(HERE / "source_valid_probe.jsonl") != manifest["output_sha256"]:
        raise RuntimeError("Diagnostic split contract mismatch")
    rows = [json.loads(line) for line in (HERE / "source_valid_probe.jsonl").read_text().splitlines()]
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, path, adapter in (
        ("a0", HERE / "a0_model", None),
        ("c14_stage0", HERE / "c14_model", HERE / "stage0_adapter.safetensors"),
        ("c21_init", HERE / "c21_model", None),
    ):
        if adapter:
            model, tok, version = load_stage0_classifier(path, adapter)
        else:
            model, tok = load(dtype=torch.bfloat16, device="cuda", model_path=path)
            version = name
        all_results[name] = one(name, model, tok, version, rows)
        del model, tok
        gc.collect(); torch.cuda.empty_cache()
    summary = {"hardware": torch.cuda.get_device_name(0),
               "metric": "input tokens per second; no output-token generation",
               "not_policy_safety_aligned": True,
               "models": {name: {"parameters": x["parameters"],
                                 "quality": x["quality"], "speed": x["speed"]}
                          for name, x in all_results.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
