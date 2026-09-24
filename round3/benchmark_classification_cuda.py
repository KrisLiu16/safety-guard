"""L20 direct-classifier quality and input-token throughput, never generation TPS."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen

import torch

from classifier_runtime import PrefixClassifier, classify_full, load_stage0_classifier
from serve_classifier import ClassificationServer
from runtime import encode, load, merge_lora
from evaluate import official

HERE = Path(__file__).resolve().parent
OUTPUT = Path("/work/output")


def quantiles_ms(seconds: list[float]) -> dict:
    values = sorted(seconds)
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    pick = lambda q: values[min(len(values) - 1, math.ceil(q * len(values)) - 1)] * 1000
    return {"n": len(values), "p50_ms": pick(0.5),
            "p95_ms": pick(0.95), "p99_ms": pick(0.99)}


def sync():
    torch.cuda.synchronize()


def make_tokens(tokenizer, needed=3000):
    phrase = tokenizer.encode("这是一段公开资料和普通英文 context。", add_special_tokens=False)
    return (phrase * (needed // len(phrase) + 1))[:needed]


def parity(model, tokenizer, version) -> dict:
    messages = [{"role": "user", "content": "请介绍一份公开资料，并说明哪些句子是事实。"}]
    ids = encode(tokenizer, messages)
    full = classify_full(model, tokenizer, version, messages)
    result = {}
    for chunk in (1, 8, 16):
        stream = PrefixClassifier(model, tokenizer, version, "user")
        for end in range(chunk, len(ids) + chunk, chunk):
            got = stream.update_ids(ids[:min(end, len(ids))])
        risk_error = max(abs(full["risk_probabilities"][k] - got["risk_probabilities"][k])
                         for k in full["risk_probabilities"])
        category_error = max(abs(full["category_distribution"][k] - got["category_distribution"][k])
                             for k in full["category_distribution"])
        if max(risk_error, category_error) > 0.03:
            raise RuntimeError("Complete/stream classification parity failed")
        result[str(chunk)] = {"risk_max_abs_error": risk_error,
                              "category_max_abs_error": category_error,
                              "input_tokens": len(ids),
                              "model_forward_input_tokens": stream.total_forwarded_tokens}
    return result


def model_itps(model, tokenizer, version) -> dict:
    tokens = make_tokens(tokenizer)
    # Warm up kernels before reporting input-token timings.
    for _ in range(3):
        PrefixClassifier(model, tokenizer, version, "user").update_ids(tokens[:256])
    sync()
    prefill = {}
    for n in (256, 1024, 2048):
        times = []
        for _ in range(5):
            sync()
            start = time.perf_counter()
            result = PrefixClassifier(model, tokenizer, version, "user").append_token_ids(tokens[:n])
            sync()
            times.append(time.perf_counter() - start)
            assert result["input_tokens_forwarded"] == n
        prefill[str(n)] = {"latency": quantiles_ms(times),
                           "median_input_tokens_per_second": n / statistics.median(times)}
    continuation = {}
    for prefix in (256, 1024):
        for chunk in (1, 8, 32):
            for concurrency in (1, 8, 32):
                streams = [PrefixClassifier(model, tokenizer, version, "user")
                           for _ in range(concurrency)]
                for stream in streams:
                    stream.append_token_ids(tokens[:prefix])
                sync()
                latencies = []
                forwarded = 0
                start = time.perf_counter()
                for step in range(4):
                    end = prefix + (step + 1) * chunk
                    for stream in streams:
                        sync()
                        one = time.perf_counter()
                        result = stream.append_token_ids(tokens[end - chunk:end])
                        sync()
                        latencies.append(time.perf_counter() - one)
                        forwarded += result["input_tokens_forwarded"]
                wall = time.perf_counter() - start
                logical = concurrency * 4 * chunk
                if forwarded != logical:
                    raise RuntimeError("Stream forwarded a different number of input tokens")
                continuation[f"prefix{prefix}_chunk{chunk}_sessions{concurrency}"] = {
                    "new_input_tokens": logical, "classification_count": concurrency * 4,
                    "model_forward_input_tokens": forwarded,
                    "wall_seconds": wall, "input_tokens_per_second": logical / wall,
                    "classifications_per_second": concurrency * 4 / wall,
                    "decision_latency": quantiles_ms(latencies),
                    "scope": "in-process synchronized exact-token append; excludes tokenization, network and external queue",
                }
    return {"prefill": prefill, "continuation": continuation}


def http_json(method, url, payload):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    request = Request(url, data=data, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def http_smoke_and_itps(model, tokenizer, version) -> dict:
    server = ClassificationServer(("127.0.0.1", 0), model, tokenizer, version)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        full = http_json("POST", root + "/classify", {"messages": [
            {"role": "user", "content": "请解释公开政策的具体文字。"}]})
        sid = http_json("POST", root + "/sessions", {"target_role": "user"})["session_id"]
        first = http_json("POST", root + f"/sessions/{sid}/append",
                          {"chunk": "请解释公开政策的", "final": False})
        last = http_json("POST", root + f"/sessions/{sid}/append",
                         {"chunk": "具体文字。", "final": True})
        if any(x["schema_version"] != "guard-classification-v1"
               or not math.isclose(sum(x["risk_probabilities"].values()), 1, abs_tol=1e-4)
               for x in (full, first, last)):
            raise RuntimeError("HTTP classification response invalid")
        http_json("DELETE", root + f"/sessions/{sid}", None)
        # Complete path includes JSON, HTTP, tokenization, model and classification readout.
        phrase = "公开资料与普通英文 context。" * 18
        latencies = []
        tokens = 0
        sync()
        began = time.perf_counter()
        for _ in range(24):
            started = time.perf_counter()
            result = http_json("POST", root + "/classify",
                               {"messages": [{"role": "user", "content": phrase}]})
            latencies.append(time.perf_counter() - started)
            tokens += result["input_tokens_forwarded"]
        wall = time.perf_counter() - began
        return {"first_append_has_classification": first["risk_level"] in ("safe", "unsafe", "controversial"),
                "final_stream_risk_max_abs_error": max(abs(full["risk_probabilities"][k] -
                                                        last["risk_probabilities"][k])
                                                   for k in full["risk_probabilities"]),
                "complete_requests": 24, "input_tokens": tokens,
                "wall_seconds": wall, "input_tokens_per_second": tokens / wall,
                "classifications_per_second": 24 / wall,
                "decision_latency": quantiles_ms(latencies),
                "scope": "localhost HTTP complete classification, one serialized GPU worker"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def one_model(name, model, tokenizer, version):
    out = OUTPUT / name
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "lm_head"):
        raise RuntimeError("Benchmark loaded an autoregressive head")
    parity_result = parity(model, tokenizer, version)
    speed = model_itps(model, tokenizer, version)
    official_metrics = official(model, tokenizer, out, scope="subset")
    result = {"model": name, "model_version": version,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "has_lm_head": False, "parity": parity_result,
              "speed": speed, "official_subset_metrics": official_metrics}
    if name == "c1_stage0_20k":
        result["http"] = http_smoke_and_itps(model, tokenizer, version)
    (out / "classification_benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    if not torch.cuda.is_available() or "L20" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Benchmark must run on NVIDIA L20")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(dtype=torch.bfloat16, device="cuda", model_path=HERE / "a0_model")
    a0 = one_model("a0_original", model, tok, "a0-original")
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    model, tok, version = load_stage0_classifier(HERE / "c1_model",
                                                  HERE / "stage0_adapter.safetensors",
                                                  merge_adapter=False)
    merge_messages = [{"role": "user", "content": "请概述一份公开文件。"}]
    before_merge = classify_full(model, tok, version, merge_messages)
    merge_lora(model)
    after_merge = classify_full(model, tok, version, merge_messages)
    merge_error = max([abs(before_merge["risk_probabilities"][key]
                           - after_merge["risk_probabilities"][key])
                       for key in before_merge["risk_probabilities"]]
                      + [abs(before_merge["category_distribution"][key]
                             - after_merge["category_distribution"][key])
                         for key in before_merge["category_distribution"]])
    if merge_error > 0.03:
        raise RuntimeError(f"Inference LoRA merge changed risk distribution: {merge_error}")
    c1 = one_model("c1_stage0_20k", model, tok, version)
    result = {"hardware": torch.cuda.get_device_name(0),
              "metric": "ITPS = accepted input tokens / wall second, not generated-token TPS",
              "a0": {"parameter_count": a0["parameter_count"],
                     "official_subset_metrics": a0["official_subset_metrics"]},
              "c1": {"parameter_count": c1["parameter_count"],
                     "merge_probability_max_abs_error": merge_error,
                     "official_subset_metrics": c1["official_subset_metrics"],
                     "http": c1["http"]},
              "not_policy_safety_aligned": True}
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": True, "a0_params": a0["parameter_count"],
                      "c1_params": c1["parameter_count"],
                      "c1_http_itps": c1["http"]["input_tokens_per_second"]}), flush=True)


if __name__ == "__main__":
    main()
