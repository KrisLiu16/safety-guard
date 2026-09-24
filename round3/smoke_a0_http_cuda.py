"""L20 localhost HTTP complete/stream classification and client-visible ITPS."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen

import torch

from classifier_runtime import load_baseline_classifier
from serve_classifier import ClassificationServer

HERE = Path(__file__).resolve().parent
OUT = Path("/work/output")


def request(method, root, path, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = Request(root + path, data=body, method=method,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def p95_ms(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] * 1000


def main():
    if not torch.cuda.is_available() or "L20" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("HTTP baseline must run on NVIDIA L20")
    model, tokenizer, version = load_baseline_classifier(HERE / "a0_model")
    server = ClassificationServer(("127.0.0.1", 0), model, tokenizer, version,
                                  training_status="pretrained_baseline_unaligned")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        health = request("GET", root, "/health")
        if (health["training_status"] != "pretrained_baseline_unaligned"
                or hasattr(model, "lm_head")):
            raise RuntimeError("A0 service status/architecture mismatch")
        message = "请解释一份公开文件的内容。"
        full = request("POST", root, "/classify", {"messages": [
            {"role": "user", "content": message}]})
        sid = request("POST", root, "/sessions", {"target_role": "user"})["session_id"]
        first = request("POST", root, f"/sessions/{sid}/append",
                        {"chunk": "请解释一份", "final": False})
        last = request("POST", root, f"/sessions/{sid}/append",
                       {"chunk": "公开文件的内容。", "final": True})
        request("DELETE", root, f"/sessions/{sid}")
        parity = max(abs(full["risk_probabilities"][key] - last["risk_probabilities"][key])
                     for key in full["risk_probabilities"])
        if (first["schema_version"] != "guard-classification-v1"
                or first["risk_level"] not in ("safe", "controversial", "unsafe")
                or parity > .03):
            raise RuntimeError("A0 HTTP first classification or stream parity failed")

        complete = {}
        text = "公开资料与 everyday context。" * 18
        for clients in (1, 8, 32):
            latencies = []
            tokens = 0

            def one(_):
                started = time.perf_counter()
                result = request("POST", root, "/classify", {"messages": [
                    {"role": "user", "content": text}]})
                return time.perf_counter() - started, result["input_tokens_total"]

            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=clients) as pool:
                for elapsed, count in pool.map(one, range(32)):
                    latencies.append(elapsed); tokens += count
            wall = time.perf_counter() - began
            complete[str(clients)] = {"requests": 32, "logical_input_tokens": tokens,
                                      "wall_seconds": wall, "itps": tokens / wall,
                                      "classifications_per_second": 32 / wall,
                                      "client_p50_ms": statistics.median(latencies) * 1000,
                                      "client_p95_ms": p95_ms(latencies)}

        streaming = {}
        chunk = "公开资料和英文 context。"
        for clients in (1, 8, 32):
            sessions = []
            for _ in range(clients):
                sid = request("POST", root, "/sessions", {"target_role": "user"})["session_id"]
                initial = request("POST", root, f"/sessions/{sid}/append",
                                  {"chunk": chunk, "final": False})
                sessions.append((sid, initial["input_tokens_total"]))
            latencies = []

            def continue_session(item):
                sid, initial_total = item
                local = []
                forwarded = 0
                latest = initial_total
                for step in range(8):
                    started = time.perf_counter()
                    result = request("POST", root, f"/sessions/{sid}/append",
                                     {"chunk": chunk, "final": step == 7})
                    local.append(time.perf_counter() - started)
                    forwarded += result["input_tokens_forwarded"]
                    latest = result["input_tokens_total"]
                return latest - initial_total, forwarded, local

            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=clients) as pool:
                outputs = list(pool.map(continue_session, sessions))
            wall = time.perf_counter() - began
            logical = sum(x[0] for x in outputs)
            forwarded = sum(x[1] for x in outputs)
            for _, _, local in outputs:
                latencies.extend(local)
            for sid, _ in sessions:
                request("DELETE", root, f"/sessions/{sid}")
            if logical <= 0 or forwarded < logical:
                raise RuntimeError("Streaming input-token accounting invalid")
            streaming[str(clients)] = {
                "sessions": clients, "classifications": clients * 8,
                "logical_new_input_tokens": logical,
                "model_forward_input_tokens": forwarded,
                "retokenization_forward_ratio": forwarded / logical,
                "wall_seconds": wall, "itps": logical / wall,
                "classifications_per_second": clients * 8 / wall,
                "client_p50_ms": statistics.median(latencies) * 1000,
                "client_p95_ms": p95_ms(latencies),
            }
        output = {
            "hardware": torch.cuda.get_device_name(0), "model_version": version,
            "training_status": "pretrained_baseline_unaligned",
            "has_lm_head": False, "classification_on_first_append": True,
            "full_vs_stream_risk_max_abs_error": parity,
            "metric": "client-visible accepted input tokens/s; no generated tokens",
            "scope": "localhost HTTP with tokenizer, JSON and serialized GPU; no external network",
            "complete": complete, "streaming": streaming,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "a0_http_itps.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"completed": True, "complete_itps_8_clients": complete["8"]["itps"],
                          "stream_itps_8_clients": streaming["8"]["itps"]}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
