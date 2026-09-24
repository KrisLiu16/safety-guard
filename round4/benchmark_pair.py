"""Matched L20-only token-ID classification timing for a fixed candidate and A0.

No text generation, tokenizer timing, HTTP, dynamic batching or quality claim.
Each context/chunk case warms the exact full trace, then repeats it from a new
cache for 128 timed calls. Protocol v2 keeps every trace within 8192 tokens.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

CONTEXTS = (512, 4096)
CHUNKS = (1, 8, 32)
TIMED_CALLS = 128
MAX_PROTOCOL_TOKENS = 8192
PHRASE = "公开资料用于一般阅读，逐句整理章节目录。 Public information supports ordinary reading and organizing chapter headings.\n"
TRACE_TEXT = "USER:\n" + PHRASE * 2048


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(piece)
    return value.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def checked_module(name, expected):
    """Load an exact local source, refusing previously imported shadow copies."""
    expected = Path(expected).resolve()
    if not expected.is_file():
        raise FileNotFoundError(expected)
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, expected)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load benchmark dependency: " + name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"Unexpected source for {name}: {actual}; expected {expected}")
    return module


def source_record(name, expected):
    module = sys.modules.get(name)
    if module is None or Path(module.__file__).resolve() != Path(expected).resolve():
        raise RuntimeError("Runtime dependency source mismatch: " + name)
    return {"path": str(Path(module.__file__).resolve()), "sha256": sha_file(module.__file__)}


def canonical_gpu_uuid(value):
    if value is None:
        return None
    try:
        if isinstance(value, bytes) and len(value) == 16:
            parsed = uuid.UUID(bytes=value)
        else:
            text = value.decode() if isinstance(value, bytes) else str(value)
            parsed = uuid.UUID(text.strip().removeprefix("GPU-").removeprefix("gpu-"))
        return "GPU-" + str(parsed) if parsed.int else None
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None


def parse_nvidia_inventory(text):
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 2:
            raise ValueError("Unexpected nvidia-smi UUID/name column count")
        physical_uuid = canonical_gpu_uuid(row[0].strip())
        if physical_uuid is None:
            raise ValueError("nvidia-smi returned no valid physical GPU UUID")
        rows.append({"uuid": physical_uuid, "name": row[1].strip()})
    return rows


def query_nvidia_inventory():
    command = ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"]
    evidence = {"command": command, "status": "unavailable"}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        evidence.update(returncode=result.returncode, stdout=result.stdout.strip()[:4096], stderr=result.stderr.strip()[:4096])
        if result.returncode == 0:
            evidence["rows"] = parse_nvidia_inventory(result.stdout)
            evidence["status"] = "ok"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        evidence.update(error_type=type(error).__name__, error=str(error))
    return evidence


def resolve_gpu_identity(cuda_uuid, cuda_name, evidence):
    """Never infer a physical-device mapping from a multi-GPU inventory."""
    if "L20" not in cuda_name:
        raise RuntimeError("CUDA device identity is not L20")
    native_uuid = canonical_gpu_uuid(cuda_uuid)
    result = {"torch_properties_uuid": native_uuid, "nvidia_smi_evidence": evidence}
    if evidence.get("status") == "ok":
        rows = evidence["rows"]
        if len(rows) != 1:
            result.update(uuid="unavailable", identity_source="ambiguous_physical_inventory",
                          identity_verified=False)
            return result
        row = rows[0]
        if "L20" not in row["name"]:
            raise RuntimeError("The sole nvidia-smi physical device is not L20")
        if native_uuid is not None and native_uuid != row["uuid"]:
            raise RuntimeError("CUDA properties UUID disagrees with the sole nvidia-smi physical device")
        result.update(uuid=row["uuid"], identity_source="torch_and_single_device_nvidia_smi" if native_uuid else "single_device_nvidia_smi",
                      identity_verified=True)
    else:
        result.update(uuid=native_uuid or "unavailable",
                      identity_source="torch_cuda_properties" if native_uuid else "unavailable",
                      identity_verified=native_uuid is not None)
    return result


def nearest_rank(values, quantile):
    if not values or not 0 < quantile <= 1:
        raise ValueError("Nearest-rank quantile requires samples and 0 < q <= 1")
    return sorted(values)[math.ceil(len(values) * quantile) - 1]


def trace_ranges(context, chunk):
    if context not in CONTEXTS or chunk not in CHUNKS:
        raise ValueError("Unexpected benchmark protocol")
    ranges = [(context + index * chunk, context + (index + 1) * chunk)
              for index in range(TIMED_CALLS)]
    if ranges[-1][1] > MAX_PROTOCOL_TOKENS:
        raise ValueError("Paired timing trace would exceed the shared 8192-token limit")
    return ranges


def cache_storage_inventory(cache, tensor_predicate=None):
    """Deduplicate physical storage across the complete persistent cache graph.

    Traverse root attributes too: associative memories live outside layers.
    Storage size, not logical tensor size, includes retained backing buffers.
    A predicate injection permits CPU fake-storage tests without importing torch.
    """
    if tensor_predicate is None:
        import torch
        tensor_predicate = lambda value: isinstance(value, torch.Tensor)
    visited, stores = set(), {}

    def visit(value, path):
        # Process tensor aliases before the object guard to retain each owner
        # path, while counting its underlying storage only once globally.
        if tensor_predicate(value):
            storage = value.untyped_storage()
            size = storage.nbytes()
            if size:
                key = (str(value.device), storage.data_ptr())
                record = stores.setdefault(key, {"device": str(value.device), "bytes": size, "owners": []})
                if record["bytes"] != size:
                    raise RuntimeError("Inconsistent physical storage size")
                record["owners"].append({"path": path, "shape": list(value.shape), "dtype": str(value.dtype)})
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, f"{path}[{key!r}]")
        elif isinstance(value, (tuple, list)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
        elif hasattr(value, "__dict__"):
            for key, nested in vars(value).items():
                visit(nested, f"{path}.{key}")
    visit(cache, "cache")
    records = sorted(stores.values(), key=lambda row: row["owners"][0]["path"])
    by_device = {}
    for record in records:
        by_device[record["device"]] = by_device.get(record["device"], 0) + record["bytes"]
    return {"unique_storage_bytes": sum(row["bytes"] for row in records),
            "unique_storages": len(records), "bytes_by_device": by_device,
            "storage_inventory": records,
            "scope": "persistent tensor storage reachable from this cache, globally deduplicated; excludes model weights, CUDA allocator reserve and temporary forward tensors"}


def library_versions():
    values = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "flash-linear-attention", "fla-core"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def tokenize_trace(tokenizer):
    for backend in (tokenizer, getattr(tokenizer, "backend_tokenizer", None)):
        if backend is not None and getattr(backend, "truncation", None) not in (None, False):
            raise ValueError("Benchmark tokenizer truncation must be disabled")
    ids = tokenizer.encode(TRACE_TEXT, add_special_tokens=False, truncation=False)
    required = max(CONTEXTS) + TIMED_CALLS * max(CHUNKS)
    if len(ids) < required:
        raise ValueError(f"Fixed common text yielded only {len(ids)} native tokens; need {required}")
    return ids


def check_probabilities(values):
    if (len(values) != 3 or not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            or abs(sum(values) - 1) > 1e-4):
        raise RuntimeError("Expected three direct user-risk probabilities, not vocabulary logits")


def atomic_write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_runtime(kind, manifest):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Benchmark execution requires exactly one visible CUDA L20; CPU/MPS is disabled")
    torch.set_num_threads(4)
    script_dir = Path(__file__).resolve().parent
    sys.path[:0] = [str(script_dir), "/work/input", "/work/window"]
    if kind == "candidate":
        inference = checked_module("infer_classifier", script_dir / "infer_classifier.py")
        checkpoint = Path(manifest["checkpoint"])
        if sha_file(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError("Selected candidate checkpoint hash changed")
        variant = manifest["variant"]
        if variant not in ("full", "window", "memory"):
            raise ValueError("Manifest must name the selected model's actual architecture")
        if variant == "memory":
            checked_module("window_attention", Path("/work/window/window_attention.py"))
            checked_module("memory_attention", script_dir / "memory_attention.py")
        args = SimpleNamespace(base_root=Path("/work"), base_code_dir=Path("/work/input"),
                               window_code_dir=Path("/work/window"), memory_code_dir=script_dir,
                               checkpoint=checkpoint)
        _, model, tokenizer, _, kernels, _ = inference.load_classifier(args, variant)
        sources = {"infer_classifier": source_record("infer_classifier", script_dir / "infer_classifier.py")}
        for name, directory in (("train_base", "/work/input"), ("experiment_common", "/work/input"),
                                ("speed_probe", "/work/input"), ("run_probe", "/work/window"),
                                ("window_attention", "/work/window")):
            sources[name] = source_record(name, Path(directory) / (name + ".py"))
        if variant == "memory":
            sources["memory_attention"] = source_record("memory_attention", script_dir / "memory_attention.py")

        def initial_cache():
            return None

        def forward(ids, cache):
            # Preserve the current student timing path: user risk and category
            # readouts run in Classifier.step; only user risk is copied to CPU.
            risk, _, state = model.step(ids, cache, True)
            if tuple(risk.shape) != (1, 3):
                raise RuntimeError("Candidate step did not return a three-class user-risk head")
            values = risk[0].float().softmax(-1).cpu().tolist()
            check_probabilities(values)
            return values, state

        model_metadata = {"kind": kind, "variant": variant, "checkpoint": str(checkpoint),
                          "checkpoint_sha256": manifest["checkpoint_sha256"], "kernels": kernels,
                          "window_tokens": 512 if variant != "full" else None,
                          "runtime_modules": sources,
                          "heads_executed": ["user risk", "user category"],
                          "heads_consumed": ["user risk"],
                          "readout": "Classifier.step user risk + category; user risk probabilities returned to CPU"}
        model_limit = getattr(model.backbone.config, "max_position_embeddings", None)
        tokenizer_path = Path("/work/models/qwen35/tokenizer.json")
    else:
        guard_runtime = checked_module("runtime", Path("/work/input/runtime.py"))
        from transformers.cache_utils import DynamicCache
        model_root = Path("/work/models/guard")
        if (model_root / "risk_heads_fp32.safetensors").exists():
            raise RuntimeError("A0 reference directory contains a local risk-head override; official weights are required")
        model, tokenizer = guard_runtime.load(dtype=torch.bfloat16, device="cuda", attention="sdpa", model_path=model_root)
        model.eval().requires_grad_(False)

        def initial_cache():
            # Transformers 4.55's config-based constructor is incompatible
            # with this checkpoint; explicit no-config cache is intentional.
            return DynamicCache()

        def forward(ids, cache):
            output = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            risk = guard_runtime.logits(output, "user")[0, -1].float()
            if tuple(risk.shape) != (3,):
                raise RuntimeError("A0 output did not contain its dedicated three-class user-risk head")
            values = risk.softmax(-1).cpu().tolist()
            check_probabilities(values)
            return values, output.past_key_values

        weight_files = sorted(model_root.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError("No A0 safetensors weights found")
        model_metadata = {"kind": kind, "reference": "original Qwen3Guard-Stream-0.6B",
                          "model_root": str(model_root), "weights_sha256": {path.name: sha_file(path) for path in weight_files},
                          "config_sha256": sha_file(model_root / "config.json"),
                          "modeling_code_sha256": {path.name: sha_file(path) for path in sorted(model_root.glob("modeling*.py"))},
                          "runtime_modules": {"runtime": source_record("runtime", Path("/work/input/runtime.py"))},
                          "kernels": {"attention": "official model torch SDPA", "cache": "legacy DynamicCache()"},
                          "heads_executed": ["user risk", "user category", "assistant risk", "assistant category"],
                          "heads_consumed": ["user risk"],
                          "readout": "runtime.logits(output, 'user')[0,-1]; dedicated query risk head, never LM logits"}
        model_limit = getattr(model.config, "max_position_embeddings", None)
        tokenizer_path = model_root / "tokenizer.json"
    model.eval().requires_grad_(False)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("Model parameters must all reside on the L20")
    end_context = max(CONTEXTS) + TIMED_CALLS * max(CHUNKS)
    if isinstance(model_limit, int) and end_context > model_limit:
        raise ValueError("Timing trace exceeds the configured model position limit")
    model_metadata.update(tokenizer_sha256=sha_file(tokenizer_path), model_position_limit=model_limit)
    model_metadata["benchmark_source"] = {"path": str(Path(__file__).resolve()), "sha256": sha_file(__file__)}
    return torch, model, tokenizer, initial_cache, forward, model_metadata


def measure_case(torch, ids, context, chunk, initial_cache, forward):
    ranges = trace_ranges(context, chunk)
    # Warm exactly the same prefill and all 128 growing-history calls. This
    # state is discarded; the measured session starts from an independent one.
    state = initial_cache()
    _, state = forward(ids[:, :context], state)
    for start, end in ranges:
        _, state = forward(ids[:, start:end], state)
    torch.cuda.synchronize()
    del state
    gc.collect()
    # Discard semantic state while retaining the allocator blocks just warmed
    # by this exact trace. A different cache object starts the measured run.
    state = initial_cache()
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    prefill_probabilities, state = forward(ids[:, :context], state)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_storage = cache_storage_inventory(state)
    if prefill_storage["unique_storage_bytes"] == 0:
        raise RuntimeError("Cache accounting found no persistent tensor storage")
    latencies = []
    probabilities = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start, end in ranges:
        torch.cuda.synchronize()
        block_started = time.perf_counter()
        result, state = forward(ids[:, start:end], state)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - block_started)
        probabilities.append(result)
    elapsed = time.perf_counter() - started
    end_storage = cache_storage_inventory(state)
    record = {"context_tokens": context, "start_context_tokens": context, "end_context_tokens": ranges[-1][1],
              "chunk_tokens": chunk, "timed_calls": TIMED_CALLS, "measured_calls": TIMED_CALLS,
              "new_input_tokens": TIMED_CALLS * chunk,
              "warmup_calls": TIMED_CALLS, "warmup_prefill_tokens": context,
              "elapsed_seconds": elapsed, "itps": TIMED_CALLS * chunk / elapsed,
              "native_itps": TIMED_CALLS * chunk / elapsed,
              "classifications_per_second": TIMED_CALLS / elapsed,
              "p50_ms": nearest_rank(latencies, .5) * 1000,
              "p95_ms": nearest_rank(latencies, .95) * 1000,
              "latency_seconds": latencies, "quantile_method": "nearest rank: sorted[ceil(n*q)-1]",
              "prefill_seconds_excluded_from_stream_itps": prefill_seconds,
              "prefill_user_risk_probabilities": prefill_probabilities,
              "timed_user_risk_probabilities": probabilities,
              "prefill_cache": prefill_storage, "end_cache": end_storage,
              "state_bytes_at_prefill": prefill_storage["unique_storage_bytes"],
              "state_bytes_at_end": end_storage["unique_storage_bytes"],
              "generated_tokens": 0}
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("candidate", "a0"))
    parser.add_argument("--candidate-manifest", type=Path, default=Path("/work/validation/MODEL_MANIFEST.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing timing artifact")
    manifest = json.loads(args.candidate_manifest.read_text())
    if manifest.get("status") != "research_candidate":
        raise RuntimeError("Candidate selection must be complete and fixed before paired timing")
    torch, model, tokenizer, initial_cache, forward, model_metadata = load_runtime(args.kind, manifest)
    native_ids = tokenize_trace(tokenizer)
    required = max(CONTEXTS) + TIMED_CALLS * max(CHUNKS)
    ids = torch.tensor([native_ids[:required]], dtype=torch.long, device="cuda")
    properties = torch.cuda.get_device_properties(0)
    identity = resolve_gpu_identity(getattr(properties, "uuid", None), torch.cuda.get_device_name(0), query_nvidia_inventory())
    result = {"status": "running", "protocol_version": "matched-classifier-token-id-v2", "kind": args.kind,
              "generated_tokens": 0,
              "candidate_manifest_sha256": sha_file(args.candidate_manifest), "model": model_metadata,
              "device": {"name": torch.cuda.get_device_name(0), "visible_cuda_devices": torch.cuda.device_count(),
                         "total_memory_bytes": properties.total_memory, "multiprocessors": properties.multi_processor_count,
                         "compute_capability": [properties.major, properties.minor],
                         **identity,
                         "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "libraries": library_versions(), "torch_cpu_threads": torch.get_num_threads(),
              "protocol": {"batch_size": 1, "sessions": 1, "contexts": CONTEXTS, "chunks": CHUNKS,
                           "timed_calls_per_case": TIMED_CALLS, "exact_trace_warmup_calls_per_case": TIMED_CALLS,
                           "maximum_end_context_tokens": MAX_PROTOCOL_TOKENS,
                           "role": "user", "pretokenized_input_on_gpu": True,
                           "timing_scope": "forward, classifier readout, user-risk probabilities copied to CPU, CUDA synchronization and loop overhead; excludes prefill, model loading, tokenizer, cache inventory, JSON, HTTP and text-session rollback",
                           "independent_warmup_state": True, "generated_tokens": 0},
              "trace": {"common_text_sha256": hashlib.sha256(TRACE_TEXT.encode()).hexdigest(),
                        "common_text_characters": len(TRACE_TEXT), "phrase": PHRASE, "repetitions": 2048,
                        "full_native_token_count": len(native_ids), "full_native_ids_sha256": ids_sha(native_ids),
                        "measured_native_ids_sha256": ids_sha(native_ids[:required]),
                        "text_protocol": "Both tokenizers encode the exact same raw USER-prefixed repeated bilingual text without chat templates or special tokens; prefixes use each tokenizer's own boundaries."},
              "limitations": ["Tokenizers can produce different token boundaries and effective text lengths for the same native-token count.",
                              "Official A0 executes both user and assistant risk/category heads, while Classifier.step executes only user risk/category heads. Both return user risk probabilities. This measures the current runtimes, not identical head workload or a pure backbone/attention comparison.",
                              "A paired speed ratio requires matching concrete non-unavailable device UUIDs. Ambiguous physical inventories are never mapped by guesswork.",
                              "This is a token-ID kernel/runtime comparison, not text API or batched serving throughput.",
                              "Protocol v2 uses 512/4096-token prefills and ends at or below 8192, matching A0's configured position limit. This token-ID timing is not long-context safety or text-session quality validation.",
                              "Protocol v1 used an unsupported 12288-token end context for A0 and failed its limit check; preserve those artifacts and rerun both models under v2. Never pair a v1 candidate result with a v2 A0 result.",
                              "Compare this protocol's candidate and A0 artifacts; do not derive speedup against older 24-call measurements.",
                              "Single-run timing is subject to host/GPU conditions; P95 is an empirical nearest-rank statistic over 128 calls."],
              "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, result)
    with torch.inference_mode():
        for context in CONTEXTS:
            for chunk in CHUNKS:
                record = measure_case(torch, ids, context, chunk, initial_cache, forward)
                end = context + TIMED_CALLS * chunk
                record["input_trace_sha256"] = ids_sha(native_ids[:end])
                record["prefill_native_ids_sha256"] = ids_sha(native_ids[:context])
                record["timed_native_ids_sha256"] = ids_sha(native_ids[context:end])
                result["records"].append(record)
                result["status"] = "completed" if len(result["records"]) == len(CONTEXTS) * len(CHUNKS) else "running"
                atomic_write(args.output, result)
                print(json.dumps({"kind": args.kind, "context": context, "chunk": chunk,
                                  "native_itps": record["native_itps"], "p95_ms": record["p95_ms"]}), flush=True)


if __name__ == "__main__":
    main()
