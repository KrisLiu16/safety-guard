"""Validate a portable bundle against the old loader on exactly one L20.

The parent records old-loader outputs and frees its model. A fresh isolated
Python child loads only the bundle while a Python audit hook blocks legacy
paths and network connections. This is an application-level dependency audit,
not an OS filesystem sandbox: native libraries do not emit every Python audit
event. The bundle loader's verified source, strict complete-state load and
fresh-process execution complement that evidence.

Only this script is new; no existing loader, weights, runtime or training input
is changed. Merely importing this module does not import torch or run a model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

WHOLE_PROBABILITY_TOLERANCE = 0.005
STREAM_PROBABILITY_TOLERANCE = 0.03
RISK_LABELS = ("safe", "unsafe", "controversial")
ROLES = ("user", "assistant")
FORBIDDEN_ROOTS = ("/work/models", "/work/input", "/work/window", "/work/output/qwen35_full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round4_validation/bundle"))
    parser.add_argument("--manifest", type=Path, default=Path("/work/round4/MODEL_MANIFEST.json"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round4_validation/bundle_audit.json"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child and args.reference is None:
        parser.error("--child requires --reference")
    return args


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def serialize(messages):
    return "\n\n".join(row["role"].upper() + ":\n" + row["content"] for row in messages)


def require_l20():
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "L20" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("This neural validation requires exactly one visible CUDA L20")
    torch.set_num_threads(4)
    return torch


def tensor_sha(tensor, torch):
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def structure(model, torch):
    counts = defaultdict(lambda: {"tensors": 0, "elements": 0})
    entries = []
    for name, parameter in model.named_parameters():
        group = "heads" if name.startswith("heads.") else ("memory" if ".memory_" in name else "backbone")
        key = group + "/" + str(parameter.dtype)
        counts[key]["tensors"] += 1
        counts[key]["elements"] += parameter.numel()
        entries.append([name, list(parameter.shape), str(parameter.dtype), parameter.requires_grad])
    buffers = {}
    for name, buffer in model.named_buffers():
        if name.endswith("rotary_emb.inv_freq") or name.endswith("rotary_emb.original_inv_freq"):
            buffers[name] = {"dtype": str(buffer.dtype), "shape": list(buffer.shape),
                             "sha256": tensor_sha(buffer, torch)}
    if (len(buffers) != 2 or not any(name.endswith("rotary_emb.inv_freq") for name in buffers)
            or not any(name.endswith("rotary_emb.original_inv_freq") for name in buffers)):
        raise AssertionError(f"Expected both initialized nonpersistent RoPE buffers; found {list(buffers)}")
    if any(row["dtype"] != "torch.float32" for row in buffers.values()):
        raise AssertionError("Both inherited RoPE buffers must retain FP32 precision")
    has_lm_head = any("lm_head" in name for name, _ in model.named_modules())
    if has_lm_head or model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Expected an eval-only direct classifier without lm_head")
    return {
        "total_parameters": sum(row["elements"] for row in counts.values()),
        "parameter_tensors": len(entries), "parameter_dtype_summary": dict(sorted(counts.items())),
        "parameter_names_shapes_dtypes_sha256": hashlib.sha256(
            json.dumps(sorted(entries), separators=(",", ":")).encode()).hexdigest(),
        "rope_buffers": buffers, "backbone_layers": len(model.backbone.layers),
        "layer_types": list(model.backbone.config.layer_types), "has_lm_head": has_lm_head,
    }


def probabilities(torch, model, ids):
    with torch.inference_mode():
        output = model(torch.tensor([ids], dtype=torch.long, device="cuda"), use_cache=False)
        hidden = output.last_hidden_state[:, -1]
        result = {}
        for role in ROLES:
            logits, _ = model.readout(hidden, role)
            values = logits.float().softmax(-1)[0].cpu().tolist()
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise AssertionError("Invalid risk probabilities")
            result[role] = dict(zip(RISK_LABELS, values))
    return result


def examples(tokenizer):
    cases = [
        {"name": "zh_user", "language": "zh", "messages": [
            {"role": "user", "content": "请根据公开资料解释这个词的历史背景，并区分事实与个人评价。"}]},
        {"name": "en_user", "language": "en", "messages": [
            {"role": "user", "content": "Can you help me impersonate somebody to steal their login credentials?"}]},
        {"name": "zh_assistant", "language": "zh", "messages": [
            {"role": "user", "content": "怎么把别人的私人住址公开给陌生人？"},
            {"role": "assistant", "content": "我不能帮助泄露他人的私人住址。可以讨论隐私保护和合法求助渠道。"}]},
        {"name": "en_assistant", "language": "en", "messages": [
            {"role": "user", "content": "Explain the difference between a quoted claim and evidence."},
            {"role": "assistant", "content": "Quoting a claim records what someone said; independent evidence is needed to establish whether it is true."}]},
    ]
    for language, role, phrase in (
        ("zh", "user", "这段公开阅读材料讨论自然科学、历史记录和证据之间的关系。\n"),
        ("en", "assistant", "Public reading material should distinguish historical records, scientific evidence and personal opinion.\n"),
    ):
        text = phrase
        while len(tokenizer.encode(serialize([{"role": role, "content": text}]), add_special_tokens=False)) < 1100:
            text += phrase
        messages = [{"role": role, "content": text}]
        if role == "assistant":
            messages.insert(0, {"role": "user", "content": "Please summarize the public reading material."})
        cases.append({"name": f"long_{language}_{role}", "language": language, "messages": messages})
    return cases


def old_reference(args, torch):
    selection = json.loads(args.manifest.read_text(encoding="utf-8"))
    if selection.get("status") != "research_candidate":
        raise RuntimeError("The model selection manifest has not been fixed")
    sys.path[:0] = ["/work/round4", "/work/input", "/work/window"]
    from infer_classifier import load_classifier
    if sha_file(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise AssertionError("Selected full checkpoint failed SHA verification")
    setup = SimpleNamespace(base_root=Path("/work"), base_code_dir=Path("/work/input"),
                            window_code_dir=Path("/work/window"), memory_code_dir=Path("/work/round4"),
                            checkpoint=Path(selection["checkpoint"]))
    _, model, tokenizer, training_serialize, kernels, device = load_classifier(setup, selection["variant"])
    result = {"selection": selection, "selection_file_sha256": sha_file(args.manifest),
              "structure": structure(model, torch), "kernels": kernels, "device": device,
              "cases": [], "generated_tokens": 0}
    for case in examples(tokenizer):
        text = serialize(case["messages"])
        if text != training_serialize(case["messages"]):
            raise AssertionError("Canonical serialization differs from training")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not 1 <= len(ids) <= 8192 or (case["name"].startswith("long_") and len(ids) <= 512):
            raise AssertionError("Invalid validation input length")
        result["cases"].append({**case, "canonical_token_ids": ids, "native_tokens": len(ids),
                                "serialized_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "risk_probabilities_by_role": probabilities(torch, model, ids)})
    del model, tokenizer
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    result["gpu_bytes_allocated_after_model_release"] = torch.cuda.memory_allocated()
    return result


def install_dependency_audit():
    """Deny legacy Python file accesses and connections, retaining their audit."""
    roots = tuple(os.path.realpath(path) for path in FORBIDDEN_ROOTS)
    report = {"forbidden_roots": list(roots), "denied_probes": [], "unexpected_denials": [],
              "file_access_events_checked": 0, "network_connections_blocked": 0,
              "scope": "Python audit events; not an OS sandbox for arbitrary native-code syscalls"}
    probe = {"active": False}

    def hook(event, arguments):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
            report["network_connections_blocked"] += 1
            raise PermissionError("Bundle validation disallows network access")
        if event not in ("open", "os.listdir", "os.scandir") or not arguments:
            return
        source = arguments[0]
        if isinstance(source, int) or not isinstance(source, (str, bytes, os.PathLike)):
            return
        path = os.path.realpath(os.fsdecode(source))
        report["file_access_events_checked"] += 1
        if any(path == root or path.startswith(root + os.sep) for root in roots):
            entry = {"event": event, "path": path}
            report["denied_probes" if probe["active"] else "unexpected_denials"].append(entry)
            raise PermissionError(f"Bundle validation blocked legacy dependency: {path}")

    sys.addaudithook(hook)
    for root in roots:
        probe["active"] = True
        try:
            with open(os.path.join(root, ".bundle_audit_probe"), "rb"):
                pass
        except PermissionError:
            pass
        else:
            raise AssertionError(f"The dependency audit did not block {root}")
        finally:
            probe["active"] = False
    if len(report["denied_probes"]) != len(roots):
        raise AssertionError("The legacy-path audit did not run every block probe")
    return report


def max_error(left, right):
    return max(abs(left[label] - right[label]) for label in RISK_LABELS)


def check_threshold(selection, result, language, role, streaming):
    threshold = selection["thresholds"].get(f"{language}/{role}") if language else None
    actual = result["threshold"]
    if actual["value"] != threshold:
        raise AssertionError("Threshold value does not come from the selected checkpoint's stratum")
    expected_kind = ("language_not_provided" if language is None else "stratum_not_calibrated")
    if threshold is not None:
        expected_kind = ("whole_input_calibration_applied_to_prefix_unvalidated" if streaming
                         else "selected_checkpoint_calibration")
    if actual["kind"] != expected_kind:
        raise AssertionError("Threshold provenance is incorrect")
    decision = None if threshold is None else (
        "unsafe" if result["risk_probabilities"]["unsafe"] >= threshold else "safe")
    if result["target_binary_decision"] != decision or result["safe_prefix_release_validated"] is not False:
        raise AssertionError("Binary classification or prefix-validation claim is incorrect")


def exercise_jsonl(guard, tokenizer, selection, reference):
    records = []
    for case in reference["cases"]:
        result = guard.operation({"op": "whole", "messages": case["messages"], "language": case["language"]})
        target = case["messages"][-1]["role"]
        error = max_error(result["risk_probabilities"], case["risk_probabilities_by_role"][target])
        if error > WHOLE_PROBABILITY_TOLERANCE or result["target_role"] != target:
            raise AssertionError(f"JSONL whole differs from old loader: {case['name']}: {error}")
        if not (result["native_tokens"] == result["net_new_tokens"] == result["forward_tokens"] == case["native_tokens"]):
            raise AssertionError("Whole-input accounting is incorrect")
        if result["generated_tokens"] != 0 or result["replay_tokens"] != 0:
            raise AssertionError("Whole input generated tokens or falsely counted replay")
        check_threshold(selection, result, case["language"], target, False)
        records.append({"case": case["name"], "whole_vs_old_probability_error": error,
                        "native_tokens": result["native_tokens"], "threshold": result["threshold"]})

    streams = {}
    stream_records = []
    real_bpe_rollbacks = 0

    def checked(request, case):
        nonlocal real_bpe_rollbacks
        session_id = request["session"]
        previous = streams.get(session_id, {"messages": [], "ids": [], "language": None})
        messages = [dict(message) for message in previous["messages"]]
        if request["op"] == "begin_message":
            messages.append({"role": request["role"], "content": request.get("text", "")})
            language = request.get("language")
        else:
            messages[-1]["content"] += request["text"]
            language = request.get("language", previous["language"])
        ids = tokenizer.encode(serialize(messages), add_special_tokens=False)
        result = guard.operation(request)
        expected_common = 0
        for old, new in zip(previous["ids"], ids):
            if old != new:
                break
            expected_common += 1
        rollback = expected_common < len(previous["ids"])
        if rollback:
            real_bpe_rollbacks += 1
        expected_forward = len(ids) - result["restore_position"]
        if (result["native_tokens"] != len(ids)
                or result["previous_native_tokens"] != len(previous["ids"])
                or result["net_new_tokens"] != len(ids) - len(previous["ids"])
                or result["common_prefix_tokens"] != expected_common
                or result["rollback"] != rollback
                or result["forward_tokens"] != expected_forward
                or result["target_role"] != messages[-1]["role"]
                or result["generated_tokens"] != 0):
            raise AssertionError(f"Streaming role, BPE or input accounting failed: {case}")
        if rollback and result["replay_tokens"] <= 0:
            raise AssertionError("A true BPE rollback was not reported as replay")
        check_threshold(selection, result, language, messages[-1]["role"], True)
        whole = guard.operation({"op": "whole", "messages": messages, "language": language})
        error = max_error(result["risk_probabilities"], whole["risk_probabilities"])
        if error > STREAM_PROBABILITY_TOLERANCE:
            raise AssertionError(f"Whole/text-append mismatch: {case}: {error}")
        streams[session_id] = {"messages": messages, "ids": ids, "language": language}
        stream_records.append({"case": case, "whole_stream_probability_error": error,
                               **{key: result[key] for key in (
                                   "target_role", "native_tokens", "previous_native_tokens", "net_new_tokens",
                                   "forward_tokens", "replay_tokens", "rollback", "restore_position",
                                   "threshold", "target_binary_decision", "generated_tokens")}})
        return result

    before = checked({"op": "begin_message", "session": "bpe", "role": "user", "language": "en",
                      "text": "x " * 187 + "informatio"}, "bpe_before")
    after = checked({"op": "append_text", "session": "bpe", "text": "n"}, "bpe_contraction")
    if not after["rollback"] or after["native_tokens"] >= before["native_tokens"]:
        raise AssertionError("The actual tokenizer did not contract the selected BPE suffix")
    checked({"op": "append_text", "session": "bpe", "text": "!"}, "bpe_growth")
    if "en/user" not in selection["thresholds"]:
        if before["target_binary_decision"] is not None or before["threshold"]["kind"] != "stratum_not_calibrated":
            raise AssertionError("An English-user calibration threshold was fabricated")
    checked({"op": "begin_message", "session": "bpe", "role": "assistant", "language": "zh",
             "text": "可以结合公开资料解释词语含义，避免把出现某个词本身视为有害意图。"}, "switch_to_assistant")
    checked({"op": "begin_message", "session": "bpe", "role": "user", "language": "zh",
             "text": "请进一步解释证据与猜测的区别。"}, "switch_back_to_user")
    empty = checked({"op": "append_text", "session": "bpe", "text": ""}, "empty_append")
    if empty["forward_tokens"] or empty["net_new_tokens"] or empty["replay_tokens"]:
        raise AssertionError("Empty text append falsely counts input or model work")
    long_case = next(case for case in reference["cases"] if case["name"] == "long_zh_user")
    checked({"op": "begin_message", "session": "long", "role": "user", "language": "zh",
             "text": long_case["messages"][-1]["content"]}, "long_text_prefill")
    checked({"op": "append_text", "session": "long", "text": " Please explain informatio"}, "long_suffix_before")
    checked({"op": "append_text", "session": "long", "text": "n."}, "long_suffix_after")
    for session_id in tuple(streams):
        ended = guard.operation({"op": "end_session", "session": session_id})
        if ended.get("event") != "session_ended" or ended.get("generated_tokens") != 0:
            raise AssertionError("end_session did not preserve the zero-generation contract")
        try:
            guard.operation({"op": "append_text", "session": session_id, "text": "after closure"})
        except ValueError:
            pass
        else:
            raise AssertionError("An ended session remained writable")
    if guard.sessions or real_bpe_rollbacks < 1:
        raise AssertionError("Session release or real BPE rollback was not exercised")
    return {"pass": True, "whole_cases": records, "stream_cases": stream_records,
            "real_bpe_rollbacks": real_bpe_rollbacks,
            "max_stream_probability_error": max(row["whole_stream_probability_error"] for row in stream_records),
            "stream_probability_tolerance": STREAM_PROBABILITY_TOLERANCE,
            "english_user_threshold_absent": "en/user" not in selection["thresholds"],
            "english_user_missing_threshold_not_fabricated": True,
            "cross_role_pass": True, "session_release_pass": True, "generated_tokens": 0}


def child_main(args):
    report = {"status": "running", "pass": False, "phase": "dependency_audit", "generated_tokens": 0}
    try:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        bundle = args.bundle.resolve()
        # No script/workspace paths remain importable in the isolated child.
        # System and /work/modern dependency paths stay available.
        sys.path[:] = [str(bundle)] + [path for path in sys.path if path and not (
            os.path.realpath(path) == "/work" or
            any(os.path.realpath(path) == root or os.path.realpath(path).startswith(root + os.sep)
                for root in (*FORBIDDEN_ROOTS, "/work/round4", "/work/validation")))]
        report["dependency_audit"] = install_dependency_audit()
        torch = require_l20()
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        from standalone_model import load_bundle
        from run_guard import JSONLGuard
        torch, model, tokenizer, metadata = load_bundle(bundle)
        if metadata["selection"] != reference["selection"]:
            raise AssertionError("Bundled selection differs from the frozen old-loader reference")
        report["phase"] = "structure_and_probability_parity"
        observed_structure = structure(model, torch)
        if observed_structure != reference["structure"]:
            report["observed_structure"] = observed_structure
            report["expected_structure"] = reference["structure"]
            raise AssertionError("Parameter structure/dtypes or exact RoPE buffer hashes differ")
        report["structure"] = observed_structure
        report["loader_cases"] = []
        for case in reference["cases"]:
            ids = tokenizer.encode(serialize(case["messages"]), add_special_tokens=False)
            if ids != case["canonical_token_ids"]:
                raise AssertionError(f"Portable tokenizer differs from old tokenizer: {case['name']}")
            observed = probabilities(torch, model, ids)
            error = max(max_error(observed[role], case["risk_probabilities_by_role"][role]) for role in ROLES)
            report["loader_cases"].append({"name": case["name"], "native_tokens": len(ids),
                                           "risk_probabilities_by_role": observed,
                                           "max_probability_error": error})
            if error > WHOLE_PROBABILITY_TOLERANCE:
                raise AssertionError(f"Portable vs old-loader probability mismatch: {case['name']}: {error}")
        guard_args = SimpleNamespace(chunk_tokens=32, prefill_chunk_tokens=2048, max_sessions=4, threshold=None)
        guard = JSONLGuard(guard_args, torch, model, tokenizer, metadata)
        report["phase"] = "jsonl_operations"
        report["jsonl"] = exercise_jsonl(guard, tokenizer, metadata["selection"], reference)
        audit = report["dependency_audit"]
        if audit["unexpected_denials"] or audit["network_connections_blocked"]:
            raise AssertionError("The portable runtime attempted a forbidden external dependency")
        loaded_sources = {}
        for name in ("standalone_model", "run_guard", "text_stream_runtime", "window_attention", "memory_attention"):
            module = sys.modules.get(name)
            if module is None:
                continue
            source = Path(module.__file__).resolve()
            if not source.is_relative_to(bundle):
                raise AssertionError(f"Runtime module came from outside the bundle: {name}: {source}")
            loaded_sources[name] = str(source)
        report.update(status="completed", phase="finished", **{"pass": True},
                      checkpoint_sha256=metadata["checkpoint_sha256"], candidate=metadata["candidate"],
                      variant=metadata["variant"], device=metadata["device"],
                      dependency_versions=metadata["runtime_versions"], runtime_module_sources=loaded_sources,
                      max_old_loader_probability_error=max(row["max_probability_error"] for row in report["loader_cases"]),
                      whole_probability_tolerance=WHOLE_PROBABILITY_TOLERANCE,
                      tolerance_rationale="0.005 maximum absolute probability error for identical BF16 whole-input weights; 0.03 for cached versus whole numerical paths; no metric/label relaxation",
                      parameter_structure_exact_match=True, rope_buffer_dtype_and_sha_exact_match=True,
                      isolated_python_process=bool(sys.flags.isolated), hf_offline=True)
        write_json(args.output, report)
        print(json.dumps({"bundle_child_audit": "passed", "output": str(args.output)}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", **{"pass": False}, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        write_json(args.output, report)
        print(json.dumps({"bundle_child_audit": "failed", "error": str(error)}), file=sys.stderr, flush=True)
        return 1


def parent_main(args):
    report = {"status": "running", "pass": False, "phase": "hardware_check", "generated_tokens": 0,
              "started_at": datetime.now(timezone.utc).isoformat()}
    reference_path = args.output.with_name(args.output.stem + ".reference.json")
    child_path = args.output.with_name(args.output.stem + ".child.json")
    child_log = args.output.with_name(args.output.stem + ".child.log")
    try:
        torch = require_l20()
        write_json(args.output, report)
        report["phase"] = "old_loader_reference"
        reference = old_reference(args, torch)
        write_json(reference_path, reference)
        report.update(phase="isolated_bundle_child", reference_path=str(reference_path),
                      reference_sha256=sha_file(reference_path),
                      checkpoint_sha256=reference["selection"]["checkpoint_sha256"],
                      candidate=reference["selection"]["candidate"], variant=reference["selection"]["variant"],
                      gpu_bytes_allocated_after_old_model_release=reference["gpu_bytes_allocated_after_model_release"])
        write_json(args.output, report)
        environment = os.environ.copy()
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            environment[key] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child",
                   "--bundle", str(args.bundle.resolve()), "--reference", str(reference_path.resolve()),
                   "--output", str(child_path.resolve())]
        with child_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=environment, cwd=args.bundle.resolve(),
                                       stdout=log, stderr=subprocess.STDOUT, timeout=3600,
                                       close_fds=True, check=False)
        report["child_returncode"] = completed.returncode
        report["child_log"] = str(child_log)
        if child_path.is_file():
            report["portable_child"] = json.loads(child_path.read_text(encoding="utf-8"))
        if completed.returncode or report.get("portable_child", {}).get("pass") is not True:
            raise RuntimeError("Portable child validation failed; inspect preserved child report/log")
        report.update(status="completed", phase="finished", **{"pass": True},
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      whole_probability_tolerance=WHOLE_PROBABILITY_TOLERANCE,
                      stream_probability_tolerance=STREAM_PROBABILITY_TOLERANCE,
                      old_base_weights_unneeded_by_bundle=True,
                      initial_h24_checkpoint_unneeded_by_bundle=True,
                      production_approval=False, third_risk_class_validated=False, category_validated=False)
        write_json(args.output, report)
        print(json.dumps({"bundle_audit": "passed", "output": str(args.output)}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", **{"pass": False}, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        write_json(args.output, report)
        print(json.dumps({"bundle_audit": "failed", "output": str(args.output), "error": str(error)}),
              file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    args = parse_args(argv)
    return child_main(args) if args.child else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
