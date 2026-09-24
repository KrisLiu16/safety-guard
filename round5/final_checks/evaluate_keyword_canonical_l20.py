"""Frozen 1024-row keyword-context regression on the final canonical32 bundle.

No training, generation, threshold fitting, template changes or ID replacement.
The parent is CPU-only and bounds an isolated L20 child to 600 seconds by default.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROBE_SHA = "9ba84695937083bcdad956957a3df3fa7cbfa176f9c47242a5901df1ca70f11f"
MANIFEST_SHA = "c1b58b122e09dc4d9e9a45c1d9ba8b02dff2741f3eae648672d15a64350f3df9"
PROOF_SHA = "bffaa24cc31237158b770b1a2781b007045353a09ab2d48dbd950eebc594e470"
VERSIONS = {"transformers": "5.17.0", "tokenizers": "0.23.2"}
TEMPLATES = {"zh_user_safe_script": ("zh", "user", "safe"),
             "zh_user_unsafe_fraud": ("zh", "user", "unsafe"),
             "en_user_unsafe_privacy": ("en", "user", "unsafe"),
             "en_assistant_safe_refusal": ("en", "assistant", "safe")}
RUNTIME_FILES = ("text_stream_runtime.py", "graph_stream.py", "canonical_block_engine.py", "canonical_text_runtime.py")
FIELDS = ("sample_id", "word", "word_family", "family", "template_id", "source_group",
          "language", "target_role", "source_label", "label_tier", "rubric", "source_licenses")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_exact(name, path, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError("Bundled source hash differs: " + name)
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError("Refusing cached module from another directory: " + name)
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_probe(root, proof_path):
    if sha(root / "manifest.json") != MANIFEST_SHA or sha(root / "probe.jsonl") != PROBE_SHA:
        raise ValueError("Frozen probe identity changed")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["output_hashes"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen probe companion file changed: " + name)
    rows = [json.loads(line) for line in (root / "probe.jsonl").read_text().splitlines() if line.strip()]
    if (len(rows) != 1024 or len({r["sample_id"] for r in rows}) != 1024
            or len({r["word_family"] for r in rows}) != 256 or len({r["family"] for r in rows}) != 256
            or dict(Counter(r["template_id"] for r in rows)) != {key: 256 for key in TEMPLATES}):
        raise ValueError("Expected the unchanged 1024 rows, 256 families and four templates")
    families = defaultdict(set)
    for row in rows:
        expected = TEMPLATES[row["template_id"]]
        if ((row["language"], row["target_role"], row["source_label"]) != expected
                or row["messages"][-1]["role"] != row["target_role"]
                or row["label_tier"] != "controlled_template_diagnostic"
                or row["training_use"] is not False or row["selection_use"] is not False
                or row["natural_semantic_gold"] is not False
                or not 1 <= len(row["ids"]) <= 8192 or any(type(i) is not int or i < 0 for i in row["ids"])):
            raise ValueError("Frozen row role/label/usage/IDs changed")
        families[row["word_family"]].add(row["template_id"])
    if any(value != set(TEMPLATES) for value in families.values()):
        raise ValueError("A word is missing a fixed paired context")
    if sha(proof_path) != PROOF_SHA:
        raise ValueError("Frozen modern tokenizer compatibility proof changed")
    proof = json.loads(proof_path.read_text())
    checks = [{"sample_id": r["sample_id"], "token_count": len(r["ids"]),
               "frozen_ids_sha256": ids_sha(r["ids"]), "reference_ids_sha256": ids_sha(r["ids"]),
               "runtime_ids_sha256": ids_sha(r["ids"]), "reference_equal": True, "runtime_equal": True} for r in rows]
    if (proof["status"] != "passed" or proof["scope"] != "exact_frozen_probe_rows_only"
            or proof["verified_rows"] != 1024 or proof["mismatched_rows"] != 0
            or proof["probe_manifest_sha256"] != MANIFEST_SHA or proof["probe_jsonl_sha256"] != PROBE_SHA
            or proof["row_checks"] != checks or proof["global_tokenizer_equivalence"] is not False
            or proof["library_versions"] != VERSIONS):
        raise ValueError("Compatibility proof does not bind all frozen rows")
    return manifest, rows, proof


def serialize(messages):
    return "\n\n".join(row["role"].upper() + ":\n" + row["content"] for row in messages)


def verify_native(tokenizer, rows, proof, assets):
    if {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError("Actual modern tokenizer versions differ")
    for name, digest in proof["runtime_tokenizer_assets_sha256"].items():
        actual = sha(assets / name) if (assets / name).is_file() else None
        if actual != digest:
            raise ValueError("Bundled tokenizer asset differs from exact-row proof: " + name)
    backend = tokenizer.backend_tokenizer
    if backend.truncation is not None or getattr(tokenizer, "truncation", None) not in (None, False):
        raise ValueError("Tokenizer truncation is forbidden")
    if (type(tokenizer).__name__ != proof["tokenizer_class"]
            or hashlib.sha256(canonical(json.loads(backend.to_str())).encode()).hexdigest()
            != proof["loaded_backend_canonical_json_sha256"]):
        raise ValueError("Loaded modern tokenizer backend differs from exact-row proof")
    positions, digest = {}, hashlib.sha256()
    for row in rows:
        text = serialize(row["messages"])
        result = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        if result["input_ids"] != row["ids"]:
            raise ValueError("Actual native IDs mismatch; replacement forbidden: " + row["sample_id"])
        start = len(text) - len(row["messages"][-1]["content"])
        target = [index for index, (_, end) in enumerate(result["offset_mapping"]) if end > start]
        if not target or target[-1] != len(row["ids"]) - 1:
            raise ValueError("No complete last-role target token span")
        positions[row["sample_id"]] = target
        digest.update((canonical([row["sample_id"], row["ids"], target]) + "\n").encode())
    return positions, {"status": "passed", "rows": len(rows), "source_proof_sha256": PROOF_SHA,
                       "actual_ids_and_target_positions_sha256": digest.hexdigest(), "replacement_ids_written": False,
                       "library_versions": VERSIONS, "neural_forward_calls": 0,
                       "target_boundary": "offset end > last message content start; prior user prompt excluded from assistant score"}


def finite_probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid probability or fixed threshold")
    return float(value)


def observed_row(runtime, row, positions, selection):
    result = runtime.classify_prefixes(row["ids"])
    n = len(row["ids"])
    physical = 32 * math.ceil(n / 32)
    if (result["native_tokens"] != n or result["forward_tokens"] != physical
            or result["padding_tokens"] != physical - n or result["real_forward_tokens"] != n
            or result["forward_calls"] != physical // 32 or result["generated_tokens"] != 0
            or result["execution_contract"] != selection["execution_contract"]):
        raise ValueError("Canonical forward accounting/contract changed")
    heads = result["probabilities_by_role_all_tokens"]
    if set(heads) != {"user", "assistant"}:
        raise ValueError("Missing role head")
    for values in heads.values():
        if len(values) != n:
            raise ValueError("Missing real prefixes or padded outputs leaked")
        for probs in values:
            if len(probs) != 3 or abs(sum(finite_probability(p) for p in probs) - 1) > 1e-4:
                raise ValueError("Invalid classifier distribution")
    values = heads[row["target_role"]]
    scores = [values[index][1] for index in positions]
    observed = {key: row[key] for key in FIELDS}
    observed.update(status="ok", native_input_tokens=n, native_ids_sha256=ids_sha(row["ids"]),
        endpoint_probs=values[-1], target_token_positions=positions, native_target_p_unsafe=scores,
        endpoint_p_unsafe=values[-1][1], native_target_max_p_unsafe=max(scores),
        endpoint_argmax_unsafe=max(range(3), key=values[-1].__getitem__) == 1,
        native_any_argmax_unsafe=any(max(range(3), key=values[index].__getitem__) == 1 for index in positions),
        forward_tokens=physical, padding_tokens=physical - n, physical_block_calls=physical // 32,
        checkpoint_sha256=selection["checkpoint_sha256"], threshold_comparison=">", generated_tokens=0)
    stratum = row["language"] + "/" + row["target_role"]
    for mode, score in (("whole", observed["endpoint_p_unsafe"]), ("stream", max(scores))):
        threshold = selection["thresholds"][mode].get(stratum)
        if stratum == "en/user" and threshold is not None:
            raise ValueError("Frozen Round5 has no English-user calibration")
        if stratum != "en/user" and threshold is None:
            raise ValueError("A required existing stratum threshold is missing")
        if threshold is not None:
            finite_probability(threshold)
        observed[mode] = {"score": score, "threshold": threshold,
            "unsafe": score > threshold if threshold is not None else None,
            "calibration_status": "bound_existing_stratum" if threshold is not None else "missing_stratum_unscored"}
    return observed


def group_metrics(rows, mode):
    positives = sum(r["source_label"] == "unsafe" for r in rows)
    negatives = len(rows) - positives
    scored = [r for r in rows if r[mode]["unsafe"] is not None]
    p = sum(r["source_label"] == "unsafe" for r in scored)
    n = len(scored) - p
    tp = sum(r[mode]["unsafe"] and r["source_label"] == "unsafe" for r in scored)
    fp = sum(r[mode]["unsafe"] and r["source_label"] == "safe" for r in scored)
    argmax = "endpoint_argmax_unsafe" if mode == "whole" else "native_any_argmax_unsafe"
    atp = sum(r[argmax] and r["source_label"] == "unsafe" for r in rows)
    afp = sum(r[argmax] and r["source_label"] == "safe" for r in rows)
    return {"n": len(rows), "safe": negatives, "unsafe": positives, "calibrated_scored": len(scored),
        "calibrated_unscored": len(rows) - len(scored), "calibrated_safe": n, "calibrated_unsafe": p,
        "tp": tp, "fp": fp, "fn": p - tp, "tn": n - fp,
        "recall": tp / p if p else None, "fpr": fp / n if n else None,
        "uncalibrated_argmax_recall": atp / positives if positives else None,
        "uncalibrated_argmax_fpr": afp / negatives if negatives else None}


def summarize(rows):
    result = {}
    for mode in ("whole", "stream"):
        groups = {}
        for name, key in (("template", lambda r: r["template_id"]),
                          ("stratum", lambda r: r["language"] + "/" + r["target_role"]),
                          ("source_group", lambda r: r["source_group"])):
            grouped = defaultdict(list)
            for row in rows:
                grouped[key(row)].append(row)
            groups[name] = {name: group_metrics(group, mode) for name, group in sorted(grouped.items())}
        result[mode] = {"all_rows": group_metrics(rows, mode), "by": groups}
    return result


def run_child(args):
    report = {"status": "running", "integrity_pass": False, "phase": "fixed_inputs", "generated_tokens": 0,
        "selection_changed": False, "thresholds_fitted_on_probe": False, "evaluator_sha256": sha(__file__),
        "coverage": {"requested": 1024, "evaluated": 0, "failed_or_unattempted": 1024},
        "stream_scope": "native target prefix maximum only; no additional real-text cuts, no arbitrary arrival-schedule FPR claim",
        "missing_template_strata": ["zh/assistant"], "missing_calibration_strata": ["en/user"],
        "natural_semantic_gold": False, "production_approval": False}
    rows, predictions, runtime = [], [], None
    began = time.monotonic()
    write_json(args.output / "metrics.json", report)
    prediction_path = args.output / "predictions.jsonl"
    try:
        manifest, rows, proof = load_probe(args.data_root, args.tokenizer_proof)
        record = json.loads((args.bundle / "BUNDLE_MANIFEST.json").read_text())
        loader = load_exact("standalone_model", args.bundle / "standalone_model.py", record["files"]["standalone_model.py"]["sha256"])
        bundle, record, selection = loader.verify_bundle(args.bundle)
        calibration = json.loads((bundle / "selection/canonical_calibration.json").read_text())
        if (calibration["checkpoint_sha256"] != selection["checkpoint_sha256"]
                or calibration["thresholds"] != selection["thresholds"]
                or calibration["engine_metadata"]["inference_engine"] != "eager"
                or selection["threshold_comparison"] != ">"):
            raise ValueError("The actual fixed bundle/calibration/checkpoint association changed")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bundle / "assets", local_files_only=True, trust_remote_code=False)
        positions, native_proof = verify_native(tokenizer, rows, proof, bundle / "assets")
        write_json(args.output / "modern_tokenizer_proof.json", native_proof)
        report.update(phase="l20_model", checkpoint_sha256=selection["checkpoint_sha256"],
            bundle_manifest_sha256=sha(bundle / "BUNDLE_MANIFEST.json"),
            model_manifest_sha256=sha(bundle / "MODEL_MANIFEST.json"),
            calibration_sha256=sha(bundle / "selection/canonical_calibration.json"),
            calibration_checkpoint_association_verified=True, thresholds=selection["thresholds"],
            threshold_comparison=">", canonical_quality_gate_pass=calibration["canonical_quality_gate_pass"],
            probe_manifest_sha256=MANIFEST_SHA, probe_jsonl_sha256=PROBE_SHA,
            compatibility_proof_sha256=PROOF_SHA, actual_modern_proof_sha256=sha(args.output / "modern_tokenizer_proof.json"),
            runtime_source_sha256=calibration["runtime_source_sha256"],
            bundled_loader_sha256=record["files"]["standalone_model.py"]["sha256"],
            historical_exclusion_receipt=manifest["excluded_splits"],
            historical_inputs_recomputed=False, execution_contract=selection["execution_contract"])
        write_json(args.output / "metrics.json", report)
        torch, model, tokenizer, metadata = loader.load_bundle(bundle)
        # Check the tokenizer actually returned by the neural loader as well.
        actual_positions, actual_proof = verify_native(tokenizer, rows, proof, bundle / "assets")
        if actual_positions != positions or actual_proof != native_proof:
            raise ValueError("Runtime tokenizer differs from pre-neural proof")
        for name in RUNTIME_FILES:
            load_exact(Path(name).stem, bundle / name, record["files"][name]["sha256"])
        runtime = sys.modules["canonical_text_runtime"].CanonicalTextRuntime(model, tokenizer,
            inference_engine="eager", pad_token_id=selection["execution_contract"]["pad_token_id"])
        if runtime.engine_metadata["execution_contract"] != selection["execution_contract"]:
            raise ValueError("Actual canonical contract differs")
        report.update(phase="observations", device=metadata["device"], runtime_versions=metadata["runtime_versions"],
                      engine_metadata=runtime.engine_metadata)
        with prediction_path.open("x") as stream:
            for source in rows:
                observed = observed_row(runtime, source, positions[source["sample_id"]], selection)
                predictions.append(observed)
                stream.write(json.dumps(observed, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if len(predictions) % 128 == 0:
                    report["coverage"] = {"requested": 1024, "evaluated": len(predictions), "failed_or_unattempted": 1024 - len(predictions)}
                    write_json(args.output / "metrics.json", report)
        if len(predictions) != 1024 or {r["sample_id"] for r in predictions} != {r["sample_id"] for r in rows}:
            raise ValueError("Incomplete probe coverage")
        # Re-verify all bundle/checkpoint/runtime bytes after execution, too.
        loader.verify_bundle(bundle)
        if sha(bundle / "BUNDLE_MANIFEST.json") != report["bundle_manifest_sha256"]:
            raise ValueError("Bundle changed during evaluation")
        report.update(status="completed", integrity_pass=True, phase="finished", **summarize(predictions),
            coverage={"requested": 1024, "evaluated": 1024, "failed_or_unattempted": 0, "word_families": 256},
            prediction_sha256=sha(prediction_path), engine_final_stats=runtime.stats(),
            inference_sequences=1024, physical_block_calls=sum(r["physical_block_calls"] for r in predictions),
            native_input_tokens=sum(r["native_input_tokens"] for r in predictions),
            physical_forward_tokens=sum(r["forward_tokens"] for r in predictions),
            padding_tokens=sum(r["padding_tokens"] for r in predictions),
            limits="Four fixed controlled templates, not natural safety gold or broad policy/linguistic coverage. "
                   "No zh/assistant samples or en/user calibrated thresholds. Native-prefix subtrajectory does not "
                   "reproduce the full cal text-cut trajectory; no tuning or model promotion from this regression.")
    except Exception as error:
        report.update(status="failed", integrity_pass=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        completed = {r["sample_id"] for r in predictions}
        with prediction_path.open("a") as stream:
            for row in rows:
                if row["sample_id"] not in completed:
                    unscored = {key: row[key] for key in FIELDS}
                    unscored.update(status="failed_or_not_attempted", reason=str(error),
                        native_ids_sha256=ids_sha(row["ids"]), checkpoint_sha256=report.get("checkpoint_sha256"),
                        whole=None, stream=None, generated_tokens=0)
                    stream.write(json.dumps(unscored, ensure_ascii=False) + "\n")
        report["coverage"] = {"requested": 1024, "evaluated": len(predictions), "failed_or_unattempted": 1024 - len(predictions)}
        report["prediction_sha256"] = sha(prediction_path)
        traceback.print_exc()
    finally:
        if runtime is not None:
            runtime.close()
        report["elapsed_seconds"] = time.monotonic() - began
        write_json(args.output / "metrics.json", report)
    return 0 if report["integrity_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("/work/output/round5/validation/bundle"))
    parser.add_argument("--data-root", type=Path, default=Path("/work/round4/data/keyword_probe_v1"))
    parser.add_argument("--tokenizer-proof", type=Path, default=Path("/work/round4/keyword_tokenizer_compatibility_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round5/final_checks/keyword_canonical"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 900:
        parser.error("timeout must be 1..900 seconds")
    if args.child:
        return run_child(args)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a frozen keyword evaluation")
    args.output.mkdir(parents=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--bundle", str(args.bundle.resolve()),
               "--data-root", str(args.data_root.resolve()), "--tokenizer-proof", str(args.tokenizer_proof.resolve()),
               "--output", str(args.output.resolve()), "--timeout-seconds", str(args.timeout_seconds)]
    with (args.output / "worker.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            path = args.output / "metrics.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            report.update(status="failed", integrity_pass=False, error_type=type(error).__name__, error=str(error),
                          parent_timeout_seconds=args.timeout_seconds, process_group_termination="SIGKILL_entire_group")
            write_json(path, report)
            return 1
    path = args.output / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    return 0 if code == 0 and result.get("status") == "completed" and result.get("integrity_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
