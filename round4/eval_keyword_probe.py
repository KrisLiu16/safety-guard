"""L20 fixed-threshold probe evaluation, plus CPU-only exact-row tokenizer proof.

The proof mode never loads a classifier. It does not claim global tokenizer
equivalence and cannot replace or regenerate any of the frozen probe IDs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

COMPATIBILITY_SCOPE = "exact_frozen_probe_rows_only"
COMPATIBILITY_VERSIONS = {"transformers": "5.17.0", "tokenizers": "0.23.2"}
TOKENIZER_ASSET_NAMES = ("tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json",
                         "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical_json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def row_ids_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        record = {"sample_id": row["sample_id"], "ids": row["ids"]}
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def row_tokenizer_checks(rows):
    checks = []
    for row in rows:
        ids_hash = hashlib.sha256(json.dumps(row["ids"], separators=(",", ":")).encode()).hexdigest()
        checks.append({"sample_id": row["sample_id"], "token_count": len(row["ids"]),
                       "frozen_ids_sha256": ids_hash, "reference_ids_sha256": ids_hash,
                       "runtime_ids_sha256": ids_hash, "reference_equal": True, "runtime_equal": True})
    return checks


def tokenizer_asset_hashes(folder):
    return {name: sha(folder / name) if (folder / name).is_file() else None for name in TOKENIZER_ASSET_NAMES}


def compatibility_versions():
    return {name: importlib.metadata.version(name) for name in COMPATIBILITY_VERSIONS}


def frozen_probe(data_root):
    manifest = json.loads((data_root / "manifest.json").read_text())
    for name, expected in manifest["output_hashes"].items():
        if sha(data_root / name) != expected:
            raise RuntimeError("Frozen keyword probe hash mismatch: " + name)
    rows = read_rows(data_root / "probe.jsonl")
    if len(rows) != manifest["examples"]:
        raise RuntimeError("Keyword probe row count mismatch")
    if any(row["label_tier"] != "controlled_template_diagnostic" or row["training_use"] or row["selection_use"] for row in rows):
        raise ValueError("Probe is not a frozen diagnostic-only dataset")
    return manifest, rows


def serialize_messages(messages):
    return "\n\n".join(message["role"].upper() + ":\n" + message["content"] for message in messages)


def write_compatibility_evidence(args, manifest, rows):
    """CPU-only tokenization proof, deliberately narrower than tokenizer equivalence."""
    destination = args.write_tokenizer_compatibility
    if destination.exists():
        raise FileExistsError("Refusing to overwrite tokenizer compatibility evidence")
    versions = compatibility_versions()
    if versions != COMPATIBILITY_VERSIONS:
        raise RuntimeError(f"Compatibility proof requires the actual frozen modern tokenizer libraries: {COMPATIBILITY_VERSIONS}; found {versions}")
    reference_path = args.reference_tokenizer_json
    if sha(reference_path) != manifest["tokenizer_sha256"]:
        raise RuntimeError("Reference tokenizer does not match the frozen probe tokenizer SHA")
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    folder = args.base_root / "models/qwen35"
    reference = Tokenizer.from_file(str(reference_path))
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    reject_tokenizer_truncation(reference)
    reject_tokenizer_truncation(tokenizer)
    checked = []
    for row in rows:
        text = serialize_messages(row["messages"])
        original = reference.encode(text, add_special_tokens=False)
        actual = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if original.overflowing or original.ids != row["ids"] or actual != row["ids"]:
            raise RuntimeError("Exact frozen row IDs mismatch while producing compatibility evidence: " + row["sample_id"])
        checked.append({"sample_id": row["sample_id"], "ids": actual})
    original_json = json.loads(reference_path.read_text())
    runtime_json = json.loads((folder / "tokenizer.json").read_text())
    backend_json = json.loads(tokenizer.backend_tokenizer.to_str())
    evidence = {
        "version": "keyword-tokenizer-row-compatibility-v1", "status": "passed",
        "scope": COMPATIBILITY_SCOPE, "global_tokenizer_equivalence": False,
        "neural_model_loaded": False, "neural_forward_calls": 0, "calibration_or_predictions_read": False,
        "library_versions": versions, "tokenizer_class": type(tokenizer).__name__,
        "reference_tokenizer_raw_sha256": sha(reference_path),
        "runtime_tokenizer_raw_sha256": sha(folder / "tokenizer.json"),
        "runtime_tokenizer_assets_sha256": tokenizer_asset_hashes(folder),
        "reference_tokenizer_canonical_json_sha256": canonical_json_sha(original_json),
        "runtime_tokenizer_canonical_json_sha256": canonical_json_sha(runtime_json),
        "loaded_backend_canonical_json_sha256": canonical_json_sha(backend_json),
        "raw_json_equal": original_json == runtime_json,
        "loaded_backend_matches_reference_json": original_json == backend_json,
        "probe_manifest_sha256": sha(args.data_root / "manifest.json"),
        "probe_jsonl_sha256": sha(args.data_root / "probe.jsonl"),
        "verified_rows": len(rows), "mismatched_rows": 0,
        "frozen_row_ids_sha256": row_ids_sha(rows), "actual_row_ids_sha256": row_ids_sha(checked),
        "row_checks": row_tokenizer_checks(rows),
        "validation_rule": "Every reference and actual native tokenizer result must equal the complete original saved IDs for each frozen sample; no replacement IDs, normalization or truncation.",
        "limitation": "Raw tokenizer JSONs can differ materially. This proof authorizes only these exact frozen inputs, not general tokenizer interchangeability.",
        "checker_source_sha256": sha(Path(__file__)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({"status": evidence["status"], "scope": evidence["scope"],
                      "verified_rows": len(rows), "mismatched_rows": 0, "library_versions": versions,
                      "global_tokenizer_equivalence": False, "evidence_sha256": sha(destination)}))


def verify_tokenizer_identity(args, manifest, rows):
    folder = args.base_root / "models/qwen35"
    runtime_sha = sha(folder / "tokenizer.json")
    if runtime_sha == manifest["tokenizer_sha256"]:
        return {"mode": "exact_original_tokenizer_raw_sha256", "runtime_tokenizer_raw_sha256": runtime_sha}
    if args.tokenizer_compatibility is None:
        raise RuntimeError("Tokenizer raw SHA differs; explicit real-modern per-row compatibility evidence is required")
    path = args.tokenizer_compatibility
    evidence = json.loads(path.read_text())
    expected = {
        "version": "keyword-tokenizer-row-compatibility-v1", "status": "passed", "scope": COMPATIBILITY_SCOPE,
        "global_tokenizer_equivalence": False,
        "neural_model_loaded": False, "neural_forward_calls": 0, "calibration_or_predictions_read": False,
        "reference_tokenizer_raw_sha256": manifest["tokenizer_sha256"],
        "runtime_tokenizer_raw_sha256": runtime_sha,
        "runtime_tokenizer_assets_sha256": tokenizer_asset_hashes(folder),
        "probe_manifest_sha256": sha(args.data_root / "manifest.json"),
        "probe_jsonl_sha256": sha(args.data_root / "probe.jsonl"),
        "verified_rows": len(rows), "mismatched_rows": 0,
        "frozen_row_ids_sha256": row_ids_sha(rows), "actual_row_ids_sha256": row_ids_sha(rows),
        "row_checks": row_tokenizer_checks(rows),
        "library_versions": compatibility_versions(),
        "checker_source_sha256": sha(Path(__file__)),
    }
    if expected["library_versions"] != COMPATIBILITY_VERSIONS:
        raise RuntimeError("Candidate environment differs from the fixed modern tokenizer libraries")
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise RuntimeError("Tokenizer compatibility evidence mismatch: " + key)
    actual_canonical = canonical_json_sha(json.loads((folder / "tokenizer.json").read_text()))
    if evidence.get("runtime_tokenizer_canonical_json_sha256") != actual_canonical:
        raise RuntimeError("Runtime tokenizer canonical JSON differs from its explicitly scoped evidence")
    return {"mode": COMPATIBILITY_SCOPE, "evidence_path": str(path), "evidence_sha256": sha(path),
            "global_tokenizer_equivalence": False, "evidence": evidence}


def reject_tokenizer_truncation(tokenizer):
    for candidate in (tokenizer, getattr(tokenizer, "backend_tokenizer", None)):
        if candidate is not None and getattr(candidate, "truncation", None) not in (None, False):
            raise ValueError("Tokenizer truncation must be disabled for diagnostic evaluation")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("candidate", "a0"))
    parser.add_argument("--data-root", type=Path, default=Path("/work/round4/data/keyword_probe_v1"))
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--calibration-metrics", type=Path)
    parser.add_argument("--tokenizer-compatibility", type=Path,
                        help="Explicit real-modern compatibility evidence scoped to every frozen probe row")
    parser.add_argument("--write-tokenizer-compatibility", type=Path,
                        help="CPU-only mode: write exact-row tokenizer evidence; never load a neural model")
    parser.add_argument("--reference-tokenizer-json", type=Path,
                        help="Original tokenizer.json whose raw SHA equals the frozen probe manifest")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", choices=("full", "window", "memory", "classification_rl"))
    parser.add_argument("--rl-base-variant", choices=("full", "window", "memory"))
    parser.add_argument("--base-root", type=Path, default=Path("/work"))
    parser.add_argument("--base-code-dir", type=Path, default=Path("/work/input"))
    parser.add_argument("--window-code-dir", type=Path, default=Path("/work/window"))
    parser.add_argument("--memory-code-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.write_tokenizer_compatibility is not None:
        if args.kind != "candidate" or args.reference_tokenizer_json is None:
            parser.error("CPU tokenizer proof requires --kind candidate and --reference-tokenizer-json")
    elif args.output_prefix is None or args.calibration_metrics is None:
        parser.error("Neural evaluation requires --output-prefix and --calibration-metrics")
    elif args.kind == "candidate" and (args.checkpoint is None or args.variant is None):
        parser.error("candidate requires --checkpoint and --variant")
    if not 1 <= args.batch_size <= 8:
        parser.error("batch size must be 1..8")
    return args


def summarize_rows(rows):
    positive = sum(row["source_label"] == "unsafe" for row in rows)
    negative = len(rows) - positive
    calibrated = [row for row in rows if row["calibrated_unsafe"] is not None]
    calibrated_positive = sum(row["source_label"] == "unsafe" for row in calibrated)
    calibrated_negative = len(calibrated) - calibrated_positive
    tp = sum(row["calibrated_unsafe"] and row["source_label"] == "unsafe" for row in calibrated)
    fp = sum(row["calibrated_unsafe"] and row["source_label"] == "safe" for row in calibrated)
    argmax_tp = sum(max(range(3), key=lambda i: row["probs"][i]) == 1 and row["source_label"] == "unsafe" for row in rows)
    argmax_fp = sum(max(range(3), key=lambda i: row["probs"][i]) == 1 and row["source_label"] == "safe" for row in rows)
    return {"n": len(rows), "safe": negative, "unsafe": positive,
            "calibrated_scored": len(calibrated), "calibrated_unscored": len(rows) - len(calibrated),
            "calibrated_safe": calibrated_negative, "calibrated_unsafe": calibrated_positive,
            "calibrated_tp": tp, "calibrated_fp": fp,
            "calibrated_recall": tp / calibrated_positive if calibrated_positive else None,
            "calibrated_fpr": fp / calibrated_negative if calibrated_negative else None,
            "uncalibrated_argmax_unsafe_recall": argmax_tp / positive if positive else None,
            "uncalibrated_argmax_unsafe_fpr": argmax_fp / negative if negative else None,
            "mean_unsafe_probability": sum(row["probs"][1] for row in rows) / len(rows) if rows else None}


def summarize(predictions, calibration):
    thresholds = {}
    for key, value in calibration.get("strata", {}).items():
        threshold = value.get("threshold_from_calibration")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid existing calibration threshold for " + key)
        thresholds[key] = threshold
    for row in predictions:
        key = row["language"] + "/" + row["target_role"]
        threshold = thresholds.get(key)
        row["existing_calibration_threshold"] = threshold
        row["calibrated_unsafe"] = row["probs"][1] >= threshold if threshold is not None else None
        row["calibration_status"] = "existing_same_stratum" if threshold is not None else "missing_stratum_unscored"
    grouped = {}
    for name, key_fn in (("template", lambda row: row["template_id"]),
                         ("stratum", lambda row: row["language"] + "/" + row["target_role"]),
                         ("source_group", lambda row: row["source_group"])):
        groups = defaultdict(list)
        for row in predictions:
            groups[key_fn(row)].append(row)
        grouped[name] = {key: summarize_rows(rows) for key, rows in sorted(groups.items())}
    return {"all_rows": summarize_rows(predictions), "by": grouped,
            "existing_thresholds": thresholds, "thresholds_tuned_on_probe": False,
            "missing_calibration_strata": sorted({r["language"] + "/" + r["target_role"] for r in predictions if r["calibrated_unsafe"] is None}),
            "argmax_diagnostic": "Explicitly uncalibrated three-logit argmax==unsafe; no threshold fallback. Single-label groups have only their defined FPR or recall, no ROC-AUC."}


def main():
    args = parse_args()
    manifest, rows = frozen_probe(args.data_root)
    if args.write_tokenizer_compatibility is not None:
        write_compatibility_evidence(args, manifest, rows)
        return
    prediction_path = Path(str(args.output_prefix) + "_predictions.jsonl")
    metric_path = Path(str(args.output_prefix) + "_metrics.json")
    if prediction_path.exists() or metric_path.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic evaluation")
    calibration = json.loads(args.calibration_metrics.read_text())
    if not isinstance(calibration.get("strata"), dict) or not calibration["strata"]:
        raise ValueError("An existing per-stratum calibration report is required")
    summarize([], calibration)  # Validate fixed thresholds before loading weights.
    sys.path[:0] = [str(args.base_code_dir), str(args.window_code_dir), str(args.memory_code_dir)]
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Model evaluation is restricted to one visible CUDA L20")
    torch.set_num_threads(4)
    if args.kind == "candidate":
        from infer_classifier import load_classifier, resolve_variant, serialize
        tokenizer_identity = verify_tokenizer_identity(args, manifest, rows)
        variant = resolve_variant(args)
        _, model, tokenizer, _, kernels, _ = load_classifier(args, variant)
        reject_tokenizer_truncation(tokenizer)
        if tokenizer_identity["mode"] == COMPATIBILITY_SCOPE:
            proof = tokenizer_identity["evidence"]
            if (type(tokenizer).__name__ != proof["tokenizer_class"]
                    or canonical_json_sha(json.loads(tokenizer.backend_tokenizer.to_str())) != proof["loaded_backend_canonical_json_sha256"]):
                raise RuntimeError("Loaded native tokenizer differs from the CPU-only compatibility evidence")
        prepared = []
        for row in rows:
            ids = tokenizer.encode(serialize(row["messages"]), add_special_tokens=False, truncation=False)
            if ids != row["ids"]:
                raise RuntimeError("Candidate native token IDs mismatch")
            prepared.append((row, row["ids"]))  # Evaluate the unchanged frozen IDs, never replacement IDs.
        tokenizer_identity["frozen_rows_rechecked_during_evaluation"] = len(rows)
        tokenizer_identity["frozen_row_ids_sha256"] = row_ids_sha(rows)
        model_metadata = {"kind": "candidate", "variant": variant, "checkpoint": str(args.checkpoint),
                          "checkpoint_sha256": sha(args.checkpoint), "kernels": kernels,
                          "serialization": "frozen training serialization", "tokenizer_identity": tokenizer_identity}
    else:
        from runtime import load, encode, logits
        path = args.base_root / "models/guard"
        model, tokenizer = load(dtype=torch.bfloat16, device="cuda", model_path=path)
        reject_tokenizer_truncation(tokenizer)
        prepared = [(row, encode(tokenizer, row["messages"])) for row in rows]
        model_metadata = {"kind": "a0", "reference": "original Qwen3Guard-Stream-0.6B",
                          "local_model_files": {file.name: sha(file) for file in sorted(path.glob("*.safetensors"))},
                          "serialization": "original published guard chat template; not architecture-only comparison"}
    if any(not 0 < len(ids) <= 8192 for _, ids in prepared):
        raise ValueError("Diagnostic token limit exceeded; no truncation is permitted")
    prepared.sort(key=lambda pair: len(pair[1]))
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(prepared), args.batch_size):
            group = prepared[begin:begin + args.batch_size]
            length = max(len(ids) for _, ids in group)
            inputs = torch.full((len(group), length), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(inputs)
            for index, (_, ids) in enumerate(group):
                inputs[index, :len(ids)] = torch.tensor(ids, device="cuda")
                mask[index, :len(ids)] = 1
            if args.kind == "candidate":
                output = model(inputs, mask)
            else:
                output = model(input_ids=inputs, attention_mask=mask,
                               position_ids=(mask.cumsum(-1) - 1).clamp(min=0), use_cache=False)
            for index, (row, ids) in enumerate(group):
                if args.kind == "candidate":
                    risk, _ = model.readout(output.last_hidden_state[index:index + 1, len(ids) - 1], row["target_role"])
                    probs = risk[0].float().softmax(-1).cpu().tolist()
                else:
                    probs = logits(output, row["target_role"])[index, len(ids) - 1].float().softmax(-1).cpu().tolist()
                if len(probs) != 3 or not all(math.isfinite(value) for value in probs):
                    raise RuntimeError("Invalid classifier probabilities")
                predictions.append({key: row[key] for key in
                                    ("sample_id", "word_family", "template_id", "source_group", "language", "target_role", "source_label")}
                                   | {"probs": probs, "native_input_tokens": len(ids)})
    predictions.sort(key=lambda row: row["sample_id"])
    result = summarize(predictions, calibration)
    result.update({"status": "completed", "model": model_metadata, "generated_tokens": 0,
                   "probe_manifest_sha256": sha(args.data_root / "manifest.json"),
                   "calibration_metrics_path": str(args.calibration_metrics),
                   "calibration_metrics_sha256": sha(args.calibration_metrics),
                   "calibration_checkpoint_association_verified": False,
                   "evaluation_scope": "controlled_template_diagnostic; complete inputs; source-preserving held-out words; no training, checkpoint selection or threshold tuning",
                   "third_risk_class_validated": False, "category_output_validated": False})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    metric_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"kind": args.kind, "all_rows": result["all_rows"],
                      "missing_calibration_strata": result["missing_calibration_strata"]}))


if __name__ == "__main__":
    main()
