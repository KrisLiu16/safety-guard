"""Calibrate fixed selected weights under the canonical32 serving execution.

CPU-safe imports. Formal execution requires exactly one L20. Calibration uses
only the frozen 900 calibration records; the 1,200 development records check
the serving gate, never change the completed training's checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

EVALUATOR_SHA = "4ee145dfaae18d3408cde321d090b156176482551efaf1254a45b3a691d06ded"
RUNTIME_FILES = ("text_stream_runtime.py", "graph_stream.py", "canonical_block_engine.py",
                 "canonical_text_runtime.py")
VERSION = "round5-canonical32-calibration-v1"


def load_local(name, path, digest=None):
    path = Path(path).resolve()
    if digest is not None and sha(path) != digest:
        raise ValueError("Frozen source SHA differs: " + name)
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise RuntimeError("Cached module comes from another path: " + name)
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid finite probability")
    return float(value)


def validate_contract(contract):
    fixed = {"name": "canonical32-v1", "block_tokens": 32, "window": 512,
             "backbone_layers": 24, "partial_cache": "not_committed",
             "readout": "all_32_positions_both_roles", "max_input_tokens": 8192,
             "risk_labels": ["safe", "unsafe", "controversial"]}
    if (not isinstance(contract, dict) or any(contract.get(k) != v for k, v in fixed.items())
            or type(contract.get("pad_token_id")) is not int or contract["pad_token_id"] < 0):
        raise ValueError("Canonical execution contract differs from the declared fixed32 algorithm")
    json.dumps(contract, allow_nan=False)


def validated_prefixes(runtime, ids, contract):
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 8192
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Invalid frozen input IDs; truncation is forbidden")
    result = runtime.classify_prefixes(ids)
    n, physical = len(ids), 32 * math.ceil(len(ids) / 32)
    if (result.get("native_tokens") != n or result.get("forward_tokens") != physical
            or result.get("real_forward_tokens") != n or result.get("padding_tokens") != physical - n
            or result.get("forward_calls") != physical // 32 or result.get("generated_tokens") != 0
            or result.get("execution_contract") != contract):
        raise ValueError("Canonical runtime returned incorrect shape, accounting or execution contract")
    execution = result["engine_execution"]
    if (execution["eager_calls"] + execution["graph_calls"] != physical // 32
            or execution["forward_tokens"] != physical or execution["real_forward_tokens"] != n
            or execution["padding_tokens"] != physical - n
            or execution.get("initialization_included") is not False):
        raise ValueError("Canonical block execution accounting changed")
    values = result["probabilities_by_role_all_tokens"]
    if set(values) != {"user", "assistant"}:
        raise ValueError("Both role heads must be observed")
    for role in values:
        if len(values[role]) != n:
            raise ValueError("Missing real prefix outputs or leaked padded-future outputs")
        for probs in values[role]:
            if len(probs) != 3 or abs(sum(probability(p) for p in probs) - 1) > 1e-4:
                raise ValueError("Invalid prefix class distribution")
    return values, {"native_tokens": n, "forward_tokens": physical, "padding_tokens": physical - n,
                    "forward_calls": physical // 32, "eager_calls": execution["eager_calls"],
                    "graph_calls": execution["graph_calls"]}


def observations(runtime, rows, *, prediction_path=None):
    """Reuse exact ID prefixes; a retokenized cut is an independent trajectory.

    No cross-record or cross-role score cache. Every endpoint comes from the
    same original trajectory used by its native-prefix maximum.
    """
    contract = runtime.engine_metadata["execution_contract"]
    validate_contract(contract)
    output = []
    totals = {k: 0 for k in ("original_calls", "retokenized_cut_calls", "exact_prefix_cuts_reused",
              "complete_input_cuts_reused", "duplicate_cut_calls_reused", "native_tokens",
              "forward_tokens", "padding_tokens", "forward_calls", "eager_calls", "graph_calls")}
    stream = Path(prediction_path).open("x") if prediction_path is not None else None
    try:
        for index, source in enumerate(rows):
            ids, role = source["ids"], source["target_role"]
            values, accounting = validated_prefixes(runtime, ids, contract)
            totals["original_calls"] += 1
            for key, value in accounting.items():
                totals[key] += value
            positions = source["target_token_positions"]
            if (not positions or positions != sorted(set(positions)) or positions[-1] != len(ids) - 1
                    or any(type(p) is not int or not 0 <= p < len(ids) for p in positions)):
                raise ValueError("Target prefix positions are invalid/incomplete")
            endpoint = values[role][-1]
            native = [values[role][position][1] for position in positions]
            cuts, repeated = [], {}
            for view in source["text_cut_views"]:
                cut_ids = view["ids"]
                if not cut_ids:
                    raise ValueError("An empty real-text cut cannot be classified")
                if cut_ids == ids[:len(cut_ids)]:
                    score = values[role][len(cut_ids) - 1][1]
                    method = "complete_input_reuse" if cut_ids == ids else "exact_native_prefix_reuse"
                    totals["complete_input_cuts_reused" if cut_ids == ids else "exact_prefix_cuts_reused"] += 1
                elif tuple(cut_ids) in repeated:
                    score = repeated[tuple(cut_ids)]
                    method = "identical_retokenized_cut_reuse"
                    totals["duplicate_cut_calls_reused"] += 1
                else:
                    cut_values, accounting = validated_prefixes(runtime, cut_ids, contract)
                    score = cut_values[role][-1][1]
                    repeated[tuple(cut_ids)] = score
                    method = "independent_retokenized_cut"
                    totals["retokenized_cut_calls"] += 1
                    for key, value in accounting.items():
                        totals[key] += value
                cuts.append({"cut_chars": view["cut_chars"], "p_unsafe": score,
                             "native_tokens": len(cut_ids), "evaluation_method": method})
            if not cuts:
                raise ValueError("Frozen text-cut schedule is empty")
            native_max, cut_max = max(native), max(item["p_unsafe"] for item in cuts)
            row = {key: source[key] for key in ("sample_id", "family", "language", "target_role", "source_label")}
            row.update(endpoint_probs=endpoint, endpoint_p_unsafe=endpoint[1],
                       native_target_max_p_unsafe=native_max, target_token_positions=positions,
                       native_target_p_unsafe=native, text_cut_observations=cuts,
                       text_cut_max_p_unsafe=cut_max, stream_max_p_unsafe=max(native_max, cut_max),
                       original_endpoint_reused_for_complete_cut=True,
                       execution_definition=contract["name"], generated_tokens=0)
            output.append(row)
            if stream is not None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"event": "canonical_observations", "completed": index + 1,
                                  "requested": len(rows), "forward_tokens": totals["forward_tokens"]}), flush=True)
    finally:
        if stream is not None:
            stream.close()
    return output, totals


def verify_tokenizer(tokenizer, rows, trainer, tokenizer_root):
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}
    if versions != {"transformers": "5.17.0", "tokenizers": "0.23.2"}:
        raise RuntimeError("Canonical verification requires the frozen modern tokenizer versions")
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError("Implicit tokenizer truncation is forbidden")
    digest, views = hashlib.sha256(), 0
    for row in rows:
        text = trainer.serialize(row["messages"])
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        start = len(text) - len(row["messages"][-1]["content"])
        positions = [i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > start]
        if (encoded["input_ids"] != row["ids"] or positions != row["target_token_positions"]
                or start != row["target_content_start_char"]):
            raise ValueError("Modern IDs or target boundary differ: " + row["sample_id"])
        for kind, text, ids in trainer.view_inputs(row, "calibration"):
            trainer.check_encoding(tokenizer, text, ids)
            digest.update((trainer.canonical([row["sample_id"], kind, ids]) + "\n").encode())
            views += 1
    return {"status": "passed", "versions": versions, "tokenizer_class": type(tokenizer).__name__,
            "original_records": len(rows), "verified_original_and_cut_views": views,
            "verified_views_sha256": digest.hexdigest(), "replacement_ids_written": False,
            "tokenizer_assets_sha256": trainer.asset_hashes(tokenizer_root), "neural_forward_calls": 0}


def source_receipt(args):
    """Bind shared serving files separately from the experiment-only loader.

    Standalone bundles reimplement the model loader and require their own L20
    audit; these loader source bytes are not falsely claimed to be identical.
    """
    runtime, model, paths, third_party = {}, {}, {}, {}
    expected = {Path(name).stem: args.runtime_code / name for name in RUNTIME_FILES}
    expected["window_attention"] = args.window_code_dir / "window_attention.py"
    expected.update({"infer_classifier": args.round4_code / "infer_classifier.py",
                     "train_base": args.base_code_dir / "train_base.py",
                     "experiment_common": args.base_code_dir / "experiment_common.py",
                     "speed_probe": args.base_code_dir / "speed_probe.py",
                     "run_probe": args.window_code_dir / "run_probe.py"})
    for name, path in expected.items():
        module = sys.modules.get(name)
        if module is None or Path(module.__file__).resolve() != path.resolve():
            raise ValueError("Actual loaded dependency differs: " + name)
        target = runtime if name in {Path(n).stem for n in RUNTIME_FILES} | {"window_attention"} else model
        target[path.name] = sha(path)
        paths[path.name] = str(path.resolve())
    for name, module in sorted(sys.modules.items()):
        if not (name.startswith("transformers.models.qwen3_5") or name == "transformers.cache_utils"
                or name == "fla" or name.startswith("fla.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file():
            third_party[name] = {"path": str(Path(filename).resolve()), "sha256": sha(filename)}
    return {"runtime_source_sha256": runtime, "model_loader_source_sha256": model,
            "source_paths": paths, "third_party_source_sha256": third_party,
            "standalone_model_loader_byte_equivalence_claimed": False}


def sources_unchanged(before, after):
    for name in ("runtime_source_sha256", "model_loader_source_sha256", "source_paths"):
        if before[name] != after[name]:
            return False
    # FLA may lazily import additional kernels on the first actual call. Record
    # those as well, but never tolerate a previously loaded source changing.
    return all(after["third_party_source_sha256"].get(name) == record
               for name, record in before["third_party_source_sha256"].items())


def prepare_modules(args):
    # Preload shared runtime dependencies explicitly; never let old copies in
    # /work/round4 or a previously imported bundle substitute for them.
    for name in RUNTIME_FILES:
        load_local(Path(name).stem, args.runtime_code / name)
    for name, directory in (("train_base", args.base_code_dir), ("experiment_common", args.base_code_dir),
                            ("speed_probe", args.base_code_dir), ("run_probe", args.window_code_dir),
                            ("window_attention", args.window_code_dir)):
        path = directory / (name + ".py")
        if not path.is_file():
            raise FileNotFoundError(path)
        previous = sys.modules.get(name)
        if previous is not None and Path(previous.__file__).resolve() != path.resolve():
            raise RuntimeError("Refusing a previously loaded model helper: " + name)
    return load_local("infer_classifier", args.round4_code / "infer_classifier.py")


def load_runtime(args, checkpoint, loader):
    options = SimpleNamespace(base_root=args.base_root, base_code_dir=args.base_code_dir,
                              window_code_dir=args.window_code_dir, memory_code_dir=args.round4_code,
                              checkpoint=checkpoint)
    torch, model, tokenizer, _, kernels, device = loader.load_classifier(options, "window")
    if torch.cuda.device_count() != 1 or "L20" not in device:
        raise RuntimeError("Exactly one visible L20 is required")
    runtime = sys.modules["canonical_text_runtime"].CanonicalTextRuntime(
        model, tokenizer, inference_engine=args.inference_engine, pad_token_id=args.pad_token_id)
    validate_contract(runtime.engine_metadata["execution_contract"])
    return torch, model, tokenizer, runtime, {"device": device, "kernels": kernels,
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "tokenizers", "flash-linear-attention",
                              "fla-core", "safetensors", "triton")}}


def calibration_artifact(audit, metrics, cal, dev, *, digest, directory, args,
                         baseline, contract, receipts, metadata, sources, reused_from=None):
    # The selection-like names in frozen stream_metrics are retained as a
    # reproducible quality gate, but no is_better/selection operation is called.
    development = metrics.evaluate(cal, dev, baseline)
    thresholds = audit.fixed_thresholds(development)
    artifact = {"version": VERSION, "status": "completed", "checkpoint_sha256": digest,
        "execution_contract": contract, "threshold_comparison": ">", "thresholds": thresholds,
        "calibrated_strata": {mode: sorted(values) for mode, values in thresholds.items()},
        "thresholds_fitted_on": "calibration only", "selection_changed": False, "selection_read_test": False,
        "arbitrary_bpe_schedule_fpr_guarantee": False,
        "calibration_receipt": {"manifest_sha256": sha(args.data_root / "manifest.json"),
            "source_sha256": sha(args.data_root / "calibration.jsonl"),
            "predictions_sha256": sha(directory / "calibration_predictions.jsonl"),
            "records": len(cal), "unique_families": len({r["family"] for r in cal})},
        "development_receipt": {"source_sha256": sha(args.data_root / "dev.jsonl"),
            "predictions_sha256": sha(directory / "dev_predictions.jsonl"),
            "records": len(dev), "unique_families": len({r["family"] for r in dev})},
        "development_metrics": development, "canonical_quality_gate_pass": development["eligible"],
        "quality_gate_is_checkpoint_selection": False,
        "canonical_initial_whole_macro_recall": development["baseline_whole_macro_recall"],
        "engine_metadata": metadata, "observation_accounting": receipts,
        "observations_reused_from": reused_from, **sources,
        "evaluator_sha256": sha(__file__), "generated_tokens": 0,
        "limits": "Empirical FPR for the frozen native-token and at most eight real-text cuts per record. "
                  "No arbitrary BPE schedule guarantee, onset gold, English-user fallback, third-class "
                  "calibration or safe-prefix release claim. A failed gate still produces this artifact; "
                  "it never changes or promotes the fixed training-selected checkpoint."}
    return artifact


def common_inputs(args, audit):
    summary, checkpoint, _ = audit.completed_training(args.training_output)
    metrics = audit.exact_module("stream_metrics", args.code_root / "stream_metrics.py", audit.METRICS_SHA)
    trainer = audit.exact_module("train_prefix", args.code_root / "train_prefix.py", audit.TRAINER_SHA)
    manifest, data = trainer.load_data(args.data_root)
    binding = summary["binding"]
    if (sha(args.data_root / "manifest.json") != binding["manifest_sha256"]
            or sha(args.training_tokenizer_proof) != binding["proof_sha256"]
            or sha(args.round4_code / "train_risk.py") != binding["round4_helper_sha256"]):
        raise ValueError("Completed training data/tokenizer/helper binding changed")
    if (args.tokenizer_root.resolve() != (args.base_root / "models/qwen35").resolve()
            or args.tokenizer_root.resolve() != Path("/work/models/qwen35")):
        raise ValueError("Model and tokenizer assets must be the frozen /work/models/qwen35")
    trainer.verify_existing_proof(SimpleNamespace(data_root=args.data_root,
        tokenizer_root=args.tokenizer_root, proof=args.training_tokenizer_proof), data)
    if len(data["calibration"]) != 900 or len(data["dev"]) != 1200:
        raise ValueError("Frozen calibration/development coverage changed")
    if sha(audit.START_PATH) != audit.START_SHA:
        raise ValueError("Initial window checkpoint bytes changed")
    return summary, checkpoint, metrics, trainer, manifest, data


def run_calibration(args, audit, report):
    summary, checkpoint, metrics, trainer, _, data = common_inputs(args, audit)
    report.update(training_summary_sha256=sha(args.training_output / "summary.json"),
                  training_binding=summary["binding"], selected_checkpoint_sha256=summary["final_checkpoint_sha256"],
                  phase="canonical_calibration_and_development", initial_checkpoint_sha256=audit.START_SHA)
    audit.write_json(args.output / "audit.json", report)
    loader, reused, baseline, common_contract = prepare_modules(args), {}, None, None
    for name, path, digest in (("initial_window", audit.START_PATH, audit.START_SHA),
                               ("selected", checkpoint, summary["final_checkpoint_sha256"])):
        directory = args.output / name
        directory.mkdir()
        if sha(path) != digest:
            raise ValueError("Checkpoint changed before canonical observations")
        if digest in reused:
            cal, dev, counts, metadata, sources, token_proof = reused[digest]
            reused_from = "initial_window"
            for split, rows in (("calibration", cal), ("dev", dev)):
                audit.write_rows(directory / (split + "_predictions.jsonl"), rows)
        else:
            torch = model = runtime = None
            try:
                torch, model, tokenizer, runtime, device = load_runtime(args, path, loader)
                sources = source_receipt(args)
                metadata = runtime.engine_metadata
                token_proof = verify_tokenizer(tokenizer, data["calibration"] + data["dev"], trainer, args.tokenizer_root)
                audit.write_json(directory / "modern_tokenizer_proof.json", token_proof)
                cal, cal_counts = observations(runtime, data["calibration"],
                    prediction_path=directory / "calibration_predictions.jsonl")
                dev, dev_counts = observations(runtime, data["dev"],
                    prediction_path=directory / "dev_predictions.jsonl")
                counts = {"calibration": cal_counts, "development": dev_counts,
                          "engine_final_stats": runtime.stats(), "startup_included": False}
                final_sources = source_receipt(args)
                if not sources_unchanged(sources, final_sources) or sha(path) != digest:
                    raise ValueError("Runtime source or checkpoint changed during calibration")
                sources = final_sources
                report["device"] = device
                reused[digest] = (cal, dev, counts, metadata, sources, token_proof)
                reused_from = None
            finally:
                if runtime is not None:
                    runtime.close()
                del runtime, model
                gc.collect()
                if torch is not None:
                    torch.cuda.empty_cache()
        audit.write_json(directory / "modern_tokenizer_proof.json", token_proof)
        for split, observed in (("calibration", cal), ("dev", dev)):
            audit.validate_observations(observed, data[split], metrics)
        contract = metadata["execution_contract"]
        if common_contract is not None and contract != common_contract:
            raise ValueError("Initial and selected canonical execution contracts differ")
        common_contract = contract
        artifact = calibration_artifact(audit, metrics, cal, dev, digest=digest, directory=directory,
            args=args, baseline=baseline, contract=contract, receipts=counts, metadata=metadata,
            sources=sources, reused_from=reused_from)
        if baseline is None:
            baseline = artifact["development_metrics"]["whole"]["macro_recall"]
        artifact.update(training_summary_sha256=report["training_summary_sha256"],
                        training_binding=summary["binding"],
                        modern_tokenizer_proof_sha256=sha(directory / "modern_tokenizer_proof.json"),
                        initial_canonical_calibration_sha256=(sha(args.output / "initial_window/calibration.json")
                                                              if name == "selected" else None))
        audit.write_json(directory / "calibration.json", artifact)
        report["results"][name] = {"checkpoint_sha256": digest,
            "calibration_artifact": str(directory / "calibration.json"),
            "calibration_artifact_sha256": sha(directory / "calibration.json"),
            "canonical_quality_gate_pass": artifact["canonical_quality_gate_pass"],
            "development_metrics": artifact["development_metrics"], "observations_reused_from": reused_from}
        audit.write_json(args.output / "audit.json", report)
    report.update(status="completed", integrity_pass=True, phase="finished",
                  neural_checkpoints_evaluated=len(reused), selection_changed=False,
                  canonical_quality_gate_pass=report["results"]["selected"]["canonical_quality_gate_pass"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Fit canonical serving thresholds (default mode)")
    parser.add_argument("--training-output", type=Path, default=Path("/work/output/round5/prefix_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runtime-code", type=Path, default=Path(__file__).resolve().parent / "runtime")
    parser.add_argument("--tokenizer-root", type=Path, default=Path("/work/models/qwen35"))
    parser.add_argument("--training-tokenizer-proof", type=Path, default=Path("/work/round5/native_tokenizer_proof.json"))
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--base-root", type=Path, default=Path("/work"))
    parser.add_argument("--base-code-dir", type=Path, default=Path("/work/input"))
    parser.add_argument("--window-code-dir", type=Path, default=Path("/work/window"))
    parser.add_argument("--inference-engine", choices=("eager", "window_cuda_graph"), default="eager")
    parser.add_argument("--pad-token-id", type=int)
    parser.add_argument("--output", type=Path, default=Path("/work/output/round5/canonical32_calibration"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Output already exists; refusing silent recalibration/overwrite")
    audit = load_local("evaluate_final_l20", args.code_root / "evaluate_final_l20.py", EVALUATOR_SHA)
    args.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "mode": "calibrate", "results": {},
              "phase": "completed_training_gate", "selection_changed": False, "selection_read_test": False,
              "training_or_weights_modified": False, "generated_tokens": 0, "evaluator_sha256": sha(__file__)}
    started = time.monotonic()
    audit.write_json(args.output / "audit.json", report)
    try:
        run_calibration(args, audit, report)
    except Exception as error:
        report.update(status="failed", integrity_pass=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        audit.write_json(args.output / "audit.json", report)
    return 0 if report["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
