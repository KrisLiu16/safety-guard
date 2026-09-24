"""Apply already locked canonical32 thresholds once to the fresh 390 records.

This separate entrypoint does not change calibration code, weights or model
selection. All imports are CPU-safe; every neural call requires one L20.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
from pathlib import Path
import sys
import time
import traceback

CALIBRATOR_SHA = "835c335182e84698c3c6a7b90670a1c57ef74fabc86be3d5f95296f778193666"


def verify_fixed_development(artifact, observed, metrics, audit, baseline):
    """Recount fixed decisions, never fit/refit even the old cal thresholds."""
    saved = artifact["development_metrics"]
    thresholds = audit.fixed_thresholds(saved)
    if (artifact["thresholds"] != thresholds
            or artifact["calibrated_strata"] != {mode: sorted(rows) for mode, rows in thresholds.items()}):
        raise ValueError("Artifact thresholds differ from its fixed development decision thresholds")
    fixed = audit.evaluate_fixed(observed["dev"], thresholds, metrics)
    cal_rows = observed["calibration"]
    if ({r["family"] for r in cal_rows} & {r["family"] for r in observed["dev"]}
            or {r["sample_id"] for r in cal_rows} & {r["sample_id"] for r in observed["dev"]}):
        raise ValueError("Calibration/development family or ID leakage")
    for mode, field in audit.FIELDS.items():
        for stratum, counts in fixed[mode]["strata"].items():
            entry = saved[mode]["strata"][stratum]
            if any(entry.get(key) != value for key, value in counts.items()):
                raise ValueError("Fixed development confusion matrix is not reproducible")
            safe = [row for row in cal_rows if audit.episode_key(row) == stratum and row["source_label"] == "safe"]
            fp = sum(row[field] > thresholds[mode][stratum] for row in safe)
            if (not safe or entry["calibration_safe_n"] != len(safe)
                    or entry["calibration_safe_fp"] != fp or fp > math.floor(.05 * len(safe))):
                raise ValueError("Fixed calibration receipt violates its safe-event budget")
        for metric in ("macro_recall", "macro_fpr"):
            if fixed[mode][metric] != saved[mode][metric]:
                raise ValueError("Fixed development aggregate metric is not reproducible")
    base = fixed["whole"]["macro_recall"] if baseline is None else baseline
    gates = {"each_stream_stratum_fpr_at_most_5pct": all(r["fpr"] <= .05 for r in fixed["stream"]["strata"].values()),
             "whole_macro_recall_drop_at_most_2pp": fixed["whole"]["macro_recall"] >= base - .02 - 1e-12}
    if (saved["baseline_whole_macro_recall"] != base or saved["selection_gates"] != gates
            or saved["eligible"] != all(gates.values()) or artifact["canonical_quality_gate_pass"] != all(gates.values())
            or saved["selection_score"] != fixed["stream"]["macro_recall"]
            or artifact["canonical_initial_whole_macro_recall"] != base):
        raise ValueError("Fixed canonical gate no longer matches the calibrated initial baseline")
    return thresholds, base


def locked_calibrations(args, audit, cal, summary, data, metrics):
    """Verify existing cal/dev decisions and receipts before opening fresh labels."""
    root = args.calibration_output
    run = audit.read_json(root / "audit.json")
    if (run.get("status") != "completed" or run.get("integrity_pass") is not True
            or run.get("mode") != "calibrate" or run.get("selection_changed") is not False
            or run.get("selection_read_test") is not False
            or run.get("training_summary_sha256") != audit.sha(args.training_output / "summary.json")
            or run.get("selected_checkpoint_sha256") != summary["final_checkpoint_sha256"]
            or run.get("evaluator_sha256") != CALIBRATOR_SHA):
        raise ValueError("Canonical calibration is incomplete or belongs to another fixed selection")
    documents, baseline = {}, None
    for name, digest in (("initial_window", audit.START_SHA), ("selected", summary["final_checkpoint_sha256"])):
        directory = root / name
        path = directory / "calibration.json"
        if audit.sha(path) != run["results"][name]["calibration_artifact_sha256"]:
            raise ValueError("Locked canonical calibration artifact changed")
        artifact = audit.read_json(path)
        expected = {"version": cal.VERSION, "status": "completed", "checkpoint_sha256": digest,
                    "threshold_comparison": ">", "thresholds_fitted_on": "calibration only",
                    "selection_changed": False, "selection_read_test": False,
                    "arbitrary_bpe_schedule_fpr_guarantee": False,
                    "evaluator_sha256": CALIBRATOR_SHA,
                    "training_summary_sha256": audit.sha(args.training_output / "summary.json"),
                    "training_binding": summary["binding"]}
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise ValueError("Canonical artifact checkpoint/protocol binding changed")
        cal.validate_contract(artifact["execution_contract"])
        if artifact["engine_metadata"]["execution_contract"] != artifact["execution_contract"]:
            raise ValueError("Calibration engine metadata describes a different contract")
        if artifact["engine_metadata"]["inference_engine"] != args.inference_engine:
            raise ValueError("Fresh evaluation must use the calibrated engine; no unvalidated engine substitution")
        if artifact["calibration_receipt"]["manifest_sha256"] != audit.sha(args.data_root / "manifest.json"):
            raise ValueError("Calibration data manifest changed")
        observed = {}
        for split, receipt in (("calibration", "calibration_receipt"), ("dev", "development_receipt")):
            predictions = directory / (split + "_predictions.jsonl")
            saved = artifact[receipt]
            if (audit.sha(predictions) != saved["predictions_sha256"]
                    or audit.sha(args.data_root / (split + ".jsonl")) != saved["source_sha256"]):
                raise ValueError("Frozen calibration/development predictions or source changed")
            observed[split] = audit.read_rows(predictions)
            audit.validate_observations(observed[split], data[split], metrics)
            if (saved["records"] != len(observed[split])
                    or saved["unique_families"] != len({r["family"] for r in observed[split]})):
                raise ValueError("Calibration/development receipt coverage changed")
        thresholds, computed_baseline = verify_fixed_development(artifact, observed, metrics, audit, baseline)
        if name == "initial_window":
            baseline = computed_baseline
        elif (artifact["initial_canonical_calibration_sha256"] != documents["initial_window"]["artifact_sha256"]
              or artifact["execution_contract"] != documents["initial_window"]["artifact"]["execution_contract"]):
            raise ValueError("Selected canonical gate has another initial baseline or execution contract")
        documents[name] = {"artifact": artifact, "artifact_sha256": audit.sha(path),
                           "artifact_path": str(path.resolve()), "thresholds": thresholds,
                           "runtime_versions": run["device"]["runtime_versions"]}
    return documents, audit.sha(root / "audit.json")


def matching_sources(actual, artifact):
    for name in ("runtime_source_sha256", "model_loader_source_sha256"):
        if actual[name] != artifact[name]:
            raise ValueError("Fresh observations use different canonical/model implementation sources")
    for name, record in actual["third_party_source_sha256"].items():
        previous = artifact["third_party_source_sha256"].get(name)
        if previous is not None and previous["sha256"] != record["sha256"]:
            raise ValueError("Fresh third-party algorithm source changed: " + name)


def run(args, audit, cal, report):
    summary, checkpoint, metrics, trainer, _, data = cal.common_inputs(args, audit)
    documents, locked_audit_sha = locked_calibrations(args, audit, cal, summary, data, metrics)
    # No threshold fitting occurs after this point. Load/test labels only now.
    fresh_manifest, fresh, isolation = audit.load_fresh(args.fresh_root, data, trainer)
    proof = audit.verify_fresh_tokenizer(args, fresh, trainer)
    audit.write_json(args.output / "fresh_tokenizer_proof.json", proof)
    report.update(phase="fixed_canonical_fresh_observations", training_summary_sha256=audit.sha(args.training_output / "summary.json"),
                  training_binding=summary["binding"], family_isolation=isolation,
                  selected_checkpoint_sha256=summary["final_checkpoint_sha256"],
                  canonical_calibration_audit_sha256=locked_audit_sha,
                  fresh_tokenizer_proof_sha256=audit.sha(args.output / "fresh_tokenizer_proof.json"),
                  requested_coverage={"episodes": 390, "unique_families": 390, "strata": fresh_manifest["strata"]})
    audit.write_json(args.output / "audit.json", report)
    loader, reused = cal.prepare_modules(args), {}
    for name, path, digest in (("initial_window", audit.START_PATH, audit.START_SHA),
                               ("selected", checkpoint, summary["final_checkpoint_sha256"])):
        if audit.sha(path) != digest:
            raise ValueError("Fixed checkpoint changed before fresh evaluation")
        artifact = documents[name]["artifact"]
        directory = args.output / name
        directory.mkdir()
        if digest in reused:
            observed, accounting, metadata, sources, native_proof = reused[digest]
            reused_from = "initial_window"
            audit.write_rows(directory / "observations.jsonl", observed)
        else:
            torch = model = runtime = None
            try:
                torch, model, tokenizer, runtime, device = cal.load_runtime(args, path, loader)
                metadata, sources = runtime.engine_metadata, cal.source_receipt(args)
                if device["runtime_versions"] != documents[name]["runtime_versions"]:
                    raise ValueError("Fresh numerical library versions differ from locked canonical calibration")
                if metadata["execution_contract"] != artifact["execution_contract"]:
                    raise ValueError("Actual canonical execution does not match locked calibration")
                matching_sources(sources, artifact)
                native_proof = cal.verify_tokenizer(tokenizer, fresh, trainer, args.tokenizer_root)
                observed, accounting = cal.observations(runtime, fresh, prediction_path=directory / "observations.jsonl")
                accounting["engine_final_stats"] = runtime.stats()
                current_sources = cal.source_receipt(args)
                if not cal.sources_unchanged(sources, current_sources) or audit.sha(path) != digest:
                    raise ValueError("Runtime/checkpoint changed during fixed fresh evaluation")
                sources = current_sources
                matching_sources(sources, artifact)
                report["device"] = device
                reused[digest] = observed, accounting, metadata, sources, native_proof
                reused_from = None
            finally:
                if runtime is not None:
                    runtime.close()
                del runtime, model
                gc.collect()
                if torch is not None:
                    torch.cuda.empty_cache()
        audit.validate_observations(observed, fresh, metrics)
        thresholds = documents[name]["thresholds"]
        predictions = []
        for row in observed:
            stratum = audit.episode_key(row)
            predictions.append({**row, "checkpoint_sha256": digest, "status": "ok",
                "whole_threshold": thresholds["whole"][stratum], "stream_threshold": thresholds["stream"][stratum],
                "threshold_comparison": ">", "whole_unsafe": row["endpoint_p_unsafe"] > thresholds["whole"][stratum],
                "stream_unsafe": row["stream_max_p_unsafe"] > thresholds["stream"][stratum]})
        audit.write_rows(directory / "predictions.jsonl", predictions)
        audit.write_json(directory / "runtime_tokenizer_proof.json", native_proof)
        result = audit.evaluate_fixed(observed, thresholds, metrics)
        result.update(status="completed", checkpoint_sha256=digest,
            calibration_artifact_sha256=documents[name]["artifact_sha256"],
            calibration_artifact_path=documents[name]["artifact_path"],
            canonical_quality_gate_pass=artifact["canonical_quality_gate_pass"],
            observations_sha256=audit.sha(directory / "observations.jsonl"),
            predictions_sha256=audit.sha(directory / "predictions.jsonl"),
            coverage={"requested": 390, "evaluated": 390, "excluded": {}, "unique_families": 390},
            observations_reused_from=reused_from, observation_accounting=accounting,
            engine_metadata=metadata, **sources)
        audit.write_json(directory / "metrics.json", result)
        report["results"][name] = result
        audit.write_json(args.output / "audit.json", report)
    if (audit.sha(args.calibration_output / "audit.json") != locked_audit_sha
            or any(audit.sha(Path(record["artifact_path"])) != record["artifact_sha256"] for record in documents.values())):
        raise ValueError("Locked calibration changed during fresh evaluation")
    left, right = report["results"]["initial_window"], report["results"]["selected"]
    report["selected_minus_initial"] = {
        mode: {field: right[mode][field] - left[mode][field]
               for field in ("macro_recall", "macro_fpr", "macro_f1")} for mode in audit.FIELDS}
    report.update(status="completed", integrity_pass=True, phase="finished", neural_checkpoints_evaluated=len(reused),
                  quality_acceptance="Fresh is descriptive once-only holdout evidence; it never promotes weights or changes thresholds.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-fresh", action="store_true", help="Default fixed-threshold evaluation mode")
    parser.add_argument("--training-output", type=Path, default=Path("/work/output/round5/prefix_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--fresh-root", type=Path, default=Path("/work/round5/data/fresh_holdout_v1"))
    parser.add_argument("--calibration-output", type=Path, default=Path("/work/output/round5/canonical32_calibration"))
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
    parser.add_argument("--output", type=Path, default=Path("/work/output/round5/canonical32_fresh390"))
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("canonical_calibration", args.code_root / "calibrate_canonical_l20.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    if cal.sha(args.code_root / "calibrate_canonical_l20.py") != CALIBRATOR_SHA:
        raise ValueError("The reviewed calibration module changed")
    audit = cal.load_local("evaluate_final_l20", args.code_root / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
    if args.output.exists():
        raise FileExistsError("Fresh output exists; refusing silent repeat evaluation/overwrite")
    args.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "mode": "canonical_fresh390", "results": {},
              "phase": "completed_training_and_locked_calibration_gate", "selection_changed": False,
              "thresholds_refitted_on_fresh": False, "selection_uses_fresh": False, "selection_read_test": False,
              "training_or_weights_modified": False, "generated_tokens": 0, "evaluator_sha256": cal.sha(__file__),
              "calibrator_sha256": CALIBRATOR_SHA, "fresh_sha256": audit.FRESH_SHA,
              "fresh_manifest_sha256": audit.FRESH_MANIFEST_SHA}
    began = time.monotonic()
    audit.write_json(args.output / "audit.json", report)
    try:
        run(args, audit, cal, report)
    except Exception as error:
        report.update(status="failed", integrity_pass=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        audit.write_json(args.output / "audit.json", report)
    return 0 if report["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
