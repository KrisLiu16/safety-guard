"""One fixed fresh-holdout evaluation, after Round5 training is complete.

Defaults:
  --training-output /work/output/round5/prefix_v2
  --fresh-root /work/round5/data/fresh_holdout_v1
  --output /work/output/round5/fresh390_final

Imports are CPU-safe. Only main's formal execution imports torch/model helpers;
all neural calls require exactly one visible L20. This script never selects a
checkpoint or fits a threshold on the fresh data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

START_SHA = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
START_PATH = Path("/work/output/round4/window/best.safetensors")
TRAINER_SHA = "1aea85c2c1228aba6066c915922e790c7fb56cc26c4ef7d21329508937c13e7c"
METRICS_SHA = "20a803f17d846b729a64dc4f0d6ad28b22d714117197854755a47bd3f163faff"
FRESH_SHA = "f571f2a93c0c5352a7835e7b522dffd48139063eee3daa229efea409c6a56cd1"
FRESH_MANIFEST_SHA = "e694f938ae3298f631242a50ac0559bc109a148dbdc7c6fdb71c5adc781f1838"
STRATA = ("en/assistant", "zh/assistant", "zh/user")
VERSIONS = {"transformers": "5.17.0", "tokenizers": "0.23.2"}
FIELDS = {"whole": "endpoint_p_unsafe", "stream": "stream_max_p_unsafe"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def exact_module(name, path, expected_sha):
    path = Path(path).resolve()
    if sha(path) != expected_sha:
        raise ValueError("Frozen module SHA changed: " + name)
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError("Refusing cached module from a different directory: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid finite probability/threshold")
    return float(value)


def episode_key(row):
    return row["language"] + "/" + row["target_role"]


def validate_observations(observed, expected, metrics):
    """Bind every saved/result row to its exact source and prefix schedule."""
    metrics.check_rows(observed)
    lookup = {row["sample_id"]: row for row in expected}
    if len(lookup) != len(expected) or len(observed) != len(expected):
        raise ValueError("Observation coverage differs from the frozen source")
    if {row["sample_id"] for row in observed} != set(lookup):
        raise ValueError("Observation IDs differ from the frozen source")
    for row in observed:
        source = lookup[row["sample_id"]]
        for key in ("family", "language", "target_role", "source_label"):
            if row[key] != source[key]:
                raise ValueError("Observation/source mismatch: " + key)
        positions = source["target_token_positions"]
        if row["target_token_positions"] != positions:
            raise ValueError("Native target-position coverage changed")
        scores = row["native_target_p_unsafe"]
        if len(scores) != len(positions) or not scores:
            raise ValueError("Native prefix scores are incomplete")
        scores = [probability(v) for v in scores]
        if max(scores) != row["native_target_max_p_unsafe"]:
            raise ValueError("Saved native maximum is not the trajectory maximum")
        endpoint_slot = positions.index(len(source["ids"]) - 1)
        if scores[endpoint_slot] != row["endpoint_p_unsafe"]:
            raise ValueError("Endpoint does not match its native trajectory position")
        probs = row["endpoint_probs"]
        if len(probs) != 3 or abs(sum(probability(v) for v in probs) - 1) > 1e-4:
            raise ValueError("Invalid endpoint class probabilities")
        if probs[1] != row["endpoint_p_unsafe"]:
            raise ValueError("Endpoint unsafe score differs from three-class result")
        cuts = row["text_cut_observations"]
        source_cuts = source["text_cut_views"]
        if [item["cut_chars"] for item in cuts] != [item["cut_chars"] for item in source_cuts]:
            raise ValueError("Frozen real-text cut coverage changed")
        if not cuts or max(probability(item["p_unsafe"]) for item in cuts) != row["text_cut_max_p_unsafe"]:
            raise ValueError("Text-cut maximum is not the saved cut maximum")
        for item, view in zip(cuts, source_cuts):
            if view["ids"] == source["ids"] and item["p_unsafe"] != row["endpoint_p_unsafe"]:
                raise ValueError("Complete cut must reuse the same original endpoint score")


def fixed_thresholds(saved_metrics):
    result = {}
    for mode in FIELDS:
        entries = saved_metrics[mode]["strata"]
        if set(entries) != set(STRATA):
            raise ValueError("The fixed calibrated strata changed")
        result[mode] = {}
        for stratum in STRATA:
            if entries[stratum]["comparison"] != ">":
                raise ValueError("Threshold operator must remain strict >")
            result[mode][stratum] = probability(entries[stratum]["threshold"])
    return result


def evaluate_fixed(observed, thresholds, metrics):
    """Only apply supplied thresholds. Never call evaluate()/threshold() here."""
    metrics.check_rows(observed)
    if {episode_key(row) for row in observed} != set(STRATA):
        raise ValueError("Fresh coverage must include the three fixed strata")
    result = {}
    for mode, field in FIELDS.items():
        if set(thresholds[mode]) != set(STRATA):
            raise ValueError("Missing fixed threshold; no stratum fallback")
        details = {}
        for stratum in STRATA:
            cutoff = probability(thresholds[mode][stratum])
            group = [row for row in observed if episode_key(row) == stratum]
            details[stratum] = metrics.confusion(group, field, cutoff)
        result[mode] = {
            "strata": details,
            "macro_recall": sum(row["recall"] for row in details.values()) / len(details),
            "macro_fpr": sum(row["fpr"] for row in details.values()) / len(details),
            "macro_f1": sum(row["f1"] for row in details.values()) / len(details),
        }
    return {**result, "n": len(observed), "unique_families": len({row["family"] for row in observed}),
            "thresholds": thresholds, "comparison": ">",
            "thresholds_fitted_on_fresh": False, "selection_uses_fresh": False,
            "metric_scope": "One source record per event; max over its native target and frozen text-cut scores. "
                            "No arbitrary BPE-arrival FPR guarantee or natural earliest-onset annotation.",
            "generated_tokens": 0}


def selected_evaluation_directory(training_root, sft, rl):
    root = Path(training_root).resolve()
    selected = rl["selected"]
    if selected["stage"] == "classification_rl":
        step = selected["step"]
        if step not in (64, 128, 192, 256):
            raise ValueError("Selected RL step was not a declared evaluation point")
        source = root / "classification_rl" / ("step_" + str(step) + ".safetensors")
        directory = root / "classification_rl" / ("evaluation_" + str(step))
    elif selected["stage"] == "sft_selected":
        if selected["step"] != 0 or rl["checkpoint_sha256"] != sft["checkpoint_sha256"]:
            raise ValueError("Inherited SFT checkpoint identity changed")
        if Path(selected["source_checkpoint"]).resolve() != root / "sft/best.safetensors":
            raise ValueError("RL inheritance points to another run")
        chosen = sft["selected"]
        if chosen["stage"] == "initial":
            if chosen["step"] != 0 or chosen["checkpoint_sha256"] != START_SHA:
                raise ValueError("Initial SFT selection identity changed")
            source, directory = START_PATH, root / "sft/evaluation_0"
        elif chosen["stage"] == "sft":
            match = re.fullmatch(r"epoch_([12])\.safetensors", Path(chosen["source_checkpoint"]).name)
            if not match:
                raise ValueError("Selected SFT source is not a declared epoch export")
            epoch = int(match.group(1))
            if chosen["step"] != epoch * 2082:
                raise ValueError("Selected SFT update does not match its epoch")
            source = root / "sft" / ("epoch_" + str(epoch) + ".safetensors")
            directory = root / "sft" / ("evaluation_" + str(epoch))
        else:
            raise ValueError("Unknown SFT selected stage")
        if Path(chosen["source_checkpoint"]).resolve() != source.resolve():
            raise ValueError("Selected SFT export path points to another run")
        if chosen["checkpoint_sha256"] != rl["checkpoint_sha256"]:
            raise ValueError("Selected SFT export hash differs from final hash")
    else:
        raise ValueError("Unknown final selected stage")
    if selected["stage"] == "classification_rl" and Path(selected["source_checkpoint"]).resolve() != source:
        raise ValueError("Selected RL source points to another run")
    if sha(source) != rl["checkpoint_sha256"]:
        raise ValueError("Selected source export bytes differ from the final checkpoint")
    return directory


def completed_training(root):
    root = Path(root).resolve()
    summary = read_json(root / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("Training is not completed; fresh holdout cannot be evaluated")
    sft, rl = read_json(root / "sft/summary.json"), read_json(root / "classification_rl/summary.json")
    if sft != summary["sft"] or rl != summary["classification_rl"]:
        raise ValueError("Final summary and separate stage summaries disagree")
    if (sft.get("status") != "completed" or rl.get("status") != "completed"
            or sft.get("steps") != 4164 or sft.get("epochs") != 2 or rl.get("updates") != 256):
        raise ValueError("The declared single SFT/RL run is incomplete")
    binding = summary["binding"]
    if sft["binding"] != binding or rl["binding"] != binding:
        raise ValueError("Stage/run input bindings differ")
    specification = read_json(root / "run_spec.json")
    if specification["binding"] != binding or specification.get("smoke_only"):
        raise ValueError("Run is a smoke run or its fixed binding differs")
    if binding["trainer_sha256"] != TRAINER_SHA or binding["metrics_sha256"] != METRICS_SHA:
        raise ValueError("This evaluator only accepts the frozen reviewed trainer/metrics")
    if binding["initial_checkpoint_sha256"] != START_SHA or binding["window"] != 512:
        raise ValueError("Training changed the fixed W512 initial checkpoint")
    if summary.get("selection_read_test") is not False or summary.get("generated_tokens") != 0:
        raise ValueError("Training summary does not preserve holdout separation")
    checkpoint = root / "classification_rl/best.safetensors"
    if Path(summary["final_checkpoint"]).resolve() != checkpoint:
        raise ValueError("Final checkpoint path points outside this completed run")
    digest = sha(checkpoint)
    if not (digest == summary["final_checkpoint_sha256"] == rl["checkpoint_sha256"]
            == rl["selected"]["checkpoint_sha256"] == rl["best_metrics"]["checkpoint_sha256"]):
        raise ValueError("Final checkpoint/SHA/best metrics do not describe the same weights")
    if sha(root / "sft/best.safetensors") != sft["checkpoint_sha256"]:
        raise ValueError("Selected SFT reference bytes changed")
    if not (rl["reference_checkpoint_sha256"] == sft["checkpoint_sha256"]
            == sft["selected"]["checkpoint_sha256"] == sft["best_metrics"]["checkpoint_sha256"]):
        raise ValueError("SFT selection and fixed RL reference do not have the same identity")
    directory = selected_evaluation_directory(root, sft, rl)
    if read_json(directory / "metrics.json") != rl["best_metrics"]:
        raise ValueError("Selected evaluation artifact differs from fixed best_metrics")
    return summary, checkpoint, directory


def calibration_receipt(directory, expected_sha, data, metrics, baseline):
    saved = read_json(directory / "metrics.json")
    if (saved["checkpoint_sha256"] != expected_sha or saved["selection_read_test"] is not False
            or saved["evaluation_weights"] != "exported_bfloat16_backbone" or saved["generated_tokens"] != 0):
        raise ValueError("Calibration metrics belong to another checkpoint/protocol")
    cal = read_rows(directory / "calibration_predictions.jsonl")
    dev = read_rows(directory / "dev_predictions.jsonl")
    validate_observations(cal, data["calibration"], metrics)
    validate_observations(dev, data["dev"], metrics)
    recomputed = metrics.evaluate(cal, dev, baseline)
    if any(saved.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Saved calibration/development metrics cannot be reproduced")
    thresholds = fixed_thresholds(saved)
    return {"evaluation_directory": str(directory.resolve()), "checkpoint_sha256": expected_sha,
            "metrics_sha256": sha(directory / "metrics.json"),
            "calibration_predictions_sha256": sha(directory / "calibration_predictions.jsonl"),
            "development_predictions_sha256": sha(directory / "dev_predictions.jsonl"),
            "calibration_records": len(cal), "development_records": len(dev),
            "calibration_unique_families": len({r["family"] for r in cal}),
            "development_unique_families": len({r["family"] for r in dev}),
            "thresholds": thresholds, "comparison": ">", "metrics_independently_recomputed": True,
            "thresholds_refitted_on_fresh": False}


def load_fresh(root, data, trainer):
    root = Path(root)
    manifest_path, path = root / "manifest.json", root / "episodes.jsonl"
    if sha(manifest_path) != FRESH_MANIFEST_SHA or sha(path) != FRESH_SHA:
        raise ValueError("The once-frozen fresh holdout changed")
    manifest, rows = read_json(manifest_path), read_rows(path)
    if (manifest["episodes"] != 390 or manifest["unique_families"] != 390
            or manifest["episode_sha256"] != FRESH_SHA
            or any(manifest[field] is not False for field in ("training_use", "calibration_use", "selection_use"))
            or manifest["neural_calls_during_preparation"] != 0):
        raise ValueError("Fresh provenance/usage contract changed")
    if len(rows) != 390 or len({r["family"] for r in rows}) != 390 or len({r["sample_id"] for r in rows}) != 390:
        raise ValueError("Fresh ID/family coverage is incomplete or duplicated")
    counts = Counter(episode_key(r) + "/" + r["source_label"] for r in rows)
    expected = {key + "/" + label: 65 for key in STRATA for label in ("safe", "unsafe")}
    if dict(counts) != expected or manifest["strata"] != expected:
        raise ValueError("Fresh stratum/label quota changed")
    all_ids, all_families = set(), set()
    for split in ("train", "calibration", "dev"):
        all_ids.update(r["sample_id"] for r in data[split])
        all_families.update(r["family"] for r in data[split])
    for row in rows:
        if row["sample_id"] in all_ids or row["family"] in all_families:
            raise ValueError("Fresh family or sample ID leaks into training/calibration/development")
        if (row["base_split"] != "fresh_holdout" or row["split_pool"] != "test"
                or row["label_tier"] != "public_source_reference"
                or row["messages"][-1]["role"] != row["target_role"] or not trainer.valid_ids(row["ids"])
                or trainer.ids_sha(row["ids"]) != row["original_ids_sha256"]):
            raise ValueError("Fresh original view identity/role is invalid")
        cuts = row["text_cut_views"]
        length = len(row["messages"][-1]["content"])
        expected_cuts = sorted({max(1, min(length, n)) for n in
                               (1, 2, length // 4, length // 2, 3 * length // 4,
                                length - 2, length - 1, length)}) if length else []
        if not cuts or [cut["cut_chars"] for cut in cuts] != expected_cuts:
            raise ValueError("Fresh text-cut schedule is incomplete")
        for cut in cuts:
            if not trainer.valid_ids(cut["ids"]) or trainer.ids_sha(cut["ids"]) != cut["ids_sha256"]:
                raise ValueError("Fresh text-cut IDs changed")
    return manifest, rows, {
        "pass": True, "verified_in_this_process_against": ["prefix_v2/train", "prefix_v2/calibration", "prefix_v2/dev"],
        "sample_id_overlap": 0, "family_overlap": 0, "fresh_unique_families": 390,
        "historical_exclusion_inputs": manifest["excluded_input_hashes"],
        "historical_files_recomputed_in_this_process": False,
        "historical_basis": "Pinned preparation manifest plus independent local CPU audit before evaluation; "
                            "the 21 historical inputs are not assumed mounted in this GPU job.",
        "text_exclusion_limit": "Preparation excluded normalized exact strings of length >=20 characters. "
                                "Shorter strings were exempt; local CPU audit found 26 short-message matches. "
                                "No claim of zero text overlap at every length or semantic decontamination.",
    }


def verify_fresh_tokenizer(args, rows, trainer):
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError("Fresh verification requires the same modern tokenizer versions")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True, trust_remote_code=False)
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError("Truncated tokenizer cannot verify fresh inputs")
    verified, digest = 0, hashlib.sha256()
    for row in rows:
        text = trainer.serialize(row["messages"])
        actual = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        content_start = len(text) - len(row["messages"][-1]["content"])
        positions = [i for i, (_, end) in enumerate(actual["offset_mapping"]) if end > content_start]
        if (actual["input_ids"] != row["ids"] or positions != row["target_token_positions"]
                or row["target_content_start_char"] != content_start):
            raise ValueError("Actual modern original IDs/target boundary mismatch: " + row["sample_id"])
        for kind, text, ids in trainer.view_inputs(row, "fresh_holdout"):
            trainer.check_encoding(tokenizer, text, ids)
            digest.update((trainer.canonical([row["sample_id"], kind, ids]) + "\n").encode())
            verified += 1
    return {"status": "passed", "versions": versions, "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_assets_sha256": trainer.asset_hashes(args.tokenizer_root),
            "original_records": 390, "verified_original_and_cut_views": verified,
            "verified_views_sha256": digest.hexdigest(), "mismatched_views": 0,
            "replacement_ids_written": False, "neural_forward_calls": 0}


def run(args):
    if args.output.exists():
        raise FileExistsError("Final holdout output already exists; refusing silent re-evaluation/overwrite")
    report = {"status": "running", "integrity_pass": False, "phase": "training_completion_gate",
              "fresh_sha256": FRESH_SHA, "fresh_manifest_sha256": FRESH_MANIFEST_SHA,
              "selection_uses_fresh": False, "thresholds_refitted_on_fresh": False,
              "training_or_weights_modified": False, "generated_tokens": 0,
              "evaluator_sha256": sha(__file__), "results": {}}
    args.output.mkdir(parents=True)
    write_json(args.output / "audit.json", report)
    began, torch = time.monotonic(), None
    try:
        summary, final_checkpoint, selected_directory = completed_training(args.training_output)
        metrics = exact_module("stream_metrics", args.code_root / "stream_metrics.py", METRICS_SHA)
        trainer = exact_module("train_prefix", args.code_root / "train_prefix.py", TRAINER_SHA)
        manifest, data = trainer.load_data(args.data_root)
        binding = summary["binding"]
        if sha(args.data_root / "manifest.json") != binding["manifest_sha256"]:
            raise ValueError("Evaluation prefix inputs differ from the completed run")
        if args.tokenizer_root.resolve() != Path("/work/models/qwen35"):
            raise ValueError("Model helper assets are fixed at /work/models/qwen35")
        if sha(args.training_tokenizer_proof) != binding["proof_sha256"]:
            raise ValueError("Completed training's tokenizer proof changed")
        proof_args = SimpleNamespace(data_root=args.data_root, tokenizer_root=args.tokenizer_root,
                                     proof=args.training_tokenizer_proof)
        trainer.verify_existing_proof(proof_args, data)
        baseline_directory = args.training_output / "sft/evaluation_0"
        baseline_metrics = read_json(baseline_directory / "metrics.json")
        fixed_baseline = baseline_metrics["whole"]["macro_recall"]
        if (summary["sft"]["baseline_whole_macro_recall"] != fixed_baseline
                or summary["classification_rl"]["baseline_whole_macro_recall"] != fixed_baseline):
            raise ValueError("Initial-window retention baseline changed")
        if sha(START_PATH) != START_SHA:
            raise ValueError("Initial window checkpoint bytes changed")
        receipts = {
            "initial_window": calibration_receipt(baseline_directory, START_SHA, data, metrics, fixed_baseline),
            "selected": calibration_receipt(selected_directory, summary["final_checkpoint_sha256"], data,
                                             metrics, fixed_baseline),
        }
        fresh_manifest, fresh, isolation = load_fresh(args.fresh_root, data, trainer)
        report.update(phase="modern_fresh_input_verification", training_summary_sha256=sha(args.training_output / "summary.json"),
                      training_binding=binding, selected_checkpoint_sha256=summary["final_checkpoint_sha256"],
                      selected_development_eligible=summary["classification_rl"]["best_metrics"]["eligible"],
                      selected_development_metrics=summary["classification_rl"]["best_metrics"],
                      calibration_receipts=receipts, family_isolation=isolation,
                      requested_coverage={"episodes": 390, "unique_families": 390, "strata": fresh_manifest["strata"]})
        write_json(args.output / "audit.json", report)
        tokenizer_proof = verify_fresh_tokenizer(args, fresh, trainer)
        write_json(args.output / "fresh_tokenizer_proof.json", tokenizer_proof)
        report["fresh_tokenizer_proof_sha256"] = sha(args.output / "fresh_tokenizer_proof.json")
        if sha(args.round4_code / "train_risk.py") != binding["round4_helper_sha256"]:
            raise ValueError("Frozen Round4 neural helper changed")
        torch, helper, load_file = trainer.import_training(SimpleNamespace(round4_code=args.round4_code))
        if sha(helper.__file__) != binding["round4_helper_sha256"]:
            raise ValueError("Imported neural helper is not the completed run's helper")
        report.update(device=torch.cuda.get_device_name(0), phase="fixed_fresh_observations",
                      visible_cuda_devices=torch.cuda.device_count(),
                      runtime_versions={name: importlib.metadata.version(name) for name in
                                        ("torch", "transformers", "tokenizers", "flash-linear-attention",
                                         "fla-core", "safetensors", "triton")})
        if torch.cuda.device_count() != 1 or "L20" not in report["device"]:
            raise RuntimeError("Exactly one visible L20 is required")
        write_json(args.output / "audit.json", report)
        reused = {}
        for name, checkpoint, digest in (
                ("initial_window", START_PATH, START_SHA),
                ("selected", final_checkpoint, summary["final_checkpoint_sha256"])):
            if sha(checkpoint) != digest:
                raise ValueError("Checkpoint changed between binding and execution")
            if digest in reused:
                observed, reused_from = reused[digest]
            else:
                model, tokenizer = trainer.make_model(helper, load_file, checkpoint)
                model.eval().requires_grad_(False)
                observed = trainer.observations(torch, helper, model, tokenizer, fresh)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                reused[digest] = (observed, name)
                reused_from = None
            validate_observations(observed, fresh, metrics)
            directory = args.output / name
            directory.mkdir()
            predictions = []
            for row in observed:
                stratum = episode_key(row)
                thresholds = receipts[name]["thresholds"]
                predictions.append({**row,
                    "whole_threshold": thresholds["whole"][stratum],
                    "stream_threshold": thresholds["stream"][stratum],
                    "threshold_comparison": ">",
                    "whole_unsafe": row["endpoint_p_unsafe"] > thresholds["whole"][stratum],
                    "stream_unsafe": row["stream_max_p_unsafe"] > thresholds["stream"][stratum],
                    "checkpoint_sha256": digest, "status": "ok", "generated_tokens": 0})
            write_rows(directory / "predictions.jsonl", predictions)
            result = evaluate_fixed(observed, receipts[name]["thresholds"], metrics)
            result.update(status="completed", checkpoint_sha256=digest,
                          calibration_receipt=receipts[name], predictions_sha256=sha(directory / "predictions.jsonl"),
                          coverage={"requested": 390, "evaluated": 390, "excluded": {}, "unique_families": 390},
                          observations_reused_from=reused_from)
            write_json(directory / "metrics.json", result)
            report["results"][name] = result
            write_json(args.output / "audit.json", report)
        left, right = report["results"]["initial_window"], report["results"]["selected"]
        report["selected_minus_initial"] = {
            mode: {field: right[mode][field] - left[mode][field]
                   for field in ("macro_recall", "macro_fpr", "macro_f1")} for mode in FIELDS}
        report.update(status="completed", integrity_pass=True, phase="finished",
                      quality_acceptance="No checkpoint promotion or threshold change is made from this holdout.",
                      neural_checkpoints_evaluated=len(reused),
                      limits="390 records/390 unique recorded families, 65 per stratum/label cell. "
                             "Source-policy labels, incomplete historical/foundation exposure knowledge. "
                             "No onset localization, third-class calibration or independent English-user stratum.")
    except Exception as error:
        report.update(status="failed", integrity_pass=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        write_json(args.output / "audit.json", report)
    return 0 if report["integrity_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output", type=Path, default=Path("/work/output/round5/prefix_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--fresh-root", type=Path, default=Path("/work/round5/data/fresh_holdout_v1"))
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tokenizer-root", type=Path, default=Path("/work/models/qwen35"))
    parser.add_argument("--training-tokenizer-proof", type=Path, default=Path("/work/round5/native_tokenizer_proof.json"))
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round5/fresh390_final"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
