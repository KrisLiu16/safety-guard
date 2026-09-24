"""Dump per-token class log-probabilities of the fixed Round5 weights (no training, no thresholds).

Why: the Round5 official evaluation saved only the two-consecutive-argmax decision, so the
decision rule (temperature, threshold, debounce) cannot be studied offline. This script
re-runs the identical frozen canonical32 eager execution on
  - the 900 calibration and 1,200 dev records of prefix_v2 (target-role head, every target token),
  - the 1,872 unique official Qwen3GuardTest sequences (assistant head, every evaluated token),
and writes log(p) for [safe, unsafe, controversial] at each position. log(p) are valid logits
(softmax is shift-invariant), so temperature scaling can be applied offline.

Official rows are observed descriptively only: nothing here fits on or selects with them.
The loaded helpers are the SHA-pinned Round5 modules; the fixed checkpoint, tokenizer IDs,
engine and pad token are verified exactly as in the Round5 official evaluation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

CALIBRATOR_SHA = "835c335182e84698c3c6a7b90670a1c57ef74fabc86be3d5f95296f778193666"
OFFICIAL_EVALUATOR_SHA = "98e5b3a1f72425a267c6cfc9436fe9cff79876b0b45f349f9899ff222382be31"
FIXED_CHECKPOINT_SHA = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
FLOOR = 1e-12


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(name, path, expected):
    path = Path(path).resolve()
    if expected is not None and sha(path) != expected:
        raise ValueError(f"pinned helper changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def log_row(probs):
    return [round(math.log(max(float(p), FLOOR)), 6) for p in probs]


def write_line(handle, value):
    handle.write((json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round5-code", type=Path, default=Path("/work/round5"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/decision_rule_v1"))
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "kind": "round6_token_logprob_dump",
              "script_sha256": sha(__file__), "training_performed": False, "thresholds_fitted": False,
              "official_used_for_fitting": False, "generated_tokens": 0}
    began = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - began
        (options.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        code = options.round5_code
        cal = load_pinned("canonical_calibration", code / "calibrate_canonical_l20.py", CALIBRATOR_SHA)
        audit = cal.load_local("evaluate_final_l20", code / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
        official = load_pinned("canonical_official_eval", code / "evaluate_canonical_official_l20.py", OFFICIAL_EVALUATOR_SHA)
        # Same path defaults as the Round5 official evaluation; only the output differs.
        args = official.parse_args(["--output", str(options.output / "unused_official_output")])
        summary, checkpoint, _, trainer, _, data = cal.common_inputs(args, audit)
        if summary["final_checkpoint_sha256"] != FIXED_CHECKPOINT_SHA or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
            raise ValueError("fixed Round5 checkpoint differs")
        locked = audit.read_json(args.calibration_output / "selected/calibration.json")
        args.inference_engine = locked["engine_metadata"]["inference_engine"]
        args.pad_token_id = locked["execution_contract"]["pad_token_id"]
        contract = locked["execution_contract"]
        official_data, official_receipt = official.load_official_inputs(args.data_dir, args.source_dir)
        report.update(checkpoint_sha256=FIXED_CHECKPOINT_SHA, inference_engine=args.inference_engine,
                      execution_contract=contract, official_input_receipt=official_receipt,
                      locked_calibration_sha256=sha(args.calibration_output / "selected/calibration.json"),
                      phase="loading_runtime")
        save()
        loader = cal.prepare_modules(args)
        torch, model, tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        try:
            if runtime.engine_metadata["execution_contract"] != contract:
                raise ValueError("runtime execution contract differs from the locked calibration")
            report["official_native_ids"] = official.verify_native_ids(tokenizer, official_data, trainer.serialize)
            report.update(device=device, sources_before=cal.source_receipt(args), phase="prefix_v2_records")
            save()
            totals = {"sequences": 0, "native_tokens": 0, "forward_tokens": 0, "padding_tokens": 0}

            def forward(ids):
                values, accounting = cal.validated_prefixes(runtime, ids, contract)
                totals["sequences"] += 1
                for key in ("native_tokens", "forward_tokens", "padding_tokens"):
                    totals[key] += accounting[key]
                return values

            for split in ("calibration", "dev"):
                rows = data[split]
                with gzip.open(options.output / f"prefix_v2_{split}.jsonl.gz", "wb") as handle:
                    for index, row in enumerate(rows):
                        values = forward(row["ids"])
                        role, positions = row["target_role"], row["target_token_positions"]
                        write_line(handle, {
                            "sample_id": row["sample_id"], "family": row["family"], "language": row["language"],
                            "target_role": role, "source_label": row["source_label"], "split": split,
                            "input_tokens": len(row["ids"]), "target_token_positions": positions,
                            "logprobs": [log_row(values[role][p]) for p in positions]})
                        if (index + 1) % 300 == 0:
                            print(json.dumps({"split": split, "done": index + 1, "of": len(rows)}), flush=True)
                report.setdefault("records", {})[split] = len(rows)
                save()
            report["phase"] = "official_unique_sequences"
            save()
            seen = {}
            with gzip.open(options.output / "official_sequences.jsonl.gz", "wb") as handle, \
                    gzip.open(options.output / "official_rows.jsonl.gz", "wb") as rows_handle:
                for split in official.SPLITS:
                    for row in official_data[split]:
                        key = official.sequence_key(row)
                        sequence_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                        if key not in seen:
                            values = forward(row["ids"])
                            start = row["eval_start_index"]
                            write_line(handle, {"sequence_id": sequence_id, "input_tokens": len(row["ids"]),
                                                "eval_start_index": start,
                                                "logprobs": [log_row(p) for p in values["assistant"][start:]]})
                            seen[key] = sequence_id
                            if len(seen) % 200 == 0:
                                print(json.dumps({"official_unique": len(seen)}), flush=True)
                        write_line(rows_handle, {"sample_id": row["sample_id"], "split": split,
                                                 "row_index": row["row_index"], "unique_id": row["unique_id"],
                                                 "label": row["label"], "sequence_id": seen[key]})
            if len(seen) != official.EXPECTED_UNIQUE:
                raise ValueError("official unique-sequence coverage differs from 1872")
            after = cal.source_receipt(args)
            if not cal.sources_unchanged(report["sources_before"], after) or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
                raise ValueError("runtime sources or weights changed during the dump")
            report.update(totals=totals, official_unique_sequences=len(seen), phase="finished",
                          files={p.name: sha(p) for p in sorted(options.output.glob("*.jsonl.gz"))},
                          status="completed", integrity_pass=True)
        finally:
            runtime.close()
            del runtime, model
            torch.cuda.empty_cache()
    except Exception as error:
        report.update(status="failed", integrity_pass=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
