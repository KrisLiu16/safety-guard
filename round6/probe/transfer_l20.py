"""T014 L20 pass: does the Run A linear probe transfer beyond Run A? No training of the model.

The T004 probe (logistic regression on the assistant head's 1024-d input, frozen Round5 weights) reached
AUC 0.995 on Run A dev, but Run A is all luna text in one template style, so the probe could be reading
generator style. This pass scores every position of
  Run A calibration + dev          (probe_input_v1.jsonl, as T004),
  prefix_v2 calibration + dev      (Round5 data, as decision_rule v1),
  the 1,872 official Qwen3GuardTest sequences (observed only, never fitted on),
with both the current head (log p) and the probe (logit), through the same frozen canonical32 eager path.
The probe is refitted inside the job from the T004 features on the PVC with the recorded L2 strength and
must reproduce T004's calibration AUC, so it is the same probe, not a new one. Thresholds are fitted later
on CPU (analyze_transfer.py) on Run A calibration only.
Integrity as in T004: hook rows cover the record, and the risk head on the captured projection reproduces
the runtime's probabilities at every real position.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dump_runA_l20 import (CALIBRATOR_SHA, FIXED_CHECKPOINT_SHA, OFFICIAL_EVALUATOR_SHA, PROB_TOLERANCE,  # noqa: E402
                           load_pinned, log_row, sha, write_line)
from probe_common import position_classes, serialize  # noqa: E402

AUC_TOLERANCE = 1e-4  # multithreaded BLAS can move the refit slightly; larger means a different probe


def refit_probe(probe_dir):
    """Refit the T004 hidden-feature probe exactly (same data, L2 and seed) and check its calibration AUC."""
    import numpy as np
    from fit_probe import load, training_view
    from probe_common import auc, fit_logistic, logistic_scores

    recorded = json.loads((probe_dir / "probe_v1_results.json").read_text())["scorers"]["probe_hidden"]
    train, calibration = load(probe_dir, "train"), load(probe_dir, "calibration")
    keep, y = training_view(train)
    model = fit_logistic(train["hidden"][keep], y, l2=recorded["l2"], seed=0)
    cal_keep, cal_y = training_view(calibration)
    s = logistic_scores(calibration["hidden"][cal_keep], model)
    reproduced = auc(s[cal_y > 0], s[cal_y == 0])
    if abs(reproduced - recorded["calibration_auc"]) > AUC_TOLERANCE:
        raise ValueError(f"refitted probe differs from T004: calibration AUC {reproduced} vs {recorded['calibration_auc']}")
    w, b, mean, std = model
    return {"w": np.asarray(w, np.float32), "b": float(b), "mean": np.asarray(mean, np.float32),
            "std": np.asarray(std, np.float32), "l2": recorded["l2"], "calibration_auc": reproduced}


class ProbeForward:
    """One record through the runtime with the head hook; returns head probs and probe logits at real positions."""

    def __init__(self, torch, model, validated_prefixes, probe):
        self.torch, self.validated = torch, validated_prefixes
        self.head = model.heads["assistant"]
        device = next(self.head.parameters()).device
        self.w = torch.tensor(probe["w"], device=device)
        self.mean = torch.tensor(probe["mean"], device=device)
        self.std = torch.tensor(probe["std"], device=device)
        self.b = probe["b"]
        self.captured = []
        self.max_prob_diff = 0.0
        self.totals = collections.Counter()
        self.handle = self.head["projection"].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = inputs[0]
        self.captured.append((hidden.detach().reshape(-1, hidden.shape[-1]).float(),
                              output.detach().reshape(-1, output.shape[-1]).float()))

    def close(self):
        self.handle.remove()

    def __call__(self, ids):
        torch = self.torch
        self.captured.clear()
        values, accounting = self.validated(ids)
        if accounting["graph_calls"] != 0:
            raise ValueError("graph execution bypasses the feature hook")
        if not self.captured:
            raise ValueError("the feature hook never fired (engine bypasses model.heads)")
        hidden = torch.cat([h for h, _ in self.captured])
        projected = torch.cat([p for _, p in self.captured])
        n = len(ids)
        if hidden.shape[0] not in (accounting["native_tokens"], accounting["forward_tokens"]):
            raise ValueError(f"captured {hidden.shape[0]} rows for {n} real / {accounting['forward_tokens']} forward tokens")
        with torch.no_grad():
            recomputed = torch.softmax(self.head["risk"](projected[:n]).float(), dim=-1).cpu()
            # The probe was fitted on float16-stored features; round the same way before scoring.
            logits = (((hidden[:n].half().float() - self.mean) / self.std) @ self.w + self.b).cpu().tolist()
        runtime_probs = torch.tensor([[float(p) for p in row] for row in values["assistant"]])
        diff = float((recomputed - runtime_probs).abs().max())
        self.max_prob_diff = max(self.max_prob_diff, diff)
        if diff > PROB_TOLERANCE:
            raise ValueError(f"hook features do not reproduce runtime probabilities (max diff {diff:.2e})")
        self.totals["sequences"] += 1
        self.totals["native_tokens"] += accounting["native_tokens"]
        self.totals["forward_tokens"] += accounting["forward_tokens"]
        return values["assistant"], logits


def run_runA(forward, fast, hf_tokenizer, input_path, output, limit):
    rows = collections.defaultdict(list)
    with Path(input_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] in ("calibration", "dev"):
                rows[row["split"]].append(row)
    counts = collections.Counter()
    for split in ("calibration", "dev"):
        with gzip.open(output / f"runA_{split}.jsonl.gz", "wb") as out:
            for row in rows[split][:limit] if limit else rows[split]:
                text = serialize(row["messages"])
                content_start = len(text) - len(row["messages"][-1]["content"])
                encoding = fast.encode(text, add_special_tokens=False)
                ids = list(encoding.ids)
                if not 1 <= len(ids) <= 8192 or (callable(hf_tokenizer) and
                                                  list(hf_tokenizer(text, add_special_tokens=False)["input_ids"]) != ids):
                    counts["runA_skipped"] += 1
                    continue
                positions, classes = position_classes(encoding.offsets, content_start, row["label"],
                                                      row.get("onset_char"), row.get("onset_end_char"))
                if not positions:
                    counts["runA_skipped"] += 1
                    continue
                probs, logits = forward(ids)
                write_line(out, {"sample_id": row["sample_id"], "family": row["family"], "split": split,
                                 "language": row["language"], "label": row["label"], "slot": row["index"],
                                 "response_style": row["response_style"], "classes": classes,
                                 "logprobs": [log_row(probs[q]) for q in positions],
                                 "probe_logits": [round(logits[q], 5) for q in positions]})
                counts[f"runA_{split}"] += 1
    return counts


def run_prefix_v2(forward, data, output, limit):
    counts = collections.Counter()
    for split in ("calibration", "dev"):
        with gzip.open(output / f"prefix_v2_{split}.jsonl.gz", "wb") as out:
            for row in data[split][:limit] if limit else data[split]:
                if row["target_role"] != "assistant":
                    continue      # the probe reads the assistant head only
                probs, logits = forward(row["ids"])
                positions = row["target_token_positions"]
                write_line(out, {"sample_id": row["sample_id"], "family": row["family"], "language": row["language"],
                                 "target_role": row["target_role"], "source_label": row["source_label"], "split": split,
                                 "logprobs": [log_row(probs[p]) for p in positions],
                                 "probe_logits": [round(logits[p], 5) for p in positions]})
                counts[f"prefix_v2_{split}"] += 1
    return counts


def run_official(forward, official, official_data, output, limit):
    seen, counts = {}, collections.Counter()
    with gzip.open(output / "official_sequences.jsonl.gz", "wb") as seq_out, \
            gzip.open(output / "official_rows.jsonl.gz", "wb") as rows_out:
        for split in official.SPLITS:
            for row in official_data[split]:
                key = official.sequence_key(row)
                sequence_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                if key not in seen:
                    if limit and len(seen) >= limit:
                        continue
                    probs, logits = forward(row["ids"])
                    start = row["eval_start_index"]
                    write_line(seq_out, {"sequence_id": sequence_id, "input_tokens": len(row["ids"]),
                                         "eval_start_index": start,
                                         "logprobs": [log_row(p) for p in probs[start:]],
                                         "probe_logits": [round(v, 5) for v in logits[start:]]})
                    seen[key] = sequence_id
                write_line(rows_out, {"sample_id": row["sample_id"], "split": split, "row_index": row["row_index"],
                                      "unique_id": row["unique_id"], "label": row["label"], "sequence_id": seen[key]})
    counts["official_unique_sequences"] = len(seen)
    if not limit and len(seen) != official.EXPECTED_UNIQUE:
        raise ValueError(f"official unique-sequence coverage {len(seen)} differs from {official.EXPECTED_UNIQUE}")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round5-code", type=Path, default=Path("/work/round5"))
    parser.add_argument("--input", type=Path, default=Path("/work/round6/probe/input_v1/probe_input_v1.jsonl"))
    parser.add_argument("--probe-dir", type=Path, default=Path("/work/output/round6/probe_v1"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/probe_transfer_v1"))
    parser.add_argument("--limit", type=int, default=0, help="records per source, 0 = all (smoke runs only)")
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "kind": "round6_probe_transfer_v1",
              "script_sha256": sha(__file__), "input_sha256": sha(options.input), "training_performed": False,
              "official_used_for_fitting": False, "generated_tokens": 0, "limit": options.limit}
    began = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - began
        (options.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    forward = None
    try:
        probe = refit_probe(options.probe_dir)
        report.update(probe={"l2": probe["l2"], "calibration_auc_reproduced": probe["calibration_auc"],
                             "features": "hidden (1024-d assistant head input)"}, phase="loading_runtime")
        save()
        code = options.round5_code
        cal = load_pinned("canonical_calibration", code / "calibrate_canonical_l20.py", CALIBRATOR_SHA)
        audit = cal.load_local("evaluate_final_l20", code / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
        official = load_pinned("canonical_official_eval", code / "evaluate_canonical_official_l20.py", OFFICIAL_EVALUATOR_SHA)
        args = official.parse_args(["--output", str(options.output / "unused_official_output")])
        summary, checkpoint, _, trainer, _, data = cal.common_inputs(args, audit)
        if summary["final_checkpoint_sha256"] != FIXED_CHECKPOINT_SHA or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
            raise ValueError("fixed Round5 checkpoint differs")
        locked = audit.read_json(args.calibration_output / "selected/calibration.json")
        args.inference_engine = locked["engine_metadata"]["inference_engine"]
        args.pad_token_id = locked["execution_contract"]["pad_token_id"]
        contract = locked["execution_contract"]
        if args.inference_engine != "eager":
            raise ValueError("feature hooks need the eager engine; locked engine is " + str(args.inference_engine))
        official_data, official_receipt = official.load_official_inputs(args.data_dir, args.source_dir)
        from tokenizers import Tokenizer
        fast = Tokenizer.from_file(str(Path(args.tokenizer_root) / "tokenizer.json"))
        if fast.truncation is not None or fast.padding is not None:
            raise ValueError("tokenizer truncation/padding must be disabled")
        loader = cal.prepare_modules(args)
        torch, model, hf_tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        try:
            if runtime.engine_metadata["execution_contract"] != contract:
                raise ValueError("runtime execution contract differs from the locked calibration")
            report["official_native_ids"] = official.verify_native_ids(hf_tokenizer, official_data, trainer.serialize)
            report.update(device=device, official_input_receipt=official_receipt, execution_contract=contract,
                          sources_before=cal.source_receipt(args), phase="runA")
            save()
            forward = ProbeForward(torch, model, lambda ids: cal.validated_prefixes(runtime, ids, contract), probe)
            counts = run_runA(forward, fast, hf_tokenizer, options.input, options.output, options.limit)
            report.update(counts=dict(counts), max_prob_diff=forward.max_prob_diff, phase="prefix_v2")
            save()
            counts.update(run_prefix_v2(forward, data, options.output, options.limit))
            report.update(counts=dict(counts), max_prob_diff=forward.max_prob_diff, phase="official")
            save()
            counts.update(run_official(forward, official, official_data, options.output, options.limit))
            after = cal.source_receipt(args)
            if not cal.sources_unchanged(report["sources_before"], after) or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
                raise ValueError("runtime sources or weights changed during the pass")
            report.update(counts=dict(counts), totals=dict(forward.totals), max_prob_diff=forward.max_prob_diff,
                          phase="finished", status="completed", integrity_pass=True,
                          files={p.name: sha(p) for p in sorted(options.output.glob("*.jsonl.gz"))})
        finally:
            if forward is not None:
                forward.close()
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
