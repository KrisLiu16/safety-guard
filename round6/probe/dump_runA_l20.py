"""T004/T005 L20 dump: per-token class log-probs and frozen head features of the fixed Round5 weights on Run A.

No training and no thresholds. For every record of probe_input_v1.jsonl (Run A dev + calibration +
a train sample, all four 2x2 responses per word) this runs the identical frozen canonical32 eager
execution used by round6/decision_rule and writes
  runA_<split>.jsonl.gz   log(p) [safe, unsafe, controversial] at every assistant target position,
                          plus the position classes (probe_common: S / P / O / U);
  features_<split>.npz    at up to 24 positions per record: the assistant head's input hidden state
                          (1024-d) and its 512-d projection, captured by a forward hook, with the
                          head's own p(unsafe) at the same positions.
Integrity: the captured rows must cover exactly the real (or real + padding) tokens of the record,
and the risk head re-applied to the captured projection must reproduce the runtime's probabilities;
otherwise the job stops at the first record instead of producing misaligned features.
Model loading reuses the SHA-pinned Round5 helpers exactly as decision_rule/dump_token_logprobs_l20.py.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_common import feature_positions, position_classes, serialize  # noqa: E402

CALIBRATOR_SHA = "835c335182e84698c3c6a7b90670a1c57ef74fabc86be3d5f95296f778193666"
OFFICIAL_EVALUATOR_SHA = "98e5b3a1f72425a267c6cfc9436fe9cff79876b0b45f349f9899ff222382be31"
FIXED_CHECKPOINT_SHA = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
FLOOR = 1e-12
PROB_TOLERANCE = 1e-3
SPLITS = ("calibration", "dev", "train")


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


def dump_records(torch, model, validated_prefixes, contract, fast, hf_tokenizer, input_path, output,
                 limit, report, save):
    """Tokenize, forward, check and write every record. Separate from main so a CPU test can drive it."""
    import numpy as np

    head = model.heads["assistant"]
    captured = []

    def hook(module, inputs, output_tensor):
        hidden = inputs[0]
        captured.append((hidden.detach().reshape(-1, hidden.shape[-1]).float(),
                         output_tensor.detach().reshape(-1, output_tensor.shape[-1]).float()))

    handle = head["projection"].register_forward_hook(hook)
    try:
        rows = collections.defaultdict(list)
        with Path(input_path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows[row["split"]].append(row)
        totals = collections.Counter()
        max_prob_diff = 0.0
        for split in SPLITS:
            records = rows[split][:limit] if limit else rows[split]
            hidden_rows, proj_rows, head_p, class_codes, record_index, meta = [], [], [], [], [], []
            with gzip.open(output / f"runA_{split}.jsonl.gz", "wb") as out:
                for number, row in enumerate(records):
                    text = serialize(row["messages"])
                    content_start = len(text) - len(row["messages"][-1]["content"])
                    encoding = fast.encode(text, add_special_tokens=False)
                    ids = list(encoding.ids)
                    if not 1 <= len(ids) <= 8192:
                        totals["skipped_length"] += 1
                        continue
                    if callable(hf_tokenizer):
                        if list(hf_tokenizer(text, add_special_tokens=False)["input_ids"]) != ids:
                            totals["skipped_tokenizer_mismatch"] += 1
                            continue
                    positions, classes = position_classes(encoding.offsets, content_start, row["label"],
                                                          row.get("onset_char"), row.get("onset_end_char"))
                    if not positions:
                        totals["skipped_no_target"] += 1
                        continue
                    captured.clear()
                    values, accounting = validated_prefixes(ids)
                    if accounting["graph_calls"] != 0:
                        raise ValueError("graph execution bypasses the feature hook")
                    if not captured:
                        raise ValueError("the feature hook never fired (engine bypasses model.heads)")
                    totals["projection_calls"] += len(captured)
                    hidden = torch.cat([h for h, _ in captured])
                    projected = torch.cat([p for _, p in captured])
                    # Padding only ever sits after the last real token, so row i is position i either way.
                    if hidden.shape[0] not in (accounting["native_tokens"], accounting["forward_tokens"]):
                        raise ValueError(f"captured {hidden.shape[0]} rows for {accounting['native_tokens']} real / "
                                         f"{accounting['forward_tokens']} forward tokens")
                    index = torch.tensor(positions, device=projected.device)
                    with torch.no_grad():
                        recomputed = torch.softmax(head["risk"](projected[index]).float(), dim=-1).cpu()
                    runtime_probs = torch.tensor([[float(p) for p in values["assistant"][q]] for q in positions])
                    diff = float((recomputed - runtime_probs).abs().max())
                    max_prob_diff = max(max_prob_diff, diff)
                    if diff > PROB_TOLERANCE:
                        raise ValueError(f"hook features do not reproduce runtime probabilities (max diff {diff:.2e})")
                    write_line(out, {
                        "sample_id": row["sample_id"], "task_key": row["task_key"], "family": row["family"],
                        "split": split, "language": row["language"], "label": row["label"],
                        "prompt_label": row["prompt_label"], "slot": row["index"],
                        "response_style": row["response_style"], "response_format": row["response_format"],
                        "input_tokens": len(ids), "classes": classes,
                        "logprobs": [log_row(values["assistant"][q]) for q in positions]})
                    keep = feature_positions(positions, classes)
                    selected = torch.tensor([positions[k] for k in keep], device=hidden.device)
                    hidden_rows.append(hidden[selected].half().cpu().numpy())
                    proj_rows.append(projected[selected].half().cpu().numpy())
                    head_p.extend(float(values["assistant"][positions[k]][1]) for k in keep)
                    class_codes.extend(classes[k] for k in keep)
                    record_index.extend([len(meta)] * len(keep))
                    meta.append({"sample_id": row["sample_id"], "family": row["family"], "slot": row["index"],
                                 "label": row["label"], "prompt_label": row["prompt_label"],
                                 "response_style": row["response_style"], "language": row["language"]})
                    totals["records"] += 1
                    totals["native_tokens"] += accounting["native_tokens"]
                    totals["forward_tokens"] += accounting["forward_tokens"]
                    if totals["records"] == 3:
                        report.update(smoke_passed=True, max_prob_diff=max_prob_diff)
                        save()
                    if (number + 1) % 500 == 0:
                        print(json.dumps({"split": split, "done": number + 1, "of": len(records),
                                          "max_prob_diff": max_prob_diff}), flush=True)
            np.savez(output / f"features_{split}.npz",
                     hidden=np.concatenate(hidden_rows) if hidden_rows else np.zeros((0, 1), np.float16),
                     projection=np.concatenate(proj_rows) if proj_rows else np.zeros((0, 1), np.float16),
                     head_p_unsafe=np.asarray(head_p, np.float32), classes=np.asarray(list(class_codes)),
                     record=np.asarray(record_index, np.int32))
            (output / f"features_{split}_records.json").write_text(json.dumps(meta, ensure_ascii=False))
            report.setdefault("records", {})[split] = len(meta)
            report.setdefault("feature_positions", {})[split] = len(head_p)
            report["totals"] = dict(totals)
            report["max_prob_diff"] = max_prob_diff
            save()
    finally:
        handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round5-code", type=Path, default=Path("/work/round5"))
    parser.add_argument("--input", type=Path, default=Path("/work/round6/probe/input_v1/probe_input_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/probe_v1"))
    parser.add_argument("--limit", type=int, default=0, help="records per split, 0 = all (smoke runs only)")
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "kind": "round6_probe_dump_v1",
              "script_sha256": sha(__file__), "common_sha256": sha(Path(__file__).with_name("probe_common.py")),
              "input_sha256": sha(options.input), "training_performed": False, "thresholds_fitted": False,
              "generated_tokens": 0, "limit": options.limit}
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
        args = official.parse_args(["--output", str(options.output / "unused_official_output")])
        summary, checkpoint, _, _, _, _ = cal.common_inputs(args, audit)
        if summary["final_checkpoint_sha256"] != FIXED_CHECKPOINT_SHA or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
            raise ValueError("fixed Round5 checkpoint differs")
        locked = audit.read_json(args.calibration_output / "selected/calibration.json")
        args.inference_engine = locked["engine_metadata"]["inference_engine"]
        args.pad_token_id = locked["execution_contract"]["pad_token_id"]
        contract = locked["execution_contract"]
        if args.inference_engine != "eager":
            raise ValueError("feature hooks need the eager engine; locked engine is " + str(args.inference_engine))

        from tokenizers import Tokenizer
        tokenizer_json = Path(args.tokenizer_root) / "tokenizer.json"
        fast = Tokenizer.from_file(str(tokenizer_json))
        if fast.truncation is not None or fast.padding is not None:
            raise ValueError("tokenizer truncation/padding must be disabled")
        report.update(checkpoint_sha256=FIXED_CHECKPOINT_SHA, inference_engine=args.inference_engine,
                      execution_contract=contract, tokenizer_json_sha256=sha(tokenizer_json), phase="loading_runtime")
        save()

        loader = cal.prepare_modules(args)
        torch, model, hf_tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        try:
            if runtime.engine_metadata["execution_contract"] != contract:
                raise ValueError("runtime execution contract differs from the locked calibration")
            report.update(device=device, sources_before=cal.source_receipt(args), phase="records",
                          hf_tokenizer_cross_check=callable(hf_tokenizer))
            save()
            dump_records(torch, model, lambda ids: cal.validated_prefixes(runtime, ids, contract), contract,
                         fast, hf_tokenizer, options.input, options.output, options.limit, report, save)
            after = cal.source_receipt(args)
            if not cal.sources_unchanged(report["sources_before"], after) or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
                raise ValueError("runtime sources or weights changed during the dump")
            report.update(phase="finished", status="completed", integrity_pass=True,
                          files={p.name: sha(p) for p in sorted(options.output.iterdir())
                                 if p.suffix in (".gz", ".npz") or p.name.endswith("_records.json")})
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
