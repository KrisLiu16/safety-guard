"""Stage 1, step 1 (L20, inference only): cache frozen assistant-head features for readout training.

Runs the fixed Round5 weights over the training and calibration records of every stage-1 source through the
same pinned canonical32 eager path as round6/probe, and stores, at labelled positions only:
  hidden      the assistant head's 1024-d input (float16)
  projection  its 512-d projection (float16) - what the risk and category heads read
  label       0 safe / 1 unsafe;  weight  the position's training weight;  record  index into the records file
Sources and position labels (round6/stage1_head/README.md):
  prefix_v2  Round5 training set, assistant role: its own anchors (weak safe interior, weight 0.1 x tier weight;
             complete-target + neutral-suffix endpoints, hard labels); calibration: endpoint + a few safe interior
  runA       stage-1 input (Run A train sample + calibration): S/P -> 0, U -> 1, onset span skipped, <= 24 positions
  s2, s5     optional extracted examples.jsonl files: every kept position safe
The current assistant head's weights are saved next to the cache for warm starts. Integrity is the T004/T014
check: captured rows cover the record and the risk head on the captured projection reproduces the runtime.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
from dump_runA_l20 import (CALIBRATOR_SHA, FIXED_CHECKPOINT_SHA, OFFICIAL_EVALUATOR_SHA, PROB_TOLERANCE,  # noqa: E402
                           load_pinned, sha)
from probe_common import feature_positions, position_classes, serialize  # noqa: E402

SHARD_POSITIONS = 200_000
PREFIX_CAL_INTERIOR = 8


class Capture:
    """One record through the runtime; returns hidden and projection rows for the real positions."""

    def __init__(self, torch, model, validated_prefixes):
        self.torch, self.validated = torch, validated_prefixes
        self.head = model.heads["assistant"]
        self.captured, self.max_prob_diff = [], 0.0
        self.totals = collections.Counter()
        self.handle = self.head["projection"].register_forward_hook(
            lambda module, inputs, output: self.captured.append(
                (inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float(),
                 output.detach().reshape(-1, output.shape[-1]).float())))

    def close(self):
        self.handle.remove()

    def __call__(self, ids):
        torch = self.torch
        self.captured.clear()
        values, accounting = self.validated(ids)
        if accounting["graph_calls"] != 0 or not self.captured:
            raise ValueError("the feature hook did not fire on an eager forward")
        hidden = torch.cat([h for h, _ in self.captured])
        projected = torch.cat([p for _, p in self.captured])
        n = len(ids)
        if hidden.shape[0] not in (accounting["native_tokens"], accounting["forward_tokens"]):
            raise ValueError(f"captured {hidden.shape[0]} rows for {n} real / {accounting['forward_tokens']} forward tokens")
        with torch.no_grad():
            recomputed = torch.softmax(self.head["risk"](projected[:n]).float(), dim=-1).cpu()
        runtime = torch.tensor([[float(p) for p in row] for row in values["assistant"]])
        diff = float((recomputed - runtime).abs().max())
        self.max_prob_diff = max(self.max_prob_diff, diff)
        if diff > PROB_TOLERANCE:
            raise ValueError(f"hook features do not reproduce runtime probabilities (max diff {diff:.2e})")
        self.totals["sequences"] += 1
        self.totals["forward_tokens"] += accounting["forward_tokens"]
        return hidden[:n], projected[:n]


class ShardWriter:
    """Accumulates labelled positions for one (source, split) and writes numbered npz shards plus a records file."""

    def __init__(self, output, source, split):
        self.output, self.name = output, f"{source}_{split}"
        self.records, self.buffers, self.shards, self.positions, self.pending = [], collections.defaultdict(list), 0, 0, 0

    def add(self, meta, hidden, projection, labels, weights):
        import numpy as np
        index = len(self.records)
        self.records.append(meta)
        self.buffers["hidden"].append(hidden.half().cpu().numpy())
        self.buffers["projection"].append(projection.half().cpu().numpy())
        self.buffers["label"].append(np.asarray(labels, np.int8))
        self.buffers["weight"].append(np.asarray(weights, np.float32))
        self.buffers["record"].append(np.full(len(labels), index, np.int32))
        self.positions += len(labels)
        self.pending += len(labels)
        if self.pending >= SHARD_POSITIONS:
            self.flush()

    def flush(self):
        import numpy as np
        if not self.buffers["label"]:
            return
        np.savez(self.output / f"cache_{self.name}_{self.shards:03d}.npz",
                 **{key: np.concatenate(value) for key, value in self.buffers.items()})
        self.buffers.clear()
        self.shards += 1
        self.pending = 0

    def close(self):
        self.flush()
        (self.output / f"cache_{self.name}_records.json").write_text(json.dumps(self.records, ensure_ascii=False))
        return {"records": len(self.records), "positions": self.positions, "shards": self.shards}


def at(torch, tensor, positions):
    return tensor[torch.tensor(positions, device=tensor.device)]


def cache_prefix_v2(torch, capture, data, output):
    stats = {}
    for split in ("train", "calibration"):
        writer = ShardWriter(output, "prefix_v2", split)
        for row in data[split]:
            if row["target_role"] != "assistant":
                continue          # stage 1 retrains the assistant readout only
            meta = {"sample_id": row["sample_id"], "family": row["family"], "language": row["language"],
                    "source_label": row["source_label"], "label_tier": row.get("label_tier")}
            if split == "train":
                views = []
                if row["anchors"]:
                    views.append(("original", row["ids"], row["anchors"]))
                views.append(("augmented", row["augmentation"]["ids"], row["augmentation"]["anchors"]))
                for view, ids, anchors in views:
                    hidden, projection = capture(ids)
                    positions = [a["token_end_exclusive"] - 1 for a in anchors]
                    writer.add({**meta, "view": view}, at(torch, hidden, positions), at(torch, projection, positions),
                               [a["label"] for a in anchors], [a["confidence"] * row["weight"] for a in anchors])
            else:
                hidden, projection = capture(row["ids"])
                targets = row["target_token_positions"]
                positions = [targets[-1]]
                labels, weights = [int(row["source_label"] == "unsafe")], [1.0]
                if row["source_label"] == "safe" and len(targets) > 1:
                    interior = [targets[round(k * (len(targets) - 2) / max(1, PREFIX_CAL_INTERIOR - 1))]
                                for k in range(PREFIX_CAL_INTERIOR)]
                    interior = sorted(set(interior) - {targets[-1]})
                    positions += interior
                    labels += [0] * len(interior)
                    weights += [0.1] * len(interior)
                writer.add({**meta, "view": "original"}, at(torch, hidden, positions), at(torch, projection, positions),
                           labels, weights)
        stats[split] = writer.close()
    return stats


def cache_text_rows(torch, capture, fast, hf_tokenizer, rows, output, source):
    """Run A (S/P/O/U classes) or S2/S5 (all safe) rows in the v14 row format; train and calibration splits."""
    stats, skipped = {}, collections.Counter()
    for split in ("train", "calibration"):
        writer = ShardWriter(output, source, split)
        for row in (r for r in rows if r.get("split") == split):
            text = serialize(row["messages"])
            content_start = len(text) - len(row["messages"][-1]["content"])
            encoding = fast.encode(text, add_special_tokens=False)
            ids = list(encoding.ids)
            if not 1 <= len(ids) <= 8192 or (callable(hf_tokenizer) and
                                              list(hf_tokenizer(text, add_special_tokens=False)["input_ids"]) != ids):
                skipped[split] += 1
                continue
            positions, classes = position_classes(encoding.offsets, content_start, row["label"],
                                                  row.get("onset_char"), row.get("onset_end_char"))
            keep = [k for k in feature_positions(positions, classes) if classes[k] != "O"]
            if not keep:
                skipped[split] += 1
                continue
            hidden, projection = capture(ids)
            chosen = [positions[k] for k in keep]
            writer.add({"sample_id": row["sample_id"], "family": row.get("family"), "language": row["language"],
                        "slot": row.get("index"), "response_style": row.get("response_style"), "classes": classes},
                       at(torch, hidden, chosen), at(torch, projection, chosen),
                       [int(classes[k] == "U") for k in keep], [1.0] * len(keep))
        stats[split] = writer.close()
    stats["skipped"] = dict(skipped)
    return stats


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round5-code", type=Path, default=Path("/work/round5"))
    parser.add_argument("--runA", type=Path, default=Path("/work/round6/stage1_head/input_v1/stage1_runA_v1.jsonl"))
    parser.add_argument("--s2", type=Path, default=None, help="optional S2 extracted examples.jsonl")
    parser.add_argument("--s5", type=Path, default=None, help="optional S5-safe extracted examples.jsonl")
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/stage1_cache_v1"))
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "kind": "round6_stage1_feature_cache_v1",
              "script_sha256": sha(__file__), "training_performed": False, "generated_tokens": 0,
              "inputs": {name: sha(path) for name, path in (("runA", options.runA), ("s2", options.s2), ("s5", options.s5))
                         if path is not None}}
    began = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - began
        (options.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    capture = None
    try:
        code = options.round5_code
        cal = load_pinned("canonical_calibration", code / "calibrate_canonical_l20.py", CALIBRATOR_SHA)
        audit = cal.load_local("evaluate_final_l20", code / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
        official = load_pinned("canonical_official_eval", code / "evaluate_canonical_official_l20.py", OFFICIAL_EVALUATOR_SHA)
        args = official.parse_args(["--output", str(options.output / "unused_official_output")])
        summary, checkpoint, _, _, _, data = cal.common_inputs(args, audit)
        if summary["final_checkpoint_sha256"] != FIXED_CHECKPOINT_SHA or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
            raise ValueError("fixed Round5 checkpoint differs")
        locked = audit.read_json(args.calibration_output / "selected/calibration.json")
        args.inference_engine = locked["engine_metadata"]["inference_engine"]
        args.pad_token_id = locked["execution_contract"]["pad_token_id"]
        contract = locked["execution_contract"]
        if args.inference_engine != "eager":
            raise ValueError("feature hooks need the eager engine; locked engine is " + str(args.inference_engine))
        from tokenizers import Tokenizer
        fast = Tokenizer.from_file(str(Path(args.tokenizer_root) / "tokenizer.json"))
        if fast.truncation is not None or fast.padding is not None:
            raise ValueError("tokenizer truncation/padding must be disabled")
        loader = cal.prepare_modules(args)
        torch, model, hf_tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        try:
            if runtime.engine_metadata["execution_contract"] != contract:
                raise ValueError("runtime execution contract differs from the locked calibration")
            init = {k: v.detach().cpu() for k, v in model.heads["assistant"].state_dict().items()}
            torch.save(init, options.output / "head_assistant_init.pt")
            report.update(device=device, head_keys=sorted(init), sources_before=cal.source_receipt(args),
                          head_init_sha256=sha(options.output / "head_assistant_init.pt"), phase="prefix_v2")
            save()
            capture = Capture(torch, model, lambda ids: cal.validated_prefixes(runtime, ids, contract))
            stats = {"prefix_v2": cache_prefix_v2(torch, capture, data, options.output)}
            report.update(stats=stats, max_prob_diff=capture.max_prob_diff, phase="runA")
            save()
            stats["runA"] = cache_text_rows(torch, capture, fast, hf_tokenizer, read_jsonl(options.runA), options.output, "runA")
            for name, path in (("s2", options.s2), ("s5", options.s5)):
                if path is not None:
                    report.update(stats=stats, phase=name)
                    save()
                    stats[name] = cache_text_rows(torch, capture, fast, hf_tokenizer, read_jsonl(path), options.output, name)
            after = cal.source_receipt(args)
            if not cal.sources_unchanged(report["sources_before"], after) or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
                raise ValueError("runtime sources or weights changed during caching")
            report.update(stats=stats, totals=dict(capture.totals), max_prob_diff=capture.max_prob_diff,
                          phase="finished", status="completed", integrity_pass=True,
                          files={p.name: sha(p) for p in sorted(options.output.glob("cache_*"))})
        finally:
            if capture is not None:
                capture.close()
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
