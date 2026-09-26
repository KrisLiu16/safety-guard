"""Stage 1, step 3 (L20, inference only): score every position with each retrained readout.

For every head_<variant>.pt from train_head.py, the weights are loaded into the live model's assistant head
(strict key match) and the same inputs as T014 are run through the same pinned canonical32 eager path:
Run A calibration + dev (round6/probe input_v1), prefix_v2 calibration + dev (assistant role) and the 1,872
official sequences (observed only). Output per variant, in T014's file format without probe logits:
eval_<variant>/runA_<split>.jsonl.gz, prefix_v2_<split>.jsonl.gz, official_sequences / official_rows.jsonl.gz.
Run A rows also carry char_ends (each position's end offset in the response), so labels from the red-line
labelling (round6/redline_v1) can be matched per position. The variant "init" loads the current head
(head_assistant_init.pt from the cache step), so every head is scored in one run with the same fields.
Integrity: the risk head applied to the captured projection must reproduce the runtime's probabilities, which
also proves the runtime is reading the newly loaded weights.
--role user (T025): the heads are loaded into the user head and scored on each distinct Run A prompt
(runA_prompts_<split>.jsonl.gz, ids "<task_key>:prompt:<label>", char_ends over the prompt) and the prefix_v2
user-role records; the official thinking set is assistant-side and is skipped.
--checkpoint (stage 2, round6/stage2): the variant "checkpoint" loads a whole trained model (backbone and heads,
strict key match) into the live model in place of the fixed Round5 weights; it must come last, since the head
variants before it are scored on the fixed backbone. The same integrity check then proves the runtime reads it.
--shard I/N (T036): score only every N-th record of each file starting at I, and write order.json next to the
files (each line's record ordinal); shard_eval.py runs N such processes on one GPU and merges them back into the
single-process files. --autotune record / pin (autotune_pin.py) fixes the Triton kernel configs across processes,
so a record scores the same in any process.
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
sys.path.insert(0, str(HERE.parent / "probe"))
sys.path.insert(0, str(HERE))
from dump_runA_l20 import (CALIBRATOR_SHA, FIXED_CHECKPOINT_SHA, OFFICIAL_EVALUATOR_SHA,  # noqa: E402
                           load_pinned, log_row, sha, write_line)
from cache_features_l20 import Capture  # noqa: E402
from probe_common import position_classes, serialize  # noqa: E402


def mine(shard, k):
    """Whether the k-th record of a file belongs to this process (--shard i/n: every n-th record from i)."""
    return shard is None or k % shard[1] == shard[0]


def note(order, name, k):
    if order is not None:
        order.setdefault(name, []).append(k)


def eval_runA(capture, probs_of, fast, hf_tokenizer, rows, output, shard=None, order=None):
    counts = collections.Counter()
    for split in ("calibration", "dev"):
        name = f"runA_{split}.jsonl.gz"
        with gzip.open(output / name, "wb") as out:
            for k, row in enumerate(r for r in rows if r["split"] == split):
                if not mine(shard, k):
                    continue
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
                capture(ids)
                probs = probs_of()
                write_line(out, {"sample_id": row["sample_id"], "family": row["family"], "split": split,
                                 "language": row["language"], "label": row["label"], "slot": row["index"],
                                 "response_style": row["response_style"], "classes": classes,
                                 "char_ends": [encoding.offsets[q][1] - content_start for q in positions],
                                 "logprobs": [log_row(probs[q]) for q in positions]})
                note(order, name, k)
                counts[f"runA_{split}"] += 1
    return counts


def eval_runA_prompts(capture, probs_of, fast, hf_tokenizer, rows, output, shard=None, order=None):
    """Each distinct Run A prompt alone (the user message), every position of it scored by the user head."""
    counts = collections.Counter()
    for split in ("calibration", "dev"):
        seen = set()
        name = f"runA_prompts_{split}.jsonl.gz"
        with gzip.open(output / name, "wb") as out:
            for row in (r for r in rows if r["split"] == split):
                sample_id = f"{row['task_key']}:prompt:{row['prompt_label']}"
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                k = len(seen) - 1
                if not mine(shard, k):
                    continue
                messages = [row["messages"][0]]
                text = serialize(messages)
                content_start = len(text) - len(messages[0]["content"])
                encoding = fast.encode(text, add_special_tokens=False)
                ids = list(encoding.ids)
                if not 1 <= len(ids) <= 8192 or (callable(hf_tokenizer) and
                                                  list(hf_tokenizer(text, add_special_tokens=False)["input_ids"]) != ids):
                    counts["runA_prompts_skipped"] += 1
                    continue
                positions = [i for i, (_, end) in enumerate(encoding.offsets) if end > content_start]
                if not positions:
                    counts["runA_prompts_skipped"] += 1
                    continue
                capture(ids)
                probs = probs_of()
                write_line(out, {"sample_id": sample_id, "family": row["family"], "split": split,
                                 "language": row["language"], "label": row["prompt_label"],
                                 "char_ends": [encoding.offsets[q][1] - content_start for q in positions],
                                 "logprobs": [log_row(probs[q]) for q in positions]})
                note(order, name, k)
                counts[f"runA_prompts_{split}"] += 1
    return counts


def eval_prefix_v2(capture, probs_of, data, output, role="assistant", shard=None, order=None):
    counts = collections.Counter()
    for split in ("calibration", "dev"):
        name = f"prefix_v2_{split}.jsonl.gz"
        with gzip.open(output / name, "wb") as out:
            for k, row in enumerate(data[split]):
                if row["target_role"] != role or not mine(shard, k):
                    continue
                capture(row["ids"])
                probs = probs_of()
                write_line(out, {"sample_id": row["sample_id"], "family": row["family"], "language": row["language"],
                                 "target_role": row["target_role"], "source_label": row["source_label"], "split": split,
                                 "logprobs": [log_row(probs[p]) for p in row["target_token_positions"]]})
                note(order, name, k)
                counts[f"prefix_v2_{split}"] += 1
    return counts


def eval_official(capture, probs_of, official, official_data, output, shard=None, order=None):
    """Sequences are sharded by first-appearance ordinal; the row file needs no model and comes from shard 0."""
    seen, r = {}, 0
    with gzip.open(output / "official_sequences.jsonl.gz", "wb") as seq_out, \
            gzip.open(output / "official_rows.jsonl.gz", "wb") as rows_out:
        for split in official.SPLITS:
            for row in official_data[split]:
                key = official.sequence_key(row)
                sequence_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                if key not in seen:
                    k = len(seen)
                    if mine(shard, k):
                        capture(row["ids"])
                        probs = probs_of()
                        start = row["eval_start_index"]
                        write_line(seq_out, {"sequence_id": sequence_id, "input_tokens": len(row["ids"]),
                                             "eval_start_index": start, "logprobs": [log_row(p) for p in probs[start:]]})
                        note(order, "official_sequences.jsonl.gz", k)
                    seen[key] = sequence_id
                if shard is None or shard[0] == 0:
                    write_line(rows_out, {"sample_id": row["sample_id"], "split": split, "row_index": row["row_index"],
                                          "unique_id": row["unique_id"], "label": row["label"], "sequence_id": seen[key]})
                    note(order, "official_rows.jsonl.gz", r)
                r += 1
    if len(seen) != official.EXPECTED_UNIQUE:
        raise ValueError(f"official unique-sequence coverage {len(seen)} differs from {official.EXPECTED_UNIQUE}")
    return {"official_unique_sequences": len(seen)}


class ProbsCapture(Capture):
    """Capture that also keeps the runtime's probabilities of the last record for its head's role."""

    def __init__(self, torch, model, validated_prefixes, role="assistant"):
        self.last = None

        def keep(ids):
            values, accounting = validated_prefixes(ids)
            self.last = values[role]
            return values, accounting

        super().__init__(torch, model, keep, role=role)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round5-code", type=Path, default=Path("/work/round5"))
    parser.add_argument("--heads", type=Path, default=Path("/work/output/round6/stage1_train_v1"),
                        help="directory with head_<variant>.pt from train_head.py")
    parser.add_argument("--variants", default="init,risk,full", help='"init" = the current head')
    parser.add_argument("--role", choices=("assistant", "user"), default="assistant")
    parser.add_argument("--init-head", type=Path, default=None,
                        help="default: head_<role>_init.pt of the stage-1 cache (stage1_cache_v1 / stage1_cache_user_v1)")
    parser.add_argument("--runA", type=Path, default=Path("/work/round6/probe/input_v1/probe_input_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/work/output/round6/stage1_eval_v1"))
    parser.add_argument("--checkpoint", type=Path, help='stage-2 model (safetensors) for the variant "checkpoint"')
    parser.add_argument("--shard", help="I/N: score every N-th record from I (see shard_eval.py)")
    parser.add_argument("--autotune", choices=("record", "pin"),
                        help="autotune_pin.py: record the Triton configs chosen, or use recorded ones (--autotune-file)")
    parser.add_argument("--autotune-file", type=Path)
    options = parser.parse_args(argv)
    shard = None
    if options.shard:
        i, n = (int(v) for v in options.shard.split("/"))
        if not 0 <= i < n:
            parser.error("--shard needs 0 <= I < N")
        shard = (i, n)
    if options.autotune:
        if options.autotune_file is None:
            parser.error("--autotune needs --autotune-file")
        import autotune_pin
        autotune_pin.install(options.autotune, options.autotune_file)
    variants = options.variants.split(",")
    if "checkpoint" in variants and (options.checkpoint is None or variants[-1] != "checkpoint"):
        parser.error('the variant "checkpoint" needs --checkpoint and must be the last variant')
    if options.init_head is None:
        options.init_head = Path("/work/output/round6/stage1_cache_v1/head_assistant_init.pt" if options.role == "assistant"
                                 else "/work/output/round6/stage1_cache_user_v1/head_user_init.pt")
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.mkdir(parents=True)
    report = {"status": "running", "integrity_pass": False, "kind": "round6_stage1_eval_v1", "role": options.role, "script_sha256": sha(__file__),
              "official_used_for_fitting": False, "generated_tokens": 0, "variants": {}, "shard": shard}
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
        summary, checkpoint, _, trainer, _, data = cal.common_inputs(args, audit)
        if summary["final_checkpoint_sha256"] != FIXED_CHECKPOINT_SHA or sha(checkpoint) != FIXED_CHECKPOINT_SHA:
            raise ValueError("fixed Round5 checkpoint differs")
        locked = audit.read_json(args.calibration_output / "selected/calibration.json")
        args.inference_engine = locked["engine_metadata"]["inference_engine"]
        args.pad_token_id = locked["execution_contract"]["pad_token_id"]
        contract = locked["execution_contract"]
        if args.inference_engine != "eager":
            raise ValueError("the integrity hook needs the eager engine; locked engine is " + str(args.inference_engine))
        official_data, _ = official.load_official_inputs(args.data_dir, args.source_dir)
        from tokenizers import Tokenizer
        fast = Tokenizer.from_file(str(Path(args.tokenizer_root) / "tokenizer.json"))
        with options.runA.open(encoding="utf-8") as handle:
            runA_rows = [json.loads(line) for line in handle if line.strip()]
        loader = cal.prepare_modules(args)
        torch, model, hf_tokenizer, runtime, device = cal.load_runtime(args, checkpoint, loader)
        try:
            if runtime.engine_metadata["execution_contract"] != contract:
                raise ValueError("runtime execution contract differs from the locked calibration")
            report["official_native_ids"] = official.verify_native_ids(hf_tokenizer, official_data, trainer.serialize)
            capture = ProbsCapture(torch, model, lambda ids: cal.validated_prefixes(runtime, ids, contract), role=options.role)
            for variant in variants:
                if variant == "checkpoint":
                    from safetensors.torch import load_file
                    weights_path = options.checkpoint
                    state = load_file(str(weights_path), device="cpu")
                    model.load_state_dict(state, strict=True)          # backbone and both heads, in place
                    del state
                else:
                    weights_path = options.init_head if variant == "init" else options.heads / f"head_{variant}.pt"
                    weights = torch.load(weights_path, map_location="cpu")
                    model.heads[options.role].load_state_dict(weights, strict=True)
                output = options.output / f"eval_{variant}"
                output.mkdir()
                report.update(phase=variant)
                save()
                order = {} if shard else None
                part = {"shard": shard, "order": order}
                if options.role == "assistant":
                    counts = eval_runA(capture, lambda: capture.last, fast, hf_tokenizer, runA_rows, output, **part)
                    counts.update(eval_prefix_v2(capture, lambda: capture.last, data, output, **part))
                    counts.update(eval_official(capture, lambda: capture.last, official, official_data, output, **part))
                else:
                    counts = eval_runA_prompts(capture, lambda: capture.last, fast, hf_tokenizer, runA_rows, output,
                                               **part)
                    counts.update(eval_prefix_v2(capture, lambda: capture.last, data, output, role="user", **part))
                if shard:
                    (output / "order.json").write_text(json.dumps(order) + "\n")
                report["variants"][variant] = {"weights_sha256": sha(weights_path), "counts": dict(counts),
                                               "files": {p.name: sha(p) for p in sorted(output.glob("*.jsonl.gz"))}}
                save()
            if sha(checkpoint) != FIXED_CHECKPOINT_SHA:
                raise ValueError("checkpoint bytes changed during evaluation")
            report.update(max_prob_diff=capture.max_prob_diff, totals=dict(capture.totals),
                          phase="finished", status="completed", integrity_pass=True)
            if options.autotune:
                report["autotune"] = autotune_pin.state()
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
