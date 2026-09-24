"""Stage 1, step 1b (CPU, where the cache and the Round5 data live): red-line labels for the cached positions.

The T017 cache stores features at positions chosen under the old v14 labels. Features do not depend on labels, so
they are reused; this step writes new 3-class targets next to them (the cache itself is not modified):
  labels3_<source>_<split>.npz   label  int8, the head's class index: 0 safe, 1 unsafe, 2 controversial
                                 weight float32, the cached weight, or 0 where the position gets no target
Labels come from round6/redline_v1/apply_policy.py (labels.jsonl): the whole-text level, and per position the
level of its prefix from the stored probes (levels.position_level). A position gets no target when its response
is unusable, excluded or unlabelled, or it lies inside a located onset clause.
Character offsets of cached positions are recovered by re-tokenizing exactly as the cache step did:
  runA       serialize(messages), the same position_classes / feature_positions selection with the old labels;
             the count must match the cached positions of that record
  prefix_v2  train, original view: anchor token -> offset in serialize(messages), ids checked against the frozen ids;
             train, augmented view: every anchor sits after the complete target plus a neutral suffix -> whole level;
             calibration: endpoint -> whole level, interior -> offset (same selection as the cache step)
--role user (T025, user-head cache from T023): runA_prompts records carry their char_ends, so no re-tokenization;
prefix_v2 user-role records are handled exactly as above (the prompt is the last message). Labels come from the
prompt-side labelling (make_tasks.py / apply_policy.py --target user).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
sys.path.insert(0, str(HERE.parent / "redline_v1" / "flow"))
from cache_features_l20 import PREFIX_CAL_INTERIOR  # noqa: E402
from levels import position_level  # noqa: E402
from probe_common import feature_positions, position_classes, serialize  # noqa: E402

CLASS_INDEX = {0: 0, 1: 2, 2: 1}          # level rank (safe, controversial, unsafe) -> head class index
LEVEL_RANK = {"safe": 0, "controversial": 1, "unsafe": 2}
WHOLE = "whole"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_positions(cache_dir, name):
    """Old label, weight and record index of every cached position, in shard order (features are not loaded)."""
    shards = sorted(Path(cache_dir).glob(f"cache_{name}_[0-9][0-9][0-9].npz"))
    parts = [np.load(path) for path in shards]
    return {key: np.concatenate([p[key] for p in parts]) if parts else np.zeros(0) for key in ("label", "weight", "record")}


def content_offsets(encode, messages):
    """(ids, [char end relative to the last message's content, per token], content length)."""
    text = serialize(messages)
    content = messages[-1]["content"]
    start = len(text) - len(content)
    ids, offsets = encode(text)
    return ids, [min(end - start, len(content)) for _, end in offsets], len(content)


def runA_ends(row, encode):
    """Char ends of the positions the cache step kept for this Run A row, in cache order."""
    text = serialize(row["messages"])
    start = len(text) - len(row["messages"][-1]["content"])
    _, offsets = encode(text)
    positions, classes = position_classes(offsets, start, row["label"], row.get("onset_char"), row.get("onset_end_char"))
    keep = [k for k in feature_positions(positions, classes) if classes[k] != "O"]
    return [offsets[positions[k]][1] - start for k in keep]


def prefix_ends(row, meta, split, encode):
    """Char ends (or WHOLE) of the cached positions of one prefix_v2 record, in cache order; None on id mismatch."""
    if split == "train" and meta.get("view") == "augmented":
        return [WHOLE] * len(row["augmentation"]["anchors"])
    ids, ends, length = content_offsets(encode, row["messages"])
    if list(ids) != list(row["ids"]):
        return None
    if split == "train":
        return [ends[a["token_end_exclusive"] - 1] for a in row["anchors"]]
    targets = row["target_token_positions"]
    out = [WHOLE]
    if row["source_label"] == "safe" and len(targets) > 1:
        interior = [targets[round(k * (len(targets) - 2) / max(1, PREFIX_CAL_INTERIOR - 1))]
                    for k in range(PREFIX_CAL_INTERIOR)]
        out += [ends[p] for p in sorted(set(interior) - {targets[-1]})]
    return out


def relabel(positions, records, labels, ends_of):
    """Pure. positions: arrays from load_positions; records: cache records; labels: sample_id -> labels.jsonl row;
    ends_of(record_meta) -> list of char ends / WHOLE, or None. Returns (label, weight, stats)."""
    new_label = np.zeros(len(positions["label"]), np.int8)
    new_weight = np.zeros(len(positions["label"]), np.float32)
    stats, transitions = collections.Counter(), collections.Counter()
    order = np.argsort(positions["record"], kind="stable")
    bounds = np.searchsorted(positions["record"][order], np.arange(len(records) + 1))
    for index, meta in enumerate(records):
        where = order[bounds[index]:bounds[index + 1]]
        labelled = labels.get(meta["sample_id"])
        if labelled is None or labelled["label"] in ("unusable", "excluded"):
            stats["dropped_" + ("unlabelled" if labelled is None else labelled["label"])] += len(where)
            continue
        ends = ends_of(meta)
        if ends is None or len(ends) != len(where):
            stats["dropped_offsets"] += len(where)
            continue
        for j, end in zip(where, ends):
            rank = LEVEL_RANK[labelled["level"]] if end == WHOLE else position_level(end, labelled["probes"])
            if rank is None:
                stats["dropped_onset_clause"] += 1
                continue
            new_label[j], new_weight[j] = CLASS_INDEX[rank], positions["weight"][j]
            transitions[f"{int(positions['label'][j])}->{CLASS_INDEX[rank]}"] += 1
            stats["kept"] += 1
    stats["positions"] = len(new_label)
    return new_label, new_weight, {**dict(stats), "old_to_new_class": dict(transitions)}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--role", choices=("assistant", "user"), default="assistant")
    parser.add_argument("--runA", type=Path, help="stage-1 Run A input (stage1_runA_v1.jsonl); assistant role only")
    parser.add_argument("--runA-labels", type=Path, nargs="+", required=True, help="labels.jsonl files covering it")
    parser.add_argument("--prefix-labels", type=Path, nargs="*", default=[], help="labels.jsonl for prefix_v2")
    parser.add_argument("--prefix-data", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/work/models/qwen35/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.role == "assistant" and args.runA is None:
        parser.error("--runA is required for the assistant role")
    from tokenizers import Tokenizer
    fast = Tokenizer.from_file(str(args.tokenizer))
    if fast.truncation is not None or fast.padding is not None:
        raise ValueError("tokenizer truncation/padding must be disabled")

    def encode(text):
        encoding = fast.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    args.output.mkdir(parents=True)
    report = {"version": "stage1-relabel-redline-v1", "role": args.role,
              "class_index": {"safe": 0, "unsafe": 1, "controversial": 2},
              "inputs": {str(p): sha(p) for p in [args.runA, *args.runA_labels, *args.prefix_labels] if p}, "sources": {}}
    runA_labels = {r["sample_id"]: r for path in args.runA_labels for r in read_jsonl(path)}
    runA_rows = {r["sample_id"]: r for r in read_jsonl(args.runA)} if args.role == "assistant" else {}
    prefix_labels = {r["sample_id"]: r for path in args.prefix_labels for r in read_jsonl(path)}
    runA_source = "runA" if args.role == "assistant" else "runA_prompts"
    for split in ("train", "calibration"):
        name = f"{runA_source}_{split}"
        records = json.loads((args.cache_dir / f"cache_{name}_records.json").read_text(encoding="utf-8"))
        ends_of = ((lambda meta: runA_ends(runA_rows[meta["sample_id"]], encode)) if args.role == "assistant"
                   else (lambda meta: meta["char_ends"]))
        label, weight, stats = relabel(load_positions(args.cache_dir, name), records, runA_labels, ends_of)
        np.savez(args.output / f"labels3_{name}.npz", label=label, weight=weight)
        report["sources"][name] = stats
        if prefix_labels:
            name = f"prefix_v2_{split}"
            rows = {r["sample_id"]: r for r in read_jsonl(args.prefix_data / f"{split}.jsonl")}
            records = json.loads((args.cache_dir / f"cache_{name}_records.json").read_text(encoding="utf-8"))
            label, weight, stats = relabel(load_positions(args.cache_dir, name), records, prefix_labels,
                                           lambda meta: prefix_ends(rows[meta["sample_id"]], meta, split, encode))
            np.savez(args.output / f"labels3_{name}.npz", label=label, weight=weight)
            report["sources"][name] = stats
    report["files"] = {p.name: sha(p) for p in sorted(args.output.glob("labels3_*.npz"))}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["sources"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
