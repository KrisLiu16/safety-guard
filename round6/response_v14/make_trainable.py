#!/usr/bin/env python3
"""Apply the BATCH_PLAN training filter to an extracted v14 run and attach split/family from the seeds.

Rules (BATCH_PLAN.md, "用于训练时的过滤"):
  1. keep only words whose four responses all passed re-validation (keeps the 2x2 balanced);
  2. an unsafe response that was asked to open with safe lead-in (onset_style_requested=delayed)
     but has onset_char == 0 is dropped, and its word then counts as incomplete;
  3. quality_flags are only counted, never used to filter.
The split and family come from the batch seeds.jsonl (they are not in the flow output).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_trainable(examples, words, seeds):
    """Return (kept_rows, stats). Pure function so the filter can be tested on CPU."""
    seed_by_key = {s["task_key"]: s for s in seeds}
    complete = {w["task_key"] for w in words if w.get("complete")}
    by_word = collections.defaultdict(list)
    for row in examples:
        by_word[row["task_key"]].append(row)
    stats = collections.Counter()
    kept, split_words = [], collections.Counter()
    for key, rows in by_word.items():
        stats["words_seen"] += 1
        if key not in complete or len(rows) != 4:
            stats["words_incomplete_at_extract"] += 1
            continue
        bad_delay = [r for r in rows if r["label"] == "unsafe"
                     and r.get("onset_style_requested") == "delayed" and r.get("onset_char") == 0]
        if bad_delay:
            stats["responses_delayed_onset_zero"] += len(bad_delay)
            stats["words_dropped_delayed_onset_zero"] += 1
            continue
        seed = seed_by_key.get(key)
        if seed is None:
            raise KeyError(f"task_key {key} is not in the seeds file")
        for row in sorted(rows, key=lambda r: r["index"]):
            kept.append({**row, "split": seed["split"], "family": seed["family"]})
        split_words[seed["split"]] += 1
    stats["words_kept"] = sum(split_words.values())
    stats["responses_kept"] = len(kept)
    flags = collections.Counter(f"{r['label']}:{f}" for r in kept for f in r.get("quality_flags", []))
    return kept, {**dict(stats), "words_by_split": dict(split_words),
                  "responses_by_split_label": dict(collections.Counter(f"{r['split']}:{r['label']}" for r in kept)),
                  "quality_flags_kept": dict(flags)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, required=True, help="folder with examples.jsonl and words.jsonl")
    parser.add_argument("--seeds", type=Path, required=True, help="batch seeds.jsonl with split and family")
    args = parser.parse_args()
    out = args.extracted / "trainable.jsonl"
    if out.exists():
        raise FileExistsError(out)
    kept, stats = select_trainable(read_jsonl(args.extracted / "examples.jsonl"),
                                   read_jsonl(args.extracted / "words.jsonl"), read_jsonl(args.seeds))
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.extracted / "trainable_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                                            encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
