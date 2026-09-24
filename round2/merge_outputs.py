#!/usr/bin/env python3
"""Merge non-overlapping Aster runs into one 1,000-term research batch."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import shutil


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-terms", type=int, default=1000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists; use a new immutable directory")
    args.out.mkdir(parents=True)
    (args.out / "by_word").mkdir()
    examples, statuses, sources = [], [], []
    seen_terms, seen_content = set(), set()
    for directory in args.inputs:
        summary = json.loads((directory / "summary.json").read_text())
        sources.append({"run_id": summary["run_id"], "run_no": summary["run_no"],
                        "word_artifacts": summary["word_artifacts"]})
        for status in read_jsonl(directory / "term_status.jsonl"):
            key = status["task_key"]
            if key in seen_terms:
                raise RuntimeError("Duplicate term across runs: " + key)
            seen_terms.add(key)
            statuses.append(status)
            shutil.copy2(directory / "by_word" / (key + ".json"), args.out / "by_word" / (key + ".json"))
        for row in read_jsonl(directory / "examples.jsonl"):
            digest = row["content_sha256"]
            if digest in seen_content:
                raise RuntimeError("Duplicate example across runs: " + row["sample_id"])
            seen_content.add(digest)
            examples.append(row)
    if len(seen_terms) != args.expected_terms:
        raise RuntimeError(f"Expected {args.expected_terms} term artifacts, got {len(seen_terms)}")
    examples.sort(key=lambda x: x["sample_id"])
    statuses.sort(key=lambda x: x["task_key"])
    write_jsonl(args.out / "examples.jsonl", examples)
    write_jsonl(args.out / "term_status.jsonl", statuses)
    summary = {"expected_terms": args.expected_terms, "word_artifacts": len(statuses),
               "complete_words": sum(x["complete"] for x in statuses),
               "valid_examples": len(examples),
               "valid_examples_by_language": dict(collections.Counter(x["language"] for x in examples)),
               "valid_examples_by_label": dict(collections.Counter(x["label"] for x in examples)),
               "error_types": dict(collections.Counter(e.split(":")[0] for x in statuses for e in x["errors"])),
               "source_runs": sources, "synthetic_labels_are_gold": False}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
