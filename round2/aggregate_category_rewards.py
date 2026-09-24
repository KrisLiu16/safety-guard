#!/usr/bin/env python3
"""Compute user-defined complete-word reward over all source categories."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "full-source-index/manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statuses", type=Path, nargs="+", required=True,
                    help="One or more extracted term_status.jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--examples-per-term", type=int, choices=(2, 10), default=2,
                    help="Current v12 uses 2; pass 10 only for historical v8 results")
    args = ap.parse_args()
    manifest = json.loads(INDEX.read_text())
    totals = {x["source_group"]: x["count"] for x in manifest["categories"]}
    seen = set()
    counts = collections.defaultdict(lambda: {"attempted": 0, "complete": 0, "raw_complete": 0,
                                              "valid_examples": 0, "filtered": 0})
    for path in args.statuses:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key, group = row["task_key"], row["source_group"]
                if key in seen:
                    raise RuntimeError("Duplicate term result: " + key)
                if group not in totals:
                    raise RuntimeError("Unknown source group: " + group)
                if not 0 <= row["valid"] <= args.examples_per_term:
                    raise RuntimeError("Unexpected valid-example count for " + key)
                if row["complete"] and row["valid"] != args.examples_per_term:
                    raise RuntimeError("Complete term has wrong example count: " + key)
                seen.add(key)
                stat = counts[group]
                stat["attempted"] += 1
                stat["complete"] += int(row["complete"])
                stat["raw_complete"] += int(row.get("raw_complete", False))
                stat["valid_examples"] += row["valid"]
                stat["filtered"] += int(any("content_filter" in e for e in row["errors"]))
    categories = []
    for group in sorted(totals):
        stat = counts[group]
        total = totals[group]
        categories.append({"source_group": group, "total_terms": total, **stat,
                           "not_yet_recorded": total - stat["attempted"],
                           "reward": stat["complete"] / total,
                           "success_among_attempted": stat["complete"] / stat["attempted"] if stat["attempted"] else None})
    total = sum(totals.values())
    overall = {"total_terms": total, "attempted": len(seen),
               "complete": sum(x["complete"] for x in categories),
               "raw_complete": sum(x["raw_complete"] for x in categories),
               "filtered": sum(x["filtered"] for x in categories),
               "not_yet_recorded": total - len(seen)}
    overall["reward"] = overall["complete"] / total
    overall["success_among_attempted"] = overall["complete"] / overall["attempted"] if overall["attempted"] else None
    result = {"definition": f"complete {args.examples_per_term}-example terms / all terms in each source category",
              "examples_per_term": args.examples_per_term,
              "category_count": len(categories), "overall": overall, "categories": categories}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"category_count": len(categories), "overall": overall}, ensure_ascii=False))


if __name__ == "__main__":
    main()
