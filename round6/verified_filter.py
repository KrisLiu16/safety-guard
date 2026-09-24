"""Keep only generated safe rows that the second judge confirmed (Mac, CPU, no model calls).

Scale-up policy for S2 and the S5 safe half (collab/BOARD.md, 2026-09-24): a generated row, labelled safe by
construction, is kept only if every judged item built from it (the whole text and its prefix probe, via
judge_v3/make_pilot.py) came back `safe`. Any unsafe, controversial or missing verdict drops the row; nothing
is relabelled. Optionally, generated user questions (S2 prompts.jsonl) are kept when their word still has at
least one verified assistant row. Human review then samples the kept rows for an audit.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify(rows, items, judgments):
    """Pure: return (kept rows, drop reasons Counter)."""
    label = {j["item_id"]: j.get("judge_label") for j in judgments}
    by_sample = collections.defaultdict(list)
    for item in items:
        by_sample[item["sample_id"]].append(label.get(item["item_id"]))
    kept, reasons = [], collections.Counter()
    for row in rows:
        verdicts = by_sample.get(row["sample_id"])
        if not verdicts:
            reasons["not_judged"] += 1
        elif any(v is None for v in verdicts):
            reasons["missing_verdict"] += 1
        elif any(v != "safe" for v in verdicts):
            reasons["judged_" + next(v for v in verdicts if v != "safe")] += 1
        else:
            kept.append({**row, "annotation_origin": "synthetic_safe_confirmed_by_second_judge"})
    return kept, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="extracted examples.jsonl (assistant rows)")
    parser.add_argument("--items", type=Path, required=True, help="judge answer key items.jsonl")
    parser.add_argument("--judgments", type=Path, required=True, help="judge judgments.jsonl")
    parser.add_argument("--prompts", type=Path, help="optional S2 prompts.jsonl (user questions)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.examples)
    kept, reasons = verify(rows, read_jsonl(args.items), read_jsonl(args.judgments))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "verified.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"rows": len(rows), "kept": len(kept), "dropped": dict(reasons),
               "kept_by_split": dict(collections.Counter(r.get("split") for r in kept)),
               "kept_by_style": dict(collections.Counter(r.get("response_style") for r in kept))}
    if args.prompts:
        words = {r["task_key"] for r in kept}
        prompts = [p for p in read_jsonl(args.prompts) if p["task_key"] in words]
        with (args.out / "verified_prompts.jsonl").open("w", encoding="utf-8") as handle:
            for row in prompts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["prompts_kept"] = len(prompts)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
