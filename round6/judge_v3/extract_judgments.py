#!/usr/bin/env python3
"""Download judge v3 archives, join with the local answer key, report agreement and category coverage.

Agreement is against the generator labels written under the old 口径 (expected_old), split by
source group, so the effect of the domestic 口径 shows up as disagreement concentrated in the
political groups. Also tallies qwen_category and gbt_codes for unsafe/controversial judgments.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aster_io import collect_archives  # noqa: E402




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="one or more run numbers (a job over 2,000 Tasks is split into several Runs)")
    parser.add_argument("--items", type=Path, required=True, help="local answer key items.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-judgments", type=Path,
                        help="judgments.jsonl of another run on the same items; its labels replace the answer key's "
                             "reference_label, so cross_judge compares the two runs directly")
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[],
                        help="frozen listings from response_v14/freeze_attempts.py, one per run in the same order; "
                             "needed for any run over 100 Tasks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    key = {json.loads(line)["item_id"]: json.loads(line) for line in args.items.open(encoding="utf-8")}
    if args.reference_judgments:
        other = {j["item_id"]: j.get("judge_label") for j in map(json.loads, args.reference_judgments.open(encoding="utf-8"))}
        for item_id, item in key.items():
            item["reference_label"] = other.get(item_id)
    archives, runs = collect_archives(args.runs, args.attempts_json, args.out / "archives")
    judged = {}
    for path in archives:
        with tarfile.open(path) as tar:
            for member in tar:
                if not (member.isfile() and member.name.startswith("flow/results/") and member.name.endswith(".json")):
                    continue
                record = json.load(tar.extractfile(member))
                if str(record.get("name", "")).startswith("item_"):
                    value = record["value"]
                    judged[value["item"]["item_id"]] = value
    rows, table = [], collections.defaultdict(collections.Counter)
    failures = collections.Counter()
    for item_id, item in key.items():
        value = judged.get(item_id)
        label = value["judgment"]["label"] if value and value.get("judgment") else None
        if label is None:
            failures[(value or {}).get("errors", ["missing"])[0].split(":")[0] if value else "missing"] += 1
        table[item["kind"]][label or "no_judgment"] += 1
        row = {**{k: item[k] for k in ("item_id", "sample_id", "word", "kind", "expected_old", "source_group", "response_style", "cut_char")},
               "judge_label": label, "judge_basis": value["judgment"]["harm_basis"] if label else None,
               "judge_qwen_category": value["judgment"]["qwen_category"] if label else None,
               "judge_gbt_codes": value["judgment"]["gbt_codes"] if label else None,
               # present only for a second-judge (fallback) run built by make_fallback.py
               **{k: item[k] for k in ("fallback_reason", "reference_label") if k in item},
               "judge_reason": value["judgment"]["reason"] if label else None,
               "errors": value.get("errors") if value else ["missing"]}
        rows.append(row)
    with (args.out / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    judged_rows = [r for r in rows if r["judge_label"]]

    def agreement(selected):
        return {kind: round(sum(1 for r in selected if r["kind"] == kind and r["judge_label"] == r["expected_old"])
                            / max(1, sum(1 for r in selected if r["kind"] == kind)), 4)
                for kind in sorted({r["kind"] for r in selected})}

    by_group = collections.defaultdict(list)
    for r in judged_rows:
        by_group[r["source_group"]].append(r)
    flagged = [r for r in judged_rows if r["judge_label"] != "safe"]
    summary = {"runs": runs, "items": len(key),
               "judged": len(judged_rows), "failures": dict(failures),
               "by_kind": {kind: dict(counter) for kind, counter in table.items()},
               "agreement_with_old_labels": agreement(judged_rows),
               "agreement_by_source_group": {g: {"items": len(rs), "by_kind": agreement(rs)} for g, rs in sorted(by_group.items())},
               "flips_old_safe_to_unsafe_by_source_group": dict(collections.Counter(
                   r["source_group"] for r in judged_rows if r["expected_old"] == "safe" and r["judge_label"] == "unsafe")),
               "qwen_category": dict(collections.Counter(r["judge_qwen_category"] for r in flagged)),
               "gbt_codes": dict(collections.Counter(c for r in flagged for c in r["judge_gbt_codes"]))}
    if any("fallback_reason" in r for r in rows):
        summary["cross_judge"] = cross_judge(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def cross_judge(rows):
    """Second judge vs the first run's verdicts, per fallback_reason (failed / political / control)."""
    out = {}
    for reason in sorted({r["fallback_reason"] for r in rows}):
        group = [r for r in rows if r["fallback_reason"] == reason]
        both = [r for r in group if r["judge_label"] and r["reference_label"]]
        out[reason] = {"items": len(group), "judged_now": sum(1 for r in group if r["judge_label"]),
                       "agreement_where_both_judged": round(sum(r["judge_label"] == r["reference_label"] for r in both)
                                                            / len(both), 4) if both else None,
                       "reference_to_now": dict(collections.Counter(
                           f"{r['reference_label'] or 'none'}->{r['judge_label'] or 'none'}" for r in group))}
    return out


if __name__ == "__main__":
    main()
