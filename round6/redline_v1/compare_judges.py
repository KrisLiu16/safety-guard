"""Compare two judges' labels on the same responses (Mac, CPU): whole-text labels and onset positions.

Each input is a labels.jsonl written by apply_policy.py from ONE judge's probes.jsonl (no fallback merge), so the
comparison is judge against judge under the same policy table. Reported: the label transition matrix, agreement
on responses both judged, agreement on the reporting stratum, and, where both located the same onset level, how
far apart the onset clauses are.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics


def compare(first, second):
    """Pure. first/second: label rows. Returns the summary."""
    other = {r["sample_id"]: r for r in second}
    both = [(a, other[a["sample_id"]]) for a in first
            if a["sample_id"] in other and a["label"] != "unusable" and other[a["sample_id"]]["label"] != "unusable"]
    shifts = collections.defaultdict(list)
    for a, b in both:
        for name in ("controversial", "unsafe"):
            if name in a["onsets"] and name in b["onsets"]:
                shifts[name].append(b["onsets"][name]["cut"] - a["onsets"][name]["cut"])
    return {"responses": len(first), "both_usable": len(both),
            "label_agreement": round(sum(a["label"] == b["label"] for a, b in both) / len(both), 4) if both else None,
            "cut_agreement": round(sum((a["level"] == "safe") == (b["level"] == "safe") for a, b in both) / len(both), 4)
            if both else None,
            "stratum_agreement": round(sum(a["stratum"] == b["stratum"] for a, b in both) / len(both), 4) if both else None,
            "transitions": dict(collections.Counter(f"{a['label']}->{b['label']}" for a, b in both)),
            "onset_shift_chars": {k: {"n": len(v), "same_cut": sum(x == 0 for x in v), "median": statistics.median(v)}
                                  for k, v in shifts.items()}}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="labels.jsonl of the first judge")
    parser.add_argument("second", type=Path, help="labels.jsonl of the second judge")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(read_jsonl(args.first), read_jsonl(args.second))
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
