"""Compare screen verdicts with local manual labels (Mac, CPU). Prints counts only, never the words.

manual_labels.tsv (local, git-ignored): one line per hand-checked word, `word<TAB>label`, label one of
insult / rumor / evasion / no / borderline. Output: a manual-label x verdict count table, the share of
manually positive words (insult, rumor, evasion) that the screen flags, and the share of manual "no"
words it flags.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "flow"))
from pipeline import FLAGGED, VERDICTS  # noqa: E402

MANUAL = ("insult", "rumor", "evasion", "borderline", "no")


def compare(manual, verdicts):
    table = collections.defaultdict(collections.Counter)
    for word, label in manual.items():
        table[label][verdicts.get(word, "not_screened")] += 1
    positives = [w for w, label in manual.items() if label in ("insult", "rumor", "evasion") and w in verdicts]
    negatives = [w for w, label in manual.items() if label == "no" and w in verdicts]
    return {"table": {label: dict(table[label]) for label in MANUAL if label in table},
            "manual_positive_flagged": (round(sum(verdicts[w] in FLAGGED for w in positives) / len(positives), 4)
                                        if positives else None),
            "manual_positive_n": len(positives),
            "manual_no_flagged": (round(sum(verdicts[w] in FLAGGED for w in negatives) / len(negatives), 4)
                                  if negatives else None),
            "manual_no_n": len(negatives)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True, help="screen.jsonl from extract_screen.py")
    args = parser.parse_args()
    manual = {}
    for line in args.manual.read_text(encoding="utf-8").splitlines():
        if line.strip():
            word, label = line.rsplit("\t", 1)
            if label not in MANUAL:
                raise ValueError(f"unknown manual label {label!r}; use one of {MANUAL}")
            manual[word] = label
    verdicts = {}
    with args.screen.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["verdict"] not in (*VERDICTS, "missing"):
                raise ValueError("screen file is not a v2 screen: " + row["verdict"])
            verdicts[row["word"]] = row["verdict"]
    print(json.dumps(compare(manual, verdicts), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
