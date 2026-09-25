"""Stage 2 spot summary (Mac, CPU): score_texts_l20.py output against labels, per checkpoint.

For every checkpoint and case: the stream score (max over the last message's tokens) of 1 - p(safe), the first
position above each --tau and, when a labels.jsonl is given, the case's label, stratum, alert flag and kind
(response_style). Printed: per checkpoint and group (kind or label), how many cases, the median stream score and
the share above each --tau. No text is printed: the cases stay local.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:       # not splitlines(): texts may hold U+2028
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--labels", type=Path, nargs="*", default=[])
    parser.add_argument("--rows", type=Path, help="source rows (for response_style)")
    parser.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--group", choices=("kind", "label", "id"), default="kind")
    args = parser.parse_args()
    labels = {r["sample_id"]: r for p in args.labels for r in read_jsonl(p)}
    styles = {r["sample_id"]: r.get("response_style") for r in read_jsonl(args.rows)} if args.rows else {}
    table = collections.defaultdict(list)
    for row in read_jsonl(args.scores):
        lab = labels.get(row["id"], {})
        if args.group == "id":
            key = row["id"]
        elif args.group == "label":
            key = lab.get("label", "?") + ("+alert" if lab.get("alert") else "")
        else:
            key = f"{styles.get(row['id'], '?')}:{lab.get('label', '?')}"
        table[(Path(row["checkpoint"]).parent.name + "/" + Path(row["checkpoint"]).name, key)].append(row["max_cut"])
    for (checkpoint, key), values in sorted(table.items()):
        shares = " ".join(f">{tau:g}: {sum(v > tau for v in values) / len(values):.0%}" for tau in args.tau)
        print(f"{checkpoint:45s} {key:40s} n={len(values):4d} median={statistics.median(values):.3f} {shares}")


if __name__ == "__main__":
    main()
