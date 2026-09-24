"""Write a local human-review sheet for a labelled pilot (Mac, CPU). The sheet holds dataset text: never commit it.

Samples responses by (old label -> new label) transition, round-robin in hash order, so every kind of change is
seen (e.g. old unsafe -> safe should be harm outside the red lines). Each entry shows the labels, the stratum, the
rules that set the level, and the response with the located onsets marked in place:
  ⟦C⟧ where the controversial onset clause starts, ⟦C|⟧ where it ends; ⟦U⟧ / ⟦U|⟧ likewise for unsafe.
The reviewer answers two questions per entry (label right under POLICY-redline-scope.md? onset clause right?);
only the counts go into the feedback.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

MARKS = (("controversial", "C"), ("unsafe", "U"))


def pick(labels, count, seed):
    pools = collections.defaultdict(list)
    for row in sorted(labels, key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest()):
        if row["label"] != "unusable":
            pools[f"{row['old_label']}->{row['label']}"].append(row)
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(chosen) < count:
                chosen.append(pools[key].pop(0))
    return chosen


def marked(text, onsets):
    inserts = []
    for name, letter in MARKS:
        if name in onsets:
            inserts += [(onsets[name]["prev_cut"], f"⟦{letter}⟧"), (onsets[name]["cut"], f"⟦{letter}|⟧")]
    for position, mark in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:position] + mark + text[position:]
    return text


def sheet(chosen, rows):
    parts = []
    for n, lab in enumerate(chosen, 1):
        row = rows[lab["sample_id"]]
        parts.append(f"## {n}. {lab['sample_id']}  旧 {lab['old_label']} → 新 {lab['label']}  分层 {lab['stratum']}\n"
                     f"规则：{', '.join(lab['rules']) or '无'}\n\n【提问】\n{row['messages'][-2]['content'] if len(row['messages']) > 1 else ''}\n\n"
                     f"【回答】\n{marked(row['messages'][-1]['content'], lab['onsets'])}\n\n"
                     "标签对不对（对 / 漏判红线 / 误判红线 / 档位错）：\n起点分句对不对（对 / 偏早 / 偏晚 / 不适用）：\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl from apply_policy.py")
    parser.add_argument("--source", type=Path, required=True, help="the v14-format rows the Tasks were built from")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", default="redline-review-v1")
    parser.add_argument("--out", type=Path, required=True, help="directory for review.txt (gitignored: review/)")
    args = parser.parse_args()
    with args.labels.open(encoding="utf-8") as handle:
        labels = [json.loads(line) for line in handle if line.strip()]
    wanted = {r["sample_id"] for r in labels}
    with args.source.open(encoding="utf-8") as handle:
        rows = {r["sample_id"]: r for r in map(json.loads, handle) if r["sample_id"] in wanted}
    chosen = pick(labels, args.count, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review.txt").write_text(sheet(chosen, rows), encoding="utf-8")
    print(json.dumps(dict(collections.Counter(f"{r['old_label']}->{r['label']}" for r in chosen)), ensure_ascii=False))


if __name__ == "__main__":
    main()
