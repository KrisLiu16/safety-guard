"""Write a local human-review sheet for a labelled pilot (Mac, CPU). The sheet holds dataset text: never commit it.

Samples responses by (old label -> new label) transition, round-robin in hash order, so every kind of change is
seen (e.g. old unsafe -> safe should be harm outside the red lines). Each entry shows the labels, the stratum, the
rules that set the level, and the response with the located onsets marked in place:
  ⟦C⟧ where the controversial onset clause starts, ⟦C|⟧ where it ends; ⟦U⟧ / ⟦U|⟧ likewise for unsafe.
The reviewer answers two questions per entry (label right under POLICY-redline-scope.md? onset clause right?);
only the counts go into the feedback.
--only-rule PREFIX reviews only responses where a rule starting with PREFIX fired (e.g. "political" or
"leader_word" for the T026 word rules, "screen" for the seed-word screen), pooled by the stratum and label before
the word rule; the span of each fired word rule is marked ⟦P⟧...⟦P|⟧.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

MARKS = (("controversial", "C"), ("unsafe", "U"))


def fired_rules(row):
    """Every rule that set a level or fired a word rule on this response."""
    return (list(row.get("rules", [])) + [f["rule"] for f in row.get("political", {}).get("fired", [])]
            + ([f"screen:{row['screen']['verdict']}"] if row.get("screen") else []))


def pick(labels, count, seed, only_rule=None):
    pools = collections.defaultdict(list)
    for row in sorted(labels, key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest()):
        if row["label"] == "unusable":
            continue
        if only_rule is None:
            pools[f"{row['old_label']}->{row['label']}"].append(row)
        elif any(rule.startswith(only_rule) for rule in fired_rules(row)):
            before = row.get("political") or {"stratum_before": row["stratum"], "label_before": row["label"]}
            pools[f"{before['stratum_before']}:{before['label_before']}->{row['label']}"].append(row)
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(chosen) < count:
                chosen.append(pools[key].pop(0))
    return chosen


def marked(text, onsets, spans=()):
    inserts = [mark for start, end in spans for mark in ((start, "⟦P⟧"), (end, "⟦P|⟧"))]
    for name, letter in MARKS:
        if name in onsets:
            inserts += [(onsets[name]["prev_cut"], f"⟦{letter}⟧"), (onsets[name]["cut"], f"⟦{letter}|⟧")]
    for position, mark in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:position] + mark + text[position:]
    return text


def spans(lab):
    return [(f["start"], f["end"]) for f in lab.get("political", {}).get("fired", [])]


def sheet(chosen, rows, target="assistant"):
    """The labelled text is the last message: the answer (assistant) or the prompt itself (user)."""
    before, labelled = ("【提问】", "【回答】") if target == "assistant" else ("【此前的对话】", "【提问（标注对象）】")
    parts = []
    for n, lab in enumerate(chosen, 1):
        row = rows[lab["sample_id"]]
        parts.append(f"## {n}. {lab['sample_id']}  旧 {lab['old_label']} → 新 {lab['label']}  分层 {lab['stratum']}\n"
                     f"规则：{', '.join(fired_rules(lab)) or '无'}\n\n{before}\n{row['messages'][-2]['content'] if len(row['messages']) > 1 else ''}\n\n"
                     f"{labelled}\n{marked(row['messages'][-1]['content'], lab['onsets'], spans(lab))}\n\n"
                     "标签对不对（对 / 漏判红线 / 误判红线 / 档位错）：\n起点分句对不对（对 / 偏早 / 偏晚 / 不适用）：\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl from apply_policy.py")
    parser.add_argument("--source", type=Path, required=True, help="the v14-format rows the Tasks were built from")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", default="redline-review-v1")
    parser.add_argument("--out", type=Path, required=True, help="directory for review.txt (gitignored: review/)")
    parser.add_argument("--target", choices=("assistant", "user"), default="assistant")
    parser.add_argument("--only-rule", help="review only responses where a rule with this prefix fired")
    args = parser.parse_args()
    with args.labels.open(encoding="utf-8") as handle:
        labels = [json.loads(line) for line in handle if line.strip()]
    wanted = {r["sample_id"] for r in labels}
    with args.source.open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    if args.target == "user":
        from make_tasks import prompt_rows
        source = prompt_rows(source)
    rows = {r["sample_id"]: r for r in source if r["sample_id"] in wanted}
    chosen = pick(labels, args.count, args.seed, args.only_rule)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review.txt").write_text(sheet(chosen, rows, args.target), encoding="utf-8")
    print(json.dumps(dict(collections.Counter(f"{r['old_label']}->{r['label']}" for r in chosen)), ensure_ascii=False))


if __name__ == "__main__":
    main()
