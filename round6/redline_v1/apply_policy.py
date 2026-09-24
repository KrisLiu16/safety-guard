"""Derive labels from stored probe facts under a policy table (Mac, CPU, no model calls).

Inputs are one or more probes.jsonl files from extract_probes.py in priority order (e.g. DeepSeek first, the luna
fallback run second); each response takes the first file where it has a usable result. Every probe's facts go
through policy.decide() with the given switches (default: the user's final table), so a policy change is a re-run
of this script, not of the judge. Output labels.jsonl keeps the per-probe levels; position labels come from
levels.position_level(). A response is usable when its whole text was judged and its probe levels are monotonic.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
from levels import RANK, monotonic  # noqa: E402
from policy import BOUNDARY_CODES, CHOICES, decide, digest, resolve  # noqa: E402

FINISHED = ("safe", "excluded", "located")


def label_response(row, switches=None):
    """Pure. Returns the label row, or None when the response has no usable result in this file."""
    if row["status"] not in FINISHED:
        return None
    probes = sorted(({"cut": p["cut"], **decide(p["facts"], switches)} for p in row["probes"]), key=lambda p: p["cut"])
    whole = next((p for p in probes if p["cut"] == row["response_chars"]), None)
    if whole is None or not monotonic(probes):
        return None
    onsets = {}
    for name, rank in (("controversial", 1), ("unsafe", 2)):
        if RANK[whole["level"]] >= rank:
            cut = min(p["cut"] for p in probes if RANK[p["level"]] >= rank)
            prev = max([p["cut"] for p in probes if RANK[p["level"]] < rank and p["cut"] < cut], default=0)
            onsets[name] = {"prev_cut": prev, "cut": cut}
    whole_facts = next(p["facts"] for p in row["probes"] if p["cut"] == row["response_chars"])
    return {"level": whole["level"], "label": whole["label"], "excluded": whole["excluded"],
            "stratum": whole["stratum"], "rules": whole["rules"], "onsets": onsets,
            "boundary": sorted({f"{h['code']}:{h['act']}" for h in whole_facts["hits"]
                                if h["code"] in BOUNDARY_CODES or h["act"] == "term_explained"}),
            "probes": [{"cut": p["cut"], "level": p["level"]} for p in probes]}


def sensitivity(rows, switches=None):
    """For each pending switch (boundary codes, term_explained): how many usable labels change if it is flipped."""
    base = resolve(switches)
    out = {}
    for name in (*BOUNDARY_CODES, "term_explained"):
        for value in CHOICES[name]:
            if value == base[name]:
                continue
            changed = collections.Counter()
            for row in rows:
                before, after = label_response(row, switches), label_response(row, {**base, name: value})
                if before is not None and after is not None and before["label"] != after["label"]:
                    changed[f"{before['label']}->{after['label']}"] += 1
            out[f"{name}={value}"] = dict(changed)
    return out


def apply(sources, switches=None):
    """Pure. sources: lists of probe rows in priority order. Returns (label rows, summary)."""
    order, chosen = [], {}
    for rank, rows in enumerate(sources):
        for row in rows:
            if row["sample_id"] not in chosen:
                order.append(row["sample_id"])
                chosen[row["sample_id"]] = (None, None, row)
            if chosen[row["sample_id"]][0] is None:
                labelled = label_response(row, switches)
                if labelled is not None:
                    chosen[row["sample_id"]] = (rank, labelled, row)
    used_rows = [row for rank, labelled, row in chosen.values() if rank is not None]
    out = []
    for sample_id in order:
        rank, labelled, row = chosen[sample_id]
        key = {k: v for k, v in row.items() if k not in ("probes", "errors", "status", "clauses")}
        out.append({**key, "old_label": row.get("label"), "source_rank": rank,
                    **(labelled or {"level": None, "label": "unusable", "excluded": None, "stratum": None,
                                    "rules": [], "onsets": {}, "boundary": [], "probes": []})})
    usable = [r for r in out if r["source_rank"] is not None]
    gaps = collections.defaultdict(list)
    for r in usable:
        for name, onset in r["onsets"].items():
            gaps[name].append(onset["cut"] - onset["prev_cut"])
    summary = {"policy_digest": digest(switches), "switches": resolve(switches), "responses": len(out),
               "usable": len(usable), "by_source_rank": dict(collections.Counter(str(r["source_rank"]) for r in out)),
               "labels_by_split": {s: dict(collections.Counter(r["label"] for r in out if r.get("split") == s))
                                   for s in sorted({str(r.get("split")) for r in out})},
               "old_to_new": dict(collections.Counter(f"{r['old_label']}->{r['label']}" for r in out)),
               "old_to_new_by_slot_style": dict(sorted(collections.Counter(
                   f"slot{r.get('index')}:{r.get('response_style')}:{r['old_label']}->{r['label']}" for r in out).items())),
               "stratum_by_label": {label: dict(collections.Counter(r["stratum"] for r in usable if r["label"] == label))
                                    for label in sorted({r["label"] for r in usable})},
               "rules_non_safe": dict(collections.Counter(rule for r in usable if r["level"] != "safe"
                                                          for rule in r["rules"]).most_common()),
               "onset_clause_chars": {k: {"median": statistics.median(v), "max": max(v)} for k, v in gaps.items()},
               "boundary_hits_by_label": {label: dict(collections.Counter(b for r in usable if r["label"] == label
                                                                         for b in r["boundary"]))
                                          for label in sorted({r["label"] for r in usable})},
               "switch_sensitivity": sensitivity(used_rows, switches)}
    return out, summary


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, nargs="+", help="probes.jsonl files, highest priority first")
    parser.add_argument("--switches", type=Path, help="JSON object overriding policy switches (default: final table)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    switches = json.loads(args.switches.read_text(encoding="utf-8")) if args.switches else None
    rows, summary = apply([read_jsonl(p) for p in args.probes], switches)
    summary["inputs"] = [str(p) for p in args.probes]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("policy_digest", "responses", "usable", "by_source_rank",
                                              "labels_by_split", "old_to_new")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
