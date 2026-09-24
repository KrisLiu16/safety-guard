"""Download onset-relocation archives and compare the judge-located onset with the v14 onset (Mac, no model calls).

For each located response the first unsafe clause is (prev_cut, cut]. Relative to the v14 quote [onset_char,
onset_end_char): v14_late when the v14 quote starts at or after `cut` (harmful text was already there),
v14_early when the v14 quote ends at or before `prev_cut` (v14 marked harm the judge did not yet see),
overlap otherwise.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from aster_io import collect_archives  # noqa: E402


def relation(result, key):
    if result.get("status") != "located":
        return None
    if key["onset_char"] >= result["cut"]:
        return "v14_late"
    if key["onset_end_char"] <= result["prev_cut"]:
        return "v14_early"
    return "overlap"


def summarize(key_rows, values):
    """Pure: join flow records with the answer key; return (rows, summary)."""
    by_id = {v["item_id"]: v for v in values}
    rows, statuses, relations = [], collections.Counter(), collections.defaultdict(collections.Counter)
    shifts, calls = collections.defaultdict(list), []
    for key in key_rows:
        value = by_id.get(key["sample_id"])
        result = value["result"] if value else {"status": "missing"}
        rel = relation(result, key)
        statuses[result["status"]] += 1
        if rel:
            relations[f"slot{key['slot']}"][rel] += 1
            shifts[f"slot{key['slot']}"].append(key["onset_char"] - result["prev_cut"])
        if value:
            calls.append(len(result.get("probes", [])))
        rows.append({**key, "status": result["status"], "cut": result.get("cut"), "prev_cut": result.get("prev_cut"),
                     "clauses": result.get("clauses"), "probes": result.get("probes"), "relation": rel,
                     "errors": (value or {}).get("errors")})
    summary = {"responses": len(rows), "statuses": dict(statuses),
               "relation_by_slot": {k: dict(v) for k, v in sorted(relations.items())},
               "v14_minus_located_start_chars": {k: {"median": statistics.median(v), "p10": sorted(v)[len(v) // 10],
                                                     "p90": sorted(v)[(9 * len(v)) // 10]} for k, v in sorted(shifts.items()) if v},
               "judge_calls_per_response": {"mean": round(statistics.mean(calls), 2), "max": max(calls)} if calls else {}}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--items", type=Path, required=True, help="local answer key from make_onset_tasks.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    archives, runs = collect_archives(args.runs, args.attempts_json, args.out / "archives")
    values = []
    for path in archives:
        with tarfile.open(path) as tar:
            for member in tar:
                if member.isfile() and member.name.startswith("flow/results/") and member.name.endswith(".json"):
                    record = json.load(tar.extractfile(member))
                    if str(record.get("name", "")).startswith("item_"):
                        values.append(record["value"])
    with args.items.open(encoding="utf-8") as handle:
        key_rows = [json.loads(line) for line in handle]
    rows, summary = summarize(key_rows, values)
    summary.update(runs=runs, archives=len(archives))
    with (args.out / "onsets.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
