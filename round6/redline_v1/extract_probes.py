"""Download red-line labelling archives and join them with the local answer key (Mac, no model calls).

Writes probes.jsonl: one row per response with the flow's status, every probe (cut, facts) and errors, plus
failed_ids.txt (responses with no usable result, for a fallback-judge run built with make_tasks.py --sample-ids).
Labels are derived afterwards by apply_policy.py, which can use another policy table without re-judging.
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

FINISHED = ("safe", "excluded", "located")


def join(key_rows, values):
    """Pure: flow records joined with the answer key; returns (rows, summary)."""
    by_id = {v["item_id"]: v for v in values}
    rows, statuses, calls = [], collections.Counter(), []
    for key in key_rows:
        value = by_id.get(key["sample_id"])
        result = value["result"] if value else {"status": "missing"}
        statuses[result["status"]] += 1
        if value:
            calls.append(len(result.get("probes", [])))
        rows.append({**key, "status": result["status"], "clauses": result.get("clauses"),
                     "probes": [{"cut": c["cut"], "facts": c["facts"]} for c in (value or {}).get("calls", [])],
                     "errors": (value or {}).get("errors", ["missing"])})
    summary = {"responses": len(rows), "statuses": dict(statuses),
               "judge_calls_per_response": {"mean": round(statistics.mean(calls), 2), "max": max(calls)} if calls else {}}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--items", type=Path, required=True, help="local answer key from make_tasks.py")
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
    rows, summary = join(key_rows, values)
    summary.update(runs=runs, archives=len(archives))
    with (args.out / "probes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    failed = [r["sample_id"] for r in rows if r["status"] not in FINISHED]
    (args.out / "failed_ids.txt").write_text("".join(i + "\n" for i in failed), encoding="utf-8")
    summary["failed_ids"] = len(failed)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
