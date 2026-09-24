#!/usr/bin/env python3
"""Download word-screen archives and join verdicts with the local word index (Mac, no model calls).

screen.jsonl : one row per word with its verdict (yes / no / unsure / missing) and source groups;
summary.json : verdict counts overall and per source group, batch errors, and recall on known positives.
Words left without a verdict are listed so they can be re-screened in a later batch.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def cli(args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"aster {' '.join(args[:2])}: {error.get('code')}: {error.get('message')}")
    return envelope["data"]


def batches(archives):
    for path in archives:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith("flow/results/") and member.name.endswith(".json"):
                    record = json.load(tar.extractfile(member))
                    if record.get("name") == "batch" and isinstance(record.get("value"), dict):
                        yield record["value"]


def join(index_rows, batch_values):
    """Pure: return (rows, summary)."""
    verdict = {}
    errors = collections.Counter()
    for value in batch_values:
        for e in value.get("errors", []):
            errors[e.split(":")[0]] += 1
        for v in value.get("verdicts", []):
            verdict[v["word"]] = v["verdict"]
    rows, by_group = [], collections.defaultdict(collections.Counter)
    for entry in index_rows:
        state = verdict.get(entry["word"], "missing")
        rows.append({**entry, "verdict": state})
        for group in entry["source_groups"]:
            by_group[group][state] += 1
    known = [r for r in rows if r.get("known_positive")]
    summary = {"words": len(rows), "verdicts": dict(collections.Counter(r["verdict"] for r in rows)),
               "batch_errors": dict(errors),
               "known_positives": len(known),
               "known_positive_verdicts": dict(collections.Counter(r["verdict"] for r in known)),
               "known_positive_recall_yes_or_unsure": (round(sum(r["verdict"] in ("yes", "unsure") for r in known) / len(known), 4)
                                                       if known else None),
               "by_source_group": {g: dict(c) for g, c in sorted(by_group.items())}}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--words", type=Path, required=True, help="words.jsonl written by make_batch.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts-json", type=Path,
                        help="frozen listing from response_v14/freeze_attempts.py when the run has over 100 samples")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = cli(["runs", "get", args.run])
    listing = json.loads(args.attempts_json.read_text()) if args.attempts_json else cli(["runs", "attempts", args.run])
    if listing.get("truncated"):
        raise RuntimeError("attempt listing truncated; freeze it first (response_v14/freeze_attempts.py)")
    archives_dir = args.out / "archives"
    archives_dir.mkdir(exist_ok=True)
    archives = []
    for attempt in listing["items"]:
        if attempt.get("state") != "completed" or not attempt.get("archive"):
            continue
        path = archives_dir / (attempt["attempt_id"] + ".tar.gz")
        if not path.exists():
            receipt = cli(["runs", "archive", args.run, attempt["attempt_id"], "--out", str(path)])
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                raise RuntimeError("archive SHA mismatch " + attempt["attempt_id"])
        archives.append(path)
    index_rows = [json.loads(line) for line in args.words.open(encoding="utf-8")]
    rows, summary = join(index_rows, batches(archives))
    summary.update(run_no=run.get("run_no"), run_status=run.get("status"), archives=len(archives))
    with (args.out / "screen.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
