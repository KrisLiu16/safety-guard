#!/usr/bin/env python3
"""Download word-screen archives and join verdicts with the local word index (Mac, no model calls).

screen.jsonl : one row per word: final verdict (a hand label where given, otherwise the model's), the
               model's merged verdict (most severe across passes), per-pass verdicts and source groups;
summary.json : verdict counts overall and per source group, batch errors, and recall on known positives
               (any verdict other than "no" counts as flagged).
Words left without a verdict are listed so they can be re-screened in a later batch.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "flow"))
from pipeline import FLAGGED, merge  # noqa: E402


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


MANUAL_TO_VERDICT = {"insult": "insult", "rumor": "rumor", "evasion": "evasion", "no": "no", "borderline": "unsure"}


def join(index_rows, batch_values, manual=None):
    """Pure: merge every pass's verdict per word (most severe wins); a human label, where given, overrides it.
    Returns (rows, summary); each row keeps the model verdict next to the final one."""
    per_word = collections.defaultdict(list)
    per_word_tagged = collections.defaultdict(dict)
    errors = collections.Counter()
    passes = set()
    for value in batch_values:
        for e in value.get("errors", []):
            errors[e.split(":")[0]] += 1
        key = value.get("batch_key", "")
        tag = key.split("-")[1] if key.count("-") >= 2 else "p0"
        passes.add(tag)
        for v in value.get("verdicts", []):
            per_word[v["word"]].append(v["verdict"])
            per_word_tagged[v["word"]][tag] = v["verdict"]
    manual = manual or {}
    rows, by_group = [], collections.defaultdict(collections.Counter)
    for entry in index_rows:
        model = merge(per_word.get(entry["word"], []))
        final, source = (MANUAL_TO_VERDICT[manual[entry["word"]]], "manual") if entry["word"] in manual else (model, "model")
        rows.append({**entry, "verdict": final, "verdict_source": source, "model_verdict": model,
                     "pass_verdicts": per_word_tagged.get(entry["word"], {})})
        for group in entry["source_groups"]:
            by_group[group][final] += 1
    known = [r for r in rows if r.get("known_positive")]
    category = [r for r in rows if {"p0", "p1"} <= set(r["pass_verdicts"])]
    summary = {"words": len(rows), "passes_seen": sorted(passes),
               "verdicts": dict(collections.Counter(r["verdict"] for r in rows)),
               "model_verdicts": dict(collections.Counter(r["model_verdict"] for r in rows)),
               "manual_overrides": sum(r["verdict_source"] == "manual" for r in rows),
               "words_with_both_category_passes": len(category),
               "pass_agreement_flagged": (round(sum((r["pass_verdicts"]["p0"] in FLAGGED) == (r["pass_verdicts"]["p1"] in FLAGGED)
                                                    for r in category) / len(category), 4) if category else None),
               "flagged_only_by_binary_pass": sum(r["model_verdict"] != "no" and all(
                   v == "no" for t, v in r["pass_verdicts"].items() if t.startswith("p")) for r in rows),
               "batch_errors": dict(errors),
               "known_positives": len(known),
               "known_positive_model_verdicts": dict(collections.Counter(r["model_verdict"] for r in known)),
               "known_positive_model_recall_flagged": (round(sum(r["model_verdict"] in FLAGGED for r in known) / len(known), 4)
                                                       if known else None),
               "by_source_group": {g: dict(c) for g, c in sorted(by_group.items())}}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="one or more run numbers (a screen over 2,000 Tasks is split into several Runs)")
    parser.add_argument("--words", type=Path, required=True, help="words.jsonl written by make_batch.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manual", type=Path,
                        help="local hand labels (word<TAB>label, git-ignored); they override the model verdict")
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[],
                        help="frozen listings from response_v14/freeze_attempts.py, one per run, in the same order; "
                             "needed for any run with over 100 Tasks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.attempts_json and len(args.attempts_json) != len(args.runs):
        raise ValueError("give one --attempts-json per run, in the same order, or none")
    archives_dir = args.out / "archives"
    archives_dir.mkdir(exist_ok=True)
    archives, runs = [], []
    for position, run_no in enumerate(args.runs):
        run = cli(["runs", "get", run_no])
        runs.append({"run_no": run.get("run_no"), "status": run.get("status")})
        if args.attempts_json:
            listing = json.loads(args.attempts_json[position].read_text())
            if listing.get("run_id") != run.get("id"):
                raise RuntimeError(f"attempts listing {position} belongs to {listing.get('run_id')}, not {run.get('id')}")
        else:
            listing = cli(["runs", "attempts", run_no])
        if listing.get("truncated"):
            raise RuntimeError("attempt listing truncated; freeze it first (response_v14/freeze_attempts.py)")
        for attempt in listing["items"]:
            if attempt.get("state") != "completed" or not attempt.get("archive"):
                continue
            path = archives_dir / (attempt["attempt_id"] + ".tar.gz")
            if not path.exists():
                receipt = cli(["runs", "archive", run_no, attempt["attempt_id"], "--out", str(path)])
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise RuntimeError("archive SHA mismatch " + attempt["attempt_id"])
            archives.append(path)
    index_rows = [json.loads(line) for line in args.words.open(encoding="utf-8")]
    manual = None
    if args.manual:
        manual = dict(line.rsplit("\t", 1) for line in args.manual.read_text(encoding="utf-8").splitlines() if line.strip())
        unknown = set(manual.values()) - set(MANUAL_TO_VERDICT)
        if unknown:
            raise ValueError(f"unknown manual labels {sorted(unknown)}")
    rows, summary = join(index_rows, batches(archives), manual)
    summary.update(runs=runs, archives=len(archives))
    with (args.out / "screen.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
