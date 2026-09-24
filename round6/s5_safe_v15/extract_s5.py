#!/usr/bin/env python3
"""Download S5-safe archives, re-validate every word from the raw response, write JSONL + summary (Mac, no model calls).

examples.jsonl is in the v14 row format (assistant rows), so judge_v3/make_pilot.py --source builds the
verification pilot directly. Validation is re-run with the frozen pipeline; the in-flow verdict is not trusted.
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, parse_json, response_text, size, stop_ok, validate  # noqa: E402


def cli(args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"aster {' '.join(args[:2])}: {error.get('code')}: {error.get('message')}")
    return envelope["data"]


def revalidate(value):
    response, wire = value.get("response"), value.get("wire_api")
    if not isinstance(response, dict):
        return value.get("errors") or ["missing_response"], []
    ok, reason = stop_ok(response, wire)
    if not ok:
        return ["stop_reason:" + reason], []
    try:
        return validate(parse_json(response_text(response, wire)), value["seed"])
    except (ValueError, TypeError, KeyError) as exc:
        return ["extract_parse:" + type(exc).__name__], []


def word_values(archives):
    for path in archives:
        with tarfile.open(path, "r:gz") as tar:
            for member in sorted((m for m in tar if m.isfile() and m.name.startswith("flow/results/")
                                  and m.name.endswith(".json")), key=lambda m: m.name):
                record = json.load(tar.extractfile(member))
                if str(record.get("name", "")).startswith("word_") and isinstance(record.get("value"), dict):
                    yield record["value"]


def quantiles(values):
    values = sorted(values)
    return {f"p{q}": values[min(len(values) - 1, int(q / 100 * len(values)))] for q in (10, 50, 90)} if values else {}


def summarize(values):
    """Pure: re-validate every word value; return (examples, words, summary)."""
    examples, words, seen = [], [], set()
    errors = collections.Counter()
    by_group, by_shape = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for value in values:
        seed = value["seed"]
        if seed["task_key"] in seen:
            raise RuntimeError("duplicate word result " + seed["task_key"])
        seen.add(seed["task_key"])
        word_errors, rows = revalidate(value)
        state = "complete" if rows else "skip" if word_errors == ["skip"] else "failed"
        by_group[seed.get("source_group", "")][state] += 1
        by_shape[seed["shape"]][state] += 1
        errors.update(e.split(":")[0] if e.startswith(("stop_reason", "request_or_parse", "extract_parse", "root_mismatch"))
                      else e for e in word_errors)
        examples.extend(rows)
        words.append({"task_key": seed["task_key"], "word": seed["word"], "shape": seed["shape"],
                      "split": seed.get("split"), "source_group": seed.get("source_group"), "state": state,
                      "errors": word_errors, "seconds": value.get("seconds"),
                      "usage": value["response"].get("usage") if isinstance(value.get("response"), dict) else None})
    lengths = collections.defaultdict(list)
    for row in examples:
        lengths[f"{row['language']}:{row['response_style']}"].append(size(row["messages"][1]["content"], row["language"]))
    summary = {"prompt_version": PROMPT_VERSION, "words": len(words),
               "states": dict(collections.Counter(w["state"] for w in words)), "assistant_rows": len(examples),
               "errors": dict(errors.most_common()),
               "length_units_by_language_style": {k: quantiles(v) for k, v in sorted(lengths.items())},
               "quality_flags": dict(collections.Counter(f for r in examples for f in r["quality_flags"])),
               "by_shape": {k: dict(v) for k, v in sorted(by_shape.items())},
               "by_source_group": {k: dict(v) for k, v in sorted(by_group.items())},
               "synthetic_labels_are_gold": False, "judge_verified": False}
    return examples, words, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
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
    examples, words, summary = summarize(word_values(archives))
    summary.update(run_no=run.get("run_no"), run_status=run.get("status"), archives=len(archives))
    for name, rows in (("examples.jsonl", examples), ("words.jsonl", words)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
