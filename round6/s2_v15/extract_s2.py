#!/usr/bin/env python3
"""Download S2 archives, re-validate every word from the raw response, write JSONL + summary (Mac, no model calls).

examples.jsonl : assistant rows (answer and reasoning), v14 row format, so judge_v3/make_pilot.py
                 can build the verification pilot from it directly (--source);
prompts.jsonl  : the generated user questions as user-side safe rows;
words.jsonl    : per-word status. Validation is re-run with the frozen pipeline; the in-flow verdict is not trusted.
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
from pipeline import PROMPT_VERSION, parse_json, response_text, stop_ok, validate  # noqa: E402


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


def summarize(values):
    """Pure: re-validate every word value; return (examples, prompts, words, summary)."""
    examples, prompts, words = [], [], []
    errors, by_group, by_form, aspects = collections.Counter(), collections.defaultdict(collections.Counter), \
        collections.defaultdict(collections.Counter), collections.Counter()
    seen = set()
    for value in values:
        seed = value["seed"]
        if seed["task_key"] in seen:
            raise RuntimeError("duplicate word result " + seed["task_key"])
        seen.add(seed["task_key"])
        word_errors, rows = revalidate(value)
        state = "complete" if rows else "skip" if word_errors == ["skip"] else "failed"
        by_group[seed.get("source_group", "")][state] += 1
        by_form[seed["form"]][state] += 1
        errors.update(e.split(":")[0] if e.startswith(("stop_reason", "request_or_parse", "extract_parse", "root_mismatch"))
                      else e for e in word_errors)
        if rows:
            aspects[rows[0]["aspect"]] += 1
            examples.extend(rows)
            prompt = rows[0]["messages"][0]["content"]
            prompts.append({**{k: rows[0][k] for k in ("language", "aspect", "form", "task_key", "word", "source_group",
                                                         "family", "split", "prompt_version", "record_version")},
                            "sample_id": f"{seed['task_key']}-s2-prompt", "target_role": "user", "label": "safe",
                            "prompt_label": "safe", "response_style": "hard_negative_prompt",
                            "messages": [{"role": "user", "content": prompt}],
                            "annotation_origin": "synthetic_unverified_pending_judge"})
        words.append({"task_key": seed["task_key"], "word": seed["word"], "language": seed["language"],
                      "form": seed["form"], "split": seed.get("split"), "source_group": seed.get("source_group"),
                      "state": state, "errors": word_errors, "seconds": value.get("seconds"),
                      "usage": (value.get("response") or {}).get("usage") if isinstance(value.get("response"), dict) else None})
    summary = {"prompt_version": PROMPT_VERSION, "words": len(words),
               "states": dict(collections.Counter(w["state"] for w in words)),
               "assistant_rows": len(examples), "prompt_rows": len(prompts),
               "errors": dict(errors.most_common()), "aspects": dict(aspects.most_common()),
               "by_form": {k: dict(v) for k, v in sorted(by_form.items())},
               "by_source_group": {k: dict(v) for k, v in sorted(by_group.items())},
               "synthetic_labels_are_gold": False, "judge_verified": False}
    return examples, prompts, words, summary


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
    examples, prompts, words, summary = summarize(word_values(archives))
    summary.update(run_no=run.get("run_no"), run_status=run.get("status"), archives=len(archives))
    for name, rows in (("examples.jsonl", examples), ("prompts.jsonl", prompts), ("words.jsonl", words)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
