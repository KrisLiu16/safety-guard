#!/usr/bin/env python3
"""Download thinking-en archives, re-validate every prompt from the raw response, write JSONL + summary (Mac, no
model calls).

examples.jsonl is in the v14 row format (assistant rows) that redline_v1/make_tasks.py takes. Validation is re-run
with the frozen pipeline; the in-flow verdict is not trusted.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT.parent))
from pipeline import PROMPT_VERSION, parse_json, response_text, size, stop_ok, validate  # noqa: E402
from aster_io import collect_archives  # noqa: E402




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
    by_group, by_shape, by_kind = (collections.defaultdict(collections.Counter) for _ in range(3))
    for value in values:
        seed = value["seed"]
        if seed["task_key"] in seen:
            raise RuntimeError("duplicate word result " + seed["task_key"])
        seen.add(seed["task_key"])
        word_errors, rows = revalidate(value)
        state = "complete" if rows else "skip" if word_errors == ["skip"] else "failed"
        by_group[seed.get("source_group", "")][state] += 1
        by_shape[seed["shape"]][state] += 1
        by_kind[seed["kind"]][state] += 1
        errors.update(e.split(":")[0] if e.startswith(("stop_reason", "request_or_parse", "extract_parse", "root_mismatch"))
                      else e for e in word_errors)
        examples.extend(rows)
        words.append({"task_key": seed["task_key"], "kind": seed["kind"], "shape": seed["shape"],
                      "split": seed.get("split"), "source_group": seed.get("source_group"), "state": state,
                      "errors": word_errors, "seconds": value.get("seconds"),
                      "usage": value["response"].get("usage") if isinstance(value.get("response"), dict) else None})
    lengths = collections.defaultdict(list)
    for row in examples:
        lengths[f"{row['language']}:{row['response_style']}"].append(size(row["messages"][1]["content"]))
    summary = {"prompt_version": PROMPT_VERSION, "words": len(words),
               "states": dict(collections.Counter(w["state"] for w in words)), "assistant_rows": len(examples),
               "errors": dict(errors.most_common()),
               "length_units_by_language_style": {k: quantiles(v) for k, v in sorted(lengths.items())},
               "quality_flags": dict(collections.Counter(f for r in examples for f in r["quality_flags"])),
               "by_shape": {k: dict(v) for k, v in sorted(by_shape.items())},
               "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
               "by_source_group": {k: dict(v) for k, v in sorted(by_group.items())},
               "synthetic_labels_are_gold": False, "judge_verified": False}
    return examples, words, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="one or more run numbers (a job over 2,000 Tasks is split into several Runs)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[],
                        help="frozen listings from response_v14/freeze_attempts.py, one per run in the same order; "
                             "needed for any run over 100 Tasks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archives, runs = collect_archives(args.runs, args.attempts_json, args.out / "archives")
    examples, words, summary = summarize(word_values(archives))
    summary.update(runs=runs, archives=len(archives))
    for name, rows in (("examples.jsonl", examples), ("words.jsonl", words)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
