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
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT.parent))
from pipeline import PROMPT_VERSION, parse_json, response_text, stop_ok, validate  # noqa: E402
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
    parser.add_argument("runs", nargs="+", help="one or more run numbers (a job over 2,000 Tasks is split into several Runs)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[],
                        help="frozen listings from response_v14/freeze_attempts.py, one per run in the same order; "
                             "needed for any run over 100 Tasks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archives, runs = collect_archives(args.runs, args.attempts_json, args.out / "archives")
    examples, prompts, words, summary = summarize(word_values(archives))
    summary.update(runs=runs, archives=len(archives))
    for name, rows in (("examples.jsonl", examples), ("prompts.jsonl", prompts), ("words.jsonl", words)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
