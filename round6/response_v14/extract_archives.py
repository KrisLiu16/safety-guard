#!/usr/bin/env python3
"""Download v14 attempt archives, re-validate every word locally and write JSONL + audit.

Same download/SHA discipline as the v12 extractor. Validation is re-run from the raw
model response with the frozen pipeline, so the flow's in-sandbox verdict is not trusted.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, response_refusal, response_text, slot_records, validate  # noqa: E402


def cli(args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"Aster {args[0]} {error.get('code')}: {error.get('message')}")
    return envelope["data"]


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def percentile(values, q):
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1] if values else None


def revalidate(value):
    seed, response = value["seed"], value.get("response")
    if not isinstance(response, dict):
        return value.get("errors") or ["missing_response"], []
    refusal = response_refusal(response)
    if refusal is not None:
        return ["model_refusal:" + refusal], []
    try:
        errors, rows = validate(json.loads(response_text(response)), seed)
    except (ValueError, TypeError, KeyError) as exc:
        return ["extract_parse:" + type(exc).__name__], []
    if response.get("status") != "completed":
        errors.append("status:" + str(response.get("status")))
        rows = []
    if any(e.startswith("root_") for e in errors):
        rows = []
    return errors, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-terms", type=int, required=True)
    parser.add_argument("--download-workers", type=int, default=5)
    parser.add_argument("--attempts-json", type=Path,
                        help="frozen per-sample listing from freeze_attempts.py; required when the run has over 100 samples")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run = cli(["runs", "get", args.run])
    if args.attempts_json:
        selected = json.loads(args.attempts_json.read_text(encoding="utf-8"))
        if selected.get("run_id") != run.get("id"):
            raise RuntimeError(f"attempts.json is for run {selected.get('run_id')}, not {run.get('id')}")
    else:
        selected = cli(["runs", "attempts", args.run])
    if selected.get("truncated"):
        raise RuntimeError("Attempt listing truncated; freeze a per-sample attempt list first (freeze_attempts.py)")
    attempts = [a for a in selected["items"] if a.get("state") == "completed" and a.get("archive")]
    unfinished = [a for a in selected["items"] if a not in attempts]
    if len({a["sample_id"] for a in attempts}) != len(attempts):
        raise RuntimeError("Expected one attempt per sample")
    (out / "selected_attempts.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    archives = out / "archives"
    archives.mkdir(exist_ok=True)
    model_id = run.get("effective_config", {}).get("models", {}).get("main", {}).get("profile_id")

    def download(attempt):
        aid = attempt["attempt_id"]
        path, digest_path = archives / (aid + ".tar.gz"), archives / (aid + ".sha256")
        if not path.exists():
            result = cli(["runs", "archive", args.run, aid, "--out", str(path)])
            digest_path.write_text(result["sha256"] + "\n")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest_path.read_text().strip():
            raise RuntimeError("Archive SHA-256 mismatch for " + aid)
        return aid, path

    seen_terms, seen_contents = set(), set()
    usage, errors, slot_errors = collections.Counter(), collections.Counter(), collections.Counter()
    refusals_by_group, words_by_group, complete_by_group = collections.Counter(), collections.Counter(), collections.Counter()
    slot_valid = collections.Counter()
    quality_flags = collections.Counter()
    onset_fraction = collections.defaultdict(list)
    durations, statuses = [], []
    complete = raw_complete = valid = repaired = artifacts = 0
    with (out / "examples.jsonl").open("w", encoding="utf-8") as examples_file, \
            (out / "words.jsonl").open("w", encoding="utf-8") as words_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            for aid, path in pool.map(download, attempts):
                with tarfile.open(path, "r:gz") as tar:
                    members = sorted((m for m in tar if m.isfile() and m.name.startswith("flow/results/")
                                      and m.name.endswith(".json")), key=lambda m: m.name)
                    for member in members:
                        obj = json.load(tar.extractfile(member))
                        name, value = obj.get("name"), obj.get("value")
                        if not isinstance(name, str) or not name.startswith("word_") or not isinstance(value, dict):
                            continue
                        artifacts += 1
                        seed, response = value["seed"], value.get("response")
                        key, group = seed["task_key"], value["source_group"]
                        if key in seen_terms:
                            raise RuntimeError("Duplicate term result " + key)
                        seen_terms.add(key)
                        word_errors, rows = revalidate(value)
                        kept = []
                        for row in rows:
                            if row["content_sha256"] in seen_contents:
                                word_errors.append("duplicate_across_terms:" + row["sample_id"])
                                continue
                            seen_contents.add(row["content_sha256"])
                            row["generation"] = {"run_id": run.get("id"), "run_no": run.get("run_no"), "attempt_id": aid,
                                                 "model_profile_id": model_id, "response_id": response.get("id")}
                            kept.append(row)
                            write_line(examples_file, row)
                            repaired += int(bool(row["repair_flags"]))
                            slot_valid[row["index"]] += 1
                            quality_flags.update(f"{row['label']}:{flag}" for flag in row["quality_flags"])
                            if row["label"] == "unsafe":
                                onset_fraction[(row["index"], row["onset_style_requested"])].append(
                                    row["onset_char"] / row["response_chars"])
                        full = len(kept) == 4 and not word_errors
                        original = value.get("records", [])
                        raw_full = len(original) == 4 and not value.get("errors")
                        complete, raw_complete, valid = complete + full, raw_complete + raw_full, valid + len(kept)
                        words_by_group[group] += 1
                        complete_by_group[group] += int(full)
                        if any(e.startswith("model_refusal") for e in word_errors):
                            refusals_by_group[group] += 1
                        errors.update(e.split(":")[0] if not e[0].isdigit() else "slot_error" for e in word_errors)
                        slot_errors.update(e for e in word_errors if e[0].isdigit())
                        status = {"task_key": key, "word": seed["word"], "source_group": group, "rotation": seed["rotation"],
                                  "unsafe_prompt_language": seed["prompts"]["unsafe"]["language"],
                                  "unsafe_prompt_basis": seed["prompts"]["unsafe"]["policy_basis"],
                                  "slots": [(s["style"], s["format"], s.get("onset_style")) for s in
                                            slot_records(seed["prompts"], seed["rotation"])],
                                  "valid": len(kept), "complete": full, "raw_valid": len(original),
                                  "raw_complete": raw_full, "errors": word_errors, "seconds": value.get("seconds"),
                                  "usage": response.get("usage") if isinstance(response, dict) else None}
                        statuses.append(status)
                        write_line(words_file, status)
                        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                            usage.update({k: response["usage"].get(k, 0) or 0
                                          for k in ("input_tokens", "output_tokens", "total_tokens")})
                        if isinstance(value.get("seconds"), (int, float)):
                            durations.append(value["seconds"])
    summary = {
        "run_id": run.get("id"), "run_no": run.get("run_no"), "run_status": run.get("status"),
        "prompt_version": PROMPT_VERSION, "attempts_with_archive": len(attempts), "attempts_without_archive": len(unfinished),
        "expected_terms": args.expected_terms, "word_artifacts": artifacts,
        "missing_word_artifacts": args.expected_terms - artifacts,
        "complete_words": complete, "raw_complete_words": raw_complete,
        "reward_over_expected": complete / args.expected_terms,
        "valid_responses": valid, "repaired_responses": repaired,
        "valid_by_slot": {str(k): slot_valid[k] for k in range(4)},
        "quality_flags": dict(quality_flags),
        "error_types": dict(errors), "slot_errors": dict(slot_errors.most_common()),
        "model_refusal_words_by_group": dict(refusals_by_group),
        "complete_by_group": {g: f"{complete_by_group[g]}/{n}" for g, n in words_by_group.items()},
        "onset_fraction_by_slot_style": {f"{slot}:{style}": {"n": len(v), "p10": percentile(v, .1), "p50": percentile(v, .5),
                                                              "p90": percentile(v, .9), "at_zero": sum(x == 0 for x in v)}
                                         for (slot, style), v in sorted(onset_fraction.items())},
        "token_usage": dict(usage),
        "tokens_per_word": {k: v / artifacts for k, v in usage.items()} if artifacts else {},
        "latency_seconds": {"mean": sum(durations) / len(durations), "p50": percentile(durations, .5),
                            "p95": percentile(durations, .95)} if durations else {},
        "synthetic_labels_are_gold": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
