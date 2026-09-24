#!/usr/bin/env python3
"""Read Aster group artifacts into per-word JSON and aggregate JSONL; no model calls."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "group_flow"))
from pipeline import response_text, validate  # noqa: E402


def cli(args: list[str]) -> dict:
    proc = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(proc.stdout or proc.stderr)
    if not envelope.get("ok"):
        err = envelope.get("error", {})
        raise RuntimeError(f"Aster {args[0]} {err.get('code')}: {err.get('message')}")
    return envelope["data"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_data = cli(["runs", "get", args.run])
    run = run_data.get("run", run_data)
    attempts = cli(["runs", "attempts", args.run])["items"]
    listing = cli(["runs", "artifacts", args.run])
    items = listing.get("items", [])
    if listing.get("truncated"):
        items = []
        for attempt in attempts:
            part = cli(["runs", "artifacts", args.run, "--attempt-id", attempt["attempt_id"]])
            if part.get("truncated"):
                raise RuntimeError("One attempt's artifact list was truncated")
            items.extend(part["items"])
    items = [x for x in items if x["path"].endswith(".json") and "results" in Path(x["path"]).parts]

    def download(item: dict) -> tuple[dict, dict]:
        short = hashlib.sha256(item["path"].encode()).hexdigest()[:16]
        destination = out / "artifacts" / item["attempt_id"] / f"{short}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cli(["runs", "download", args.run, item["attempt_id"], item["path"], "--out", str(destination)])
        if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Artifact SHA-256 mismatch: " + str(destination))
        return item, json.loads(destination.read_text())

    bundles: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for item, obj in pool.map(download, items):
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "value" in obj:
                bundles[item["attempt_id"]][obj["name"]] = obj["value"]
    examples, statuses, rejected = [], [], []
    seen_content = set()
    for attempt_id, results in sorted(bundles.items()):
        for name, item in sorted(results.items()):
            if not name.startswith("word_") or not isinstance(item, dict):
                continue
            seed = item["seed"]
            response = item.get("response")
            if isinstance(response, dict):
                try:
                    payload = json.loads(response_text(response))
                    recorded = item.get("records", [])
                    expected_version = recorded[0].get("prompt_version") if recorded else None
                    if expected_version not in ("guard-mimo-bilingual-v1", "guard-mimo-bilingual-v2", "guard-mimo-bilingual-v3", "guard-mimo-bilingual-v4"):
                        expected_version = "guard-mimo-bilingual-v4"
                    errors, rows = validate(payload, seed, expected_version=expected_version)
                except (ValueError, TypeError, KeyError) as exc:
                    errors, rows = ["extract_parse:" + type(exc).__name__], []
                if response.get("stop_reason") != "end_turn":
                    errors.append("stop_reason:" + str(response.get("stop_reason")))
                    rows = []
            else:
                errors, rows = item.get("errors", ["missing_response"]), []
            if any(x.startswith("root_") for x in errors):
                rows = []
            kept = []
            for row in rows:
                if row["content_sha256"] in seen_content:
                    errors.append("duplicate_across_terms:" + row["sample_id"])
                    continue
                seen_content.add(row["content_sha256"])
                row["generation"] = {"run_id": run.get("id"), "run_no": run.get("run_no"),
                                     "attempt_id": attempt_id, "model_profile_id": "mdl_01M33V8C39GGNX33TFS5ETDNR3",
                                     "response_id": response.get("id") if isinstance(response, dict) else None}
                kept.append(row)
            word_record = {"seed": seed, "source_group": item["source_group"], "errors": errors,
                           "records": kept, "seconds": item.get("seconds"),
                           "usage": response.get("usage") if isinstance(response, dict) else None,
                           "annotation_status": "synthetic_unverified"}
            write_json(out / "by_word" / (seed["task_key"] + ".json"), word_record)
            examples.extend(kept)
            statuses.append({"task_key": seed["task_key"], "word": seed["word"],
                             "source_group": item["source_group"], "valid": len(kept),
                             "complete": len(kept) == 10 and not errors, "errors": errors})
            if errors:
                rejected.append({"task_key": seed["task_key"], "errors": errors})
    examples.sort(key=lambda x: x["sample_id"])
    statuses.sort(key=lambda x: x["task_key"])
    write_jsonl(out / "examples.jsonl", examples)
    write_jsonl(out / "term_status.jsonl", statuses)
    write_jsonl(out / "rejected.jsonl", rejected)
    summary = {"run_id": run.get("id"), "run_no": run.get("run_no"), "run_status": run.get("status"),
               "word_artifacts": len(statuses), "valid_examples": len(examples),
               "complete_words": sum(x["complete"] for x in statuses),
               "languages": dict(collections.Counter(x["language"] for x in examples)),
               "labels": dict(collections.Counter(x["label"] for x in examples)),
               "error_types": dict(collections.Counter(e.split(":")[0] for s in statuses for e in s["errors"])),
               "synthetic_labels_are_gold": False}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
