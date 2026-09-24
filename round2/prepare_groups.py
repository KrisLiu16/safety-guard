#!/usr/bin/env python3
"""Pack selected terms into source-category Tasks with serial calls per Task."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTED = ROOT / "batch-1000-v2/seeds.jsonl"


def source_group(seed: dict) -> str:
    refs = seed.get("source_refs", [])
    path = refs[0].get("path", "unknown") if refs else "unknown"
    if seed["data_layer"] == "noncommercial":
        return "citizenlab/" + path.split("/")[0]
    vocab = [r["path"].split("/")[-1] for r in refs if "/Vocabulary/" in r.get("path", "")]
    return "permissive/" + (vocab[0] if vocab else path.split("/")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words-per-task", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.max_words_per_task <= 300:
        ap.error("max-words-per-task must be 1..300")
    if args.out.exists():
        ap.error("output exists")
    groups = collections.defaultdict(list)
    source_bytes = SELECTED.read_bytes()
    for line in source_bytes.splitlines():
        seed = json.loads(line)
        groups[source_group(seed)].append(seed)
    tasks_root = args.out / "tasks"
    tasks_root.mkdir(parents=True)
    manifest = {"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "grouping": "source category, never inferred safety category",
                "max_words_per_task": args.max_words_per_task,
                "categories": {}, "tasks": [], "planned_model_requests": 0,
                "planned_examples": 0, "run_state": "not_submitted"}
    for group_name, seeds in sorted(groups.items()):
        seeds.sort(key=lambda s: s["task_key"])
        manifest["categories"][group_name] = len(seeds)
        digest = hashlib.sha256(group_name.encode()).hexdigest()[:12]
        for start in range(0, len(seeds), args.max_words_per_task):
            chunk = seeds[start:start + args.max_words_per_task]
            task_key = f"grp-{digest}-{start // args.max_words_per_task:03d}"
            task = tasks_root / task_key
            task.mkdir()
            instruction = {"group_key": task_key, "source_group": group_name,
                           "terms": chunk, "calls_per_term": 1, "examples_per_term": 10}
            (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False))
            (task / "task.toml").write_text('version = "1.0"\n\n[metadata]\nname = "' + task_key + '"\ncategory = "guard-data-generation"\n')
            (task / "tests").mkdir()
            test = task / "tests/test.sh"
            test.write_text('#!/bin/sh\n# Custom flow performs generation and validation.\nexit 1\n')
            test.chmod(0o755)
            manifest["tasks"].append({"task_key": task_key, "source_group": group_name,
                                      "term_count": len(chunk), "first_term_key": chunk[0]["task_key"]})
            manifest["planned_model_requests"] += len(chunk)
            manifest["planned_examples"] += 10 * len(chunk)
    assert manifest["planned_model_requests"] == 1000
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_categories": len(groups), "tasks": len(manifest["tasks"]),
                      "planned_model_requests": 1000, "planned_examples": 10000,
                      "max_words_per_task": args.max_words_per_task}, ensure_ascii=False))


if __name__ == "__main__":
    main()
