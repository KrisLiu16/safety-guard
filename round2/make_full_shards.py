#!/usr/bin/env python3
"""Make restartable Aster execution shards under fixed source categories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "full-source-index"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words-per-shard", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.max_words_per_shard <= 1000:
        ap.error("max-words-per-shard must be 1..1000; Aster has a 24-hour Task limit")
    if args.out.exists():
        ap.error("output exists")
    manifest = json.loads((INDEX / "manifest.json").read_text())
    if manifest["category_count"] > 100 or manifest["total_terms"] != 449575:
        raise RuntimeError("Unexpected full index")
    task_root = args.out / "tasks"
    task_root.mkdir(parents=True)
    shards = []
    for category in manifest["categories"]:
        group = category["source_group"]
        path = INDEX / category["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != category["sha256"]:
            raise RuntimeError("Index category hash changed: " + group)
        group_hash = hashlib.sha256(group.encode()).hexdigest()[:12]
        with path.open() as source:
            shard_no = 0
            offset = 0
            while True:
                terms = []
                for _ in range(args.max_words_per_shard):
                    line = source.readline()
                    if not line:
                        break
                    seed = json.loads(line)
                    terms.append([seed["task_key"], seed["word"], seed["source_licenses"]])
                if not terms:
                    break
                task_key = f"all-{group_hash}-{shard_no:05d}"
                task = task_root / task_key
                task.mkdir()
                instruction = {"group_key": task_key, "source_group": group,
                               "category_total_terms": category["count"],
                               "start_offset": offset, "terms": terms,
                               "calls_per_term": 1, "examples_per_term": 10}
                (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False, separators=(",", ":")))
                (task / "task.toml").write_text('version = "1.0"\n\n[metadata]\nname = "' + task_key + '"\ncategory = "guard-data-generation"\n')
                (task / "tests").mkdir()
                test = task / "tests/test.sh"
                test.write_text('#!/bin/sh\n# Custom flow performs validation.\nexit 1\n')
                test.chmod(0o755)
                shards.append({"task_key": task_key, "source_group": group,
                               "start_offset": offset, "term_count": len(terms)})
                offset += len(terms)
                shard_no += 1
        if offset != category["count"]:
            raise RuntimeError(f"Category {group} count mismatch")
    result = {"source_index_sha256": hashlib.sha256((INDEX / "manifest.json").read_bytes()).hexdigest(),
              "logical_categories": manifest["category_count"], "physical_tasks": len(shards),
              "total_terms": sum(x["term_count"] for x in shards),
              "max_words_per_shard": args.max_words_per_shard,
              "reward_definition": "complete_terms / category_total_terms, aggregated across shards",
              "shards": shards, "model_requests_submitted": 0}
    if result["total_terms"] != 449575:
        raise RuntimeError("Shard count mismatch")
    (args.out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("logical_categories", "physical_tasks", "total_terms", "max_words_per_shard")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
