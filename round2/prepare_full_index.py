#!/usr/bin/env python3
"""Partition every text seed into fewer than 100 provenance categories."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "lexicon/research_nc/citizenlab-1eeb5e6/combined_task_seeds.jsonl"


def source_group(seed: dict) -> str:
    refs = seed.get("source_refs", seed.get("source_examples", []))
    path = refs[0].get("path", "unknown") if refs else "unknown"
    if seed["data_layer"] == "noncommercial":
        return "citizenlab/" + path.split("/")[0]
    vocab = [ref["path"].split("/")[-1] for ref in refs if "/Vocabulary/" in ref.get("path", "")]
    return "permissive/" + (vocab[0] if vocab else path.split("/")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists; use a new immutable directory")
    groups_dir = args.out / "groups"
    groups_dir.mkdir(parents=True)
    writers = {}
    paths = {}
    counts = collections.Counter()
    seen_keys = set()
    source_sha = hashlib.sha256()
    try:
        with SOURCE.open("rb") as source:
            for raw in source:
                source_sha.update(raw)
                seed = json.loads(raw)
                key = seed["task_key"]
                if key in seen_keys:
                    raise RuntimeError("duplicate task_key: " + key)
                seen_keys.add(key)
                group = source_group(seed)
                if group not in writers:
                    slug = hashlib.sha256(group.encode()).hexdigest()[:16]
                    path = groups_dir / (slug + ".jsonl")
                    writers[group] = path.open("w")
                    paths[group] = path
                record = {field: seed[field] for field in
                          ("task_key", "word", "origin_group_id", "data_layer", "source_licenses")}
                record["source_group"] = group
                writers[group].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[group] += 1
    finally:
        for writer in writers.values():
            writer.close()
    if len(seen_keys) != 449575:
        raise RuntimeError(f"Expected 449575 unique text seeds, got {len(seen_keys)}")
    if len(counts) > 100:
        raise RuntimeError(f"Category cap exceeded: {len(counts)}")
    categories = []
    for name in sorted(counts):
        path = paths[name]
        categories.append({"source_group": name, "count": counts[name],
                           "file": str(path.relative_to(args.out)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"input": str(SOURCE), "input_sha256": source_sha.hexdigest(),
                "category_count": len(categories), "total_terms": sum(counts.values()),
                "grouping": "source_file_or_source_family; not a safety label",
                "categories": categories, "model_requests_submitted": 0}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"category_count": len(categories), "total_terms": sum(counts.values()),
                      "largest_category": max(counts.values()), "input_sha256": source_sha.hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
