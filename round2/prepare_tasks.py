#!/usr/bin/env python3
"""Deterministically pick a balanced, readable first batch; no network calls."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "lexicon/research_nc/citizenlab-1eeb5e6/combined_task_seeds.jsonl"
NC_CANDIDATES = ROOT.parent / "lexicon/research_nc/citizenlab-1eeb5e6/candidate_terms.jsonl"
PILOT = ROOT.parent / "pilot/seeds.jsonl"
SALT = "mimo-bilingual-first-batch-v2"


def suitable(word: str) -> bool:
    if not isinstance(word, str) or not 2 <= len(word) <= 12:
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fff]+", word))


def pick(count: int) -> tuple[list[dict], dict]:
    prior = {json.loads(line)["word"] for line in PILOT.open() if line.strip()}
    # Multiple independent archive families are a coarse automatic quality
    # signal. They are not votes on whether a term should be blocked.
    nc_multifamily = set()
    with NC_CANDIDATES.open() as handle:
        for line in handle:
            term = json.loads(line)
            if term.get("new_vs_permissive_base") and len(term.get("families", [])) >= 2:
                nc_multifamily.add(term["id"])
    target = {"permissive": count // 2, "noncommercial": count - count // 2}
    heaps: dict[str, list[tuple[int, str, dict]]] = {k: [] for k in target}
    audit = {"source_rows": 0, "unsuitable": 0, "previous_pilot": 0,
             "single_family_nc_skipped": 0}
    source_hash = hashlib.sha256()
    with SOURCE.open("rb") as handle:
        for raw in handle:
            source_hash.update(raw)
            seed = json.loads(raw)
            audit["source_rows"] += 1
            if seed["word"] in prior:
                audit["previous_pilot"] += 1
                continue
            if not suitable(seed["word"]):
                audit["unsuitable"] += 1
                continue
            layer = seed["data_layer"]
            if layer == "noncommercial" and seed["task_key"].removeprefix("lex-") not in nc_multifamily:
                audit["single_family_nc_skipped"] += 1
                continue
            rank = int.from_bytes(hashlib.sha256((SALT + ":" + seed["task_key"]).encode()).digest()[:8], "big")
            heap = heaps[layer]
            item = (-rank, seed["task_key"], seed)
            if len(heap) < target[layer]:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if any(len(heaps[k]) != target[k] for k in target):
        raise RuntimeError("not enough eligible terms")
    selected = [item[2] for heap in heaps.values() for item in heap]
    selected.sort(key=lambda s: s["task_key"])
    return selected, {**audit, "input_sha256": source_hash.hexdigest(), "selection_salt": SALT,
                      "eligibility": "2-12 CJK ideographs; NC term observed in at least two archive families",
                      "target_by_layer": target, "selected": len(selected)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.count < 2 or args.count % 2:
        ap.error("count must be even and at least 2")
    if args.out.exists():
        ap.error("output exists; pick a new immutable batch directory")
    selected, audit = pick(args.count)
    tasks = args.out / "tasks"
    tasks.mkdir(parents=True)
    with (args.out / "seeds.jsonl").open("w") as seed_file:
        for seed in selected:
            record = {k: seed[k] for k in ("task_key", "word", "origin_group_id", "data_layer", "source_licenses")}
            record["source_refs"] = seed.get("source_refs", seed.get("source_examples", []))
            seed_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            task = tasks / seed["task_key"]
            task.mkdir()
            (task / "instruction.md").write_text(json.dumps(record, ensure_ascii=False))
            (task / "task.toml").write_text('version = "1.0"\n\n[metadata]\nname = "' + seed["task_key"] + '"\ncategory = "guard-data-generation"\n')
            (task / "tests").mkdir()
            test = task / "tests/test.sh"
            test.write_text('#!/bin/sh\n# Harbor envelope only; the custom Aster flow validates outputs.\nexit 1\n')
            test.chmod(0o755)
    audit["seeds_sha256"] = hashlib.sha256((args.out / "seeds.jsonl").read_bytes()).hexdigest()
    audit["planned_examples"] = 10 * len(selected)
    audit["planned_model_requests"] = len(selected)
    audit["model_profile_id"] = "mdl_01M33V8C39GGNX33TFS5ETDNR3"
    audit["run_state"] = "not_submitted"
    (args.out / "manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
