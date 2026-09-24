"""Build the frozen v14 50k-word batch from the full aster-dev-272 extraction.

Selection rules (all deterministic, SHA-256 ordered):
- pool: v12 words whose zh+en prompts are both valid, minus every lexicon key already used
  by round4/round5 data and the v13/v14 pilots;
- family: NFKC + casefold + drop whitespace/punctuation of the word; one word per family;
- allocation: sqrt(family count) water-filling over source groups, so small groups are taken
  whole and the largest groups are capped;
- holdout: ~6% of each group's selected families go to dev / calibration (alternating),
  never a family whose normalized form appears in previously used words;
- shards: one source group per Task, at most 150 words per Task (~105 KB instruction), one Task per archive.
The v14 flow and pipeline are used unchanged; split lives in seeds.jsonl, not in the flow.

--rest-of <batch_50k/seeds.jsonl> builds Run B instead: every remaining family not in Run A,
all assigned to train (Run A already holds the dev/calibration families).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import unicodedata

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT, schema  # noqa: E402

SNAPSHOT = PROJECT / "round2/oneword2_v12/full_snapshot_final_v2"
USED_SOURCES = ("round4/data", "round5/data")
PILOTS = ("round6/response_v13/pilot/seeds.jsonl", "round6/response_v14/pilot/seeds.jsonl")
TOTAL_WORDS = 50000
HOLDOUT_FRACTION = 0.06
WORDS_PER_TASK = 150  # 300-word Tasks (~208 KB instruction) were rejected by the upload endpoint; 196 KB passed
SALT = "v14-batch50k-2026-09-24"


def rank(text):
    return hashlib.sha256(f"{SALT}:{text}".encode()).hexdigest()


def family_of(word):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", word).casefold()) or word


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool():
    complete = set()
    with open(SNAPSHOT / "term_status.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["complete"] and row["valid"] == 2:
                complete.add(row["task_key"])
    grouped, words = collections.defaultdict(list), {}
    with open(SNAPSHOT / "examples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            words[row["task_key"]] = row["word"]
            if row["task_key"] in complete:
                grouped[row["task_key"]].append(row)
    seeds = {}
    for key, rows in grouped.items():
        labels = {r["label"]: r for r in rows}
        if len(rows) != 2 or set(labels) != {"safe", "unsafe"} or {r["language"] for r in rows} != {"zh", "en"}:
            continue
        first = rows[0]
        seeds[key] = {
            "task_key": key, "word": first["word"], "origin_group_id": first["origin_group_id"],
            "data_layer": first["data_layer"], "source_licenses": first["source_licenses"],
            "source_group": first["source_group"],
            "prompts": {label: {"language": r["language"], "text": r["messages"][0]["content"],
                                "policy_basis": r["policy_basis"], "sample_id": r["sample_id"]}
                        for label, r in labels.items()}}
    return seeds, words


def used_keys():
    used = set()
    for folder in USED_SOURCES:
        for path in (PROJECT / folder).rglob("*.jsonl"):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    used.update(re.findall(r"lex-[0-9a-f]{20}", line))
    for path in PILOTS:
        with open(PROJECT / path, encoding="utf-8") as handle:
            used.update(json.loads(line)["task_key"] for line in handle)
    return used


def allocate(available, total):
    """sqrt water-filling: groups whose share exceeds availability are taken whole."""
    allocation, rest, budget = {}, dict(available), total
    while True:
        weights = {g: math.sqrt(n) for g, n in rest.items()}
        scale = budget / sum(weights.values())
        full = [g for g in rest if weights[g] * scale >= rest[g]]
        if not full:
            break
        for group in full:
            allocation[group] = rest.pop(group)
            budget -= allocation[group]
    weights = {g: math.sqrt(n) for g, n in rest.items()}
    shares = {g: budget * w / sum(weights.values()) for g, w in weights.items()}
    base = {g: int(s) for g, s in shares.items()}
    for group in sorted(shares, key=lambda g: shares[g] - base[g], reverse=True)[:budget - sum(base.values())]:
        base[group] += 1
    allocation.update(base)
    return allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "batch_50k")
    parser.add_argument("--rest-of", type=Path, help="Run A seeds.jsonl; build all remaining families instead")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds, words = load_pool()
    used = used_keys()
    used_families = {family_of(words[k]) for k in used if k in words}
    # One word per family across ALL groups (first in SHA order), so a family can never be
    # selected twice or land in two splits via different source groups.
    chosen_family, by_group = {}, collections.defaultdict(dict)
    for key in sorted(seeds, key=rank):
        if key in used:
            continue
        chosen_family.setdefault(family_of(seeds[key]["word"]), seeds[key])
    run_a_families = set()
    if args.rest_of:
        with open(args.rest_of, encoding="utf-8") as handle:
            run_a_families = {json.loads(line)["family"] for line in handle}
    for family, seed in chosen_family.items():
        if hashlib.sha256(family.encode()).hexdigest() not in run_a_families:
            by_group[seed["source_group"]][family] = seed
    if args.rest_of:
        allocation = {g: len(f) for g, f in by_group.items() if f}
    else:
        allocation = allocate({g: len(f) for g, f in by_group.items()}, TOTAL_WORDS)
    selected, split_counts = [], collections.Counter()
    for group in sorted(allocation, key=lambda g: (-allocation[g], g)):
        # Alternate zh-unsafe / en-unsafe words (each list in SHA order) so both orientations
        # of the 2x2 design stay balanced; the longer list fills in once the other runs out.
        lanes = {lang: [f for f in sorted(by_group[group], key=rank)
                        if by_group[group][f]["prompts"]["unsafe"]["language"] == lang] for lang in ("zh", "en")}
        families = []
        while len(families) < allocation[group]:
            for lang in ("zh", "en"):
                if lanes[lang] and len(families) < allocation[group]:
                    families.append(lanes[lang].pop(0))
        holdout_n = 0 if args.rest_of else round(allocation[group] * HOLDOUT_FRACTION)
        holdout = [f for f in sorted(families, key=lambda f: rank("holdout:" + f)) if f not in used_families][:holdout_n]
        assignment = {f: ("dev", "calibration")[i % 2] for i, f in enumerate(holdout)}
        for family in families:
            seed = dict(by_group[group][family])
            seed["family"] = hashlib.sha256(family.encode()).hexdigest()
            seed["split"] = assignment.get(family, "train")
            split_counts[seed["split"]] += 1
            selected.append(seed)
    counters = collections.Counter()
    for seed in selected:
        orientation = seed["prompts"]["unsafe"]["language"]
        seed["rotation"] = counters[orientation]
        counters[orientation] += 1
    tasks_dir, archives = args.output / "tasks", args.output / "archives"
    archives.mkdir(parents=True)
    group_ids = {g: i for i, g in enumerate(sorted(allocation, key=lambda g: (-allocation[g], g)))}
    task_rows, grouped_seeds = [], collections.defaultdict(list)
    for seed in selected:
        grouped_seeds[seed["source_group"]].append(seed)
    with open(args.output / "seeds.jsonl", "w", encoding="utf-8") as seeds_file:
        for group, rows in grouped_seeds.items():
            for chunk_index, start in enumerate(range(0, len(rows), WORDS_PER_TASK)):
                chunk = rows[start:start + WORDS_PER_TASK]
                key = f"v14-{'rest' if args.rest_of else '50k'}-g{group_ids[group]:02d}-{chunk_index:03d}"
                task = tasks_dir / key
                (task / "tests").mkdir(parents=True)
                terms = [{k: v for k, v in s.items() if k not in ("split", "family")} for s in chunk]
                instruction = {"group_key": key, "source_group": group, "terms": terms,
                               "calls_per_term": 1, "responses_per_term": 4, "prompt_version": PROMPT_VERSION}
                (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False), encoding="utf-8")
                (task / "task.toml").write_text(
                    f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-data-generation"\n', encoding="utf-8")
                test = task / "tests/test.sh"
                test.write_text("#!/bin/sh\nexit 1\n")
                test.chmod(0o755)
                with tarfile.open(archives / f"{key}.tar.gz", "w:gz") as output:
                    output.add(task, arcname=key)
                task_rows.append({"task_key": key, "source_group": group, "words": len(chunk)})
                for seed in chunk:
                    seeds_file.write(json.dumps({**seed, "group_key": key}, ensure_ascii=False) + "\n")
    orientation = collections.Counter(s["prompts"]["unsafe"]["language"] for s in selected)
    manifest = {
        "prompt_version": PROMPT_VERSION, "words": len(selected), "tasks": len(task_rows),
        "words_per_task_max": WORDS_PER_TASK, "responses_per_word": 4, "calls_per_word": 1,
        "splits": dict(split_counts), "unsafe_prompt_language": dict(orientation),
        "allocation": {g: {"selected": allocation[g], "available_families": len(by_group[g])}
                       for g in sorted(allocation, key=lambda g: -allocation[g])},
        "excluded_used_task_keys": len(used), "holdout_excludes_used_families": len(used_families),
        "family_rule": "NFKC + casefold + remove whitespace/punctuation; exact match only, no alias or near-duplicate merge",
        "selection_salt": SALT, "source_snapshot": str(SNAPSHOT.relative_to(PROJECT)),
        "flow_frozen_from_pilot": "aster-dev-275",
        "run_a_seeds_sha256": sha256_file(args.rest_of) if args.rest_of else None,
        "run_a_families_excluded": len(run_a_families),
        "sha256": {"pipeline": sha256_file(ROOT / "flow/pipeline.py"), "flow": sha256_file(ROOT / "flow/flow.py"),
                   "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                   "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                   "seeds": sha256_file(args.output / "seeds.jsonl")},
        "tasks_index": task_rows}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("words", "tasks", "splits", "unsafe_prompt_language")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
