"""Build the 50-word v14 pilot from complete v12 words, one Task per source group.

Selection is deterministic (SHA-256 order within each source group), stratified
across source groups, and excludes every lexicon task_key already present in the
round4/round5 datasets and the v13 pilot so later dev/calibration splits stay clean.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT, request_body, schema  # noqa: E402

SNAPSHOT = PROJECT / "round2/oneword2_v12/completed_snapshot_v2"
USED_SOURCES = ("round4/data", "round5/data")
PILOT_WORDS = 50
SALT = "v14-pilot-2026-09-24"
PREVIOUS_PILOTS = ("round6/response_v13/pilot/seeds.jsonl",)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def used_task_keys():
    used = set()
    for folder in USED_SOURCES:
        for path in (PROJECT / folder).rglob("*.jsonl"):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    used.update(re.findall(r"lex-[0-9a-f]{20}", line))
    for path in PREVIOUS_PILOTS:
        with open(PROJECT / path, encoding="utf-8") as handle:
            used.update(json.loads(line)["task_key"] for line in handle)
    return used


def complete_words():
    """task_key -> seed with exactly one safe and one unsafe v12 user prompt."""
    complete = set()
    with open(SNAPSHOT / "term_status.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["complete"] and row["valid"] == 2:
                complete.add(row["task_key"])
    grouped = collections.defaultdict(list)
    with open(SNAPSHOT / "examples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
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
    return seeds


def select(seeds, used, total):
    by_group = collections.defaultdict(list)
    for key, seed in seeds.items():
        if key not in used:
            by_group[seed["source_group"]].append(seed)
    for rows in by_group.values():
        rows.sort(key=lambda s: hashlib.sha256(f"{SALT}:{s['task_key']}".encode()).hexdigest())
    order = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    quota = collections.Counter()
    while sum(quota.values()) < total:
        progressed = False
        for group in order:
            if sum(quota.values()) == total:
                break
            if quota[group] < len(by_group[group]):
                quota[group] += 1
                progressed = True
        if not progressed:
            raise ValueError("not enough eligible words")
    # Within each group keep SHA order, but prefer the currently rarer unsafe-prompt
    # language so zh-unsafe and en-unsafe words stay balanced overall.
    chosen, balance = {}, collections.Counter()
    for group in order:
        pool = list(by_group[group])
        picks = []
        for _ in range(quota[group]):
            wanted = min(("zh", "en"), key=lambda lang: balance[lang])
            pick = next((s for s in pool if s["prompts"]["unsafe"]["language"] == wanted), pool[0])
            pool.remove(pick)
            picks.append(pick)
            balance[pick["prompts"]["unsafe"]["language"]] += 1
        if picks:
            chosen[group] = picks
    # Style rotation is counted separately for zh-unsafe and en-unsafe words so both
    # orientations see every safe-response style and unsafe onset/format combination.
    counters = collections.Counter()
    for group in order:
        for seed in chosen.get(group, []):
            orientation = seed["prompts"]["unsafe"]["language"]
            seed["rotation"] = counters[orientation]
            counters[orientation] += 1
    return chosen, {group: len(by_group[group]) for group in order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    used = used_task_keys()
    chosen, eligible = select(complete_words(), used, PILOT_WORDS)
    tasks = args.output / "tasks"
    task_paths, seeds_out = [], []
    for ordinal, (group, seeds) in enumerate(chosen.items()):
        key = f"pilot-response-v14-{ordinal:02d}"
        task = tasks / key
        (task / "tests").mkdir(parents=True)
        instruction = {"group_key": key, "source_group": group, "terms": seeds,
                       "calls_per_term": 1, "responses_per_term": 4, "prompt_version": PROMPT_VERSION}
        (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(
            f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-data-generation"\n', encoding="utf-8")
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        task_paths.append(task)
        seeds_out += [{**seed, "group_key": key} for seed in seeds]
    with open(args.output / "seeds.jsonl", "w", encoding="utf-8") as handle:
        for seed in seeds_out:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    archive = args.output / "pilot.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for task in task_paths:
            output.add(task, arcname=task.name)
    example = request_body("gpt-5.6-luna", seeds_out[0])
    (args.output / "example_request.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    orientation = collections.Counter(s["prompts"]["unsafe"]["language"] for s in seeds_out)
    manifest = {
        "prompt_version": PROMPT_VERSION, "words": len(seeds_out), "tasks": len(task_paths),
        "responses_per_word": 4, "calls_per_word": 1,
        "unsafe_prompt_language": dict(orientation),
        "rotation_mod4_by_orientation": {lang: dict(collections.Counter(s["rotation"] % 4 for s in seeds_out
                                                                        if s["prompts"]["unsafe"]["language"] == lang))
                                         for lang in orientation},
        "words_per_group": {group: len(seeds) for group, seeds in chosen.items()},
        "eligible_unused_words_per_group": eligible,
        "excluded_used_task_keys": len(used), "selection_salt": SALT,
        "v12_snapshot": str(SNAPSHOT.relative_to(PROJECT)),
        "sha256": {"pipeline": sha256_file(ROOT / "flow/pipeline.py"), "flow": sha256_file(ROOT / "flow/flow.py"),
                   "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                   "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                   "seeds": sha256_file(args.output / "seeds.jsonl"), "archive": sha256_file(archive)}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("words", "tasks", "unsafe_prompt_language", "rotation_mod4_by_orientation")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
