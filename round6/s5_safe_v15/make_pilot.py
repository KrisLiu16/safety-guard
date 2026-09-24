"""Build the S5-safe pilot from Run B seeds (Mac, CPU): words spread over source groups, one Task per word.

Each Run B seed already carries the word's two frozen v12 user questions (one unsafe, one safe); they are
passed through unchanged. The program assigns the thinking shape (rotation over SHAPES). The family split
uses exactly S2's rule and salt, so a family is dev / calibration / train in both slices alike. Words
already in the S2 pilot are excluded.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import LENGTHS, PROMPT_VERSION, SHAPES, SYSTEM_PROMPT, request_body, schema  # noqa: E402

DEFAULT_SEEDS = ROOT.parent / "response_v14/batch_rest/seeds.jsonl"
DEFAULT_EXCLUDE = (ROOT.parent / "s2_v15/pilot/seeds.jsonl",)
SALT = "s5-safe-v1-2026-09-24"
FAMILY_SPLIT_SALT = "s2-v1-2026-09-24"  # shared with round6/s2_v15/make_pilot.py on purpose
FIELDS = ("task_key", "word", "origin_group_id", "data_layer", "source_licenses", "source_group", "family", "prompts")


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_of(family):
    bucket = number(f"{FAMILY_SPLIT_SALT}:split:{family}") % 100
    return "dev" if bucket < 3 else "calibration" if bucket < 6 else "train"


def pick(seeds, words, exclude=frozenset()):
    by_group = collections.defaultdict(list)
    for seed in seeds:
        if seed["task_key"] not in exclude:
            by_group[seed["source_group"]].append(seed)
    for rows in by_group.values():
        rows.sort(key=lambda s: number(f"{SALT}:{s['task_key']}"))
    chosen = []
    while len(chosen) < words and any(by_group.values()):
        for group in sorted(by_group):
            if by_group[group] and len(chosen) < words:
                chosen.append(by_group[group].pop(0))
    return chosen


def plan(seeds):
    shapes = sorted(SHAPES)
    out = []
    for ordinal, seed in enumerate(seeds):
        row = {k: seed.get(k) for k in FIELDS}
        row["prompts"] = {label: {k: p.get(k) for k in ("language", "text", "sample_id")}
                          for label, p in seed["prompts"].items()}
        row.update(shape=shapes[ordinal % len(shapes)], length=list(LENGTHS)[(ordinal // len(shapes)) % len(LENGTHS)],
                   split=split_of(seed["family"]))
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--exclude", type=Path, action="append", default=None,
                        help="seeds.jsonl files whose task_keys are skipped (default: the S2 pilot)")
    parser.add_argument("--words", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    exclude = set()
    for path in args.exclude if args.exclude is not None else DEFAULT_EXCLUDE:
        if path.exists():
            exclude.update(json.loads(line)["task_key"] for line in path.open(encoding="utf-8"))
    seeds = [json.loads(line) for line in args.seeds.open(encoding="utf-8")]
    chosen = plan(pick(seeds, args.words, frozenset(exclude)))
    paths = []
    for ordinal, seed in enumerate(chosen):
        name = f"s5safe-pilot-{ordinal:03d}"
        task = args.output / "tasks" / name
        (task / "tests").mkdir(parents=True)
        instruction = {"group_key": name, "source_group": seed["source_group"], "terms": [seed],
                       "calls_per_term": 1, "prompt_version": PROMPT_VERSION}
        (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{name}"\ncategory = "guard-data-generation"\n')
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        paths.append(task)
    with tarfile.open(args.output / "pilot.tar.gz", "w:gz") as archive:
        for task in paths:
            archive.add(task, arcname=task.name)
    with (args.output / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in chosen:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"prompt_version": PROMPT_VERSION, "words": len(chosen), "tasks": len(paths),
                "excluded_task_keys": len(exclude),
                "unsafe_prompt_language": dict(collections.Counter(s["prompts"]["unsafe"]["language"] for s in chosen)),
                "shapes": dict(collections.Counter(s["shape"] for s in chosen)),
                "lengths": dict(collections.Counter(s["length"] for s in chosen)),
                "splits": dict(collections.Counter(s["split"] for s in chosen)),
                "source_groups": dict(collections.Counter(s["source_group"] for s in chosen)),
                "seeds_source": str(args.seeds),
                "sha256": {"pipeline": digest(ROOT / "flow/pipeline.py"), "flow": digest(ROOT / "flow/flow.py"),
                           "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("words", "tasks", "unsafe_prompt_language", "shapes", "splits")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
