"""Build the S2 pilot from Run B seeds (Mac, CPU): words spread over source groups, one Task per word.

Run B words (round6/response_v14/batch_rest/seeds.jsonl) are disjoint from Run A by family, so S2 never
reuses a Run A dev/calibration family. The program assigns the question form (rotation over FORMS) and
the language (alternating zh/en), so both are balanced; the model only picks the aspect.
S2 holds its own dev/calibration families (~6%, by family hash) for later threshold fitting.
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
from pipeline import FORMS, PROMPT_VERSION, SYSTEM_PROMPT, request_body, schema  # noqa: E402

DEFAULT_SEEDS = ROOT.parent / "response_v14/batch_rest/seeds.jsonl"
SALT = "s2-v1-2026-09-24"
FIELDS = ("task_key", "word", "origin_group_id", "data_layer", "source_licenses", "source_group", "family")


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_of(family):
    bucket = number(f"{SALT}:split:{family}") % 100
    return "dev" if bucket < 3 else "calibration" if bucket < 6 else "train"


def pick(seeds, words):
    """Round-robin over source groups, SHA order inside each group."""
    by_group = collections.defaultdict(list)
    for seed in seeds:
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
    forms = sorted(FORMS)
    out = []
    for ordinal, seed in enumerate(seeds):
        row = {k: seed.get(k) for k in FIELDS}
        row.update(language=("zh", "en")[ordinal % 2], form=forms[(ordinal // 2) % len(forms)],
                   split=split_of(seed["family"]))
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--words", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = [json.loads(line) for line in args.seeds.open(encoding="utf-8")]
    chosen = plan(pick(seeds, args.words))
    paths = []
    for ordinal, seed in enumerate(chosen):
        name = f"s2-pilot-{ordinal:03d}"
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
                "languages": dict(collections.Counter(s["language"] for s in chosen)),
                "forms": dict(collections.Counter(s["form"] for s in chosen)),
                "splits": dict(collections.Counter(s["split"] for s in chosen)),
                "source_groups": dict(collections.Counter(s["source_group"] for s in chosen)),
                "seeds_source": str(args.seeds),
                "sha256": {"pipeline": digest(ROOT / "flow/pipeline.py"), "flow": digest(ROOT / "flow/flow.py"),
                           "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    example = request_body("deepseek/deepseek-flash", chosen[0], "chat_completions")
    (args.output / "example_request.json").write_text(json.dumps(example, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ("words", "tasks", "languages", "forms", "splits")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
