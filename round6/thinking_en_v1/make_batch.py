"""Build the English thinking batch (T037) from Aegis 2.0 train prompts (Mac, CPU): one Task per prompt.

Prompts come from english_redline_v1's candidate pools (same source file, same benchmark hold-out, same groups), in
this batch's own hash order, so a prompt T032 already used as a user row may come back here with a new assistant
passage. A prompt text appears once. The split follows the Aegis row id with T032's rule, so a row's prompt,
response and thinking share a split. kind is risk for the harm groups, helpful for safe; the program rotates shape
and length.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "english_redline_v1"))
from pipeline import LENGTHS, PROMPT_VERSION, SHAPES, SYSTEM_PROMPT, schema  # noqa: E402
from task_pack import write_generation_tasks  # noqa: E402
import make_rows as aegis  # noqa: E402

SALT = "thinking-en-v1-2026-09-26"
# other_harm first: most of v4's official-set false positives reason about harm outside the red lines
QUOTAS = {"other_harm": 900, "safe": 500, "weapons_drugs": 350, "violence": 300, "sexual": 200, "self_harm": 150}


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def pick(pools, quotas=QUOTAS):
    """Pure: per group, rows in this batch's hash order, one per distinct prompt text, up to the quota."""
    seen, chosen = set(), []
    for group, quota in quotas.items():
        taken = 0
        for row in sorted(pools.get(group, []), key=lambda r: number(f"{SALT}:{r['id']}")):
            if taken == quota:
                break
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            chosen.append((group, row))
            taken += 1
    return chosen


def plan(chosen):
    shapes, lengths = sorted(SHAPES), list(LENGTHS)
    seeds = []
    for ordinal, (group, row) in enumerate(chosen):
        seeds.append({"task_key": f"thinking-en-v1-{ordinal:05d}", "word": f"thinking-en-v1-{ordinal:05d}",
                      "prompt": row["prompt"], "group": group, "kind": "helpful" if group == "safe" else "risk",
                      "shape": shapes[ordinal % len(shapes)], "length": lengths[(ordinal // len(shapes)) % len(lengths)],
                      "family": f"aegis2en:{row['id']}", "split": aegis.split_of(row["id"]),
                      "source_group": f"thinking_en_v1/{group}"})
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "english_redline_v1/data/raw/aegis2_train.json")
    parser.add_argument("--bench-cases", type=Path, default=ROOT.parent / "qwen3guard_bench/data/cases.jsonl")
    parser.add_argument("--name-prefix", default="thinking-en-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "batch")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    held_out = set()
    with args.bench_cases.open(encoding="utf-8") as handle:
        for line in handle:
            held_out.update(m["content"] for m in json.loads(line)["messages"])
    seeds = plan(pick(aegis.candidates(rows, "user", held_out)))
    packed = write_generation_tasks(seeds, args.output, args.name_prefix,
                                    {"calls_per_term": 1, "prompt_version": PROMPT_VERSION}, 1, stem="batch")
    with (args.output / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps({**seed, "group_key": packed["task_of"][seed["task_key"]]}, ensure_ascii=False) + "\n")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"prompt_version": PROMPT_VERSION, "prompts": len(seeds), "tasks": packed["tasks"],
                "archives": packed["archives"], "quotas": QUOTAS,
                "groups": dict(collections.Counter(s["group"] for s in seeds)),
                "kinds": dict(collections.Counter(s["kind"] for s in seeds)),
                "shapes": dict(collections.Counter(s["shape"] for s in seeds)),
                "lengths": dict(collections.Counter(s["length"] for s in seeds)),
                "splits": dict(collections.Counter(s["split"] for s in seeds)),
                "source_sha256": digest(args.source), "bench_cases_sha256": digest(args.bench_cases),
                "sha256": {"pipeline": digest(ROOT / "flow/pipeline.py"), "flow": digest(ROOT / "flow/flow.py"),
                           "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("prompts", "tasks", "archives", "groups", "kinds", "splits")}))


if __name__ == "__main__":
    main()
