"""Build a leader-sensitivity generation batch (T031, Mac, CPU): program-assigned seeds, one Task per seed.

Seeds rotate deterministically over kind (cut kinds weighted 11/20, see --cut-share), leader, language (English
--english-share, else Chinese), shape and length. Each seed is its own family; the split comes from a salted hash
(dev 5%, calibration 5%, train 90%). --offset starts the rotation later, so a scale-up batch does not repeat the
pilot's seeds.
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
from pipeline import CUT_KINDS, LEADERS, LENGTHS, PASS_KINDS, PROMPT_VERSION, SHAPES, SYSTEM_PROMPT, schema  # noqa: E402
from task_pack import write_generation_tasks  # noqa: E402

SALT = "leader-v1-2026-09-25"
NO_LEADER = ("cartoon_only", "cartoon_china", "insult_assistant")
# The pilot (aster-dev-370) skipped satire and rumour seeds for leaders with no widely reported mocking comparison
# or rumour; from --leader-pools on, those kinds draw from leaders with well-documented cases (the cartoon-bear
# comparison of the current leader weighted most, as in the case that was missed).
LEADER_POOLS = {
    "satire_restated": ("Xi Jinping",) * 5 + ("Jiang Zemin", "Wen Jiabao", "Mao Zedong", "Hu Jintao"),
    "rumor_restated": ("Xi Jinping",) * 2 + ("Jiang Zemin", "Wen Jiabao", "Hu Jintao", "Li Keqiang", "Mao Zedong",
                                           "Deng Xiaoping"),
}


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_of(family):
    bucket = number(f"{SALT}:split:{family}") % 100
    return "dev" if bucket < 5 else "calibration" if bucket < 10 else "train"


def plan(count, offset=0, cut_share=0.55, english_share=0.7, pools=False):
    """Pure: the seeds for ordinals offset .. offset + count - 1. pools: per-kind leader pools (LEADER_POOLS)."""
    seeds = []
    shapes, lengths = sorted(SHAPES), sorted(LENGTHS)
    for ordinal in range(offset, offset + count):
        cut = number(f"{SALT}:cut:{ordinal}") % 1000 < cut_share * 1000
        kinds = CUT_KINDS if cut else PASS_KINDS
        kind = kinds[(ordinal // 2) % len(kinds)]
        language = "en" if number(f"{SALT}:lang:{ordinal}") % 1000 < english_share * 1000 else "zh"
        choices = LEADER_POOLS.get(kind, LEADERS) if pools else LEADERS
        leader = None if kind in NO_LEADER else choices[number(f"{SALT}:leader:{ordinal}") % len(choices)]
        task_key = f"leader-v1-{ordinal:06d}"
        seeds.append({"task_key": task_key, "word": task_key, "kind": kind, "leader": leader, "language": language,
                      "shape": shapes[ordinal % len(shapes)], "length": lengths[(ordinal // 3) % len(lengths)],
                      "family": task_key, "split": split_of(task_key), "source_group": f"leader_v1/{kind}"})
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cut-share", type=float, default=0.55)
    parser.add_argument("--english-share", type=float, default=0.7)
    parser.add_argument("--leader-pools", action="store_true", help="per-kind leader pools for satire and rumour")
    parser.add_argument("--name-prefix", default="leader-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = plan(args.count, args.offset, args.cut_share, args.english_share, args.leader_pools)
    packed = write_generation_tasks(seeds, args.output, args.name_prefix,
                                    {"calls_per_term": 1, "prompt_version": PROMPT_VERSION}, 1)
    with (args.output / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps({**seed, "group_key": packed["task_of"][seed["task_key"]]}, ensure_ascii=False) + "\n")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"prompt_version": PROMPT_VERSION, "seeds": len(seeds), "offset": args.offset,
                "leader_pools": args.leader_pools, "tasks": packed["tasks"],
                "archives": packed["archives"],
                "kinds": dict(collections.Counter(s["kind"] for s in seeds)),
                "languages": dict(collections.Counter(s["language"] for s in seeds)),
                "leaders": dict(collections.Counter(str(s["leader"]) for s in seeds)),
                "splits": dict(collections.Counter(s["split"] for s in seeds)),
                "sha256": {"pipeline": digest(ROOT / "flow/pipeline.py"), "flow": digest(ROOT / "flow/flow.py"),
                           "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("seeds", "tasks", "archives", "kinds", "languages", "splits")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
