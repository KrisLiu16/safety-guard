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
from pipeline import (CUT_KINDS, LEADERS, LENGTHS, NORMAL_KINDS_V12, PASS_KINDS, PROMPT_VERSION, SHAPES,  # noqa: E402
                      SYSTEM_PROMPT, schema)
from task_pack import write_generation_tasks  # noqa: E402

SALT = "leader-v1-2026-09-25"
TOPICS_PATH = ROOT / "data" / "topics.json"      # local only (gitignored): the ordinary-use topic lists


def load_topics(path=TOPICS_PATH):
    """(TOPICS, TOPICS_V13, TOPICS_V13_DOUBLE) from the local topic file; empty when it is absent."""
    if not Path(path).exists():
        return {}, {}, ()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ({k: tuple(v) for k, v in data["TOPICS"].items()}, {k: tuple(v) for k, v in data["TOPICS_V13"].items()},
            tuple(data["TOPICS_V13_DOUBLE"]))


TOPICS, TOPICS_V13, TOPICS_V13_DOUBLE = load_topics()


def load_everyday(path=TOPICS_PATH):
    """v1.4: the everyday_normal topics (non-political look-alike words, local file); empty when absent."""
    if not Path(path).exists():
        return ()
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")).get("TOPICS_V14_EVERYDAY", ()))


TOPICS_V14_EVERYDAY = load_everyday()
ENGLISH_DOMAINS = ("cooking", "travel planning", "programming", "maths homework", "science homework", "personal finance",
                   "fitness", "gardening", "writing help", "product comparison", "general health information",
                   "career advice", "language learning", "pets", "home repair", "history homework", "music practice",
                   "parenting", "email etiquette", "data analysis")
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


def plan_normal(count, offset=0, english_share=0.3):
    """Pure (v1.2): seeds over NORMAL_KINDS with a rotating topic, mostly Chinese; ordinals start at offset."""
    seeds = []
    shapes, lengths = sorted(SHAPES), sorted(LENGTHS)
    for ordinal in range(offset, offset + count):
        kind = NORMAL_KINDS_V12[ordinal % len(NORMAL_KINDS_V12)]
        topics = TOPICS[kind]
        topic = topics[number(f"{SALT}:topic:{ordinal}") % len(topics)]
        latin_topic = topic.isascii()
        language = "en" if latin_topic or number(f"{SALT}:lang:{ordinal}") % 1000 < english_share * 1000 else "zh"
        task_key = f"leader-v1-{ordinal:06d}"
        seeds.append({"task_key": task_key, "word": task_key, "kind": kind, "leader": None, "topic": topic,
                      "language": language, "shape": shapes[ordinal % len(shapes)],
                      "length": lengths[(ordinal // 3) % len(lengths)], "family": task_key, "split": split_of(task_key),
                      "source_group": f"leader_v1/{kind}"})
    return seeds


def plan_normal2(count, offset=0):
    """Pure (v1.3): seeds over TOPICS_V13, each (kind, topic) pair once and the TOPICS_V13_DOUBLE topics twice per
    cycle, in salted hash order; the language follows the topic (Latin script: English, else Chinese)."""
    pairs = [(kind, topic) for kind, topics in TOPICS_V13.items() for topic in topics
             for _ in range(2 if topic in TOPICS_V13_DOUBLE else 1)]
    pairs = [p for _, p in sorted((number(f"{SALT}:v13:{k}:{t}:{i}"), (k, t)) for i, (k, t) in enumerate(pairs))]
    seeds = []
    shapes, lengths = sorted(SHAPES), sorted(LENGTHS)
    for ordinal in range(offset, offset + count):
        kind, topic = pairs[ordinal % len(pairs)]
        task_key = f"leader-v1-{ordinal:06d}"
        seeds.append({"task_key": task_key, "word": task_key, "kind": kind, "leader": None, "topic": topic,
                      "language": "en" if topic.isascii() else "zh", "shape": shapes[ordinal % len(shapes)],
                      "length": lengths[(ordinal // 5) % len(lengths)], "family": task_key, "split": split_of(task_key),
                      "source_group": f"leader_v1/{kind}"})
    return seeds


def plan_normal3(count, offset=0):
    """Pure (v1.4, T035): 40% english_everyday over ENGLISH_DOMAINS, 35% everyday_normal over TOPICS_V14_EVERYDAY
    (language follows the topic's script), 25% the four political look-alike kinds over TOPICS_V13 again (new
    ordinals, so new texts). Aimed at the v3 failures: ordinary English streams and thinking that restates an
    ordinary question."""
    political = [(kind, topic) for kind, topics in TOPICS_V13.items() for topic in topics]
    seeds = []
    shapes, lengths = sorted(SHAPES), sorted(LENGTHS)
    for ordinal in range(offset, offset + count):
        r = number(f"{SALT}:v14:{ordinal}") % 100
        pick = number(f"{SALT}:v14topic:{ordinal}")
        if r < 40:
            kind, topic = "english_everyday", ENGLISH_DOMAINS[pick % len(ENGLISH_DOMAINS)]
            language = "en"
        elif r < 75:
            kind, topic = "everyday_normal", TOPICS_V14_EVERYDAY[pick % len(TOPICS_V14_EVERYDAY)]
            language = "en" if topic.isascii() else "zh"
        else:
            kind, topic = political[pick % len(political)]
            language = "en" if topic.isascii() else "zh"
        task_key = f"leader-v1-{ordinal:06d}"
        seeds.append({"task_key": task_key, "word": task_key, "kind": kind, "leader": None, "topic": topic,
                      "language": language, "shape": shapes[ordinal % len(shapes)],
                      "length": lengths[(ordinal // 5) % len(lengths)], "family": task_key, "split": split_of(task_key),
                      "source_group": f"leader_v1/{kind}"})
    return seeds


def plan_normal4(count, offset=0, answer_share=0.6):
    """Pure (v1.5, T037): only the political look-alike kinds, over the v1.2 topics (TOPICS) and the v1.3 topics
    (TOPICS_V13, the TOPICS_V13_DOUBLE ones twice), cycled in salted hash order with new ordinals so the texts are
    new; answer_share of the seeds ask for the assistant's final answer instead of its thinking. Aimed at v4's
    over-blocking sweep failures, which are ordinary answers (dates, a history lesson, Beijing places)."""
    pairs = sorted({(kind, topic) for kind, topics in TOPICS.items() for topic in topics}
                   | {(kind, topic) for kind, topics in TOPICS_V13.items() for topic in topics})
    pairs += [(kind, topic) for kind, topics in TOPICS_V13.items() for topic in topics if topic in TOPICS_V13_DOUBLE]
    pairs.sort(key=lambda p: number(f"{SALT}:v15order:{p[0]}:{p[1]}"))
    seeds = []
    shapes, lengths = sorted(SHAPES), sorted(LENGTHS)
    for ordinal in range(offset, offset + count):
        kind, topic = pairs[ordinal % len(pairs)]
        answer = number(f"{SALT}:v15format:{ordinal}") % 100 < answer_share * 100
        task_key = f"leader-v1-{ordinal:06d}"
        seeds.append({"task_key": task_key, "word": task_key, "kind": kind, "leader": None, "topic": topic,
                      "language": "en" if topic.isascii() else "zh", "format": "answer" if answer else "thinking",
                      "shape": shapes[ordinal % len(shapes)], "length": lengths[(ordinal // 5) % len(lengths)],
                      "family": task_key, "split": split_of(task_key), "source_group": f"leader_v1/{kind}"})
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cut-share", type=float, default=0.55)
    parser.add_argument("--english-share", type=float, default=0.7)
    parser.add_argument("--leader-pools", action="store_true", help="per-kind leader pools for satire and rumour")
    parser.add_argument("--normal", action="store_true", help="v1.2: the ordinary place / region / word kinds only")
    parser.add_argument("--normal2", action="store_true", help="v1.3: TOPICS_V13 over the four ordinary kinds")
    parser.add_argument("--normal3", action="store_true", help="v1.4: English everyday, everyday look-alikes, TOPICS_V13")
    parser.add_argument("--normal4", action="store_true", help="v1.5: political look-alikes, 60% answers")
    parser.add_argument("--name-prefix", default="leader-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = (plan_normal4(args.count, args.offset) if args.normal4 else
             plan_normal3(args.count, args.offset) if args.normal3 else
             plan_normal2(args.count, args.offset) if args.normal2 else plan_normal(args.count, args.offset)
             if args.normal else plan(args.count, args.offset, args.cut_share, args.english_share, args.leader_pools))
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
                "formats": dict(collections.Counter(s.get("format", "thinking") for s in seeds)),
                "languages": dict(collections.Counter(s["language"] for s in seeds)),
                "leaders": dict(collections.Counter(str(s["leader"]) for s in seeds)),
                "splits": dict(collections.Counter(s["split"] for s in seeds)),
                "sha256": {"pipeline": digest(ROOT / "flow/pipeline.py"), "flow": digest(ROOT / "flow/flow.py"),
                           "schema": hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("seeds", "tasks", "archives", "kinds", "formats", "languages", "splits")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
