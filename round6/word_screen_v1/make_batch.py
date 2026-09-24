"""Build word-screen Tasks from v14 seeds files (Mac, CPU): each distinct word is screened once.

Full mode screens every distinct word of the given seeds files (Run A + Run B). Pilot mode (--pilot N)
takes N words spread over source groups plus every word in --include (a local, uncommitted file with one
known positive per line, e.g. the words T006 found; they are tagged so recall can be measured).
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
from pipeline import BINARY_PROMPT, PROMPT_VERSION, SYSTEM_PROMPT  # noqa: E402

DEFAULT_SEEDS = (ROOT.parent / "response_v14/batch_50k/seeds.jsonl", ROOT.parent / "response_v14/batch_rest/seeds.jsonl")
SALT = "word-screen-v1"
BATCH_WORDS = 200
PART_TASKS = 2000   # Aster: at most 2,000 Tasks per Run and per dataset version in a Run


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def distinct_words(seed_rows):
    """word -> {source_groups, task_keys}; the word string is the unit that gets screened."""
    words = {}
    for row in seed_rows:
        entry = words.setdefault(row["word"], {"word": row["word"], "source_groups": set(), "task_keys": []})
        entry["source_groups"].add(row.get("source_group", ""))
        entry["task_keys"].append(row["task_key"])
    return words


def pilot_selection(words, count, include):
    by_group = collections.defaultdict(list)
    for word, entry in words.items():
        by_group[min(entry["source_groups"])].append(word)
    for group in by_group.values():
        group.sort(key=lambda w: number(f"{SALT}:{w}"))
    chosen = [w for w in include if w in words] + [w for w in include if w not in words]
    picked = set(chosen)
    while len(picked) < count + len(include) and any(by_group.values()):
        for group in sorted(by_group):
            while by_group[group] and by_group[group][0] in picked:
                by_group[group].pop(0)
            if by_group[group] and len(picked) < count + len(include):
                word = by_group[group].pop(0)
                chosen.append(word)
                picked.add(word)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, action="append", default=None)
    parser.add_argument("--pilot", type=int, default=0, help="pilot word count; 0 = every distinct word")
    parser.add_argument("--include", type=Path, help="local file of known positives, one word per line")
    parser.add_argument("--passes", type=int, default=2, help="category-mode passes per word, each with its own batch order")
    parser.add_argument("--binary-passes", type=int, default=1, help="binary-mode (yes / unsure / no) passes per word")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for path in (args.seeds or DEFAULT_SEEDS) for line in path.open(encoding="utf-8")]
    words = distinct_words(rows)
    include = list(dict.fromkeys(w.strip() for w in args.include.read_text(encoding="utf-8").splitlines() if w.strip())) \
        if args.include else []
    order = pilot_selection(words, args.pilot, include) if args.pilot else sorted(words, key=lambda w: number(f"{SALT}:{w}"))
    known = set(include)
    entries = [{"word": w, "known_positive": w in known,
                "source_groups": sorted(words[w]["source_groups"]) if w in words else ["include-only"],
                "task_keys": words[w]["task_keys"] if w in words else []} for w in order]
    paths = []
    plan = [("category", f"p{p}") for p in range(args.passes)] + [("binary", f"b{p}") for p in range(args.binary_passes)]
    for p, (mode, tag) in enumerate(plan):
        # Each pass puts every word in a different batch with different neighbours.
        ordered = sorted(entries, key=lambda e: number(f"{SALT}:pass{p}:{e['word']}")) if p else entries
        for start in range(0, len(ordered), BATCH_WORDS):
            key = f"screen-{tag}-{start // BATCH_WORDS:05d}"
            task = args.output / "tasks" / key
            (task / "tests").mkdir(parents=True)
            batch = {"batch_key": key, "prompt_version": PROMPT_VERSION, "mode": mode,
                     "words": [{"word": e["word"]} for e in ordered[start:start + BATCH_WORDS]]}   # nothing but the word is sent
            (task / "instruction.md").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
            (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-word-screen"\n')
            test = task / "tests/test.sh"
            test.write_text("#!/bin/sh\nexit 1\n")
            test.chmod(0o755)
            paths.append(task)
            for e in ordered[start:start + BATCH_WORDS]:
                e.setdefault("batch_keys", []).append(key)
    # One archive per Run: a single screen.tar.gz when it fits, otherwise screen_partN.tar.gz of <= PART_TASKS each.
    parts = [paths[i:i + PART_TASKS] for i in range(0, len(paths), PART_TASKS)]
    names = ["screen.tar.gz"] if len(parts) == 1 else [f"screen_part{i}.tar.gz" for i in range(len(parts))]
    for name, part in zip(names, parts):
        with tarfile.open(args.output / name, "w:gz") as archive:
            for task in part:
                archive.add(task, arcname=task.name)
    with (args.output / "words.jsonl").open("w", encoding="utf-8") as handle:     # local index, not uploaded
        for e in entries:
            handle.write(json.dumps(e, ensure_ascii=False) + "\n")
    manifest = {"prompt_version": PROMPT_VERSION, "mode": "pilot" if args.pilot else "full", "words": len(entries),
                "known_positives": sum(e["known_positive"] for e in entries), "tasks": len(paths), "passes": args.passes,
                "binary_passes": args.binary_passes,
                "archives": {name: len(part) for name, part in zip(names, parts)},
                "batch_words": BATCH_WORDS, "seed_rows": len(rows),
                "sha256": {"pipeline": hashlib.sha256((ROOT / "flow/pipeline.py").read_bytes()).hexdigest(),
                           "flow": hashlib.sha256((ROOT / "flow/flow.py").read_bytes()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                           "binary_prompt": hashlib.sha256(BINARY_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("mode", "words", "known_positives", "passes", "binary_passes", "tasks")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
