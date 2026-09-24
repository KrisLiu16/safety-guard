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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT.parent))
from pipeline import FORMS, PROMPT_VERSION, SYSTEM_PROMPT, request_body, schema  # noqa: E402
from task_pack import screen_exclusions, write_generation_tasks  # noqa: E402

DEFAULT_SEEDS = ROOT.parent / "response_v14/batch_rest/seeds.jsonl"
SALT = "s2-v1-2026-09-24"
FIELDS = ("task_key", "word", "origin_group_id", "data_layer", "source_licenses", "source_group", "family")


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_of(family):
    bucket = number(f"{SALT}:split:{family}") % 100
    return "dev" if bucket < 3 else "calibration" if bucket < 6 else "train"


def pick(seeds, words, exclude_keys=frozenset(), exclude_words=frozenset()):
    """Round-robin over source groups, SHA order inside each group; skipped task_keys and words never enter."""
    by_group = collections.defaultdict(list)
    for seed in seeds:
        if seed["task_key"] not in exclude_keys and seed["word"] not in exclude_words:
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
    parser.add_argument("--screen", type=Path, help="word screen screen.jsonl; words not judged `no` are left out")
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="seeds.jsonl files whose task_keys are skipped")
    parser.add_argument("--words-per-task", type=int, default=1)
    parser.add_argument("--name-prefix", default="s2-pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.seeds.open(encoding="utf-8") as handle:
        seeds = [json.loads(line) for line in handle]
    exclude_keys = set()
    for path in args.exclude:
        with path.open(encoding="utf-8") as handle:
            exclude_keys.update(json.loads(line)["task_key"] for line in handle)
    exclude_words, unscreened = set(), 0
    if args.screen:
        exclude_words, screened = screen_exclusions(args.screen)
        unscreened_words = {s["word"] for s in seeds} - screened
        exclude_words |= unscreened_words          # never generate for a word the screen did not see
        unscreened = len(unscreened_words)
    chosen = plan(pick(seeds, args.words, frozenset(exclude_keys), frozenset(exclude_words)))
    packed = write_generation_tasks(chosen, args.output, args.name_prefix,
                                    {"calls_per_term": 1, "prompt_version": PROMPT_VERSION}, args.words_per_task)
    with (args.output / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in chosen:
            handle.write(json.dumps({**seed, "group_key": packed["task_of"][seed["task_key"]]}, ensure_ascii=False) + "\n")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"prompt_version": PROMPT_VERSION, "words": len(chosen), "tasks": packed["tasks"],
                "words_per_task": args.words_per_task, "archives": packed["archives"],
                "excluded_task_keys": len(exclude_keys), "excluded_by_screen": len(exclude_words), "unscreened_words": unscreened,
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
    print(json.dumps({k: manifest[k] for k in ("words", "tasks", "archives", "excluded_by_screen", "languages", "forms", "splits")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
