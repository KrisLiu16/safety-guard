"""Build onset-relocation Tasks from Run A unsafe responses (Mac, CPU).

Picks unsafe responses from the chosen splits, balanced over slot (1 compliance, 3 drift) and language,
deterministically by sample hash. Only the user prompt and the response text are sent; the v14 onset and every
label stay in the local answer key (items.jsonl) for comparison after the run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
sys.path.insert(0, str(HERE.parent))
from onset import ONSET_VERSION  # noqa: E402
from pipeline import PROMPT_VERSION as JUDGE_VERSION  # noqa: E402
from task_pack import write_generation_tasks  # noqa: E402

DEFAULT_SOURCE = HERE.parent / "response_v14/batch_50k/extracted/trainable.jsonl"
SALT = "onset-v1"


def select(rows, splits, count):
    pools = collections.defaultdict(list)
    for row in rows:
        if row["split"] in splits and row["label"] == "unsafe":
            pools[(row["index"], row["language"])].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r: hashlib.sha256(f"{SALT}:{r['sample_id']}".encode()).hexdigest())
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(chosen) < count:
                chosen.append(pools[key].pop(0))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--name-prefix", default="onset-pilot")
    parser.add_argument("--output", type=Path, default=HERE / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    chosen = select(rows, set(args.splits.split(",")), args.count)
    items = [{"task_key": r["sample_id"], "user_prompt": r["messages"][0]["content"],
              "assistant_text": r["messages"][1]["content"]} for r in chosen]          # blind: nothing else is sent
    packed = write_generation_tasks(items, args.output, args.name_prefix,
                                    {"prompt_version": ONSET_VERSION, "judge_version": JUDGE_VERSION},
                                    args.per_task, category="guard-onset")
    with (args.output / "items.jsonl").open("w", encoding="utf-8") as handle:          # local answer key
        for r in chosen:
            handle.write(json.dumps({"sample_id": r["sample_id"], "task_key": r["task_key"], "word": r["word"],
                                     "split": r["split"], "language": r["language"], "slot": r["index"],
                                     "response_style": r["response_style"], "source_group": r.get("source_group"),
                                     "onset_char": r["onset_char"], "onset_end_char": r["onset_end_char"],
                                     "onset_style_requested": r.get("onset_style_requested"),
                                     "response_chars": len(r["messages"][1]["content"]),
                                     "group_key": packed["task_of"][r["sample_id"]]}, ensure_ascii=False) + "\n")
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"onset_version": ONSET_VERSION, "judge_version": JUDGE_VERSION, "responses": len(chosen),
                "tasks": packed["tasks"], "archives": packed["archives"], "splits": args.splits,
                "by_slot_language": dict(collections.Counter(f"{r['index']}:{r['language']}" for r in chosen)),
                "sha256": {name: digest(HERE / "flow" / name) for name in ("flow.py", "onset.py", "pipeline.py", "taxonomy.py")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("responses", "tasks", "archives", "by_slot_language")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
