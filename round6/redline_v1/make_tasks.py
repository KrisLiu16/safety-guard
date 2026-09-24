"""Build red-line labelling Tasks from v14-format rows (Run A trainable.jsonl, the stage-1 input, S2 / S5 examples).

Only the user prompt and the response text are sent. Every label, the v14 onset and the source metadata stay in the
local answer key (items.jsonl). --sample-ids restricts the build to a list (e.g. the failures of a first run, for a
fallback judge).

--target user labels prompts instead (judge user mode): one item per distinct prompt. Run A / stage-1 rows give
one unsafe and one safe prompt per word, with sample ids "<task_key>:prompt:<label>" (the ids of the T023 user-head
cache); rows exported with export_prefix_v2.py --role user are used as they are (earlier turns become context).
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
from levels import LEVEL_VERSION  # noqa: E402
from pipeline import PROMPT_VERSION, USER_PROMPT_VERSION  # noqa: E402
from policy import digest  # noqa: E402
from task_pack import write_generation_tasks  # noqa: E402

KEY_FIELDS = ("sample_id", "task_key", "word", "split", "language", "family", "index", "response_style",
              "source_group", "label", "label_tier", "onset_char", "onset_end_char")


def context(messages):
    """What the judge sees before the judged text: the user turn, or every earlier turn when there are several."""
    before = messages[:-1]
    if len(before) == 1 and before[0]["role"] == "user":
        return before[0]["content"]
    return "\n\n".join(m["role"].upper() + ":\n" + m["content"] for m in before)


def order(row):
    return hashlib.sha256(f"redline-v1:{row['sample_id']}".encode()).hexdigest()


def pick(rows, count):
    """Deterministic sample balanced over (slot, old label, label tier, language), round-robin in hash order."""
    pools = collections.defaultdict(list)
    for row in sorted(rows, key=order):
        pools[tuple(str(row.get(f)) for f in ("index", "label", "label_tier", "language"))].append(row)
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(chosen) < count:
                chosen.append(pools[key].pop(0))
    return chosen


def prompt_rows(rows):
    """One row per distinct prompt, in the row format build() takes; the prompt is the last message."""
    out, seen = [], set()
    for r in rows:
        if "prompt_label" not in r:           # already a prompt row (export_prefix_v2.py --role user)
            out.append(r)
            continue
        sample_id = f"{r['task_key']}:prompt:{r['prompt_label']}"
        if sample_id in seen:
            continue
        seen.add(sample_id)
        out.append({"sample_id": sample_id, "task_key": r["task_key"], "word": r.get("word"), "split": r.get("split"),
                    "language": r.get("language"), "family": r.get("family"), "index": None, "response_style": None,
                    "source_group": r.get("source_group"), "label": r["prompt_label"], "messages": [r["messages"][0]]})
    return out


def build(rows, splits=None, sample_ids=None, count=0, target="assistant"):
    """Pure: (blind items in a stable order, answer-key rows). count > 0 takes a balanced sample."""
    if target == "user":
        rows = prompt_rows(rows)
    chosen = [r for r in rows if (not splits or r.get("split") in splits)
              and (sample_ids is None or r["sample_id"] in sample_ids)]
    if count:
        chosen = pick(chosen, count)
    chosen.sort(key=order)
    if target == "user":
        items = [{"task_key": r["sample_id"], "context": context(r["messages"]), "user_prompt": r["messages"][-1]["content"]}
                 for r in chosen]
    else:
        items = [{"task_key": r["sample_id"], "user_prompt": context(r["messages"]),
                  "assistant_text": r["messages"][-1]["content"]} for r in chosen]
    key = [{**{f: r.get(f) for f in KEY_FIELDS}, "response_chars": len(r["messages"][-1]["content"])} for r in chosen]
    return items, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="v14-format rows (jsonl)")
    parser.add_argument("--splits", default="", help="comma-separated splits to keep (default: all)")
    parser.add_argument("--sample-ids", type=Path, help="optional file with one sample_id per line")
    parser.add_argument("--count", type=int, default=0, help="balanced sample of this many responses (0 = all)")
    parser.add_argument("--per-task", type=int, default=10, help="responses per Task (size-capped)")
    parser.add_argument("--name-prefix", default="redline-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("assistant", "user"), default="assistant")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = set(args.sample_ids.read_text(encoding="utf-8").split()) if args.sample_ids else None
    items, key = build(rows, set(args.splits.split(",")) if args.splits else None, ids, args.count, args.target)
    judge_version = PROMPT_VERSION if args.target == "assistant" else USER_PROMPT_VERSION
    base = {"prompt_version": LEVEL_VERSION, "judge_version": judge_version, "policy_digest": digest()}
    if args.target == "user":
        base["target"] = "user"
    packed = write_generation_tasks(items, args.output, args.name_prefix, base, args.per_task, category="guard-redline-label")
    with (args.output / "items.jsonl").open("w", encoding="utf-8") as handle:          # local answer key
        for row in key:
            handle.write(json.dumps({**row, "group_key": packed["task_of"][row["sample_id"]]}, ensure_ascii=False) + "\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"level_version": LEVEL_VERSION, "judge_version": judge_version, "target": args.target, "policy_digest": digest(),
                "source": str(args.source), "source_sha256": sha(args.source), "splits": args.splits or "all",
                "responses": len(items), "tasks": packed["tasks"], "archives": packed["archives"],
                "count": args.count, "by_split_label": dict(collections.Counter(f"{r['split']}:{r['label']}" for r in key)),
                "by_slot_language": dict(collections.Counter(f"{r['index']}:{r['language']}" for r in key)),
                "sha256": {name: sha(HERE / "flow" / name) for name in ("flow.py", "levels.py", "pipeline.py", "policy.py")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("responses", "tasks", "archives", "by_split_label")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
