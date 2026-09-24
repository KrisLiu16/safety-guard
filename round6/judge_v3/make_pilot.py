"""Build a judge v3 pilot from an extracted v14 examples.jsonl (the v14 pilot now, Run A later).

Items per response are the same probes as judge_v2, so v2 and v3 agreement are comparable:
  safe response   : whole text, and one prefix cut at 30-80% of its length   (generator label: safe)
  unsafe response : prefix before the onset quote (if >= 8 chars, generator label: safe),
                    prefix through the end of the onset quote and the whole text (generator label: unsafe)
The generator labels were written under the old 口径, so disagreement on items from political
source groups is expected and is what this pilot measures. Words are sampled deterministically,
spread across source groups. One Task per word; nothing but user_prompt and assistant_text is sent.
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
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT  # noqa: E402

DEFAULT_SOURCE = ROOT.parent / "response_v14/pilot/extracted/examples.jsonl"
SALT = "judge-v3-pilot-2026-09-24"


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def items_for(row):
    text, prompt = row["messages"][1]["content"], row["messages"][0]["content"]
    base = {"sample_id": row["sample_id"], "word": row["word"], "task_key": row["task_key"],
            "source_group": row.get("source_group", ""), "generator_label": row["label"],
            "response_style": row["response_style"], "user_prompt": prompt}
    if row["label"] == "safe":
        cut = int(len(text) * (0.3 + 0.5 * (number("mid:" + row["sample_id"]) % 1000) / 1000))
        return [{**base, "kind": "safe_whole", "expected_old": "safe", "cut_char": len(text), "assistant_text": text},
                {**base, "kind": "safe_mid_prefix", "expected_old": "safe", "cut_char": cut, "assistant_text": text[:cut]}]
    out = []
    if row["onset_char"] >= 8:
        out.append({**base, "kind": "unsafe_before_onset", "expected_old": "safe", "cut_char": row["onset_char"],
                    "assistant_text": text[:row["onset_char"]]})
    out.append({**base, "kind": "unsafe_through_onset", "expected_old": "unsafe", "cut_char": row["onset_end_char"],
                "assistant_text": text[:row["onset_end_char"]]})
    out.append({**base, "kind": "unsafe_whole", "expected_old": "unsafe", "cut_char": len(text), "assistant_text": text})
    return out


def pick_words(rows, words):
    """Round-robin over source groups, SHA order inside each group."""
    by_group = collections.defaultdict(dict)
    for row in rows:
        by_group[row.get("source_group", "")][row["task_key"]] = True
    queues = {g: sorted(keys, key=lambda k: number(f"{SALT}:{k}")) for g, keys in by_group.items()}
    chosen = []
    while len(chosen) < words and any(queues.values()):
        for group in sorted(queues):
            if queues[group] and len(chosen) < words:
                chosen.append(queues[group].pop(0))
    return set(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="extracted v14 examples.jsonl")
    parser.add_argument("--words", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    keep = pick_words(rows, args.words)
    items = [item for row in rows if row["task_key"] in keep for item in items_for(row)]
    for item in items:
        item["item_id"] = hashlib.sha256(f"{item['sample_id']}:{item['kind']}".encode()).hexdigest()[:20]
    by_word = collections.defaultdict(list)
    for item in items:
        by_word[item["task_key"]].append(item)
    task_paths = []
    for ordinal, (key, group) in enumerate(sorted(by_word.items())):
        group.sort(key=lambda it: number("order:" + it["item_id"]))
        name = f"judge-v3-pilot-{ordinal:03d}"
        task = args.output / "tasks" / name
        (task / "tests").mkdir(parents=True)
        blind = [{"item_id": it["item_id"], "user_prompt": it["user_prompt"], "assistant_text": it["assistant_text"]}
                 for it in group]
        (task / "instruction.md").write_text(json.dumps({"group_key": name, "prompt_version": PROMPT_VERSION,
                                                         "items": blind}, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{name}"\ncategory = "guard-judge"\n')
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        task_paths.append(task)
    with tarfile.open(args.output / "pilot.tar.gz", "w:gz") as archive:
        for task in task_paths:
            archive.add(task, arcname=task.name)
    with (args.output / "items.jsonl").open("w", encoding="utf-8") as handle:     # local answer key, never uploaded
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"prompt_version": PROMPT_VERSION, "items": len(items), "tasks": len(task_paths),
                "words": len(by_word), "kinds": dict(collections.Counter(i["kind"] for i in items)),
                "source_groups": dict(collections.Counter(i["source_group"] for i in items)),
                "source": str(args.source),
                "sha256": {"pipeline": hashlib.sha256((ROOT / "flow/pipeline.py").read_bytes()).hexdigest(),
                           "flow": hashlib.sha256((ROOT / "flow/flow.py").read_bytes()).hexdigest(),
                           "taxonomy": hashlib.sha256((ROOT / "flow/taxonomy.py").read_bytes()).hexdigest(),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("items", "tasks", "words", "kinds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
