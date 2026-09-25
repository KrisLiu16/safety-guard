"""v3 (T033) teacher inputs: every text the stage-2 student trains on, as {id, source, role, messages} (CPU).

Same sources and sample ids as build_targets.py: Run A answers ("<sample_id>"), Run A prompts
("<task_key>:prompt:<prompt_label>"), prefix_v2 train and calibration rows of both roles ("<sample_id>"; augmented
views reuse their row's teacher scores), leader_v1 conversations and their prompts ("<sample_id>:prompt"), and the
--extra v14-row sources (English red-line rows). The last message is the one the student scores; the teacher
(teacher_label_l20.py) scores the same message with the head of its role. Output JSONL is local data (not committed).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cases(runA, prefix, leader, extra):
    """Pure: the teacher cases, one per (sample id, role), in a stable order."""
    seen = set()

    def case(sample_id, source, messages):
        key = (sample_id, messages[-1]["role"])
        if key in seen:
            return None
        seen.add(key)
        return {"id": sample_id, "source": source, "role": messages[-1]["role"], "messages": messages}

    out = []
    for row in runA:
        out.append(case(row["sample_id"], "runA", row["messages"]))
        out.append(case(f"{row['task_key']}:prompt:{row['prompt_label']}", "runA_prompts", [row["messages"][0]]))
    for split_rows in prefix.values():
        for row in split_rows:
            out.append(case(row["sample_id"], "prefix_v2", row["messages"]))
    for row in leader:
        out.append(case(row["sample_id"], "leader_v1", row["messages"]))
        out.append(case(row["sample_id"] + ":prompt", "leader_v1_prompts", row["messages"][:-1]))
    for source, rows in extra:
        for row in rows:
            out.append(case(row["sample_id"], source, row["messages"]))
    return [c for c in out if c is not None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runA", type=Path, default=Path("/work/round6/stage1_head/input_v1/stage1_runA_v1.jsonl"))
    parser.add_argument("--prefix-data", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--leader-rows", type=Path, default=Path("/work/round6/leader_v1/examples.jsonl"))
    parser.add_argument("--extra", nargs=2, action="append", default=[], metavar=("SOURCE", "ROWS"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefix = {split: read_jsonl(args.prefix_data / f"{split}.jsonl") for split in ("train", "calibration")}
    made = cases(read_jsonl(args.runA), prefix, read_jsonl(args.leader_rows),
                 [(source, read_jsonl(rows)) for source, rows in args.extra])
    with args.output.open("w", encoding="utf-8") as handle:
        for c in made:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(made), "by_source_role": dict(collections.Counter(f"{c['source']}/{c['role']}"
                                                                                       for c in made))}))


if __name__ == "__main__":
    main()
