"""Export the Round5 prefix_v2 assistant records as v14-format rows for red-line labelling (cluster or Mac, CPU).

Reads the frozen /work/round5/data/prefix_v2/{train,calibration,dev}.jsonl without changing them and keeps the
assistant-role records. Messages are copied unchanged, so the stage-1 relabel step can re-tokenize
probe_common.serialize(messages) and check it against the frozen ids. The output holds dataset text: keep it on
the PVC or the Mac, never commit it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "calibration", "dev")


def export(row):
    return {"sample_id": row["sample_id"], "task_key": row["sample_id"], "word": None, "split": row["base_split"],
            "language": row.get("language"), "family": row["family"], "index": None, "response_style": None,
            "source_group": "prefix_v2/" + row["label_tier"], "label": row["source_label"],
            "label_tier": row["label_tier"], "messages": row["messages"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--output", type=Path, required=True, help="output jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, sources = collections.Counter(), {}
    with args.output.open("w", encoding="utf-8") as out:
        for split in SPLITS:
            path = args.data_root / f"{split}.jsonl"
            sources[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["target_role"] != "assistant":
                        continue
                    if row["messages"][-1]["role"] != "assistant":
                        raise ValueError("assistant target must be the last message: " + row["sample_id"])
                    out.write(json.dumps(export(row), ensure_ascii=False) + "\n")
                    counts[f"{split}:{row['source_label']}:{row['label_tier']}"] += 1
    manifest = {"version": "prefix-v2-assistant-export-v1", "source_sha256": sources,
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "records": sum(counts.values()), "by_split_label_tier": dict(sorted(counts.items()))}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
