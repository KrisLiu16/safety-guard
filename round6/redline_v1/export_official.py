"""Export the official Qwen3GuardTest thinking split as v14-format rows for red-line labelling (T027; standard library).

The stage-1 evaluation (eval_head_l20.py) scores the official thinking split, but its Safe / Unsafe labels follow the
official policy, not ours. This export lets redline_v1 judge the same assistant turns under the red-line policy, so
analyze_redline.py --official-labels can report the new heads' cuts on the official set by our labels.
Rows keep the evaluation sample_id (Qwen3GuardTest:thinking:<index>:<unique_id>) and the original messages; the
labelled text is the assistant content (<think> ... </think> and the answer), the prompt is the context. Only rows
with status "ready" are exported (the ones the evaluation scores). The output holds dataset text: never commit it;
commit only the manifest.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def language(text):
    cjk = sum("一" <= ch <= "鿿" for ch in text)
    return "zh" if cjk >= max(1, len(text) // 20) else "en"


def export(rows):
    """Pure: prepared official rows -> v14-format rows."""
    out = []
    for row in rows:
        if row.get("status") != "ready":
            continue
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("last message is not the assistant turn: " + row["sample_id"])
        out.append({"sample_id": row["sample_id"], "task_key": row["sample_id"], "split": "official_thinking",
                    "label": row["label"].lower(), "language": language(messages[-1]["content"]),
                    "family": row["sample_id"], "unsafe_type": row.get("unsafe_type"), "source": row.get("source"),
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/work/validation/official_adapted_v1"))
    parser.add_argument("--split", default="thinking")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.data_dir / f"{args.split}.jsonl"
    with source.open(encoding="utf-8") as handle:
        rows = export(json.loads(line) for line in handle if line.strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "official-thinking-export-v1", "split": args.split,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "records": len(rows),
                "by_label_language": dict(sorted(collections.Counter(f"{r['label']}:{r['language']}" for r in rows).items())),
                "unsafe_type": dict(sorted(collections.Counter(str(r["unsafe_type"]) for r in rows).items()))}
    manifest_path = args.output.with_name(args.output.name.replace(".jsonl", ".manifest.json"))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
