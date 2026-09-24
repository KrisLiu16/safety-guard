"""Build the T004/T005 input file from Run A trainable.jsonl (Mac, CPU, no model calls).

Keeps every dev and calibration word and a deterministic sample of train words (by family hash),
all four responses per word, with only the fields the L20 dump needs. Output is uploaded to the PVC.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "response_v14/batch_50k/extracted/trainable.jsonl"
FIELDS = ("sample_id", "task_key", "family", "split", "language", "label", "prompt_label", "index",
          "response_style", "response_format", "onset_style_requested", "onset_char", "onset_end_char",
          "messages", "quality_flags")
SALT = "probe-v1-train-words"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(rows, train_words):
    by_word = collections.defaultdict(list)
    for row in rows:
        by_word[row["task_key"]].append(row)
    train = sorted((k for k, v in by_word.items() if v[0]["split"] == "train"),
                   key=lambda k: hashlib.sha256(f"{SALT}:{by_word[k][0]['family']}".encode()).hexdigest())
    keep = set(train[:train_words]) | {k for k, v in by_word.items() if v[0]["split"] in ("dev", "calibration")}
    return [{f: r.get(f) for f in FIELDS} for k in sorted(keep) for r in sorted(by_word[k], key=lambda r: r["index"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-words", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=HERE / "input_v1/probe_input_v1.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    chosen = select(rows, args.train_words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "probe-input-v1", "source": str(args.source), "source_sha256": sha(args.source),
                "output_sha256": sha(args.output), "records": len(chosen),
                "words_by_split": dict(collections.Counter(r["split"] for r in chosen if r["index"] == 0)),
                "records_by_split_label": dict(collections.Counter(f"{r['split']}:{r['label']}" for r in chosen)),
                "records_by_slot_style": dict(collections.Counter(f"{r['index']}:{r['response_style']}" for r in chosen)),
                "train_words_requested": args.train_words, "salt": SALT}
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
