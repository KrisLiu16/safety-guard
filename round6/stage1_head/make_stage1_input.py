"""Build the stage-1 Run A input (Mac, CPU): a train sample for readout training plus every calibration word.

Dev words are left out on purpose: dev is only for the final comparison (eval step). The train sample is
deterministic by family hash, as in round6/probe/make_probe_input.py, with its own salt.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
from make_probe_input import FIELDS  # noqa: E402

DEFAULT_SOURCE = HERE.parent / "response_v14/batch_50k/extracted/trainable.jsonl"
SALT = "stage1-v1-train-words"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(rows, train_words):
    by_word = collections.defaultdict(list)
    for row in rows:
        by_word[row["task_key"]].append(row)
    train = sorted((k for k, v in by_word.items() if v[0]["split"] == "train"),
                   key=lambda k: hashlib.sha256(f"{SALT}:{by_word[k][0]['family']}".encode()).hexdigest())
    keep = set(train[:train_words]) | {k for k, v in by_word.items() if v[0]["split"] == "calibration"}
    return [{f: r.get(f) for f in FIELDS} for k in sorted(keep) for r in sorted(by_word[k], key=lambda r: r["index"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-words", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=HERE / "input_v1/stage1_runA_v1.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    chosen = select(rows, args.train_words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "stage1-runA-input-v1", "source_sha256": sha(args.source), "output_sha256": sha(args.output),
                "records": len(chosen), "train_words_requested": args.train_words,
                "words_by_split": dict(collections.Counter(r["split"] for r in chosen if r["index"] == 0)),
                "records_by_split_label": dict(collections.Counter(f"{r['split']}:{r['label']}" for r in chosen)),
                "salt": SALT}
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
