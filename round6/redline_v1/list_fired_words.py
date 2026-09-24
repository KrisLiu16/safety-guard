"""List the political terms that fired in labels from apply_policy.py --political-terms (Mac, CPU; T026 v1.3).

For every labels.jsonl given (with the rows its Tasks were built from, for the texts), each fired word rule
(the `political` record: rule and span) is traced back to the word in the text. Output, one line per word, a
local review sheet (TSV) sorted by how often the word raised ordinary text:
  word, rule, rows it fired in, rows raised from normal / non_redline_harm / other_sensitive / redline_topic,
  up to three example sample ids, and an empty column for the reviewer ("exclude" or blank).
The reviewer checks the words that raised ordinary text and copies the ones that are not specifically political
into a local exclusion file for apply_policy.py --exclude-words. The sheet and the file hold the words: the sheet
must go under a review/ or review_*/ directory (gitignored), and neither is ever committed.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ORDINARY = ("normal", "non_redline_harm", "other_sensitive", "redline_topic")


def fired_words(labels, texts):
    """Pure. labels: label rows; texts: sample_id -> text. Yields (word, rule, stratum_before, raised, sample_id)."""
    for row in labels:
        record = row.get("political")
        if not record or row["sample_id"] not in texts:
            continue
        text = texts[row["sample_id"]]
        raised = record["label_before"] != record["label_after"]
        for fired in record["fired"]:
            yield text[fired["start"]:fired["end"]], fired["rule"], record["stratum_before"], raised, row["sample_id"]


def tally(events):
    """Pure: per (word, rule) counts; sorted by rows raised from ordinary text, then by rows."""
    table = collections.defaultdict(lambda: {"rows": 0, **{s: 0 for s in ORDINARY}, "examples": []})
    for word, rule, stratum, raised, sample_id in events:
        entry = table[(word, rule)]
        entry["rows"] += 1
        if raised and stratum in ORDINARY:
            entry[stratum] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(sample_id)
    return sorted(table.items(), key=lambda kv: (-sum(kv[1][s] for s in ORDINARY[:3]), -kv[1]["rows"], kv[0]))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True, help="one per --labels, same order")
    parser.add_argument("--target", choices=("assistant", "user"), nargs="+", required=True, help="one per --labels")
    parser.add_argument("--out", type=Path, required=True, help="a .tsv under a review/ or review_*/ directory")
    args = parser.parse_args()
    if not (len(args.labels) == len(args.source) == len(args.target)):
        parser.error("give one --source and one --target per --labels")
    if not any(part == "review" or part.startswith("review_") for part in args.out.parent.parts):
        parser.error("--out must be under a review/ or review_*/ directory (gitignored): the sheet holds the words")
    events = []
    for labels_path, source, target in zip(args.labels, args.source, args.target):
        rows = read_jsonl(source)
        if target == "user":
            from make_tasks import prompt_rows
            rows = prompt_rows(rows)
        texts = {r["sample_id"]: r["messages"][-1]["content"] for r in rows}
        events.extend(fired_words(read_jsonl(labels_path), texts))
    table = tally(events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(["word", "rule", "rows", *ORDINARY, "examples", "exclude"]) + "\n")
        for (word, rule), entry in table:
            handle.write("\t".join([word, rule, str(entry["rows"]), *(str(entry[s]) for s in ORDINARY),
                                    ",".join(entry["examples"]), ""]) + "\n")
    raised = sum(1 for _, e in table if any(e[s] for s in ORDINARY[:3]))
    print(json.dumps({"words": len(table), "words_raising_normal_harm_or_other": raised}))


if __name__ == "__main__":
    main()
