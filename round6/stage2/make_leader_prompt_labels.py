"""User-side labels for the T031 pass-kind conversations (Mac, CPU): stage 2 v2 (T030).

Pass kinds are written to be ordinary (news about a leader, cartoons, places, regions, everyday words); their last
user message is labelled safe as a whole (one probe at the end). insult_assistant prompts insult the assistant: the
alert tier (POLICY section 9), alert from the last clause. Cut kinds get no user-side label: their prompts need the
prompt-mode judge and are left out of the user head. With --assistant-labels (v1.1), a prompt is labelled only
when the red-line judge found its conversation's thinking usable and safe (alert included): a generated "ordinary"
conversation that drifted into a red line keeps no user-side label. Output: labels.jsonl rows in apply_policy.py's format with
sample id "<sample_id>:prompt", and summary.json.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "leader_v1" / "flow"))
from pipeline import CUT_KINDS  # noqa: E402


def label(row, assistant=None):
    """Pure: the user-side label row for one conversation, or None for cut kinds and, when assistant labels are
    given ({sample_id: label row}), for conversations whose thinking is not usable and safe."""
    if row["response_style"] in CUT_KINDS:
        return None
    if assistant is not None:
        judged = assistant.get(row["sample_id"])
        if not judged or judged.get("excluded") or judged.get("level") != "safe":
            return None
    prompt = row["messages"][-2]["content"]
    alert = {"from": None} if row["response_style"] == "insult_assistant" else None
    out = {"sample_id": row["sample_id"] + ":prompt", "split": row["split"], "language": row["language"],
           "response_style": row["response_style"], "level": "safe", "label": "safe", "excluded": False,
           "stratum": "non_redline_harm" if alert else "normal", "rules": [], "onsets": {}, "boundary": [],
           "probes": [{"cut": len(prompt), "level": "safe"}], "source_rank": 0}
    if alert:
        out["alert"] = alert
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, nargs="+", help="leader_v1 examples.jsonl files")
    parser.add_argument("--assistant-labels", type=Path, help="apply_policy labels.jsonl of the same rows")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.rows:                         # line by line, not splitlines(): texts may hold U+2028
        with path.open(encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    assistant = None
    if args.assistant_labels:
        assistant = {}
        for line in args.assistant_labels.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                assistant[r["sample_id"]] = r
    labels = [lab for lab in (label(r, assistant) for r in rows) if lab is not None]
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for lab in labels:
            handle.write(json.dumps(lab, ensure_ascii=False) + "\n")
    summary = {"rows": len(rows), "labels": len(labels),
               "by_kind": dict(collections.Counter(f"{l['response_style']}:{'alert' if l.get('alert') else 'safe'}"
                                                   for l in labels)),
               "by_split": dict(collections.Counter(l["split"] for l in labels))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
