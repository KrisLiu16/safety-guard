"""Build a second-judge pilot from a finished judge v3 run: re-judge what the first judge could not
answer or called political, plus a fixed control sample, so the two judges can be compared.

Selection uses the first run's own output, not source-group names:
  failed    : items with no judgment in the first run (request errors, token cap, bad JSON);
  political : items the first judge labelled Politically Sensitive Topics or gave any political
              GB/T code (taxonomy.QWEN_TO_GBT["Politically Sensitive Topics"]);
  control   : a deterministic sample of the remaining judged items, for the cross-judge baseline.
The first judge's verdict is kept in the local answer key (reference_*) and never sent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from make_pilot import number, write_pilot  # noqa: E402
from taxonomy import QWEN_TO_GBT  # noqa: E402

POLITICAL_CATEGORY = "Politically Sensitive Topics"
POLITICAL_CODES = set(QWEN_TO_GBT[POLITICAL_CATEGORY])
SALT = "judge-v3-fallback-2026-09-24"


def select(items, judgments, controls):
    """Return the chosen items, each tagged with fallback_reason and the reference verdict."""
    verdict = {j["item_id"]: j for j in judgments}
    chosen, rest = [], []
    for item in items:
        ref = verdict.get(item["item_id"], {})
        label = ref.get("judge_label")
        tagged = {**item, "reference_label": label, "reference_qwen_category": ref.get("judge_qwen_category"),
                  "reference_gbt_codes": ref.get("judge_gbt_codes")}
        if label is None:
            chosen.append({**tagged, "fallback_reason": "failed"})
        elif ref.get("judge_qwen_category") == POLITICAL_CATEGORY or POLITICAL_CODES & set(ref.get("judge_gbt_codes") or []):
            chosen.append({**tagged, "fallback_reason": "political"})
        else:
            rest.append({**tagged, "fallback_reason": "control"})
    rest.sort(key=lambda it: number(f"{SALT}:{it['item_id']}"))
    return chosen + rest[:controls]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "pilot/items.jsonl", help="first run's local answer key")
    parser.add_argument("--judgments", type=Path, default=ROOT / "pilot/extracted/judgments.jsonl")
    parser.add_argument("--controls", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot_fallback")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = [json.loads(line) for line in args.items.open(encoding="utf-8")]
    judgments = [json.loads(line) for line in args.judgments.open(encoding="utf-8")]
    chosen = select(items, judgments, args.controls)
    reasons = {}
    for item in chosen:
        reasons[item["fallback_reason"]] = reasons.get(item["fallback_reason"], 0) + 1
    manifest = write_pilot(chosen, args.output, "judge-v3-fallback",
                           {"reference_items": str(args.items), "reference_judgments": str(args.judgments),
                            "fallback_reasons": reasons})
    print(json.dumps({k: manifest[k] for k in ("items", "tasks", "words", "fallback_reasons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
