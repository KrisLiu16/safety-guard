"""Benchmark cases for our red-line judge (Mac, CPU): the policy-aligned view of the Qwen3Guard benchmarks.

The benches' harm categories are broader than our red lines (T032: our judge finds about half of Aegis 2.0's
weapons / drugs prompts and a fifth of its self-harm prompts to be red-line content), so recall on the category
groups of redline_subset.py overstates what the guard should cut. This picks, per level, every unsafe case of the
four red-line groups plus a hash-ordered sample of OTHER_PER_LEVEL other-harm and SAFE_PER_LEVEL safe cases, and
writes them as v14 rows for redline_v1/make_tasks.py (prompt level with --target user). The labels from
apply_policy.py then give recall on red-line cases and false positives on the rest (policy_metrics.py).
Evaluation only: these rows must never enter training. Output under data/ (gitignored).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch  # noqa: E402
from redline_subset import groups_by_id  # noqa: E402

RED = ("weapons_drugs", "self_harm", "sexual", "violence")
OTHER_PER_LEVEL, SAFE_PER_LEVEL = 400, 400


def order(case_id):
    return hashlib.sha256(f"bench-judge-v1:{case_id}".encode()).hexdigest()


def pick(cases, groups):
    """Pure: level -> chosen cases (all red-line groups, then samples of other_harm and safe)."""
    out = collections.defaultdict(list)
    by = collections.defaultdict(list)
    for case in sorted(cases, key=lambda c: order(c["id"])):
        group = groups.get(case["id"])
        if group:
            by[(case["level"], group)].append(case)
    for level in ("prompt", "response"):
        for group in RED:
            out[level] += by[(level, group)]
        out[level] += by[(level, "other_harm")][:OTHER_PER_LEVEL] + by[(level, "safe")][:SAFE_PER_LEVEL]
    return out


def row(case, group):
    return {"sample_id": case["id"], "task_key": case["id"], "word": None, "split": "test", "language": "en",
            "family": case["id"], "index": None, "response_style": group, "source_group": f"bench/{case['bench']}",
            "label": "unsafe" if case["label"] else "safe", "messages": case["messages"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE / "data" / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args()
    groups = groups_by_id({n: fetch.read_source(n) for n in ("sst", "harmbench_behaviors", "aegis2", "beavertails",
                                                             "openaimod")})
    with args.cases.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle]
    chosen = pick(cases, groups)
    summary = {}
    for level, name in (("prompt", "user"), ("response", "assistant")):
        path = args.out / f"judge_rows_{name}.jsonl"
        path.write_text("".join(json.dumps(row(c, groups[c["id"]]), ensure_ascii=False) + "\n" for c in chosen[level]),
                        encoding="utf-8")
        summary[name] = {"rows": len(chosen[level]),
                         "by_group": dict(collections.Counter(groups[c["id"]] for c in chosen[level])),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.out / "judge_rows_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
