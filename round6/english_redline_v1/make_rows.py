"""English red-line data v1 (T032): Aegis 2.0 train rows for the red-line judge (Mac, CPU).

Stage 2 v1 lost most of Round5's English red-line recall on the Qwen3Guard benchmarks (weapons / drugs, self-harm,
sexual, violence; round6/qwen3guard_bench/redline_subset.py): its training data is almost all Chinese. This builds
English prompts and prompt + response pairs from the Aegis 2.0 *train* split (nvidia/Aegis-AI-Content-Safety-
Dataset-2.0, CC BY 4.0, not gated, the revision pinned in qwen3guard_bench/fetch.py) in the v14 row format that
redline_v1/make_tasks.py takes. The Aegis labels only pick the sample; the training labels come from our red-line
judge and apply_policy.py, so the rows follow POLICY-redline-scope.md (other harm becomes the alert tier).

- Groups (qwen3guard_bench/redline_subset.py): weapons_drugs, self_harm, sexual, violence, other_harm, safe; a
  deterministic hash-ordered sample per group and target (QUOTAS; a group smaller than its quota is taken whole).
- Left out: redacted or empty prompts, texts over MAX_CHARS, and any row whose prompt or response also appears as
  a message of the benchmark cases (qwen3guard_bench/data/cases.jsonl), so the benchmarks stay held out.
- Split by the Aegis row id (5% dev, 5% calibration); a row's prompt and response share it.
Output (data/, gitignored): rows_user.jsonl, rows_assistant.jsonl, manifest.json (counts and SHA256s only).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "qwen3guard_bench"
sys.path.insert(0, str(BENCH))
from redline_subset import AEGIS2, first_group  # noqa: E402

VERSION = "english-redline-rows-v1"
SALT = "english-redline-v1"
SOURCE_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
MAX_CHARS = 6000
QUOTAS = {
    "user": {"weapons_drugs": 1300, "self_harm": 700, "sexual": 800, "violence": 800, "other_harm": 700, "safe": 800},
    "assistant": {"weapons_drugs": 700, "self_harm": 100, "sexual": 450, "violence": 450, "other_harm": 600,
                  "safe": 900},
}


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def split_of(row_id):
    bucket = number(f"{SALT}:split:{row_id}") % 100
    return "dev" if bucket < 5 else "calibration" if bucket < 10 else "train"


def categories(row):
    return [x.strip() for x in (row.get("violated_categories") or "").split(",") if x.strip()]


def usable_text(text):
    return isinstance(text, str) and text.strip() and text.strip() != "REDACTED" and len(text) <= MAX_CHARS


def make(row, target, group):
    """Pure: one v14-format row for the judge (make_tasks.py) and build_targets.py."""
    sample_id = f"aegis2en:{row['id']}:{target}"
    messages = [{"role": "user", "content": row["prompt"]}]
    if target == "assistant":
        messages.append({"role": "assistant", "content": row["response"]})
    return {"sample_id": sample_id, "task_key": sample_id, "word": None, "split": split_of(row["id"]), "language": "en",
            "family": f"aegis2en:{row['id']}", "index": None, "response_style": group,
            "source_group": f"aegis2_en/{group}", "label": "safe" if group == "safe" else "unsafe", "messages": messages}


def candidates(rows, target, held_out):
    """Pure: group -> rows usable for the target (prompt, or prompt + response), in hash order."""
    pools = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: number(f"{SALT}:{target}:{r['id']}")):
        if not usable_text(row.get("prompt")) or row["prompt"] in held_out:
            continue
        if target == "user":
            if row.get("prompt_label") not in ("safe", "unsafe"):
                continue
            group = "safe" if row["prompt_label"] == "safe" else first_group(categories(row), AEGIS2)
        else:
            if not usable_text(row.get("response")) or row["response"] in held_out \
                    or row.get("response_label") not in ("safe", "unsafe"):
                continue
            group = "safe" if row["response_label"] == "safe" else first_group(categories(row), AEGIS2)
        pools[group].append(row)
    return pools


def build(rows, held_out, quotas=QUOTAS):
    out = {}
    for target, quota in quotas.items():
        pools = candidates(rows, target, held_out)
        out[target] = [make(r, target, g) for g in quota for r in pools.get(g, [])[:quota[g]]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=HERE / "data" / "raw" / "aegis2_train.json")
    parser.add_argument("--bench-cases", type=Path, default=BENCH / "data" / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args()
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    held_out = set()
    with args.bench_cases.open(encoding="utf-8") as handle:
        for line in handle:
            held_out.update(m["content"] for m in json.loads(line)["messages"])
    built = build(rows, held_out)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"version": VERSION, "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0 train.json",
                "revision": SOURCE_REVISION, "license": "CC BY 4.0", "source_sha256": sha(args.source),
                "bench_cases_sha256": sha(args.bench_cases), "quotas": QUOTAS}
    for target, made in built.items():
        path = args.out / f"rows_{target}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in made), encoding="utf-8")
        manifest[target] = {"rows": len(made), "sha256": sha(path),
                            "by_group": dict(collections.Counter(r["response_style"] for r in made)),
                            "by_split": dict(collections.Counter(r["split"] for r in made))}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({t: manifest[t] for t in built}, indent=1))


if __name__ == "__main__":
    main()
