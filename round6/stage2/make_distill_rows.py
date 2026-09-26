"""v5 (T037) teacher-only rows: texts the general head learns from Qwen3Guard-Stream-4B without any red-line label
(Mac, CPU). They move the general head, and the shared backbone through it, never the red-line head.

  bench_fit      the "fit" half of the benchmark cases (qwen3guard_bench/bench_split.py; Think is held whole):
                 prompt cases as user rows, response cases as assistant rows
  aegis_distill  Aegis 2.0 *train* prompts and prompt + response pairs that T032 did not use, every benchmark text
                 held out, one row per distinct scored text
Only train-split rows are written (Aegis rows keep T032's split by row id; bench rows are all train), so the
calibration and dev records keep their red-line labels only. Outputs (local data, not committed): rows_<source>_<role>.jsonl
in the v14 row format that build_targets.py --distill-only takes, teacher_cases.jsonl for teacher_label_l20.py
(assistant rows, per token) and teacher_cases_user.jsonl (user rows, --end-user), manifest.json (counts, SHA256s).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen3guard_bench"))
sys.path.insert(0, str(HERE.parent / "english_redline_v1"))
from bench_split import half  # noqa: E402
import make_rows as aegis  # noqa: E402


def bench_rows(cases):
    """Pure: the fit half as v14 rows (split train); prompt cases end with the user message, response cases with the
    assistant message."""
    out = []
    for case in cases:
        if half(case) != "fit":
            continue
        role = "user" if case["level"] == "prompt" else "assistant"
        if case["messages"][-1]["role"] != role:
            raise ValueError(f"{case['id']}: last message is not the {role} message")
        out.append({"sample_id": f"bench:{case['bench']}:{case['id']}", "task_key": f"bench:{case['bench']}:{case['id']}",
                    "split": "train", "language": "en", "family": f"bench:{case['bench']}:{case['id']}",
                    "source_group": f"bench_fit/{case['bench']}", "role": role, "messages": case["messages"]})
    return out


def aegis_rows(rows, held_out, used_user, used_assistant):
    """Pure: T032-unused Aegis train texts (train split), prompts as user rows and prompt + response as assistant
    rows; a scored text that repeats is kept once, benchmark texts never."""
    out, seen = [], set()
    for row in sorted(rows, key=lambda r: r["id"]):
        if aegis.split_of(row["id"]) != "train":
            continue
        prompt, response = row.get("prompt"), row.get("response")
        if aegis.usable_text(prompt) and prompt not in held_out and row["id"] not in used_user \
                and ("user", prompt) not in seen:
            seen.add(("user", prompt))
            out.append({"sample_id": f"aegis2en:{row['id']}:user:distill", "role": "user",
                        "messages": [{"role": "user", "content": prompt}]})
        if aegis.usable_text(prompt) and aegis.usable_text(response) and prompt not in held_out \
                and response not in held_out and row["id"] not in used_assistant and ("assistant", response) not in seen:
            seen.add(("assistant", response))
            out.append({"sample_id": f"aegis2en:{row['id']}:assistant:distill", "role": "assistant",
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]})
    for r in out:
        r.update(task_key=r["sample_id"], split="train", language="en", family=f"aegis2en:{r['sample_id'].split(':')[1]}",
                 source_group=f"aegis_distill/{r['role']}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = HERE.parent
    parser.add_argument("--bench-cases", type=Path, default=root / "qwen3guard_bench/data/cases.jsonl")
    parser.add_argument("--aegis", type=Path, default=root / "english_redline_v1/data/raw/aegis2_train.json")
    parser.add_argument("--aegis-used", type=Path, default=root / "english_redline_v1/data")
    parser.add_argument("--output", type=Path, default=root / "stage2/data/distill_v5")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.bench_cases.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle]
    held_out = {m["content"] for c in cases for m in c["messages"]}
    used = {}
    for role in ("user", "assistant"):
        with (args.aegis_used / f"rows_{role}.jsonl").open(encoding="utf-8") as handle:
            used[role] = {json.loads(line)["sample_id"].split(":")[1] for line in handle}
    made = {"bench_fit": bench_rows(cases),
            "aegis_distill": aegis_rows(json.loads(args.aegis.read_text(encoding="utf-8")), held_out,
                                        used["user"], used["assistant"])}
    args.output.mkdir(parents=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"version": "distill-rows-v5", "bench_cases_sha256": sha(args.bench_cases), "aegis_sha256": sha(args.aegis),
                "halves": dict(collections.Counter(f"{c['bench']}:{half(c)}" for c in cases)), "files": {}}
    teacher = {"assistant": [], "user": []}
    for source, rows in made.items():
        for role in ("user", "assistant"):
            part = [r for r in rows if r["role"] == role]
            path = args.output / f"rows_{source}_{role}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
            manifest["files"][path.name] = {"rows": len(part), "sha256": sha(path),
                                            "by_group": dict(collections.Counter(r["source_group"] for r in part))}
            teacher[role] += [{"id": r["sample_id"], "source": source, "role": role, "messages": r["messages"]} for r in part]
    for role, name in (("assistant", "teacher_cases.jsonl"), ("user", "teacher_cases_user.jsonl")):
        path = args.output / name
        path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in teacher[role]), encoding="utf-8")
        manifest["files"][name] = {"rows": len(teacher[role]), "sha256": sha(path)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["rows"] for k, v in manifest["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
