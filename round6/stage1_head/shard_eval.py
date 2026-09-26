"""Run eval_head_l20.py as N processes on one GPU, each scoring every N-th record, then merge them (T036).

eval_head_l20 streams one record at a time through the pinned canonical32 eager path: one process keeps an L20 at
about a quarter of its use (v4: 23% utilisation, 97 W, 2.2 GB, one CPU core at 100%). A record's computation does not
depend on which process scores it, so the merged files hold the same lines in the same order as a single-process run
(only the gzip headers differ) once the Triton kernel configs are fixed: processes sharing a GPU time each other's
autotuning, and even a lone process can pick either of two near-equal configs from run to run (l2norm BT 8 or 32),
so every shard is pinned (autotune_pin.py) to --autotune-file, or, without it, to what one process scoring every
50th record alone picks first. Merging checks that every shard passed its
integrity check, that all shards scored the same weights and that no record ordinal appears twice.

  shard_eval.py --shards 4 --output OUT [--compare DIR] -- <eval_head_l20.py arguments without --output>

writes OUT/report.json and OUT/eval_<variant>/*.jsonl.gz as eval_head_l20 would, keeps the shard outputs and logs in
OUT.shards/, and with --compare reports, per file, whether the merged lines equal those of an earlier run's DIR.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    with gzip.open(path, "rb") as handle:
        return handle.read().splitlines(keepends=True)


def merge_files(dirs, out):
    """The lines of every *.jsonl.gz in the shard dirs, put back in record order (order.json in each dir)."""
    orders = [json.loads((d / "order.json").read_text()) for d in dirs]
    names = sorted({p.name for d in dirs for p in d.glob("*.jsonl.gz")})
    out.mkdir(parents=True)
    for name in names:
        keyed = []
        for d, order in zip(dirs, orders):
            lines = read_lines(d / name) if (d / name).exists() else []
            ordinals = order.get(name, [])
            if len(ordinals) != len(lines):
                raise ValueError(f"{d / name}: {len(lines)} lines but {len(ordinals)} ordinals")
            keyed.extend(zip(ordinals, lines))
        keyed.sort(key=lambda kv: kv[0])
        ordinals = [k for k, _ in keyed]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError(f"{name}: a record was scored by two shards")
        with gzip.open(out / name, "wb") as handle:
            for _, line in keyed:
                handle.write(line)
    return names


def merge_reports(reports, output):
    """One report as eval_head_l20 writes it: counts and totals summed over shards, coverage taken whole."""
    failed = [i for i, r in enumerate(reports) if not r.get("integrity_pass")]
    if failed:
        raise ValueError(f"shards {failed} did not pass their integrity check")
    merged = {k: v for k, v in reports[0].items()
              if k not in ("variants", "totals", "max_prob_diff", "elapsed_seconds", "shard")}
    totals = collections.Counter()
    for r in reports:
        totals.update(r["totals"])
    merged.update(shards=len(reports), elapsed_seconds=max(r["elapsed_seconds"] for r in reports),
                  shard_elapsed_seconds=[r["elapsed_seconds"] for r in reports],
                  max_prob_diff=max(r["max_prob_diff"] for r in reports), totals=dict(totals), variants={})
    for variant in reports[0]["variants"]:
        weights = {r["variants"][variant]["weights_sha256"] for r in reports}
        if len(weights) != 1:
            raise ValueError(f"{variant}: shards scored different weights")
        counts = collections.Counter()
        for r in reports:
            counts.update(r["variants"][variant]["counts"])
        if "official_unique_sequences" in counts:            # every shard checks the full coverage itself
            counts["official_unique_sequences"] = reports[0]["variants"][variant]["counts"]["official_unique_sequences"]
        folder = output / f"eval_{variant}"
        merged["variants"][variant] = {"weights_sha256": weights.pop(), "counts": dict(counts),
                                       "files": {p.name: sha(p) for p in sorted(folder.glob("*.jsonl.gz"))}}
    return merged


def logprob_diff(a, b):
    """Largest absolute difference between the logprobs of two records with the same fields, or None."""
    x, y = json.loads(a), json.loads(b)
    if {k: v for k, v in x.items() if k != "logprobs"} != {k: v for k, v in y.items() if k != "logprobs"}:
        return None
    if len(x["logprobs"]) != len(y["logprobs"]):
        return None
    return max((abs(p - q) for u, v in zip(x["logprobs"], y["logprobs"]) for p, q in zip(u, v)), default=0.0)


def compare_dirs(a, b):
    """Per file: line counts, identical lines, and for differing lines the largest logprob difference."""
    out = {}
    for name in sorted({p.name for p in a.glob("*.jsonl.gz")} | {p.name for p in b.glob("*.jsonl.gz")}):
        la = read_lines(a / name) if (a / name).exists() else []
        lb = read_lines(b / name) if (b / name).exists() else []
        same = sum(1 for x, y in zip(la, lb) if x == y)
        diffs = [logprob_diff(x, y) for x, y in zip(la, lb) if x != y]
        out[name] = {"lines": [len(la), len(lb)], "identical_lines": same, "identical": la == lb,
                     "other_field_differences": sum(1 for d in diffs if d is None),
                     "max_abs_logprob_diff": max((d for d in diffs if d is not None), default=0.0)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="an earlier eval_head_l20 output dir to compare against")
    parser.add_argument("--threads", type=int, default=2, help="OMP_NUM_THREADS of each shard process")
    parser.add_argument("--no-pin-autotune", dest="pin", action="store_false",
                        help="let every shard autotune for itself (records may differ slightly from one process)")
    parser.add_argument("--autotune-file", type=Path,
                        help="a recorded table (e.g. autotune_l20_v1.json) to pin every shard to, instead of a warm-up")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="-- then eval_head_l20.py arguments")
    args = parser.parse_args(argv)
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    if "--output" in rest or "--shard" in rest:
        parser.error("--output and --shard are set per shard")
    if args.output.exists():
        raise FileExistsError(args.output)
    work = args.output.with_name(args.output.name + ".shards")
    work.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    began = time.monotonic()
    pin = []
    if args.pin and args.autotune_file:
        pin = ["--autotune", "pin", "--autotune-file", str(args.autotune_file)]
    elif args.pin:
        table = work / "autotune.json"
        with (work / "warmup.log").open("w") as log:
            code = subprocess.run([sys.executable, "-B", str(HERE / "eval_head_l20.py"), *rest, "--shard", "0/50",
                                   "--autotune", "record", "--autotune-file", str(table), "--output",
                                   str(work / "warmup")], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if code or not table.exists():
            print(json.dumps({"warmup_exit_code": code}), flush=True)
            return 1
        pin = ["--autotune", "pin", "--autotune-file", str(table)]
    procs = []
    for i in range(args.shards):
        log = (work / f"shard_{i}.log").open("w")
        cmd = [sys.executable, "-B", str(HERE / "eval_head_l20.py"), *rest, "--shard", f"{i}/{args.shards}", *pin,
               "--output", str(work / f"shard_{i}")]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    wall = time.monotonic() - began
    if any(codes):
        print(json.dumps({"shard_exit_codes": codes, "wall_seconds": wall}), flush=True)
        return 1
    dirs = [work / f"shard_{i}" for i in range(args.shards)]
    reports = [json.loads((d / "report.json").read_text()) for d in dirs]
    args.output.mkdir()
    for variant in reports[0]["variants"]:
        merge_files([d / f"eval_{variant}" for d in dirs], args.output / f"eval_{variant}")
    merged = merge_reports(reports, args.output)
    merged["wall_seconds"] = wall
    merged["autotune"] = [r.get("autotune") for r in reports]
    if args.compare:
        merged["compare"] = {variant: compare_dirs(args.output / f"eval_{variant}", args.compare / f"eval_{variant}")
                             for variant in merged["variants"]}
    (args.output / "report.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: merged[k] for k in ("shards", "wall_seconds", "shard_elapsed_seconds", "totals", "autotune")
                      + (("compare",) if args.compare else ())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
