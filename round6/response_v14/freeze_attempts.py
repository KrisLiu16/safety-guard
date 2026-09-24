#!/usr/bin/env python3
"""Freeze one successful archived attempt per sample for a finished v14 run.

The run-wide `aster runs attempts` listing stops at 100 items, so samples are paged first and each
sample's attempts are read on their own (same method as round2/oneword2_v12/freeze_full_attempts.py).
The output attempts.json has the same shape as the run-wide listing with truncated=false and is
consumed by extract_archives.py --attempts-json. No model calls are made.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def cli(args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"aster {' '.join(args[:2])}: {error.get('code')}: {error.get('message')}")
    return envelope["data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--out", type=Path, required=True, help="output attempts.json")
    parser.add_argument("--expected-samples", type=int, required=True, help="physical Tasks in the run (Run A: 352)")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    run = cli(["runs", "get", args.run])
    samples, page = [], 1
    while True:
        data = cli(["runs", "samples", args.run, "--page", str(page), "--page-size", "100"])
        samples += data["items"]
        if len(data["items"]) < 100:
            break
        page += 1
    ids = {s["id"] for s in samples}
    if len(samples) != args.expected_samples or len(ids) != args.expected_samples:
        raise RuntimeError(f"expected {args.expected_samples} unique samples, got {len(samples)} ({len(ids)} unique)")

    def pick(sample):
        items = cli(["runs", "attempts", args.run, "--sample-id", sample["id"]])["items"]
        good = [a for a in items if a.get("state") == "completed" and a.get("result_status") == "succeeded" and a.get("archive")]
        if len(good) != 1:
            raise RuntimeError(f"sample {sample.get('sample_no', sample['id'])}: {len(good)} successful archived attempts")
        return good[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        attempts = list(pool.map(pick, samples))
    if len({a["sample_id"] for a in attempts}) != len(attempts):
        raise RuntimeError("an attempt was selected for more than one sample")
    frozen = {"run_id": run.get("id"), "run_no": run.get("run_no"), "created_at": datetime.now(timezone.utc).isoformat(),
              "selection": "one successful archived attempt per sample of the finished run; no model calls",
              "total_run_samples": len(samples), "items": attempts, "truncated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_no": frozen["run_no"], "samples": len(samples), "attempts": len(attempts)}))


if __name__ == "__main__":
    main()
