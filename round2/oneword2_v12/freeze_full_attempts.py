#!/usr/bin/env python3
"""Freeze one successful archived attempt per sample for the finished aster-dev-272 run.

The run-wide attempts endpoint caps at 100 items, so samples are paged first and each
sample's attempts are read individually. Output matches completed_snapshot_v2/attempts.json
and is consumed by extract_archives.py --attempts-json. Already downloaded archives from an
earlier snapshot are hard-linked (never copied or modified) so they are not fetched again.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

RUN = "aster-dev-272"
RUN_ID = "run_01M36NWG2G4NF17NYJAX95HPN5"
TOTAL = 1517


def cli(args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"Aster {' '.join(args[:2])} {error.get('code')}: {error.get('message')}")
    return envelope["data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-archives", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples, page = [], 1
    while True:
        data = cli(["runs", "samples", RUN, "--page", str(page), "--page-size", "100"])
        (args.out / f"samples_page_{page:03d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        samples += data["items"]
        if len(data["items"]) < 100:
            break
        page += 1
    if len(samples) != TOTAL or len({s["id"] for s in samples}) != TOTAL:
        raise RuntimeError(f"expected {TOTAL} unique samples, got {len(samples)}")
    (args.out / "samples_snapshot.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")

    def pick(sample):
        items = cli(["runs", "attempts", RUN, "--sample-id", sample["id"]])["items"]
        good = [a for a in items if a.get("state") == "completed" and a.get("result_status") == "succeeded" and a.get("archive")]
        if len(good) != 1:
            raise RuntimeError(f"{sample['sample_no']}: {len(good)} successful archived attempts")
        return good[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        attempts = list(pool.map(pick, samples))
    frozen = {"run_id": RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(),
              "selection": "all successful archived samples of the finished run; one attempt per sample; no model calls submitted",
              "total_run_samples": TOTAL, "successful_samples_seen": len(attempts), "awaiting_archive": 0,
              "items": attempts, "truncated": False}
    (args.out / "attempts.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    archives = args.out / "archives"
    archives.mkdir(exist_ok=True)
    linked = 0
    wanted = {a["attempt_id"] for a in attempts}
    for folder in args.reuse_archives:
        for path in folder.glob("*.tar.gz"):
            aid = path.name[:-len(".tar.gz")]
            digest = path.with_name(aid + ".sha256")
            if aid in wanted and digest.exists() and not (archives / path.name).exists():
                os.link(path, archives / path.name)
                os.link(digest, archives / digest.name)
                linked += 1
    print(json.dumps({"samples": len(samples), "attempts": len(attempts), "reused_archives": linked}))


if __name__ == "__main__":
    main()
