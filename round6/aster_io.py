"""Shared Aster helpers for local extractors (Mac): the CLI envelope and multi-Run archive download.

A job over 2,000 Tasks has to be split into several Runs (platform limit), and any Run over 100 Tasks needs a
frozen per-sample attempt listing (response_v14/freeze_attempts.py). collect_archives() takes one or more Runs,
optionally with their frozen listings in the same order, downloads every completed archive once and checks
its SHA-256 against the platform receipt.
"""
from __future__ import annotations

import hashlib
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


def collect_archives(runs, attempts_jsons, out_dir, call=cli):
    """Return (archive paths, run infos). attempts_jsons: [] or one frozen listing per run, same order."""
    if attempts_jsons and len(attempts_jsons) != len(runs):
        raise ValueError("give one frozen attempts listing per run, in the same order, or none")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archives, infos = [], []
    for position, run_no in enumerate(runs):
        run = call(["runs", "get", run_no])
        infos.append({"run_no": run.get("run_no"), "status": run.get("status")})
        if attempts_jsons:
            listing = json.loads(Path(attempts_jsons[position]).read_text(encoding="utf-8"))
            if listing.get("run_id") != run.get("id"):
                raise RuntimeError(f"attempts listing {position} belongs to {listing.get('run_id')}, not {run.get('id')}")
        else:
            listing = call(["runs", "attempts", run_no])
        if listing.get("truncated"):
            raise RuntimeError("attempt listing truncated; freeze it first (response_v14/freeze_attempts.py)")
        for attempt in listing["items"]:
            if attempt.get("state") != "completed" or not attempt.get("archive"):
                continue
            path = out_dir / (attempt["attempt_id"] + ".tar.gz")
            if not path.exists():
                receipt = call(["runs", "archive", run_no, attempt["attempt_id"], "--out", str(path)])
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise RuntimeError("archive SHA mismatch " + attempt["attempt_id"])
            archives.append(path)
    return archives, infos
