#!/usr/bin/env python3
"""Resume-safe Aster draft upload; stops on the first server error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def cli(args: list[str]) -> dict:
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"{error.get('code')}: {error.get('message')} ({error.get('hint')})")
    return envelope["data"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version_id")
    ap.add_argument("chunks", type=Path)
    args = ap.parse_args()
    archives = sorted(args.chunks.glob("*.tar.gz"))
    if not archives:
        ap.error("no .tar.gz chunks")
    version = cli(["datasets", "version", args.version_id])
    if version["version"]["status"] != "draft":
        ap.error("dataset version is not a draft")
    known = {task["task_key"] for task in version["tasks"]}
    for i, archive in enumerate(archives, 1):
        with tarfile.open(archive, "r:gz") as tar:
            names = {member.name.split("/", 1)[0] for member in tar.getmembers() if member.name}
        if names <= known:
            continue
        result = cli(["datasets", "add", args.version_id, str(archive.resolve())])
        invalid = [task["task_key"] for task in result.get("tasks", []) if task.get("harbor_validation") == "invalid"]
        if invalid:
            raise RuntimeError(f"Invalid Task(s) in {archive.name}: {invalid[:4]}")
        known.update(result.get("added", []))
        known.update(result.get("unchanged", []))
        known.update(result.get("replaced", []))
        if i % 10 == 0 or i == len(archives):
            print(json.dumps({"archive": i, "of": len(archives), "task_count": result["version"]["task_count"]}), flush=True)
    print(json.dumps({"status": "uploaded", "task_count": len(known)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
