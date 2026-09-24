"""Pack new v9 Harbor Tasks into small, resumable upload archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=ROOT / "full-shards")
    parser.add_argument("--tasks-per-archive", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.tasks_per_archive <= 50:
        parser.error("tasks-per-archive must be 1..50")
    manifest = json.loads((args.shards / "manifest.json").read_text())
    tasks = [args.shards / "tasks" / item["task_key"] for item in manifest["shards"]]
    archives = args.shards / "chunks"
    if archives.exists():
        parser.error("chunks directory already exists")
    archives.mkdir()
    for start in range(0, len(tasks), args.tasks_per_archive):
        path = archives / f"{start // args.tasks_per_archive:04d}.tar.gz"
        with tarfile.open(path, "w:gz") as output:
            for task in tasks[start:start + args.tasks_per_archive]:
                output.add(task, arcname=task.name)
    print(json.dumps({"archives": len(list(archives.glob("*.tar.gz"))),
                      "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
