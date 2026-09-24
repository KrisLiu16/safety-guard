"""Build one ten-word source-faithful smoke Task for the frozen v9 contract."""
from __future__ import annotations

import json
from pathlib import Path
import tarfile

from flow.pipeline import PROMPT_VERSION


ROOT = Path(__file__).resolve().parent
GROUP = "permissive/贪腐词库.txt"
KEY = "smoke-batch10-v9-corruption-0001"


def main() -> None:
    output = ROOT / "smoke"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((ROOT / "full-shards/manifest.json").read_text())
    match = next(item for item in manifest["shards"] if item["source_group"] == GROUP)
    source = ROOT / "full-shards/tasks" / match["task_key"] / "instruction.md"
    instruction = json.loads(source.read_text())
    instruction.update({
        "group_key": KEY,
        "category_total_terms": 10,
        "start_offset": 0,
        "terms": instruction["terms"][:10],
        "batch_size": 10,
        "examples_per_term": 2,
        "prompt_version": PROMPT_VERSION,
    })
    task = output / "tasks" / KEY
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text(
        json.dumps(instruction, ensure_ascii=False, separators=(",", ":"))
    )
    (task / "task.toml").write_text(
        f'version = "1.0"\n\n[metadata]\nname = "{KEY}"\ncategory = "guard-data-generation"\n'
    )
    test = task / "tests/test.sh"
    test.write_text("#!/bin/sh\nexit 1\n")
    test.chmod(0o755)
    with tarfile.open(output / "smoke.tar.gz", "w:gz") as archive:
        archive.add(task, arcname=KEY)
    print(json.dumps({"task_key": KEY, "terms": len(instruction["terms"]),
                      "archive": str(output / "smoke.tar.gz")}))


if __name__ == "__main__":
    main()
