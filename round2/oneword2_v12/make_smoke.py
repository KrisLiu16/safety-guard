"""Two small source-faithful Tasks covering both v12 label orientations."""
from __future__ import annotations

import json
from pathlib import Path
import tarfile

from flow.pipeline import PROMPT_VERSION


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "full-source-index"
GROUPS = ("permissive/贪腐词库.txt", "permissive/民生词库.txt")


def main() -> None:
    output = ROOT / "smoke"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    output.mkdir()
    task_paths = []
    for ordinal, group_name in enumerate(GROUPS):
        category = next(x for x in manifest["categories"] if x["source_group"] == group_name)
        seeds = [json.loads(line) for line in (SOURCE / category["file"]).read_text().splitlines()[:4]]
        terms = [[seed["task_key"], seed["word"], seed["source_licenses"]] for seed in seeds]
        key = f"smoke-oneword2-v12-{ordinal:02d}"
        task = output / "tasks" / key
        (task / "tests").mkdir(parents=True)
        instruction = {
            "group_key": key, "source_group": group_name,
            "category_total_terms": len(terms), "start_offset": 0,
            "terms": terms, "calls_per_term": 1,
            "examples_per_term": 2, "prompt_version": PROMPT_VERSION,
        }
        (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False))
        (task / "task.toml").write_text(
            f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-data-generation"\n'
        )
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        task_paths.append(task)
    with tarfile.open(output / "smoke.tar.gz", "w:gz") as archive:
        for task in task_paths:
            archive.add(task, arcname=task.name)
    print(json.dumps({"tasks": len(task_paths), "terms": 8,
                      "archive": str(output / "smoke.tar.gz")}))


if __name__ == "__main__":
    main()
