"""Shared helpers for batch builders (Mac, CPU): word-screen exclusions and Task packing within platform limits.

Aster accepts at most 2,000 Tasks per Run, and instructions around 208 KB were rejected (196 KB passed).
write_generation_tasks() packs whole seeds into Tasks of at most words_per_task words and max_task_bytes, and
writes one archive per Run-sized part: <stem>.tar.gz when everything fits in one, else <stem>_partN.tar.gz.
"""
from __future__ import annotations

import json
from pathlib import Path
import tarfile

PART_TASKS = 2000
MAX_TASK_BYTES = 180_000


def screen_exclusions(path):
    """Words to keep out of safe-labelled generation: every word whose final screen verdict is not "no"."""
    excluded, screened = set(), set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            screened.add(row["word"])
            if row["verdict"] != "no":
                excluded.add(row["word"])
    return excluded, screened


def write_generation_tasks(seeds, output, name_prefix, instruction_base, words_per_task=1,
                           max_task_bytes=MAX_TASK_BYTES, category="guard-data-generation", stem="pilot"):
    """seeds: planned seed dicts in order. instruction_base: fields shared by every Task (prompt_version etc.).
    Returns {"tasks": n, "archives": {name: task count}, "task_of": {task_key: task name}}."""
    output = Path(output)
    tasks, current = [], []

    def size(terms):
        return len(json.dumps({**instruction_base, "group_key": "x", "terms": terms}, ensure_ascii=False).encode())

    for seed in seeds:
        if size([seed]) > max_task_bytes:
            raise ValueError(f"seed {seed['task_key']} alone exceeds {max_task_bytes} bytes")
        if current and (len(current) >= words_per_task or size(current + [seed]) > max_task_bytes):
            tasks.append(current)
            current = []
        current.append(seed)
    if current:
        tasks.append(current)
    paths, task_of = [], {}
    for ordinal, terms in enumerate(tasks):
        name = f"{name_prefix}-{ordinal:03d}"
        task = output / "tasks" / name
        (task / "tests").mkdir(parents=True)
        instruction = {**instruction_base, "group_key": name, "source_group": terms[0].get("source_group", ""),
                       "terms": terms}
        (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{name}"\ncategory = "{category}"\n')
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        paths.append(task)
        task_of.update({t["task_key"]: name for t in terms})
    parts = [paths[i:i + PART_TASKS] for i in range(0, len(paths), PART_TASKS)]
    names = [f"{stem}.tar.gz"] if len(parts) <= 1 else [f"{stem}_part{i}.tar.gz" for i in range(len(parts))]
    for archive_name, part in zip(names, parts):
        with tarfile.open(output / archive_name, "w:gz") as archive:
            for task in part:
                archive.add(task, arcname=task.name)
    return {"tasks": len(paths), "archives": {n: len(p) for n, p in zip(names, parts)}, "task_of": task_of}
