"""DeepSeek v14 generation pilot: 6 Run B words from each Run B source group, one Task per word.

The seeds (prompts, rotation) are taken unchanged from batch_rest/seeds.jsonl, so if Run B later
runs on DeepSeek these words simply get regenerated under the identical contract.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION  # noqa: E402

RUN_B = ROOT.parent / "response_v14/batch_rest/seeds.jsonl"
PER_GROUP = 6
SALT = "v14ds-pilot-2026-09-24"


def main():
    output = ROOT / "pilot"
    if output.exists():
        raise FileExistsError(output)
    by_group = collections.defaultdict(list)
    for line in RUN_B.open(encoding="utf-8"):
        seed = json.loads(line)
        by_group[seed["source_group"]].append(seed)
    chosen = []
    for group in sorted(by_group):
        pool = sorted(by_group[group], key=lambda s: hashlib.sha256(f"{SALT}:{s['task_key']}".encode()).hexdigest())
        lanes = {lang: [s for s in pool if s["prompts"]["unsafe"]["language"] == lang] for lang in ("zh", "en")}
        picks = []
        while len(picks) < min(PER_GROUP, len(pool)):
            for lang in ("zh", "en"):
                if lanes[lang] and len(picks) < PER_GROUP:
                    picks.append(lanes[lang].pop(0))
        chosen += picks
    tasks = output / "tasks"
    paths = []
    for ordinal, seed in enumerate(chosen):
        key = f"pilot-v14ds-{ordinal:03d}"
        task = tasks / key
        (task / "tests").mkdir(parents=True)
        term = {k: v for k, v in seed.items() if k not in ("split", "family", "group_key")}
        instruction = {"group_key": key, "source_group": seed["source_group"], "terms": [term],
                       "calls_per_term": 1, "responses_per_term": 4, "prompt_version": PROMPT_VERSION}
        (task / "instruction.md").write_text(json.dumps(instruction, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-data-generation"\n')
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        paths.append(task)
    with tarfile.open(output / "pilot.tar.gz", "w:gz") as archive:
        for task in paths:
            archive.add(task, arcname=task.name)
    with (output / "seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in chosen:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    manifest = {"prompt_version": PROMPT_VERSION, "words": len(chosen), "tasks": len(paths),
                "per_group": dict(collections.Counter(s["source_group"] for s in chosen)),
                "unsafe_prompt_language": dict(collections.Counter(s["prompts"]["unsafe"]["language"] for s in chosen)),
                "pipeline_sha256": hashlib.sha256((ROOT / "flow/pipeline.py").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
