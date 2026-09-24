"""Build political-screen Tasks (Mac, CPU): 200 words per Task, only the words are sent.

Pass 1 screens every distinct word of the T012 word index (word_screen_v1/full/words.jsonl, Run A + Run B seeds).
Pass 2 (--only confirm_words.txt from extract.py) re-screens the words pass 1 put in event / figure / org / unsure,
in a different batch order with different neighbours; extract.py merges the passes (most alerting wins).
The word index and the output hold the words themselves: keep them local (gitignored), never commit them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT  # noqa: E402

BATCH_WORDS = 200
PART_TASKS = 2000          # Aster: at most 2,000 Tasks per Run


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def batches(words, tag):
    ordered = sorted(words, key=lambda w: number(f"political-screen:{tag}:{w}"))
    return [(f"political-{tag}-{i // BATCH_WORDS:05d}", ordered[i:i + BATCH_WORDS]) for i in range(0, len(ordered), BATCH_WORDS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=Path, default=ROOT.parent / "word_screen_v1/full/words.jsonl")
    parser.add_argument("--only", type=Path, help="screen only these words (one per line), e.g. confirm_words.txt")
    parser.add_argument("--tag", default="p1", help="pass tag; each pass gets its own batch order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        parser.error("--tag must be letters and digits only (extract.py reads it from the batch key)")
    if args.output.exists():
        raise FileExistsError(args.output)
    words = sorted({json.loads(line)["word"] for line in args.words.open(encoding="utf-8") if line.strip()})
    if args.only:
        wanted = {w for w in args.only.read_text(encoding="utf-8").splitlines() if w}
        words = [w for w in words if w in wanted]
    paths = []
    for key, group in batches(words, args.tag):
        task = args.output / "tasks" / key
        (task / "tests").mkdir(parents=True)
        batch = {"batch_key": key, "prompt_version": PROMPT_VERSION, "mode": "screen", "words": [{"word": w} for w in group]}
        (task / "instruction.md").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
        (task / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nname = "{key}"\ncategory = "guard-political-screen"\n')
        test = task / "tests/test.sh"
        test.write_text("#!/bin/sh\nexit 1\n")
        test.chmod(0o755)
        paths.append(task)
    parts = [paths[i:i + PART_TASKS] for i in range(0, len(paths), PART_TASKS)]
    names = ["screen.tar.gz"] if len(parts) == 1 else [f"screen_part{i}.tar.gz" for i in range(len(parts))]
    for name, part in zip(names, parts):
        with tarfile.open(args.output / name, "w:gz") as archive:
            for task in part:
                archive.add(task, arcname=task.name)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"prompt_version": PROMPT_VERSION, "tag": args.tag, "words": len(words), "tasks": len(paths),
                "archives": {name: len(part) for name, part in zip(names, parts)}, "batch_words": BATCH_WORDS,
                "sha256": {"pipeline": sha(ROOT / "flow/pipeline.py"), "flow": sha(ROOT / "flow/flow.py"),
                           "system_prompt": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("tag", "words", "tasks", "archives")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
