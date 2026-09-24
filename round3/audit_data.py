"""Inventory local safety corpora without emitting or relabeling their text."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NEMOTRON_REVISION = "a3f7ecb3433d1933701a83f18de16c36934a7f51"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def nemotron_split(path: Path) -> tuple[dict, set[str], set[str]]:
    counters = {key: Counter() for key in (
        "language", "prompt_label", "response_label", "prompt_label_source",
        "response_label_source", "tag",
    )}
    ids: set[str] = set()
    texts: set[str] = set()
    duplicate_ids = duplicate_texts = empty_prompts = empty_responses = rows = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path.name}:{number}: expected object")
            rows += 1
            for key, counter in counters.items():
                value = item.get(key)
                counter["<null>" if value is None else str(value)] += 1
            identity = str(item.get("id", ""))
            if identity in ids:
                duplicate_ids += 1
            ids.add(identity)
            prompt = item.get("prompt") or ""
            response = item.get("response") or ""
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(f"{path.name}:{number}: prompt/response must be strings or null")
            empty_prompts += not bool(prompt.strip())
            empty_responses += not bool(response.strip())
            content_hash = fingerprint(prompt + "\n<response>\n" + response)
            if content_hash in texts:
                duplicate_texts += 1
            texts.add(content_hash)
    return ({
        "path": str(path.resolve()), "sha256": sha256_file(path), "rows": rows,
        "label_counts": {k: dict(v) for k, v in counters.items()},
        "empty_prompt_rows": empty_prompts, "empty_response_rows": empty_responses,
        "duplicate_id_rows": duplicate_ids, "duplicate_text_rows": duplicate_texts,
    }, ids, texts)


def cold_split(path: Path) -> tuple[dict, set[str]]:
    labels, topics = Counter(), Counter()
    texts: set[str] = set()
    duplicates = empty = rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for number, item in enumerate(csv.DictReader(stream), 2):
            rows += 1
            value = item.get("TEXT")
            if not isinstance(value, str):
                raise ValueError(f"{path.name}:{number}: missing TEXT")
            empty += not bool(value.strip())
            key = fingerprint(value)
            if key in texts:
                duplicates += 1
            texts.add(key)
            labels[str(item.get("label"))] += 1
            topics[str(item.get("topic"))] += 1
    return ({
        "path": str(path.resolve()), "sha256": sha256_file(path), "rows": rows,
        "labels": dict(labels), "topics": dict(topics),
        "empty_text_rows": empty, "duplicate_text_rows": duplicates,
    }, texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron-dir", type=Path,
                        default=ROOT / "data/raw/nemotron_safety_guard_v3_zh")
    parser.add_argument("--cold-dir", type=Path,
                        default=ROOT.parent / "lexicon/corpora/COLDataset/COLDataset")
    parser.add_argument("--out", type=Path, default=ROOT / "data/audit_report.json")
    args = parser.parse_args()
    nemotron, n_ids, n_texts = {}, {}, {}
    for split in ("train", "valid"):
        nemotron[split], n_ids[split], n_texts[split] = nemotron_split(
            args.nemotron_dir / f"{split}.jsonl"
        )
    cold, c_texts = {}, {}
    for split in ("train", "dev", "test"):
        cold[split], c_texts[split] = cold_split(args.cold_dir / f"{split}.csv")
    report = {
        "audit_version": "guard-corpus-audit-v1",
        "text_exported": False,
        "nemotron_zh": {
            "source": "https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3",
            "revision": NEMOTRON_REVISION,
            "license_card": "CC BY 4.0",
            "splits": nemotron,
            "train_valid_shared_ids": len(n_ids["train"] & n_ids["valid"]),
            "train_valid_shared_texts": len(n_texts["train"] & n_texts["valid"]),
        },
        "cold": {
            "source": "https://github.com/thu-coai/COLDataset",
            "license_repository": "Apache-2.0",
            "splits": cold,
            "train_dev_shared_texts": len(c_texts["train"] & c_texts["dev"]),
            "train_test_shared_texts": len(c_texts["train"] & c_texts["test"]),
            "dev_test_shared_texts": len(c_texts["dev"] & c_texts["test"]),
        },
        "training_gate": {
            "ready": False,
            "missing": [
                "versioned project safety policy and source-label mapping",
                "independent adjudicated Chinese gold anchors and prefix evidence locations",
                "train/eval source-family and benchmark contamination exclusions",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.out.resolve()),
        "nemotron_train_rows": nemotron["train"]["rows"],
        "nemotron_valid_rows": nemotron["valid"]["rows"],
        "cold_train_rows": cold["train"]["rows"],
        "training_gate_ready": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

