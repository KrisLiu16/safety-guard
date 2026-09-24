"""Prepare source-faithful candidates; never assign project safety labels."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from audit_data import ROOT, NEMOTRON_REVISION, fingerprint, sha256_file


RAW = ROOT / "data/raw/nemotron_safety_guard_v3_zh"
COLD = ROOT.parent / "lexicon/corpora/COLDataset/COLDataset"
OUT = ROOT / "data/candidates"


def nemotron_eval_fingerprints() -> set[str]:
    excluded = set()
    with (RAW / "valid.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("prompt", "response"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    excluded.add(fingerprint(value))
    return excluded


def cold_eval_fingerprints() -> set[str]:
    excluded = set()
    for split in ("dev", "test"):
        with (COLD / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("TEXT")
                if value and value.strip():
                    excluded.add(fingerprint(value))
    return excluded


def write_candidate(stream, *, source, source_revision, source_id, license_id,
                    role, messages, label, label_source, category_raw=None):
    conversation_hash = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "candidate_version": "guard-candidate-v1",
        "candidate_id": hashlib.sha256(
            f"{source}:{source_id}:{role}:{conversation_hash}".encode()
        ).hexdigest(),
        "source": source,
        "source_revision": source_revision,
        "source_id": str(source_id),
        "source_split": "train",
        "source_license": license_id,
        "target_role": role,
        "messages": messages,
        "source_label": label,
        "source_label_origin": label_source,
        "source_category": category_raw,
        "project_policy_label": None,
        "eligible_for_training": False,
    }
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare_nemotron(out_dir: Path) -> dict:
    excluded = nemotron_eval_fingerprints()
    seen = set()
    counts = Counter()
    source = "nvidia/Nemotron-Safety-Guard-Dataset-v3/zh"
    with (RAW / "train.jsonl").open(encoding="utf-8") as input_stream, \
         (out_dir / "nemotron_user_candidates.jsonl").open("w", encoding="utf-8") as users, \
         (out_dir / "nemotron_assistant_candidates.jsonl").open("w", encoding="utf-8") as assistants:
        for line in input_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row["id"]
            prompt = row.get("prompt") or ""
            response = row.get("response") or ""
            if prompt.strip() and row.get("prompt_label") in ("safe", "unsafe"):
                key = ("user", fingerprint(prompt))
                if key[1] in excluded:
                    counts["user_eval_overlap_skipped"] += 1
                elif key in seen:
                    counts["user_duplicate_skipped"] += 1
                else:
                    seen.add(key)
                    write_candidate(users, source=source, source_revision=NEMOTRON_REVISION,
                                    source_id=source_id, license_id="CC BY 4.0", role="user",
                                    messages=[{"role": "user", "content": prompt}],
                                    label=row["prompt_label"],
                                    label_source=row.get("prompt_label_source"),
                                    category_raw=row.get("violated_categories"))
                    counts["user_written"] += 1
            if response.strip() and row.get("response_label") in ("safe", "unsafe"):
                key = ("assistant", fingerprint(prompt + "\n<response>\n" + response))
                if fingerprint(prompt) in excluded or fingerprint(response) in excluded:
                    counts["assistant_eval_overlap_skipped"] += 1
                elif key in seen:
                    counts["assistant_duplicate_skipped"] += 1
                else:
                    seen.add(key)
                    messages = ([{"role": "user", "content": prompt}] if prompt.strip() else []) + [
                        {"role": "assistant", "content": response}
                    ]
                    write_candidate(assistants, source=source, source_revision=NEMOTRON_REVISION,
                                    source_id=source_id, license_id="CC BY 4.0", role="assistant",
                                    messages=messages, label=row["response_label"],
                                    label_source=row.get("response_label_source"),
                                    category_raw=row.get("violated_categories"))
                    counts["assistant_written"] += 1
    return dict(counts)


def prepare_cold(out_dir: Path) -> dict:
    excluded = cold_eval_fingerprints()
    revision = "sha256:" + sha256_file(COLD / "train.csv")
    seen = set()
    counts = Counter()
    with (COLD / "train.csv").open(encoding="utf-8-sig", newline="") as input_stream, \
         (out_dir / "cold_candidates.jsonl").open("w", encoding="utf-8") as output:
        for row in csv.DictReader(input_stream):
            text = row.get("TEXT") or ""
            if not text.strip():
                counts["empty_skipped"] += 1
                continue
            if row.get("label") not in ("0", "1"):
                counts["unknown_label_skipped"] += 1
                continue
            key = fingerprint(text)
            if key in excluded:
                counts["eval_overlap_skipped"] += 1
            elif key in seen:
                counts["duplicate_skipped"] += 1
            else:
                seen.add(key)
                write_candidate(output, source="thu-coai/COLDataset",
                                source_revision=revision,
                                source_id=row.get("") or row.get("Unnamed: 0") or key,
                                license_id="Apache-2.0", role="user",
                                messages=[{"role": "user", "content": text}],
                                label=row["label"], label_source="dataset_label",
                                category_raw=row.get("topic"))
                counts["written"] += 1
    return dict(counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate_version": "guard-candidate-v1",
        "training_ready": False,
        "note": "Original labels retained; project policy labels deliberately null.",
        "nemotron": prepare_nemotron(OUT),
        "cold": prepare_cold(OUT),
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
