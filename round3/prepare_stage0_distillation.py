"""Build a bilingual, source-traceable teacher-distillation pilot without policy labels."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "data/candidates"
BEAVER = ROOT / "data/raw/beavertails_round0_30k/train.jsonl.gz"
BEAVER_REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
PILOT_QUOTAS = {
    "nemotron_user": (256, 24),
    "nemotron_assistant": (256, 24),
    "cold_user": (256, 24),
    "beaver_user": (128, 16),
    "beaver_assistant": (128, 16),
}
SCALE_QUOTAS = {
    "nemotron_user": (5000, 500),
    "nemotron_assistant": (4000, 400),
    "cold_user": (3000, 300),
    "beaver_user": (4000, 400),
    "beaver_assistant": (4000, 400),
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(row: dict) -> str:
    return digest("stage0-v1:" + row["sample_id"])


def convert_candidate(row: dict, pool: str) -> dict:
    source_group = row["source"] + ":" + row["source_id"]
    messages = row["messages"]
    return {
        "sample_id": row["candidate_id"],
        "family": digest(source_group),
        "target_role": row["target_role"],
        "messages": messages,
        "source_pool": pool,
        "source": row["source"],
        "source_revision": row["source_revision"],
        "source_license": row["source_license"],
        "training_objective": "causal_prefix_teacher_distillation_only",
        "project_policy_label": None,
    }


def load_candidates(path: Path, pool: str) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [convert_candidate(json.loads(line), pool) for line in stream if line.strip()]


def beaver_rows() -> dict[str, list[dict]]:
    result = {"beaver_user": [], "beaver_assistant": []}
    with gzip.open(BEAVER, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            prompt, response = row.get("prompt") or "", row.get("response") or ""
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            if not prompt.strip() or not response.strip():
                continue
            if len(prompt) > 2000 or len(response) > 2000:
                continue
            family = digest("beavertails:" + prompt)
            base = {
                "family": family,
                "source": "PKU-Alignment/BeaverTails/round0/30k",
                "source_revision": BEAVER_REVISION,
                "source_license": "CC BY-NC 4.0",
                "training_objective": "causal_prefix_teacher_distillation_only",
                "project_policy_label": None,
            }
            for role, messages, pool in (
                ("user", [{"role": "user", "content": prompt}], "beaver_user"),
                ("assistant", [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": response}],
                 "beaver_assistant"),
            ):
                result[pool].append({**base,
                                     "sample_id": digest(f"beavertails:{index}:{role}:{prompt}:{response}"),
                                     "target_role": role, "messages": messages,
                                     "source_pool": pool})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "scale20k"), default="pilot")
    args = parser.parse_args()
    quotas = PILOT_QUOTAS if args.profile == "pilot" else SCALE_QUOTAS
    out = ROOT / "data" / ("stage0_distillation" if args.profile == "pilot"
                            else "stage0_distillation_20k")
    sources = {
        "nemotron_user": load_candidates(CANDIDATES / "nemotron_user_candidates.jsonl", "nemotron_user"),
        "nemotron_assistant": load_candidates(CANDIDATES / "nemotron_assistant_candidates.jsonl", "nemotron_assistant"),
        "cold_user": load_candidates(CANDIDATES / "cold_candidates.jsonl", "cold_user"),
    }
    sources.update(beaver_rows())
    selected = {"train": [], "dev": []}
    seen_content = set()
    seen_family_split = {}
    counts = Counter()
    for pool, (train_quota, dev_quota) in quotas.items():
        rows = sorted(sources[pool], key=rank)
        local = Counter()
        for row in rows:
            split = "dev" if int(row["family"][:8], 16) % 10 == 0 else "train"
            limit = dev_quota if split == "dev" else train_quota
            if local[split] >= limit:
                continue
            family = row["family"]
            if family in seen_family_split and seen_family_split[family] != split:
                raise AssertionError("Family crosses train/dev")
            content = digest(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True))
            if content in seen_content:
                counts[pool + "_duplicate_skipped"] += 1
                continue
            seen_content.add(content)
            seen_family_split[family] = split
            selected[split].append(row)
            local[split] += 1
            if local["train"] == train_quota and local["dev"] == dev_quota:
                break
        if local["train"] != train_quota or local["dev"] != dev_quota:
            raise ValueError(f"Not enough source records in {pool}: {dict(local)}")
        counts[pool + "_train"] = local["train"]
        counts[pool + "_dev"] = local["dev"]
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        selected[split].sort(key=rank)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_version": ("stage0-open-corpus-distillation-v1"
                            if args.profile == "pilot" else "stage0-open-corpus-distillation-v2-20k"),
        "objective": "match a frozen pretrained stream classifier on the same visible prefixes",
        "uses_source_safety_labels": False,
        "project_safety_alignment": False,
        "train_rows": len(selected["train"]),
        "dev_rows": len(selected["dev"]),
        "pool_counts": dict(counts),
        "source_sha256": {
            "nemotron_user_candidates": file_hash(CANDIDATES / "nemotron_user_candidates.jsonl"),
            "nemotron_assistant_candidates": file_hash(CANDIDATES / "nemotron_assistant_candidates.jsonl"),
            "cold_candidates": file_hash(CANDIDATES / "cold_candidates.jsonl"),
            "beavertails_train_gz": file_hash(BEAVER),
        },
        "output_sha256": {split: file_hash(out / f"{split}.jsonl") for split in ("train", "dev")},
        "licenses": ["CC BY 4.0", "Apache-2.0", "CC BY-NC 4.0"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"train_rows": manifest["train_rows"],
                      "dev_rows": manifest["dev_rows"], "pools": dict(counts)}))


if __name__ == "__main__":
    main()
