"""Freeze a balanced source-label diagnostic split; never train on it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/nemotron_safety_guard_v3_zh/valid.jsonl"
OUT = ROOT / "data/source_valid_probe.jsonl"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Refusing to replace frozen diagnostic split")
    audit = json.loads((ROOT / "data/audit_report.json").read_text())
    expected = audit["nemotron_zh"]["splits"]["valid"]["sha256"]
    if sha256(SOURCE) != expected:
        raise RuntimeError("Nemotron valid source checksum mismatch")
    pools = {(role, label): [] for role in ("user", "assistant")
             for label in ("safe", "unsafe")}
    for line in SOURCE.open():
        row = json.loads(line)
        if (row.get("prompt_label_source") == "human"
                and row.get("prompt_label") in ("safe", "unsafe")
                and isinstance(row.get("prompt"), str) and row["prompt"].strip()):
            pools[("user", row["prompt_label"])].append({
                "source_id": row["id"], "target_role": "user",
                "messages": [{"role": "user", "content": row["prompt"]}],
                "source_label": row["prompt_label"], "source_label_origin": "human",
            })
        if (row.get("response_label_source") == "llm_jury"
                and row.get("response_label") in ("safe", "unsafe")
                and isinstance(row.get("response"), str)
                and row["response"].strip() not in ("", "None")
                and isinstance(row.get("prompt"), str) and row["prompt"].strip()):
            pools[("assistant", row["response_label"])].append({
                "source_id": row["id"], "target_role": "assistant",
                "messages": [{"role": "user", "content": row["prompt"]},
                             {"role": "assistant", "content": row["response"]}],
                "source_label": row["response_label"],
                "source_label_origin": "llm_jury",
            })
    rng = random.Random(20260923)
    picked = []
    used_ids = set()
    for role, label, count in (("user", "safe", 200), ("user", "unsafe", 200),
                               ("assistant", "safe", 100), ("assistant", "unsafe", 100)):
        candidates = pools[(role, label)]
        rng.shuffle(candidates)
        rows = []
        for row in candidates:
            if row["source_id"] in used_ids:
                continue
            rows.append(row)
            used_ids.add(row["source_id"])
            if len(rows) == count:
                break
        if len(rows) != count:
            raise RuntimeError(f"Insufficient {role}/{label} source-valid records")
        picked.extend(rows)
    rng.shuffle(picked)
    with OUT.open("w") as stream:
        for row in picked:
            row.update(source="nvidia/Nemotron-Safety-Guard-Dataset-v3/zh",
                       source_split="valid", project_policy_label=None,
                       training_use=False)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"source_sha256": expected, "output_sha256": sha256(OUT),
                "rows": len(picked), "role_label_counts": {
                    f"{role}/{label}": sum(r["target_role"] == role and r["source_label"] == label
                                           for r in picked)
                    for role in ("user", "assistant") for label in ("safe", "unsafe")},
                "label_scope": "original source labels only; not project policy gold",
                "training_use": False}
    (ROOT / "data/source_valid_probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
