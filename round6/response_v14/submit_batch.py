#!/usr/bin/env python3
"""Publish one uploaded v14 batch draft, preflight, submit one Run and write its lock file.

Checks before submit: the draft holds exactly the local Task keys, the local flow files
still match the pilot-frozen SHAs in the batch manifest. After submit: the platform flow
snapshot is downloaded and compared byte-for-byte with the local flow directory.
Concurrency is passed explicitly (user decision); max attempts and repetitions are 1.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
MODEL_ID = "mdl_01M2W0QMTH0KFNJQFTBYY2QTVG"
PLACEMENT = "sg-public"


def cli(*args):
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"aster {' '.join(args[:2])}: {error.get('code')}: {error.get('message')} ({error.get('hint')})")
    return envelope["data"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draft_task_keys(version_id):
    keys, page = set(), 1
    while True:
        data = cli("datasets", "tasks", version_id, "--page", str(page), "--page-size", "200")
        keys.update(t["task_key"] for t in data["items"])
        if len(data["items"]) < 200:
            return keys
        page += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--idempotency-key", required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    ids = json.loads((batch / "dataset_ids.json").read_text())
    flow_dir = ROOT / "flow"
    for cache in flow_dir.glob("__pycache__"):
        raise RuntimeError("remove flow/__pycache__ before submitting")
    if sha(flow_dir / "pipeline.py") != manifest["sha256"]["pipeline"] or sha(flow_dir / "flow.py") != manifest["sha256"]["flow"]:
        raise RuntimeError("local flow differs from the batch manifest")
    expected = {t["task_key"] for t in manifest["tasks_index"]}
    version = cli("datasets", "version", ids["version_id"])["version"]
    if version["status"] == "draft":
        actual = draft_task_keys(ids["version_id"])
        if actual != expected:
            raise RuntimeError(f"draft Task set differs: missing={len(expected - actual)} extra={len(actual - expected)}")
        published = cli("datasets", "publish", ids["version_id"])
        (batch / "dataset_publish.json").write_text(json.dumps(published, ensure_ascii=False, indent=2) + "\n")
        version = cli("datasets", "version", ids["version_id"])["version"]
    if version["status"] != "published" or version["task_count"] != len(expected):
        raise RuntimeError(f"version not publishable: {version['status']} {version['task_count']}")
    common = ["--name", args.name, "--model-profile-id", MODEL_ID, "--placement-id", PLACEMENT,
              "--tasks", ids["version_id"], "--repetition-count", "1", "--concurrency", str(args.concurrency),
              "--max-attempts", "1", "--flow", str(flow_dir), "--tito-enabled", "false"]
    plan = cli("runs", "plan", *common)
    (batch / "run_plan.json").write_text(json.dumps(plan["plan"], ensure_ascii=False, indent=2) + "\n")
    if plan["plan"]["blocking"] or plan["plan"]["task_count"] != len(expected):
        raise RuntimeError(f"preflight blocked: {plan['plan']['blocking']}")
    submitted = cli("runs", "submit", "--idempotency-key", args.idempotency_key, *common)
    run = submitted["run"]
    snapshot = batch / "flow_snapshot"
    cli("runs", "flow", run["run_no"], "--out", str(snapshot))
    comparison = filecmp.dircmp(flow_dir, snapshot)
    identical = not (comparison.left_only or comparison.right_only or comparison.diff_files or comparison.funny_files)
    lock = {"run_id": run["id"], "run_no": run["run_no"], "status_at_lock": run["status"], "name": args.name,
            "dataset_id": ids["dataset_id"], "dataset_version_id": ids["version_id"],
            "dataset_tree_sha256": version.get("tree_sha256"), "config_digest": submitted.get("config_digest"),
            "model_profile_id": MODEL_ID, "placement_id": PLACEMENT, "concurrency": args.concurrency,
            "max_attempts": 1, "repetition_count": 1, "physical_tasks": len(expected), "total_terms": manifest["words"],
            "splits": manifest["splits"], "prompt_version": manifest["prompt_version"],
            "flow_sha256": manifest["sha256"], "flow_snapshot_identical": identical,
            "seeds_sha256": manifest["sha256"]["seeds"], "idempotency_key": args.idempotency_key}
    (batch / "run_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: lock[k] for k in ("run_no", "status_at_lock", "physical_tasks", "total_terms", "concurrency",
                                           "flow_snapshot_identical")}, ensure_ascii=False))
    if not identical:
        raise RuntimeError("platform flow snapshot differs from local flow directory")


if __name__ == "__main__":
    main()
