#!/usr/bin/env python3
"""Publish the verified v12 full dataset, plan, then submit one fixed Run."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VERSION_ID = "dsv_01M36JEXZBQ0E1X2BWZ94A8GFP"
MODEL_ID = "mdl_01M2W0QMTH0KFNJQFTBYY2QTVG"
TASKS = 1517
TERMS = 449575
RUN_NAME = "guard-v12-oneword-two-full-20260923"
IDEMPOTENCY_KEY = "guard-v12-oneword2-full-20260923"


def cli(*args: str) -> dict:
    process = subprocess.run(["aster", *args], capture_output=True, text=True)
    raw = process.stdout or process.stderr
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Aster response was not JSON; exit={process.returncode}") from exc
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"{error.get('code')}: {error.get('message')} ({error.get('hint')})")
    return envelope["data"]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    version_data = cli("datasets", "version", VERSION_ID)
    version = version_data["version"]
    count = version["task_count"]
    if count != TASKS or len(version_data["tasks"]) != TASKS:
        raise RuntimeError(f"Draft incomplete: {count} / {TASKS} Tasks")
    source_manifest = json.loads((HERE / "full-shards-300/manifest.json").read_text())
    expected_keys = {shard["task_key"] for shard in source_manifest["shards"]}
    actual_keys = {task["task_key"] for task in version_data["tasks"]}
    if (actual_keys != expected_keys or len(expected_keys) != TASKS
            or source_manifest["total_terms"] != TERMS
            or sum(shard["term_count"] for shard in source_manifest["shards"]) != TERMS
            or source_manifest["logical_categories"] != 33
            or source_manifest["calls_per_term"] != 1
            or source_manifest["examples_per_term"] != 2):
        raise RuntimeError("Published Task keys or one-word/two-example contract differ")
    if version["status"] == "draft":
        published = cli("datasets", "publish", VERSION_ID)
        published_version = published.get("version", published)
        print(json.dumps({"dataset_published": VERSION_ID,
                          "status": published_version.get("status")}), flush=True)
        version = (published_version if published_version.get("tree_sha256")
                   else cli("datasets", "version", VERSION_ID)["version"])
    if version["status"] != "published":
        raise RuntimeError(f"Unexpected dataset status {version['status']}")

    shared = ["--name", RUN_NAME, "--model-profile-id", MODEL_ID,
              "--placement-id", "sg-public", "--tasks", VERSION_ID,
              "--repetition-count", "1", "--concurrency", "40",
              "--max-attempts", "1", "--flow", str(HERE / "flow"),
              "--tito-enabled", "false", "--model-access", "gateway"]
    plan = cli("runs", "plan", *shared)
    # Do not persist or print the plan's validation token or internal URLs.
    sample_count = plan.get("plan", {}).get("total_samples")
    if not plan.get("valid") or plan.get("plan", {}).get("blocking"):
        raise RuntimeError("Aster plan has blocking validation findings")
    if sample_count != TASKS:
        raise RuntimeError(f"Plan sample count differs: {sample_count} != {TASKS}")
    print(json.dumps({"plan_ok": True, "tasks": TASKS,
                      "config_digest": plan.get("config_digest"),
                      "flow_digest": plan.get("effective_config", {}).get("flow", {}).get("digest")}), flush=True)
    model_config = plan.get("effective_config", {}).get("models", {}).get("main", {})
    if model_config.get("profile_id") != MODEL_ID or model_config.get("model_name") != "gpt-5.6-luna":
        raise RuntimeError("Planned model profile differs from user-selected Luna")
    submitted = cli("runs", "submit", "--idempotency-key", IDEMPOTENCY_KEY, *shared)
    run = submitted.get("run", submitted)
    run_id = run.get("id", run.get("run_id"))
    run_no = run.get("run_no", run.get("number"))
    if not run_id:
        raise RuntimeError("Submission returned no Run ID")

    spec = importlib.util.spec_from_file_location("v12_pipeline", HERE / "flow/pipeline.py")
    assert spec and spec.loader
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    lock = {
        "run_id": run_id, "run_no": run_no, "status_at_lock": run.get("status"),
        "dataset_version_id": VERSION_ID,
        "dataset_tree_sha256": version.get("tree_sha256"),
        "plan_config_digest": plan.get("config_digest"),
        "config_digest": run.get("config_digest", plan.get("config_digest")),
        "flow_digest": run.get("flow_digest", plan.get("effective_config", {}).get("flow", {}).get("digest")),
        "model_profile_id": MODEL_ID, "placement_id": "sg-public",
        "model_name": model_config.get("model_name"),
        "model_secret_revision": model_config.get("secret_revision"),
        "physical_tasks": TASKS, "logical_categories": 33, "total_terms": TERMS,
        "calls_per_term": 1, "examples_per_term": 2,
        "languages_per_term": ["zh", "en"], "concurrency": 40,
        "max_attempts": 1, "repetition_count": 1,
        "prompt_version": pipeline.PROMPT_VERSION,
        "record_version": pipeline.RECORD_VERSION,
        "schema_sha256": canonical_sha(pipeline.schema()),
        "system_prompt_sha256": hashlib.sha256(pipeline.SYSTEM_PROMPT.encode()).hexdigest(),
        "pipeline_sha256": sha_file(HERE / "flow/pipeline.py"),
        "flow_sha256": sha_file(HERE / "flow/flow.py"),
        "shard_manifest_sha256": sha_file(HERE / "full-shards-300/manifest.json"),
        "source_manifest": {key: source_manifest[key] for key in
                            ("total_terms", "logical_categories", "physical_tasks")},
    }
    (HERE / "full_run_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"submitted": True, "run_id": run_id,
                      "run_no": run_no, "status": run.get("status"),
                      "lock": str(HERE / "full_run_lock.json")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
