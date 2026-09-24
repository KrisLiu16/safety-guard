#!/usr/bin/env python3
"""Download Aster result artifacts and export validated, deduplicated RL candidates.

Read-only platform access. This script never submits, retries, or cancels a run.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import validate

def cli(args):
    result = subprocess.run(["aster", *args], capture_output=True, text=True)
    envelope = json.loads(result.stdout or result.stderr)
    if not envelope.get("ok"):
        error = envelope.get("error", {})
        raise RuntimeError(f"Aster {args[0]} failed: {error.get('code')}: {error.get('message')}")
    return envelope["data"]

def dump_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    temporary.replace(path)

def quality_flags(row):
    """Conservative review flags, not a second semantic classifier."""
    full = "\n".join(m["content"] for m in row["messages"])
    flags = []
    if any(prefix + row["word"] in full for prefix in ["虚构人物", "虚构角色", "虚构的", "虚构组织"]):
        flags.append("seed_entity_recast_as_fictional")
    if row["label"] == "unsafe" and any(w in full for w in ["虚构", "小说", "剧本", "角色扮演"]):
        flags.append("fiction_context_requires_semantic_review")
    if row["target_role"] == "assistant" and row["input_label"] == "safe":
        user = row["messages"][0]["content"]
        if re.search(r"(如何|怎么|怎样|帮我|请).{0,25}(伪造|掩盖|骚扰|威胁|袭击|骗取|规避|查不出|降低.{0,8}识别)", user):
            flags.append("safe_input_label_has_risk_cues")
    return sorted(set(flags))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run = cli(["runs", "get", args.run])
    run = run.get("run", run)
    attempts = cli(["runs", "attempts", args.run])["items"]
    artifacts = cli(["runs", "artifacts", args.run])
    items = artifacts["items"]
    if artifacts.get("truncated"):
        items = []
        for attempt in attempts:
            page = cli(["runs", "artifacts", args.run, "--attempt-id", attempt["attempt_id"]])
            if page.get("truncated"):
                raise RuntimeError("A single attempt's artifacts are truncated; export incomplete")
            items.extend(page["items"])
    # Results.record stores numbered JSON files; their content identifies the result name.
    items = [x for x in items if x["path"].endswith(".json") and "results" in Path(x["path"]).parts]
    (out / "artifact_manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))

    def download(item):
        digest = hashlib.sha256(item["path"].encode()).hexdigest()[:16]
        destination = out / "artifacts" / item["attempt_id"] / (digest + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            cli(["runs", "download", args.run, item["attempt_id"], item["path"], "--out", str(destination)])
        if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Artifact digest mismatch: " + str(destination))
        obj = json.loads(destination.read_text())
        return item, obj

    results = collections.defaultdict(dict)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for item, obj in pool.map(download, items):
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "value" in obj:
                results[item["attempt_id"]][obj["name"]] = obj["value"]
    attempt_map = {a["attempt_id"]: a for a in attempts}
    valid, rejected, usage, per_word = [], [], [], []
    seen_ids, seen_contents = set(), set()
    for attempt_id, bundles in sorted(results.items()):
        dataset = bundles.get("dataset")
        if dataset:
            seed = dataset["seed"]
            requested = dataset["requested"]
        else:
            batches = [v for k, v in bundles.items() if k.startswith("batch_")]
            if not batches: continue
            seed = batches[0]["seed"]
            requested = sum(len(v.get("records", [])) for v in batches)
        accepted_for_word = 0
        for key, raw in sorted(bundles.items()):
            if not key.startswith("response_"): continue
            start = int(key.removeprefix("response_"))
            request = bundles.get(f"request_{start:04d}", {})
            count = request.get("text", {}).get("format", {}).get("schema", {}).get("properties", {}).get("examples", {}).get("minItems")
            if not isinstance(count, int):
                rejected.append({"attempt_id": attempt_id, "batch": start, "errors": ["missing_request_schema"]}); continue
            from pipeline import response_text
            try:
                payload = json.loads(response_text(raw))
                expected_policy = json.loads(request["input"][-1]["content"])["policy_version"]
                errors, rows = validate(payload, seed, start, count, policy_version=expected_policy)
            except (ValueError, TypeError, KeyError) as exc:
                errors, rows = ["parse:"+type(exc).__name__], []
            if any(e.startswith("root_") for e in errors) or raw.get("status") not in (None, "completed"):
                rows = []
            if errors:
                rejected.append({"attempt_id": attempt_id, "seed_id": seed["seed_id"], "batch": start, "errors": errors})
            if isinstance(raw.get("usage"), dict): usage.append(raw["usage"])
            for row in rows:
                if row["sample_id"] in seen_ids or row["content_sha256"] in seen_contents:
                    rejected.append({"sample_id": row["sample_id"], "errors": ["duplicate_id_or_content"]}); continue
                seen_ids.add(row["sample_id"]); seen_contents.add(row["content_sha256"])
                row["generation"] = {"run_id": run.get("id"), "run_no": run.get("run_no"),
                    "attempt_id": attempt_id, "response_id": raw.get("id"), "model": raw.get("model"),
                    "model_profile_id": "mdl_01M2TK4P8Z0DE5QCJYJ8RSK8YV"}
                valid.append(row); accepted_for_word += 1
        per_word.append({"seed_id": seed["seed_id"], "word": seed["word"], "requested": requested,
                         "valid": accepted_for_word, "attempt_id": attempt_id})
    valid.sort(key=lambda x: x["sample_id"])
    for row in valid:
        row["quality_flags"] = quality_flags(row)
    review_required = [r for r in valid if r["quality_flags"]]
    provisional = [r for r in valid if not r["quality_flags"]]
    dump_jsonl(out / "examples.jsonl", valid)
    dump_jsonl(out / "review_required.jsonl", review_required)
    dump_jsonl(out / "rejected.jsonl", rejected)
    # The training adapter must pass ONLY observation to the Guard; reward data is separate.
    rl = [{"id": x["sample_id"], "origin_group_id": x["origin_group_id"], "split": "unassigned",
           "observation": {"messages": x["messages"], "target_role": x["target_role"],
                           "target_message_index": x["target_message_index"]},
           "reward_spec": {"expected_label": x["label"], "input_label": x["input_label"],
                           "policy_id": x["policy_id"], "policy_version": x["policy_version"],
                           "evidence_spans": x["evidence_spans"], "decidable_at_char": x["decidable_at_char"],
                           "label_origin": "luna_synthetic_unverified"},
           "provenance": {"word": x["word"], "source": x["source"], "generation": x["generation"]}}
          for x in provisional]
    dump_jsonl(out / "rl_tasks.jsonl", rl)
    summary = {"run_id": run.get("id"), "run_no": run.get("run_no"), "status": run.get("status"),
               "total_tasks": run.get("total_samples"), "processed_tasks": run.get("processed_count"),
               "succeeded_tasks": run.get("succeeded_count"), "failed_tasks": run.get("failed_count"),
               "valid_examples": len(valid), "words_with_results": len(per_word),
               "rl_candidates": len(provisional), "review_required": len(review_required),
               "quality_flags": dict(collections.Counter(f for r in valid for f in r["quality_flags"])),
               "labels": dict(collections.Counter(x["label"] for x in valid)),
               "target_roles": dict(collections.Counter(x["target_role"] for x in valid)),
               "policies": dict(collections.Counter(x["policy_id"] for x in valid)),
               "rejection_events": len(rejected), "per_word": per_word,
               "usage": {k: sum(v.get(k, 0) or 0 for v in usage) for k in ["input_tokens", "output_tokens", "total_tokens"]},
               "semantic_review": "not_automatically_verified", "data_split": "unassigned"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_word"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
