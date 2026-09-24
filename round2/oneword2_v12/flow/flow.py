"""Source-grouped Luna calls: one word per request, two user examples."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import PROMPT_VERSION, request_body, response_text, validate


def expand_seed(item, group_name):
    if isinstance(item, dict):
        return item
    if isinstance(item, list) and len(item) == 3:
        key, word, licenses = item
        return {"task_key": key, "word": word, "origin_group_id": key,
                "data_layer": "noncommercial" if group_name.startswith("citizenlab/") else "permissive",
                "source_licenses": licenses}
    raise ValueError("Invalid compact term record")


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if (group.get("calls_per_term") != 1 or group.get("examples_per_term") != 2
            or group.get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Unexpected task generation contract")
    model = ctx.models["main"]
    if model.wire_api != "responses":
        raise ValueError("Expected Responses model profile")
    counts = {"terms": len(group["terms"]), "attempted": 0, "valid_examples": 0,
              "complete_terms": 0, "unfit_terms": 0, "model_errors": 0}
    for position, item in enumerate(group["terms"]):
        seed = expand_seed(item, group["source_group"])
        seed = {**seed, "source_group": group["source_group"]}
        global_index = group["start_offset"] + position
        body = request_body(model.name, seed, global_index)
        request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        started = time.monotonic()
        response = None
        errors, rows = [], []
        try:
            # Exactly one model request for this term, with no in-flow retry.
            response = model.request(body, timeout=600)
            payload = json.loads(response_text(response))
            errors, rows = validate(payload, seed, global_index=global_index)
            if response.get("status") != "completed":
                errors.append("status:" + str(response.get("status")))
                rows = []
            if any(e.startswith("root_") for e in errors):
                rows = []
        except Exception as exc:
            errors = ["request_or_parse:" + type(exc).__name__ + ":" + str(exc)[:200]]
            counts["model_errors"] += 1
        counts["attempted"] += 1
        counts["valid_examples"] += len(rows)
        counts["complete_terms"] += int(len(rows) == 2 and not errors)
        counts["unfit_terms"] += int("unfit_seed" in errors)
        ctx.results.record(f"word_{position:05d}", {
            "group_key": group["group_key"], "source_group": group["source_group"],
            "seed": seed, "global_index": global_index,
            "request_sha256": request_hash, "response": response,
            "seconds": time.monotonic() - started, "errors": errors, "records": rows,
        })
    ctx.results.record("summary", counts)
    # User-defined success: a term counts only if both examples pass.
    ctx.evaluation.score({"reward": counts["complete_terms"] / counts["terms"]},
                         detail="Complete words / total words; filtered and partial words count as 0")
