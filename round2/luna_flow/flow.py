"""Source-grouped Luna calls: read each word serially and record one JSON per word."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import request_body, response_text, validate


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if group.get("calls_per_term") != 1 or group.get("examples_per_term") != 10:
        raise ValueError("Unexpected task generation contract")
    model = ctx.models["main"]
    if model.wire_api != "responses":
        raise ValueError("Expected Responses model profile")
    counts = {"terms": len(group["terms"]), "attempted": 0, "valid_examples": 0,
              "complete_terms": 0, "unfit_terms": 0, "model_errors": 0}
    for position, seed in enumerate(group["terms"]):
        body = request_body(model.name, seed)
        request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        started = time.monotonic()
        response = None
        errors, rows = [], []
        try:
            # Exactly one model request for this term, with no in-flow retry.
            response = model.request(body, timeout=600)
            payload = json.loads(response_text(response))
            errors, rows = validate(payload, seed)
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
        counts["complete_terms"] += int(len(rows) == 10 and not errors)
        counts["unfit_terms"] += int("unfit_seed" in errors)
        ctx.results.record(f"word_{position:05d}", {
            "group_key": group["group_key"], "source_group": group["source_group"],
            "seed": seed, "request_sha256": request_hash, "response": response,
            "seconds": time.monotonic() - started, "errors": errors, "records": rows,
        })
    ctx.results.record("summary", counts)
    # User-defined success: a term counts only if all 10 examples pass.
    ctx.evaluation.score({"reward": counts["complete_terms"] / counts["terms"]},
                         detail="Complete words / total words; filtered and partial words count as 0")
