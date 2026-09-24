"""Read a source category's word list and call MiMo serially, one word at a time."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import request_body, response_text, validate


@flow(setup_timeout=60, finish_timeout=60, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 20000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if group.get("calls_per_term") != 1 or group.get("examples_per_term") != 10:
        raise ValueError("Unexpected task generation contract")
    model = ctx.models["main"]
    if model.wire_api != "anthropic_messages":
        raise ValueError("Expected Anthropic Messages model profile")
    counts = {"terms": len(group["terms"]), "attempted": 0, "valid_examples": 0,
              "complete_terms": 0, "unfit_terms": 0, "model_errors": 0}
    for position, seed in enumerate(group["terms"]):
        body = request_body(model.name, seed)
        request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        started = time.monotonic()
        response = None
        errors, rows = [], []
        try:
            # One request for this word; no application-level retry.
            response = model.request(body, timeout=600)
            payload = json.loads(response_text(response))
            errors, rows = validate(payload, seed)
            if response.get("stop_reason") != "end_turn":
                errors.append("stop_reason:" + str(response.get("stop_reason")))
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
        # Each record becomes a JSON result artifact. It can be downloaded and
        # independently revalidated without replaying any model request.
        ctx.results.record(f"word_{position:03d}", {
            "group_key": group["group_key"], "source_group": group["source_group"],
            "seed": seed, "request_sha256": request_hash, "response": response,
            "seconds": time.monotonic() - started, "errors": errors, "records": rows,
        })
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["valid_examples"] / (10 * counts["terms"])},
                         detail="Structural validity only; synthetic labels unverified")
