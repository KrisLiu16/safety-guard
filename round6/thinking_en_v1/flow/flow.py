"""English long thinking (T037): one Aegis 2.0 train prompt per request, one thinking passage; either wire API."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import PROMPT_VERSION, WIRE_APIS, parse_json, request_body, response_text, stop_ok, validate


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if group.get("prompt_version") != PROMPT_VERSION or group.get("calls_per_term") != 1:
        raise ValueError("Unexpected thinking-en generation contract")
    model = ctx.models["main"]
    if model.wire_api not in WIRE_APIS:
        raise ValueError("Expected a model profile with wire API in " + ", ".join(WIRE_APIS))
    counts = {"terms": len(group["terms"]), "complete": 0, "skipped": 0, "invalid": 0,
              "filtered_or_truncated": 0, "model_errors": 0}
    for position, seed in enumerate(group["terms"]):
        body = request_body(model.name, seed, model.wire_api)
        started = time.monotonic()
        response, errors, rows = None, [], []
        try:
            # Exactly one model request per prompt, no in-flow retry.
            response = model.request(body, timeout=600)
            ok, reason = stop_ok(response, model.wire_api)
            if not ok:
                errors = ["stop_reason:" + reason]
                counts["filtered_or_truncated"] += 1
            else:
                errors, rows = validate(parse_json(response_text(response, model.wire_api)), seed)
        except Exception as exc:
            errors = ["request_or_parse:" + type(exc).__name__ + ":" + str(exc)[:200]]
            counts["model_errors"] += 1
        if rows:
            counts["complete"] += 1
        elif errors == ["skip"]:
            counts["skipped"] += 1
        elif not any(e.startswith(("stop_reason", "request_or_parse")) for e in errors):
            counts["invalid"] += 1
        ctx.results.record(f"word_{position:05d}", {
            "group_key": group["group_key"], "seed": seed, "wire_api": model.wire_api,
            "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "response": response, "seconds": time.monotonic() - started, "errors": errors, "records": rows})
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["complete"] / max(1, counts["terms"])},
                         detail="Prompts with a valid thinking passage / total prompts; skips count as 0")
