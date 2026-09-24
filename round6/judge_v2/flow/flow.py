"""Blind judge flow v2: one request per assistant text; Anthropic Messages or chat_completions."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import PROMPT_VERSION, parse, request_body, response_text, stop_ok


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if group.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Unexpected judge contract")
    model = ctx.models["main"]
    if model.wire_api not in ("anthropic_messages", "chat_completions"):
        raise ValueError("Expected an Anthropic Messages or chat_completions model profile")
    counts = {"items": len(group["items"]), "parsed": 0, "refused_or_filtered": 0, "model_errors": 0}
    for position, item in enumerate(group["items"]):
        body = request_body(model.name, item, model.wire_api)
        started = time.monotonic()
        response, errors, judgment = None, [], None
        try:
            # Exactly one request per judged text, no in-flow retry.
            response = model.request(body, timeout=300)
            ok, reason = stop_ok(response, model.wire_api)
            if not ok:
                errors.append("stop_reason:" + reason)
                counts["refused_or_filtered"] += 1
            else:
                errors, judgment = parse(response_text(response, model.wire_api))
        except Exception as exc:
            errors = ["request:" + type(exc).__name__ + ":" + str(exc)[:200]]
            counts["model_errors"] += 1
        counts["parsed"] += int(judgment is not None)
        ctx.results.record(f"item_{position:05d}", {
            "group_key": group["group_key"], "item": item, "judgment": judgment, "errors": errors,
            "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "response": response, "seconds": time.monotonic() - started})
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["parsed"] / max(1, counts["items"])},
                         detail="Parsed judgments / judged texts (agreement is analysed offline)")
