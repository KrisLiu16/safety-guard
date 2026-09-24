"""Political screen flow (same shape as the word screen): one Task = one batch of words = one request."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import MODES, PROMPT_VERSION, WIRE_APIS, parse, request_body, response_text, stop_ok


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 7200})
def run(ctx: Context):
    batch = json.loads(ctx.task.instruction)
    if batch.get("prompt_version") != PROMPT_VERSION or batch.get("mode", "screen") not in MODES:
        raise ValueError("Unexpected political screen contract")
    model = ctx.models["main"]
    if model.wire_api not in WIRE_APIS:
        raise ValueError("Expected a model profile with wire API in " + ", ".join(WIRE_APIS))
    body = request_body(model.name, batch, model.wire_api)
    started = time.monotonic()
    response, errors, verdicts = None, [], []
    try:
        # Exactly one request per batch, no in-flow retry; words that come back invalid are re-screened later.
        response = model.request(body, timeout=900)
        ok, reason = stop_ok(response, model.wire_api)
        if ok:
            errors, verdicts = parse(response_text(response, model.wire_api), batch)
        else:
            errors = ["stop_reason:" + reason]
    except Exception as exc:
        errors = ["request:" + type(exc).__name__ + ":" + str(exc)[:200]]
    ctx.results.record("batch", {
        "batch_key": batch["batch_key"], "mode": batch.get("mode", "screen"), "wire_api": model.wire_api,
        "words": len(batch["words"]),
        "verdicts": verdicts, "errors": errors, "response": response, "seconds": time.monotonic() - started,
        "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    ctx.evaluation.score({"reward": len(verdicts) / max(1, len(batch["words"]))},
                         detail="Words with a valid verdict / words in the batch")
