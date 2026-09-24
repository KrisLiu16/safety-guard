"""One Aster Task per term, one MiMo Messages request per Task."""
import json
import time

from aster_flow import Context, flow
from pipeline import request_body, response_text, validate


@flow(setup_timeout=60, finish_timeout=60, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 900})
def run(ctx: Context):
    seed = json.loads(ctx.task.instruction)
    model = ctx.models["main"]
    if model.wire_api != "anthropic_messages":
        raise ValueError("Expected Anthropic Messages model profile")
    body = request_body(model.name, seed)
    ctx.results.record("request", body)
    started = time.monotonic()
    response = model.request(body, timeout=600)
    ctx.results.record("response", response)
    ctx.results.record("timing", {"seconds": time.monotonic() - started,
                                  "usage": response.get("usage"), "model": response.get("model")})
    try:
        payload = json.loads(response_text(response))
        errors, rows = validate(payload, seed)
    except (ValueError, TypeError, KeyError) as exc:
        errors, rows = ["parse_or_validation:" + type(exc).__name__], []
    if response.get("stop_reason") != "end_turn":
        errors.append("stop_reason:" + str(response.get("stop_reason")))
        rows = []
    if any(e.startswith("root_") for e in errors):
        rows = []
    ctx.results.record("dataset", {"seed": seed, "errors": errors, "records": rows,
                                   "requested": 10, "valid_count": len(rows)})
    ctx.evaluation.score({"reward": len(rows) / 10},
                         detail="Structural validity only; synthetic labels are unverified")
