import json
import time
from aster_flow import Context, flow
from pipeline import request_body, response_text, validate

@flow(setup_timeout=60, finish_timeout=60, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 6620})
def run(ctx: Context):
    seed = json.loads(ctx.task.instruction)
    settings = json.loads(ctx.flow_files.read_text("settings.json"))
    total, batch = settings["examples_per_word"], settings["examples_per_request"]
    model = ctx.models["main"]
    if model.wire_api != "responses":
        raise ValueError("This flow requires the configured Responses protocol")
    accepted, all_errors = [], []
    for start in range(0, total, batch):
        count = min(batch, total - start)
        body = request_body(model.name, seed, start, count)
        ctx.results.record(f"request_{start:04d}", body)
        began = time.monotonic()
        # Exactly one call per batch; no automatic application-level retries.
        response = model.request(body, timeout=600)
        ctx.results.record(f"response_{start:04d}", response)
        ctx.results.record(f"timing_{start:04d}", {"seconds": time.monotonic()-began,
                          "usage": response.get("usage"), "model": response.get("model")})
        try:
            payload = json.loads(response_text(response))
            errors, rows = validate(payload, seed, start, count)
        except (ValueError, TypeError, KeyError) as exc:
            errors, rows = ["parse_or_validation:"+type(exc).__name__], []
        if response.get("status") not in (None, "completed"):
            errors.append("response_status:"+str(response.get("status")))
            rows = []
        # Preserve independently valid rows, but never promote them as gold labels.
        if any(e.startswith("root_") for e in errors): rows = []
        accepted.extend(rows); all_errors.extend(errors)
        ctx.results.record(f"batch_{start:04d}", {"seed":seed,"errors":errors,"records":rows})
    ctx.results.record("dataset", {"seed": seed, "requested": total,
                        "valid_count": len(accepted), "errors": all_errors, "records": accepted})
    ctx.evaluation.score({"reward": len(accepted)/total},
                         detail="Structural validity only; semantic labels remain unverified")
