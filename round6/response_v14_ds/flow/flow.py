"""Source-grouped v14 generation over chat_completions: one word per request, four responses."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from chat_adapter import chat_body, chat_text, finish_reason, parse_json
from pipeline import PROMPT_VERSION, validate


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if (group.get("calls_per_term") != 1 or group.get("responses_per_term") != 4
            or group.get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Unexpected task generation contract")
    model = ctx.models["main"]
    if model.wire_api != "chat_completions":
        raise ValueError("Expected a chat_completions model profile")
    counts = {"terms": len(group["terms"]), "attempted": 0, "valid_responses": 0,
              "complete_terms": 0, "filtered_or_truncated": 0, "model_errors": 0}
    for position, seed in enumerate(group["terms"]):
        seed = {**seed, "source_group": group["source_group"]}
        body = chat_body(model.name, seed)
        request_hash = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        started = time.monotonic()
        response, errors, rows = None, [], []
        try:
            # Exactly one model request for this term, with no in-flow retry.
            response = model.request(body, timeout=600)
            reason = finish_reason(response)
            if reason != "stop":
                errors = ["finish_reason:" + reason]
                counts["filtered_or_truncated"] += 1
            else:
                errors, rows = validate(parse_json(chat_text(response)), seed)
                if any(e.startswith("root_") for e in errors):
                    rows = []
        except Exception as exc:
            errors = ["request_or_parse:" + type(exc).__name__ + ":" + str(exc)[:200]]
            counts["model_errors"] += 1
        counts["attempted"] += 1
        counts["valid_responses"] += len(rows)
        counts["complete_terms"] += int(len(rows) == 4 and not errors)
        ctx.results.record(f"word_{position:05d}", {
            "group_key": group["group_key"], "source_group": group["source_group"],
            "seed": seed, "request_sha256": request_hash, "response": response,
            "seconds": time.monotonic() - started, "errors": errors, "records": rows,
        })
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["complete_terms"] / counts["terms"]},
                         detail="Complete words (4/4 responses) / total words; filtered, refused and partial words count as 0")
