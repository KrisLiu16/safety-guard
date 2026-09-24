"""Source-grouped Aster Tasks, sequential batch-of-ten Luna calls."""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from pipeline import PROMPT_VERSION, request_body, response_text, validate


def expand_seed(item: list, group_name: str) -> dict:
    if not isinstance(item, list) or len(item) != 3:
        raise ValueError("Invalid compact term")
    key, word, licenses = item
    return {
        "task_key": key, "word": word, "origin_group_id": key,
        "data_layer": "noncommercial" if group_name.startswith("citizenlab/") else "permissive",
        "source_licenses": licenses,
        "source_group": group_name,
    }


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if (group.get("batch_size") != 10 or group.get("examples_per_term") != 2
            or group.get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Unexpected v9 task contract")
    model = ctx.models["main"]
    if model.wire_api != "responses":
        raise ValueError("Expected Responses profile")
    terms = group["terms"]
    counts = {"terms": len(terms), "attempted_requests": 0,
              "complete_terms": 0, "valid_examples": 0,
              "model_errors": 0, "filtered_batches": 0}
    for offset in range(0, len(terms), 10):
        seeds = [expand_seed(item, group["source_group"]) for item in terms[offset:offset + 10]]
        batch_key = f"{group['group_key']}-b{offset // 10:04d}"
        first_global_index = group["start_offset"] + offset
        body = request_body(model.name, batch_key, seeds, first_global_index)
        request_hash = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        response = None
        started = time.monotonic()
        try:
            response = model.request(body, timeout=900)
            if response.get("status") != "completed":
                raise ValueError("response_status:" + str(response.get("status")))
            payload = json.loads(response_text(response))
            term_results = validate(payload, batch_key, seeds,
                                    first_global_index, group["source_group"])
        except Exception as exc:
            error = "request_or_parse:" + type(exc).__name__ + ":" + str(exc)[:200]
            term_results = [{"errors": [error], "records": []} for _ in seeds]
            counts["model_errors"] += 1
            counts["filtered_batches"] += int("filter" in str(exc).lower())
        counts["attempted_requests"] += 1
        for term in term_results:
            counts["valid_examples"] += len(term["records"])
            counts["complete_terms"] += int(len(term["records"]) == 2 and not term["errors"])
        ctx.results.record(f"batch_{offset // 10:04d}", {
            "batch_key": batch_key, "group_key": group["group_key"],
            "source_group": group["source_group"],
            "first_global_index": first_global_index,
            "seeds": seeds, "request_sha256": request_hash,
            "response": response, "seconds": time.monotonic() - started,
            "term_results": term_results,
        })
    ctx.results.record("summary", counts)
    ctx.evaluation.score(
        {"reward": counts["complete_terms"] / counts["terms"]},
        detail="Each word requires one valid Chinese and one valid English case; filtered words score zero",
    )
