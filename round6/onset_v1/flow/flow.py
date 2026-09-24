"""Onset relocation flow: per unsafe response, bisect clause cut points with the domestic judge (v3.2, unchanged).

The judge sees only the user prompt and a prefix of the response, exactly as in a blind judge run; it never sees
the v14 onset or any label. One Task holds several responses; each response costs about log2(clauses) + 2 calls.
"""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from onset import ONSET_VERSION, bisect, clause_cuts
from pipeline import PROMPT_VERSION, WIRE_APIS, parse, request_body, response_text, stop_ok


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if group.get("prompt_version") != ONSET_VERSION or group.get("judge_version") != PROMPT_VERSION:
        raise ValueError("Unexpected onset contract")
    model = ctx.models["main"]
    if model.wire_api not in WIRE_APIS:
        raise ValueError("Expected a model profile with wire API in " + ", ".join(WIRE_APIS))
    counts = {"responses": len(group["terms"]), "located": 0, "calls": 0}
    for position, item in enumerate(group["terms"]):
        started, calls, errors = time.monotonic(), [], []

        def judge(cut):
            body = request_body(model.name, {"user_prompt": item["user_prompt"],
                                              "assistant_text": item["assistant_text"][:cut]}, model.wire_api)
            try:
                response = model.request(body, timeout=300)   # one request per probe, no in-flow retry
                ok, reason = stop_ok(response, model.wire_api)
                if not ok:
                    errors.append(f"{cut}:stop_reason:{reason}")
                    return None
                problems, judgment = parse(response_text(response, model.wire_api))
                if judgment is None:
                    errors.append(f"{cut}:parse:{','.join(problems)}")
                    return None
                calls.append({"cut": cut, "judgment": judgment,
                              "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
                return judgment["label"]
            except Exception as exc:
                errors.append(f"{cut}:request:{type(exc).__name__}:{str(exc)[:160]}")
                return None

        result = bisect(clause_cuts(item["assistant_text"]), judge)
        counts["calls"] += len(result["probes"])
        counts["located"] += result["status"] == "located"
        ctx.results.record(f"item_{position:05d}", {"group_key": group["group_key"], "item_id": item["task_key"],
                                                    "result": result, "calls": calls, "errors": errors,
                                                    "seconds": time.monotonic() - started})
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["located"] / max(1, counts["responses"])},
                         detail="Responses with a monotonic located onset / responses")
