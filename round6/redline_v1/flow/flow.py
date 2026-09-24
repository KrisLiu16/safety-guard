"""Red-line labelling flow: per response, judge v4 on the whole text, then two-level bisection when it is not safe.

The judge sees only the user prompt and a prefix of the response, exactly as in a blind judge run. The bisection
targets come from policy.py with its default table; every probe's facts are recorded, so labels can be re-derived
offline under another table. One Task holds several responses; one request per probe, no in-flow retry.
"""
from __future__ import annotations

import hashlib
import json
import time

from aster_flow import Context, flow
from levels import LEVEL_VERSION, clause_cuts, locate
from pipeline import PROMPT_VERSION, WIRE_APIS, parse, request_body, response_text, stop_ok
from policy import decide, digest


@flow(setup_timeout=60, finish_timeout=600, requires=["instruction"],
      limits={"max_sandboxes": 1, "max_creations": 1, "cpu": 1,
              "memory_gib": 1, "storage_gib": 1, "wall_seconds": 80000})
def run(ctx: Context):
    group = json.loads(ctx.task.instruction)
    if (group.get("prompt_version"), group.get("judge_version"), group.get("policy_digest")) != (
            LEVEL_VERSION, PROMPT_VERSION, digest()):
        raise ValueError("Unexpected red-line labelling contract")
    model = ctx.models["main"]
    if model.wire_api not in WIRE_APIS:
        raise ValueError("Expected a model profile with wire API in " + ", ".join(WIRE_APIS))
    counts = {"responses": len(group["terms"]), "finished": 0, "calls": 0}
    for position, item in enumerate(group["terms"]):
        started, calls, errors = time.monotonic(), [], []

        def judge(cut):
            body = request_body(model.name, {"user_prompt": item["user_prompt"],
                                              "assistant_text": item["assistant_text"][:cut]}, model.wire_api)
            try:
                response = model.request(body, timeout=300)
                ok, reason = stop_ok(response, model.wire_api)
                if not ok:
                    errors.append(f"{cut}:stop_reason:{reason}")
                    return None
                problems, facts = parse(response_text(response, model.wire_api))
                if facts is None:
                    errors.append(f"{cut}:parse:{','.join(problems)}")
                    return None
                verdict = decide(facts)
                calls.append({"cut": cut, "facts": facts, "level": verdict["level"], "excluded": verdict["excluded"],
                              "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
                return verdict["level"], verdict["excluded"]
            except Exception as exc:
                errors.append(f"{cut}:request:{type(exc).__name__}:{str(exc)[:160]}")
                return None

        result = locate(clause_cuts(item["assistant_text"]), judge)
        counts["calls"] += len(result["probes"])
        counts["finished"] += result["status"] in ("safe", "excluded", "located")
        ctx.results.record(f"item_{position:05d}", {"group_key": group["group_key"], "item_id": item["task_key"],
                                                    "result": result, "calls": calls, "errors": errors,
                                                    "seconds": time.monotonic() - started})
    ctx.results.record("summary", counts)
    ctx.evaluation.score({"reward": counts["finished"] / max(1, counts["responses"])},
                         detail="Responses labelled (safe, excluded or monotonic located) / responses")
