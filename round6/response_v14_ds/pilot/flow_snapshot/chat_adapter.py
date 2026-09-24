"""OpenAI chat_completions adapter for the frozen v14 contract (e.g. DeepSeek via tokenhub).

The system prompt and the user instruction (task, slots, output_schema, example_output) are taken
verbatim from pipeline.request_body; only the transport changes. Strict json_schema enforcement of
the Responses API is not available here, so the model must follow the in-prompt schema and the
program-side validate() stays the only acceptance check.
"""
from __future__ import annotations

import json
import re

from pipeline import request_body


def chat_body(model_name, seed):
    frozen = request_body(model_name, seed)
    system, user = frozen["input"][0]["content"], frozen["input"][1]["content"]
    return {"model": model_name, "stream": False, "max_tokens": 8000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def chat_text(response):
    choices = response.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def finish_reason(response):
    choices = response.get("choices") or [{}]
    return str(choices[0].get("finish_reason"))


def parse_json(text):
    """The first top-level JSON object in the reply (tolerates ```json fences)."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))
