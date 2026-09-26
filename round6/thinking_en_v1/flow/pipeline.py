"""English long thinking (T037, v5 data): one Aegis 2.0 train prompt per request, one first-person thinking passage.

v4 cuts 6.2% of the safe thinking streams in the official thinking set (v3: 2.5%): 7.3% of those that reason about
harm outside the red lines, 27% of those that stay on a red-line topic without writing anything that crosses it.
Our training data has almost no English thinking. This writes the S5-safe kind of passage (round6/s5_safe_v15) for
English prompts of the kind the official set is built from: Aegis 2.0 *train* prompts, benchmark cases held out.
  kind risk    (Aegis unsafe prompt): restate the ask in general terms, weigh risk and consequences, look for the
               legitimate need, decide not to give harmful specifics, plan a safe and useful reply;
  kind helpful (Aegis safe prompt):   understand the ask, work through the knowledge, plan a helpful answer.
Nothing harmful is generated: the prompts are the dataset's own, the passages must stay safe at every cut. The
labels do not come from here: every passage goes through the red-line judge (redline_v1, assistant mode) and
apply_policy.py, so restating a red-line ask is labelled as the policy says.
"""
from __future__ import annotations

import hashlib
import json
import re

PROMPT_VERSION = "guard-thinking-en-v1"
RECORD_VERSION = "guard-response-record-v1"
WIRE_APIS = ("chat_completions", "responses")
KINDS = ("risk", "helpful")
SHAPES = {
    "stepwise": "Reason step by step: list the considerations one by one, then reach a conclusion.",
    "interpretations": "First consider several ways the request could be meant, then judge how best to respond to each.",
    "draft_revise": "Sketch a draft reply in your head, check it critically, revise it, then settle the plan.",
    "context_check": "Check the request against law, platform rules and common sense point by point, then decide.",
}
LENGTHS = {"short": "about 450 words", "medium": "about 750 words", "long": "about 1100 words"}
SYSTEM_PROMPT = """You write synthetic samples of an assistant's long private reasoning for offline content-moderation research. One request covers one user prompt. Output only a JSON object that satisfies the JSON Schema.
The samples teach a moderation model that restating and analysing a sensitive or harmful request inside the reasoning is not itself a violation.
The input gives one user prompt taken unchanged from a public safety dataset, and a kind. Do not rewrite the prompt. Write the assistant's first-person internal reasoning before it replies, in English. Write the reasoning only, never the final reply.

kind "risk": restate in your own words what the user wants (you may name the topic and the gist of the request), analyse the risks and likely consequences, consider what legitimate need might lie behind it, decide not to provide harmful help, and plan a safe reply that is still useful (safety information, lawful alternatives, where to get help).
The reasoning must stay safe if it is cut off at any point: summarise the request only; never write the harmful content itself; no actionable details, steps, recipes, quantities, code, targets, times, places or persuasive wording; never write something harmful and then take it back.

kind "helpful": understand the question, work through the relevant knowledge and common mistakes, think about how to answer accurately and completely, and plan the structure of the answer. If the prompt turns out to carry harmful intent, reason as for "risk".

Follow length_hint and organise the reasoning as shape_hint says; make it substantive and do not pad or repeat.
Keep the stance of mainland Chinese law and mainstream positions: respect the national system, ethnic unity, heroes and martyrs and settled history; write no real contact details, addresses or ID numbers; do not describe real people as having committed specific crimes. Do not say the text is fictional or a simulation.
State uncertain specifics (names of laws and lists, article numbers, figures, dates) in general terms rather than as facts.
The output is a JSON string: write line breaks as JSON escapes, not as a literal backslash followed by n.
If the prompt asks for insults, slurs or demeaning nicknames of a current or former Chinese state leader, set status to skip, leave reasoning empty and give the reason in notes.
example_output only shows the format; do not imitate its content or wording. All labels are synthetic first-pass labels, not human gold labels."""


def schema():
    string = {"type": "string"}
    return {"type": "object", "additionalProperties": False,
            "required": ["task_key", "prompt_version", "status", "reasoning", "notes"],
            "properties": {"task_key": string, "prompt_version": string,
                           "status": {"type": "string", "enum": ["ready", "skip"]},
                           "reasoning": string, "notes": string}}


EXAMPLE = {
    "risk": "The user wants help writing an untrue story about a coworker to post in a group chat. That would be an "
            "accusation without evidence, which can be defamatory and hurt the person... I should not help write "
            "that. A better reply explains why and suggests raising the actual problem directly or with a manager... "
            "(a real output must reach the length target; the example is shortened)",
    "helpful": "The user asks how to keep basil alive indoors. This is an everyday gardening question. First I recall "
               "what basil needs: light, warmth, well-drained soil... (a real output must reach the length target; "
               "the example is shortened)",
}


def request_task(seed):
    return {"task_key": seed["task_key"], "prompt_version": PROMPT_VERSION, "kind": seed["kind"],
            "shape": seed["shape"], "shape_hint": SHAPES[seed["shape"]], "length_hint": LENGTHS[seed["length"]],
            "user_prompt": seed["prompt"], "output_schema": schema(),
            "example_output": {"task_key": "example-task", "prompt_version": PROMPT_VERSION, "status": "ready",
                               "reasoning": EXAMPLE[seed["kind"]], "notes": ""}}


def request_body(model_name, seed, wire_api):
    system, user = SYSTEM_PROMPT, json.dumps(request_task(seed), ensure_ascii=False)
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 12000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 10000,
                "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_thinking_en_v1",
                                    "strict": True, "schema": schema()}}}
    raise ValueError("unsupported wire API " + str(wire_api))


def response_text(response, wire_api):
    if wire_api == "chat_completions":
        choices = response.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""
    parts = [block.get("text", "") for item in response.get("output", []) if item.get("type") == "message"
             for block in item.get("content", []) if block.get("type") == "output_text"]
    if not parts and isinstance(response.get("output_text"), str):
        parts.append(response["output_text"])
    return "".join(parts)


def stop_ok(response, wire_api):
    if wire_api == "chat_completions":
        choices = response.get("choices") or [{}]
        return choices[0].get("finish_reason") == "stop", str(choices[0].get("finish_reason"))
    if any(block.get("type") == "refusal" for item in response.get("output", [])
           if item.get("type") == "message" for block in item.get("content", [])):
        return False, "refusal"
    return response.get("status") == "completed", str(response.get("status"))


def parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))


WORDS = (330, 1500)
FICTION_MARKER = re.compile(r"fictional|imaginary|simulation", re.IGNORECASE)


def clean(text):
    """Undo one extra level of JSON escaping (a literal backslash-n instead of a line break); report whether it did."""
    text = text.strip()
    if "\\n" in text and "\n" not in text:
        return text.replace("\\n", "\n"), True
    return text, False


def size(text):
    return len(re.findall(r"[A-Za-z]+", text))


def mostly_english(text):
    letters = re.findall(r"[^\W\d_]", text)
    return bool(letters) and sum(1 for c in letters if c.isascii()) / len(letters) >= 0.95


def validate(payload, seed):
    """Return (errors, rows). A skip is not an error; it yields no rows."""
    if not isinstance(payload, dict):
        return ["root_not_object"], []
    errors = ["root_mismatch:" + k for k, v in (("task_key", seed["task_key"]), ("prompt_version", PROMPT_VERSION))
              if payload.get(k) != v]
    if errors:
        return errors, []
    if payload.get("status") == "skip":
        return ["skip"], []
    if payload.get("status") != "ready":
        return ["bad_status"], []
    text = payload.get("reasoning")
    if not isinstance(text, str) or not text.strip():
        return ["reasoning:empty"], []
    text, unescaped = clean(text)
    if not WORDS[0] <= size(text) <= WORDS[1]:
        errors.append("reasoning:length")
    if not mostly_english(text):
        errors.append("reasoning:not_english")
    if re.sub(r"\s+", " ", text).strip() == re.sub(r"\s+", " ", seed["prompt"]).strip():
        errors.append("reasoning:copies_prompt")
    if errors:
        return errors, []
    messages = [{"role": "user", "content": seed["prompt"]}, {"role": "assistant", "content": text}]
    return [], [{
        "index": 0, "language": "en", "target_role": "assistant", "label": "safe", "prompt_label": seed["group"],
        "response_style": "risk_reasoning_long" if seed["kind"] == "risk" else "helpful_reasoning_long",
        "response_format": "reasoning", "shape": seed["shape"], "messages": messages,
        "harm_basis": "benign", "onset_quote": "", "onset_char": None, "onset_end_char": None,
        "response_chars": len(text),
        "quality_flags": (["fiction_marker"] if FICTION_MARKER.search(text) else [])
                         + (["escaped_newline_repaired"] if unescaped else []),
        "length_target": seed["length"], "sample_id": f"{seed['task_key']}-thinking-00", "task_key": seed["task_key"],
        "origin_group_id": seed["family"], "word": None, "source_licenses": "CC BY 4.0", "data_layer": "synthetic",
        "source_group": seed["source_group"], "family": seed["family"], "split": seed["split"],
        "prompt_version": PROMPT_VERSION, "record_version": RECORD_VERSION,
        "annotation_origin": "synthetic_unverified_pending_judge",
        "content_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}]
