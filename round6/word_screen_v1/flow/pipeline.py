"""Word screen v1: flag lexicon words that insult current or former national leaders (T012).

Why: under the user's strict rule (2026-09-24), writing out such a word is unsafe (A.1.b) even in
neutral analysis. The S2 pilot showed the generator does not reliably recognise these words (2 of 20
reviewed words slipped through, one a homophone it "explained" harmlessly, which the judge then missed),
so words are screened once, directly, before generation and before the Run A re-judge.

One request classifies a batch of existing lexicon words on this single question; nothing is generated.
Output per word: yes / no / unsure. Downstream, yes and unsure are both kept out of safe-labelled data.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "guard-word-screen-leader-v1"
WIRE_APIS = ("responses", "chat_completions")
VERDICTS = ("yes", "no", "unsure")
SYSTEM_PROMPT = """你是内容安全词表审核员。给定一批词条，逐条判断：这个词条是不是针对中国国家领导人（现任或历任）的侮辱性称呼、蔑称、贬损性谐音、贬损性代称、贬损性绰号或贬损性说法。只判断这一件事，只输出一个 JSON 对象。
- yes：是。包括用谐音、拆字、缩写、外号、代称等方式指向国家领导人并带有贬损、嘲讽、侮辱意味的写法。
- no：不是。包括领导人的正式姓名、职务、正式称谓和中性简称；与国家领导人无关的词，即使它属于其他敏感类型（色情、赌博、暴力、诈骗、其他政治话题等），也填 no。
- unsure：拿不准它是否指向国家领导人，或拿不准是否带贬损意味。
results 必须逐条对应输入的每个词条，index 和 word 原样照抄，不要遗漏、合并或改写。"""


def schema(count):
    item = {"type": "object", "additionalProperties": False, "required": ["index", "word", "verdict"],
            "properties": {"index": {"type": "integer"}, "word": {"type": "string"},
                           "verdict": {"type": "string", "enum": list(VERDICTS)}}}
    return {"type": "object", "additionalProperties": False, "required": ["batch_key", "prompt_version", "results"],
            "properties": {"batch_key": {"type": "string"}, "prompt_version": {"type": "string"},
                           "results": {"type": "array", "items": item, "minItems": count, "maxItems": count}}}


def request_body(model_name, batch, wire_api):
    words = [{"index": i, "word": w["word"]} for i, w in enumerate(batch["words"])]
    user = json.dumps({"batch_key": batch["batch_key"], "prompt_version": PROMPT_VERSION, "words": words,
                       "output_schema": schema(len(words))}, ensure_ascii=False)
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 16000,
                "input": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_word_screen_leader_v1",
                                    "strict": True, "schema": schema(len(words))}}}
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 16000, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
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


def parse(text, batch):
    """Return (errors, verdicts). Every input word must come back exactly once, index and word unchanged."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return ["no_json"], []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ["bad_json"], []
    if payload.get("batch_key") != batch["batch_key"] or payload.get("prompt_version") != PROMPT_VERSION:
        return ["root_mismatch"], []
    results = payload.get("results")
    if not isinstance(results, list):
        return ["results"], []
    words = batch["words"]
    seen, verdicts, errors = set(), [], []
    for item in results:
        index = item.get("index") if isinstance(item, dict) else None
        if type(index) is not int or not 0 <= index < len(words) or index in seen:
            errors.append("bad_index")
            continue
        seen.add(index)
        if item.get("word") != words[index]["word"]:
            errors.append("word_changed")
            continue
        if item.get("verdict") not in VERDICTS:
            errors.append("verdict")
            continue
        verdicts.append({**words[index], "verdict": item["verdict"]})
    missing = len(words) - len(verdicts)   # includes words whose item came back invalid
    if missing:
        errors.append(f"missing:{missing}")
    return errors, verdicts
