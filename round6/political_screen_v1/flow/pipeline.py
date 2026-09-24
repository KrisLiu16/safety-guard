"""Political screen v1 (T026): classify lexicon words for the political "high alert" rule.

User decision (2026-09-25): a special political event is cut as soon as it is mentioned (controversial);
national leaders mentioned in normal news pass, but negative, mocking, rumour or variant-spelled mentions are cut;
other political topics follow the red-line rules. The judge cannot be relied on to recognise every coded or
variant name of an event, so the lexicon is screened once, like the leader screen (word_screen_v1, T012):
  event          a Chinese politically sensitive event, movement or anniversary, or a code name for one
  figure         a politically sensitive person who is not a national leader (dissidents, activists, symbols
                 of a sensitive event, officials purged for political reasons)
  org            an organisation banned or designated as separatist, terrorist or a cult, or its slogan
  leader         about a national leader (handled by the leader screen, T012; recorded here for coverage)
  other_political  ordinary political vocabulary: policies, institutions, official terms, general politics
  no             not political;  unsure   cannot tell
plus ambiguous: whether the string often means something harmless in ordinary text (a number, a common word),
so that its mere presence must not trigger a cut. Downstream (redline_v1/apply_policy.py --political-terms) event /
figure / org words, and leader words the T012 screen flags, are matched verbatim in every training text; an
ambiguous one fires only in a text where the judge found a political red line. Nothing is generated; only words
are sent.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "guard-political-screen-v1"
WIRE_APIS = ("responses", "chat_completions")
VERDICTS = ("event", "figure", "org", "leader", "other_political", "unsure", "no")
MODES = {"screen": VERDICTS}
MERGE_ORDER = ("event", "figure", "org", "leader", "unsure", "other_political", "no")   # most to least alerting

SYSTEM_PROMPT = """你是内容安全词表审核员。给定一批词条，逐条判断它和中国政治敏感内容的关系，只输出一个 JSON 对象。
每个词条选一个 verdict：
- event：指向某个中国政治敏感事件、运动、纪念日，或它们的代称、隐晦说法、谐音、数字写法（例如六四及其各种代称）。
- figure：政治敏感人物，但不是党和国家领导人：异见人士、维权人士、敏感事件的标志性人物、因政治原因被查处的官员，或他们的代称。
- org：被取缔或被定性为分裂、恐怖、邪教的组织，以及它们的口号、代称。
- leader：与党和国家领导人（现任和历任的政治局常委、总书记、国家主席和副主席、国务院总理、全国人大常委会委员长、全国政协主席、中央军委主席、开国领袖）有关的任何写法。
- other_political：普通的政治词汇：政策、机构、官方用语、一般时政话题。
- no：与政治无关（包括色情、赌博、诈骗、脏话等其他敏感类型）。
- unsure：拿不准。拿不准是否属于 event、figure、org 时选 unsure，不要选 no。
另外逐条给出 ambiguous：这个字符串在普通文本里是否常常是别的意思（例如一个数字、一个常用词、一个常见人名），以至于单凭它出现不能判断在说敏感内容。true 或 false。
results 必须逐条对应输入的每个词条，index 和 word 原样照抄，不要遗漏、合并或改写。"""


def merge(verdicts):
    """Most alerting verdict across passes wins; a word is ambiguous if any pass said so."""
    present = [v for v, _ in verdicts if v in MERGE_ORDER]
    verdict = min(present, key=MERGE_ORDER.index) if present else "missing"
    return verdict, any(a for _, a in verdicts)


def schema(count):
    item = {"type": "object", "additionalProperties": False, "required": ["index", "word", "verdict", "ambiguous"],
            "properties": {"index": {"type": "integer"}, "word": {"type": "string"},
                           "verdict": {"type": "string", "enum": list(VERDICTS)}, "ambiguous": {"type": "boolean"}}}
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
                "text": {"format": {"type": "json_schema", "name": "guard_political_screen_v1",
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
        if item.get("verdict") not in VERDICTS or not isinstance(item.get("ambiguous"), bool):
            errors.append("verdict")
            continue
        verdicts.append({**words[index], "verdict": item["verdict"], "ambiguous": item["ambiguous"]})
    missing = len(words) - len(verdicts)
    if missing:
        errors.append(f"missing:{missing}")
    return errors, verdicts
