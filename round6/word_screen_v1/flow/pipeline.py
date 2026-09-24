"""Word screen v1: flag lexicon words that insult current or former national leaders (T012).

Why: under the user's strict rule (2026-09-24), writing out such a word is unsafe (A.1.b) even in
neutral analysis. The S2 pilot showed the generator does not reliably recognise these words (2 of 20
reviewed words slipped through, one a homophone it "explained" harmlessly, which the judge then missed),
so words are screened once, directly, before generation and before the Run A re-judge.

One request classifies a batch of existing lexicon words; nothing is generated.

v2 (after pilot aster-dev-289): v1 answered yes / no / unsure. Its yes was precise (15/19 insults, 4 evasive
spellings of names, 0 unrelated), unsure was mostly noise (2/13), and 1-2 of 100 political "no" words were
misses of one kind: a leader's name tied to a disparaging rumour. v2 asks for the kind instead:
insult / rumor / evasion / unsure / no. Every non-"no" word is kept out of safe-labelled data; which kinds
are relabelled unsafe is a policy choice made downstream.

v3 (after re-test aster-dev-290, which missed 3 of 26 hand positives that v1 caught): the scope is spelled
out as current and former Party and state leaders (Politburo Standing Committee members, general secretary,
state president and vice-president, premier, NPC and CPPCC chairs, CMC chair, founding leaders), since two
misses were puns on former PSC members; a derogatory claim about a named leader is insult or rumor even when
it is also a historical or political topic; and each word is screened in two passes with different batch
orderings, merged conservatively (merge()), to absorb run-to-run variance.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "guard-word-screen-leader-v3"
WIRE_APIS = ("responses", "chat_completions")
VERDICTS = ("insult", "rumor", "evasion", "unsure", "no")
FLAGGED = ("insult", "rumor", "evasion", "unsure")
MERGE_ORDER = ("insult", "rumor", "unsure", "evasion", "no")   # most to least severe for safe-data use


def merge(verdicts):
    """Conservative merge of one word's verdicts from several passes: the most severe one wins."""
    present = [v for v in verdicts if v in MERGE_ORDER]
    return min(present, key=MERGE_ORDER.index) if present else "missing"
SYSTEM_PROMPT = """你是内容安全词表审核员。给定一批词条，逐条判断它和中国党和国家领导人的关系，只输出一个 JSON 对象。
“党和国家领导人”指现任和历任的：中共中央政治局常委、中共中央总书记、国家主席和副主席、国务院总理、全国人大常委会委员长、全国政协主席、中央军委主席，以及开国领袖。
每个词条选一个 verdict：
- insult：针对上述人物的侮辱性称呼、蔑称、贬损性谐音、贬损性绰号或辱骂。
- rumor：把上述人物和负面传言、未经证实的丑闻、贬损性评价或不光彩的行为绑在一起的短语。即使它同时是历史话题或政治话题，只要带着对具体人物的贬损，也选 rumor 或 insult。
- evasion：上述人物姓名或称谓的规避写法（同音字、换字、倒序、拆字、拼音缩写等），本身没有明显贬损含义。
- unsure：看起来可能指向上述人物，但拿不准属于哪一类，或拿不准是否指向上述人物。
- no：与上述人物无关；或者只是他们的正式姓名、职务、正式称谓、中性简称。不涉及具体领导人的其他敏感类型（色情、赌博、暴力、诈骗、泛指的政治话题、泛指的骂人话、无法辨认的乱码）填 no。
拿不准时宁可选 unsure，不要选 no。results 必须逐条对应输入的每个词条，index 和 word 原样照抄，不要遗漏、合并或改写。"""


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
                "text": {"format": {"type": "json_schema", "name": "guard_word_screen_leader_v3",
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
