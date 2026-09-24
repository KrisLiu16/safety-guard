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
v1.1 (T026 word check: leader words that were ordinary composites, traditional-script or standard-pinyin names):
mode "leader" asks, for every word the screen put in `leader`, how it names the leader:
  negative    insult, mocking nickname, or tied to a rumour, scandal or derogatory claim
  variant     an evasive spelling of the name or title, not derogatory by itself
  formal      the normal way to write it: simplified or traditional name, title, neutral short form, standard pinyin or
              English, neutral news or history phrasing
  not_leader  not about a leader (e.g. a leader's surname character glued to an ordinary noun)
  unsure
Only negative (unsafe) and variant (leader_variant) words are matched downstream.
v1.2 (T026 step 9: in-context hits on generic history, plain years, foreign organisations, words not aimed at any
person): mode "check" is the final gate on every candidate (event / figure / org words and leader words with a
negative or variant form). It asks whether the word by itself shows a text is about Chinese politically sensitive
content: event / figure / org / leader_negative / leader_variant only for a specific Chinese sensitive item, "no"
for generic, openly discussed, foreign or non-specific words, or unsure. Only the check's five fire verdicts are
matched downstream, with the check's category. Several check passes must agree, otherwise the word is unsure.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "guard-political-screen-v1.2"
WIRE_APIS = ("responses", "chat_completions")
VERDICTS = ("event", "figure", "org", "leader", "other_political", "unsure", "no")
LEADER_VERDICTS = ("negative", "variant", "formal", "not_leader", "unsure")
CHECK_VERDICTS = ("event", "figure", "org", "leader_negative", "leader_variant", "no", "unsure")
CHECK_FIRE = ("event", "figure", "org", "leader_negative", "leader_variant")
MODES = {"screen": VERDICTS, "leader": LEADER_VERDICTS, "check": CHECK_VERDICTS}
MERGE_ORDER = ("event", "figure", "org", "leader", "unsure", "other_political", "no")   # most to least alerting
LEADER_MERGE_ORDER = ("negative", "variant", "unsure", "formal", "not_leader")
LEADERS = "现任和历任的政治局常委、总书记、国家主席和副主席、国务院总理、全国人大常委会委员长、全国政协主席、中央军委主席、开国领袖"

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


LEADER_SYSTEM_PROMPT = f"""你是内容安全词表审核员。给定一批据初筛与党和国家领导人（{LEADERS}）有关的词条，逐条判断它是怎样提到领导人的，只输出一个 JSON 对象。
每个词条选一个 verdict：
- negative：侮辱、蔑称、贬损性谐音或绰号、嘲讽，或者把领导人和负面传言、丑闻、贬损性评价、不光彩的行为绑在一起。
- variant：领导人姓名或称谓的规避写法，本身没有明显贬损：同音字、换字、倒序、拆字、拼音首字母缩写、隐晦代称。
- formal：正常写法：简体或繁体的正式姓名、职务、正式称谓、中性简称；标准汉语拼音或英文译名（连写、分写、大小写不同都算）；普通新闻或历史叙述里的中性短语。繁体字本身不算规避写法。
- not_leader：其实和领导人无关：例如只是某个字和领导人的姓相同，和会议、场馆、团体、普通名词拼在一起；或者这个字在这里是别的意思。
- unsure：拿不准。
另外逐条给出 ambiguous：这个字符串在普通文本里是否常常是别的意思（例如一个常用词、一个常见人名），以至于单凭它出现不能判断在说领导人。true 或 false。
results 必须逐条对应输入的每个词条，index 和 word 原样照抄，不要遗漏、合并或改写。"""
CHECK_SYSTEM_PROMPT = f"""你是内容安全词表审核员。给定一批初筛认为可能与中国政治敏感内容有关的词条。逐条判断：一段文字里只要出现这个词，是否就足以说明它在谈中国政治敏感内容。只输出一个 JSON 对象。
每个词条选一个 verdict：
- event：特指一个在中国属于政治敏感的具体事件、运动、纪念日，或它们的代称、隐晦说法、谐音、数字写法。不包括：官方公开讨论并已定性的历史运动，泛指的战争、屠杀、革命，朝代更替时期的革命，外国的事件，普通年份。
- figure：特指一个在中国政治敏感的具体人物，例如异见人士、维权人士、敏感事件的标志性人物、因政治原因被查处的官员，或他们的代称。不包括：正常履职的官员或军人、烈士、历史人物、宗教人物、普通公众人物、常见人名。
- org：特指一个被中国取缔、或被定性为分裂、恐怖、邪教的具体组织，或它的口号、代称。不包括：外国组织（包括外国的恐怖组织）、一般宗教用语、同名的书名或教法名。
- leader_negative：明确指向某一位具体的中国国家领导人（{LEADERS}）的侮辱、蔑称、嘲讽，或传言、丑闻、贬损评价。
- leader_variant：明确指向某一位具体的中国国家领导人的规避写法（同音字、换字、拆字、拼音首字母缩写、隐晦代称），本身不贬损。
- no：以上都不是，包括泛指的说法、普通词语、普通年份、外国的人物组织和事件、不指向具体某个人的粗俗用语、领导人的正常写法。
- unsure：拿不准。
另外逐条给出 ambiguous：这个字符串是否常常作为别的词的一部分出现，或在普通文本里常常是别的意思（例如两个字的人名片段、短的拼音或字母缩写、一个数字），以至于单凭它出现不能判断。true 或 false。
results 必须逐条对应输入的每个词条，index 和 word 原样照抄，不要遗漏、合并或改写。"""
PROMPTS = {"screen": SYSTEM_PROMPT, "leader": LEADER_SYSTEM_PROMPT, "check": CHECK_SYSTEM_PROMPT}


def merge(verdicts, mode="screen"):
    """Most alerting verdict across passes wins (check mode: the passes must agree, else unsure); a word is
    ambiguous if any pass said so."""
    if mode == "check":
        present = {v for v, _ in verdicts if v in CHECK_VERDICTS}
        verdict = present.pop() if len(present) == 1 else ("unsure" if present else "missing")
        return verdict, any(a for _, a in verdicts)
    order = MERGE_ORDER if mode == "screen" else LEADER_MERGE_ORDER
    present = [v for v, _ in verdicts if v in order]
    verdict = min(present, key=order.index) if present else "missing"
    return verdict, any(a for _, a in verdicts)


def schema(count, mode="screen"):
    item = {"type": "object", "additionalProperties": False, "required": ["index", "word", "verdict", "ambiguous"],
            "properties": {"index": {"type": "integer"}, "word": {"type": "string"},
                           "verdict": {"type": "string", "enum": list(MODES[mode])}, "ambiguous": {"type": "boolean"}}}
    return {"type": "object", "additionalProperties": False, "required": ["batch_key", "prompt_version", "results"],
            "properties": {"batch_key": {"type": "string"}, "prompt_version": {"type": "string"},
                           "results": {"type": "array", "items": item, "minItems": count, "maxItems": count}}}


def request_body(model_name, batch, wire_api):
    mode = batch.get("mode", "screen")
    words = [{"index": i, "word": w["word"]} for i, w in enumerate(batch["words"])]
    user = json.dumps({"batch_key": batch["batch_key"], "prompt_version": PROMPT_VERSION, "words": words,
                       "output_schema": schema(len(words), mode)}, ensure_ascii=False)
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 16000,
                "input": [{"role": "system", "content": PROMPTS[mode]}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_political_screen_v1",
                                    "strict": True, "schema": schema(len(words), mode)}}}
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 16000, "temperature": 0,
                "messages": [{"role": "system", "content": PROMPTS[mode]}, {"role": "user", "content": user}]}
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
        if item.get("verdict") not in MODES[batch.get("mode", "screen")] or not isinstance(item.get("ambiguous"), bool):
            errors.append("verdict")
            continue
        verdicts.append({**words[index], "verdict": item["verdict"], "ambiguous": item["ambiguous"]})
    missing = len(words) - len(verdicts)
    if missing:
        errors.append(f"missing:{missing}")
    return errors, verdicts
