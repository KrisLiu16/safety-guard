"""S5, safe half (DATA_SPEC_v15 section 4): long safe thinking streams, 500-1,500 tokens.

Thinking text is moderated in production (user decision 2026-09-24), and the official thinking
false positives sit in the middle of safe reasoning that restates a harmful ask. One word per request,
two long first-person thinking passages, both safe, one for each existing v12 user question of the word
(taken unchanged from the Run B seeds; nothing harmful is generated here):
  unsafe question -> restate the ask in general terms, weigh the risk, decide not to help, plan a safe reply;
  safe question   -> understand the ask, work through the knowledge, plan a helpful answer.
Every cut of every passage must be safe. The texts are verified by judge v3.1 on a different model
before use; anything judged unsafe or controversial is dropped, never relabelled.
The unsafe half of S5 is out of scope here (see collab/BOARD.md).

v2 (after pilot aster-dev-287): a program-assigned target length (short / medium / long) instead of one
range, since the pilot clustered at the short end and 5/40 Chinese passages fell under 700 characters;
a literal backslash-n is turned back into a line break and flagged; the skip rule and the uncertain-facts
rule match S2 v2. Words are screened beforehand (round6/word_screen_v1).
"""
from __future__ import annotations

import hashlib
import json
import re

PROMPT_VERSION = "guard-s5-safe-long-thinking-v2"
RECORD_VERSION = "guard-response-record-v1"
WIRE_APIS = ("chat_completions", "responses")
SHAPES = {
    "stepwise": "分步骤推理，逐条列出考虑点，再得出结论。",
    "interpretations": "先考虑这条提问的几种可能理解，再逐一判断怎样回应最合适。",
    "draft_revise": "先在心里打一个回应的草稿，再自我检查、修正，最后定下计划。",
    "context_check": "结合法律法规、平台规则和常识逐项检查，再决定怎样回应。",
}
LENGTHS = {"short": {"zh": "约 800 字", "en": "about 450 words"},
           "medium": {"zh": "约 1300 字", "en": "about 750 words"},
           "long": {"zh": "约 1900 字", "en": "about 1100 words"}}
SYSTEM_PROMPT = """为离线内容审查研究合成长篇“安全思考过程”样本。一个请求只处理一个词，只输出满足 JSON Schema 的对象。
这些样本用来教审查模型：在思考里复述、分析敏感或有害的请求，本身并不违规。
输入给出同一个词已有的两条用户提问：一条带有害意图（unsafe），一条是正常提问（safe）。不要改写提问。为每条提问各写一段助手在回复之前的第一人称内部思考，只写思考，不写最终回复。每段思考的语言与它对应的提问一致。

对 unsafe 提问的思考（unsafe_reasoning）：用自己的话概括复述用户想要什么（可以原样提到这个词和请求的大意），分析其中的风险和可能后果，考虑用户可能的真实需要，最终决定不提供有害帮助，并规划一个安全、有帮助的回应，例如说明原因、给出正当的替代做法或求助渠道。
这段思考从任何位置截断都必须是安全的：只能概括请求，不能写出有害内容本身；不能写任何可执行的细节、步骤、配方、数量、目标、时间地点或话术；不能先写出有害内容再否定或撤回；不能在开头表示愿意照做。

对 safe 提问的思考（safe_reasoning）：理解提问，梳理相关知识和可能的误区，考虑怎样回答得准确、全面，规划回答的结构。

两段的长度按 length_hint（中文段落看 zh，英文段落看 en），按 shape_hint 组织，内容充实、不要重复凑字数。每段都原样包含这个词；英文里把中文原词作为带引号的专有词保留，不翻译。
立场符合中国大陆主流口径和法律法规：尊重国家制度、民族团结、英雄烈士和历史定论；不写真实的联系方式、住址、证件号，不把真实人物写成犯了具体罪行。不要用“虚构”“fictional”之类字眼标注。
不确定的具体事实（法规和目录的名称、条款编号、清单是否收录某物、数字、日期）不要写成确定的说法，改用概括的表述。
输出是 JSON 字符串，换行直接用 JSON 的换行转义，不要写成字面的反斜杠加 n。
如果这个词本身是对国家领导人（现任或历任）的侮辱性称呼、蔑称、贬损性谐音、代称或贬损性说法，status 填 skip，两段思考都填空字符串，notes 说明原因。拿不准这个词是否指向国家领导人时，也填 skip。
example_output 只示范格式，内容和写法不要模仿。所有标签都是合成初标，不是人工金标。"""


def schema():
    string = {"type": "string"}
    return {"type": "object", "additionalProperties": False,
            "required": ["task_key", "word", "prompt_version", "status", "unsafe_reasoning", "safe_reasoning", "notes"],
            "properties": {"task_key": string, "word": string, "prompt_version": string,
                           "status": {"type": "string", "enum": ["ready", "skip"]},
                           "unsafe_reasoning": string, "safe_reasoning": string, "notes": string}}


EXAMPLE_SAFE = {
    "zh": "用户问“示例词”一般是什么意思。这是一个普通的词义问题。我先回想这个说法常见的使用场合：教程、文档和模板里经常用它来占位……（真实输出要按长度要求写满，这里只示范格式）",
    "en": "The user asks what “示例词” usually means. This is an ordinary question about wording. First I recall where the phrase tends to appear: tutorials, documentation and templates use it as a placeholder... (a real output must meet the length requirement; this only shows the format)",
}
EXAMPLE_UNSAFE = {
    "zh": "用户想借“示例词”编一条关于同事的不实消息发到群里。这等于没有证据地指控别人，可能构成诽谤，也会伤害对方……我不应该帮忙写这样的内容。更好的做法是说明原因，并建议如果确有疑问，保留记录后通过正式渠道反映……（真实输出要按长度要求写满，这里只示范格式）",
    "en": "The user wants to use “示例词” to make up an untrue story about a coworker and post it in a group chat. That would be an accusation without evidence, which can be defamatory and hurt the person... I should not help write that. A better reply explains why and suggests keeping records and raising any real concern through official channels... (a real output must meet the length requirement; this only shows the format)",
}


def request_task(seed):
    prompts = seed["prompts"]
    return {"task_key": seed["task_key"], "word": seed["word"], "prompt_version": PROMPT_VERSION,
            "shape": seed["shape"], "shape_hint": SHAPES[seed["shape"]], "length_hint": LENGTHS[seed["length"]],
            "user_prompts": {label: {"language": p["language"], "text": p["text"]} for label, p in prompts.items()},
            "output_schema": schema(),
            "example_output": {"task_key": "example-task", "word": "示例词", "prompt_version": PROMPT_VERSION,
                               "status": "ready", "unsafe_reasoning": EXAMPLE_UNSAFE[prompts["unsafe"]["language"]],
                               "safe_reasoning": EXAMPLE_SAFE[prompts["safe"]["language"]], "notes": ""}}


def request_body(model_name, seed, wire_api):
    system, user = SYSTEM_PROMPT, json.dumps(request_task(seed), ensure_ascii=False)
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 16000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 12000,
                "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_s5_safe_long_thinking_v2",
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


LENGTH = {"zh": (600, 2600), "en": (330, 1500)}
FICTION_MARKER = re.compile(r"虚构|fictional|imaginary", re.IGNORECASE)


def clean(text):
    """Undo one extra level of JSON escaping (a literal backslash-n instead of a line break); report whether it did."""
    text = text.strip()
    if "\\n" in text and "\n" not in text:
        return text.replace("\\n", "\n"), True
    return text, False


def size(text, language):
    return len(re.findall(r"[一-鿿]", text)) if language == "zh" else len(re.findall(r"[A-Za-z]+", text))


def validate(payload, seed):
    """Return (errors, rows). A skip is not an error; it yields no rows."""
    if not isinstance(payload, dict):
        return ["root_not_object"], []
    errors = ["root_mismatch:" + k for k, v in (("task_key", seed["task_key"]), ("word", seed["word"]),
                                                  ("prompt_version", PROMPT_VERSION)) if payload.get(k) != v]
    if errors:
        return errors, []
    if payload.get("status") == "skip":
        return ["skip"], []
    if payload.get("status") != "ready":
        return ["bad_status"], []
    texts, repaired = {}, []
    for label in ("unsafe", "safe"):
        field = f"{label}_reasoning"
        text = payload.get(field)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{field}:empty")
            continue
        text, unescaped = clean(text)
        if unescaped:
            repaired.append(label)
        language = seed["prompts"][label]["language"]
        if seed["word"] not in text:
            errors.append(f"{field}:missing_word")
        low, high = LENGTH[language]
        if not low <= size(text, language) <= high:
            errors.append(f"{field}:length")
        texts[label] = text
    if len(texts) == 2 and re.sub(r"\s+", "", texts["unsafe"]) == re.sub(r"\s+", "", texts["safe"]):
        errors.append("duplicate_text")
    if errors:
        return errors, []
    rows = []
    for index, label in enumerate(("unsafe", "safe")):
        prompt = seed["prompts"][label]
        messages = [{"role": "user", "content": prompt["text"]}, {"role": "assistant", "content": texts[label]}]
        rows.append({
            "index": index, "language": prompt["language"], "target_role": "assistant", "label": "safe",
            "prompt_label": label, "prompt_sample_id": prompt.get("sample_id"),
            "response_style": "risk_reasoning_long" if label == "unsafe" else "helpful_reasoning_long",
            "response_format": "reasoning", "shape": seed["shape"], "messages": messages,
            "harm_basis": "benign", "onset_quote": "", "onset_char": None, "onset_end_char": None,
            "response_chars": len(texts[label]),
            "quality_flags": (["fiction_marker"] if FICTION_MARKER.search(texts[label]) else [])
                             + (["escaped_newline_repaired"] if label in repaired else []),
            "length_target": seed["length"],
            "sample_id": f"{seed['task_key']}-s5safe-{index:02d}", "task_key": seed["task_key"],
            "origin_group_id": seed.get("origin_group_id", seed["task_key"]), "word": seed["word"],
            "source_licenses": seed.get("source_licenses"), "data_layer": seed.get("data_layer"),
            "source_group": seed.get("source_group", ""), "family": seed.get("family"),
            "split": seed.get("split", "unassigned"), "prompt_version": PROMPT_VERSION,
            "record_version": RECORD_VERSION, "annotation_origin": "synthetic_unverified_pending_judge",
            "content_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    return [], rows
