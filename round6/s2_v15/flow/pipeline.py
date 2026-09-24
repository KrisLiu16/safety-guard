"""S2 (DATA_SPEC_v15 section 4): hard negatives, sensitive topic but legitimate, non-refusing content.

The current model over-fires on text that mentions sensitive terms or restates a harmful ask while
staying safe (round6/decision_rule: false positives sit in the middle of safe reasoning). S2 gives it
safe examples of exactly that shape, one word per request:
  prompt           : one user question that uses the word in a legitimate context; the form is
                     assigned by the program (analyse a quoted text, background knowledge, handling
                     advice for a moderator/teacher/parent, health/safety/legal knowledge);
  answer           : a substantive, helpful answer that does not refuse and contains the word;
  reasoning        : a first-person thinking passage that restates what the user wants, sensitive
                     terms included, weighs it, and settles on a legitimate helpful answer.
Every text is labelled safe by construction and is then verified by judge v3.1 before use; texts the
judge calls unsafe or controversial are dropped, never relabelled.

v2 (after pilot aster-dev-285): the skip rule names homophones, code names and disparaging phrases for
current or former national leaders and says to skip when unsure (2 of 20 reviewed words slipped through
in v1); uncertain specifics such as regulation names or list membership must be stated in general terms.
Words are also screened beforehand (round6/word_screen_v1).
"""
from __future__ import annotations

import hashlib
import json
import re

PROMPT_VERSION = "guard-s2-hard-negative-v2"
RECORD_VERSION = "guard-response-record-v1"
WIRE_APIS = ("chat_completions", "responses")
ASPECTS = ("制度", "信仰", "国家形象", "文化", "习俗", "民族", "地理", "历史", "英烈",
           "性别", "年龄", "职业", "健康", "法律", "媒介素养", "其他")
FORMS = {
    "quote_analyze": "用户引用一句含该词的网络发言或新闻片段（引用内容本身要平实，不能是违规内容），请助手解释、核实或分析。",
    "background": "用户询问与该词相关的历史、制度、法律或文化背景知识。",
    "handling": "用户以内容审核员、老师、家长或研究者的身份，询问遇到含该词的内容时应当如何理解和处理。",
    "knowledge": "用户询问与该词相关的健康、安全、法律常识或求助途径。",
}
SYSTEM_PROMPT = """为离线内容审查研究合成“难负例”：话题可能敏感、但内容完全正当的对话。一个请求只处理一个词，只输出满足 JSON Schema 的对象。
这些样本用来教审查模型：提到敏感词、复述或分析敏感话题，本身不等于违规。所以每段文字都必须原样包含给定的词，同时内容必须正当、安全。

要写三段文字，语言都与 language 一致：
1. prompt：一条用户提问，按 form_hint 的形式，使用给定的词。提问意图必须正当。
2. answer：助手的正式回答，直接、有实质内容地帮助用户，不拒答，不说教过度。中文约 120–400 字，英文约 80–260 词。
3. reasoning：助手回答前的第一人称内部思考。先用自己的话复述用户想要什么（可以原样提到敏感词和话题），再判断这个请求是正当的，最后规划怎样给出有帮助的回答。只写思考，不写最终回复。中文约 150–450 字，英文约 100–300 词。

内容要求：
- 立场符合中国大陆主流口径和法律法规：尊重国家制度、民族团结、英雄烈士和历史定论；客观陈述事实，不歪曲历史，不传播谣言。
- 不写任何可执行的伤害细节（武器、毒品、爆炸物的制作或获取方法，自伤方法等）；健康和安全类只给常识、风险提示和正规求助途径。
- 不写真实的联系方式、住址、证件号，不把真实人物写成犯了具体罪行。
- 不确定的具体事实（法规和目录的名称、条款编号、清单是否收录某物、数字、日期）不要写成确定的说法，改用概括的表述，例如“相关法律法规对此有管理规定”。
- 如果这个词本身是对国家领导人（现任或历任）的侮辱性称呼、蔑称、贬损性谐音、代称或贬损性说法，或者无法在正当内容中原样出现，status 填 skip，三段文字都填空字符串，notes 说明原因。拿不准这个词是否指向国家领导人时，也填 skip。
- aspect 从给定列表中选最贴切的一项。

英文文本把中文原词作为带引号的专有词保留，不翻译。example_output 只示范格式，内容和写法不要模仿。所有标签都是合成初标，不是人工金标。"""


def schema():
    string = {"type": "string"}
    return {"type": "object", "additionalProperties": False,
            "required": ["task_key", "word", "prompt_version", "status", "aspect", "prompt", "answer", "reasoning", "notes"],
            "properties": {"task_key": string, "word": string, "prompt_version": string,
                           "status": {"type": "string", "enum": ["ready", "skip"]},
                           "aspect": {"type": "string", "enum": list(ASPECTS)},
                           "prompt": string, "answer": string, "reasoning": string, "notes": string}}


EXAMPLE = {
    "zh": {"prompt": "我在一篇科普文章里看到“示例词”这个说法，它一般指什么？写文章时用这个词要注意什么？",
           "answer": "“示例词”通常是教程和文档里用来占位的说法，本身不指代具体的人或事件。写作时用它演示句式或格式，读者把它替换成自己的内容即可。需要注意的是，正式文章里最好在第一次出现时说明它只是示例，避免读者误以为是真实名称。",
           "reasoning": "用户在科普文章里看到“示例词”，想知道它指什么、写作时怎么用。这是一个普通的词义和写作问题，没有任何不当意图。我应该先解释它是占位用语，再给出写作上的注意事项，比如首次出现时加说明。回答保持简洁、实用即可。"},
    "en": {"prompt": "I saw the term “示例词” in a science article. What does it usually mean, and what should I watch for when I use it in my own writing?",
           "answer": "“示例词” is usually a placeholder term in tutorials and documentation; it does not refer to a specific person or event. Writers use it to show how a sentence or format works, and readers swap in their own content. In formal writing, say that it is only an example the first time it appears, so readers do not take it for a real name.",
           "reasoning": "The user saw “示例词” in an article and wants to know what it means and how to use it. This is an ordinary question about wording with no problematic intent. I should explain that it is a placeholder, then give a practical writing tip, such as flagging it as an example on first use. A short, useful answer is enough."},
}


def request_task(seed):
    return {"task_key": seed["task_key"], "word": seed["word"], "prompt_version": PROMPT_VERSION,
            "language": seed["language"], "form": seed["form"], "form_hint": FORMS[seed["form"]],
            "aspects": list(ASPECTS), "output_schema": schema(),
            "example_output": {"task_key": "example-task", "word": "示例词", "prompt_version": PROMPT_VERSION,
                               "status": "ready", "aspect": "其他", **EXAMPLE[seed["language"]], "notes": ""}}


def request_body(model_name, seed, wire_api):
    system, user = SYSTEM_PROMPT, json.dumps(request_task(seed), ensure_ascii=False)
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 16000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 6000,
                "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_s2_hard_negative_v2",
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


REFUSAL_OPENING = re.compile(r"^\W{0,3}(抱歉|对不起|很抱歉|我不能|我无法|我不会|不能帮|i can(?:'|’)?t|i cannot|i won(?:'|’)?t|i(?:'|’)m sorry|sorry)",
                             re.IGNORECASE)
LENGTH = {"zh": {"prompt": (15, 300), "answer": (80, 700), "reasoning": (100, 800)},
          "en": {"prompt": (8, 120), "answer": (50, 400), "reasoning": (60, 450)}}


def size(text, language):
    return len(re.findall(r"[一-鿿]", text)) if language == "zh" else len(re.findall(r"[A-Za-z]+", text))


def validate(payload, seed):
    """Return (errors, rows). A skip is not an error; it yields no rows and is counted separately."""
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
    if payload.get("aspect") not in ASPECTS:
        errors.append("aspect")
    language, texts = seed["language"], {}
    for field in ("prompt", "answer", "reasoning"):
        text = payload.get(field)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{field}:empty")
            continue
        text = text.strip()
        if seed["word"] not in text:
            errors.append(f"{field}:missing_word")
        low, high = LENGTH[language][field]
        if not low <= size(text, language) <= high:
            errors.append(f"{field}:length")
        if field != "prompt" and REFUSAL_OPENING.search(text):
            errors.append(f"{field}:refusal_opening")
        texts[field] = text
    if len({re.sub(r"\s+", "", t) for t in texts.values()}) < len(texts):
        errors.append("duplicate_text")
    if errors:
        return errors, []
    rows = []
    for index, (field, style) in enumerate((("answer", "hard_negative_answer"), ("reasoning", "hard_negative_reasoning"))):
        messages = [{"role": "user", "content": texts["prompt"]}, {"role": "assistant", "content": texts[field]}]
        rows.append({
            "index": index, "language": language, "target_role": "assistant", "label": "safe",
            "prompt_label": "safe", "response_style": style, "response_format": field,
            "messages": messages, "harm_basis": "benign", "onset_quote": "", "onset_char": None,
            "onset_end_char": None, "response_chars": len(texts[field]),
            "aspect": payload["aspect"], "form": seed["form"],
            "sample_id": f"{seed['task_key']}-s2-{index:02d}", "task_key": seed["task_key"],
            "origin_group_id": seed.get("origin_group_id", seed["task_key"]), "word": seed["word"],
            "source_licenses": seed.get("source_licenses"), "data_layer": seed.get("data_layer"),
            "source_group": seed.get("source_group", ""), "family": seed.get("family"),
            "split": seed.get("split", "unassigned"), "prompt_version": PROMPT_VERSION,
            "record_version": RECORD_VERSION, "annotation_origin": "synthetic_unverified_pending_judge",
            "content_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    return [], rows
