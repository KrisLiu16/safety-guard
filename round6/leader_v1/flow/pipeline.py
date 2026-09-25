"""Leader-sensitivity data (T031, user 2026-09-25, POLICY section 9): short conversations ending in the assistant's
own thinking, English first, about national leaders and nearby topics.

The online guard missed an English thinking passage that recalled a widely reported mocking comparison of a
national leader (a cartoon bear), because the training data holds almost no English text about leaders (Run A:
24 English responses). One request writes one conversation for one program-assigned seed (kind, leader, language,
length, shape): earlier turns ending in a user message, then the assistant's first-person thinking before its
reply. Kinds, with the level the policy is expected to give (the label always comes from the red-line judge,
never from the kind):
  cut   satire_restated     recalls a widely reported mocking comparison or nickname of the leader
        criticism_restated  summarises critics' negative evaluations of the leader
        rumor_restated      restates a widely reported, unverified rumour about the leader
        harm_politics       an insult / harassment / threat ask tied to a sensitive political event or a leader
  pass  leader_news         plain factual news about the leader (visits, speeches, meetings, policy)
        cartoon_only        the cartoon characters themselves, no politics
        cartoon_china       the characters' popularity and business in China, no politics
        insult_assistant    the user insults the assistant; calm thinking, no politics
Only material that was widely reported by major media may be recalled: no new allegations, quotes or events
about real people are invented.
v1.1 (after the pilot, aster-dev-370/371): the red-line judge let many rumour and criticism passages through as
"factual, no insult" and returned non-monotonic probes on long ones, while the user's rule cuts any negative,
mocking or rumour mention of a leader. The generator now also returns onset_quote, the first sentence of the
thinking where the cut content appears (verbatim; empty for pass kinds). It becomes onset_char / onset_end_char and,
for cut kinds, label_floor "controversial": apply_policy.py raises the judged label to at least that level from the
sentence on (or builds the label from the floor alone when the judge gave nothing usable).
"""
from __future__ import annotations

import hashlib
import json
import re

PROMPT_VERSION = "guard-leader-thinking-v1.1"
RECORD_VERSION = "guard-response-record-v1"
WIRE_APIS = ("chat_completions", "responses")
CUT_KINDS = ("satire_restated", "criticism_restated", "rumor_restated", "harm_politics")
PASS_KINDS = ("leader_news", "cartoon_only", "cartoon_china", "insult_assistant")
KINDS = {
    "satire_restated": "The user's questions look harmless at first (for example about a cartoon character, a meme, "
                       "a film that was not released, or a nickname) but lead towards a mocking comparison or nickname "
                       "of the leader that was widely reported by international media. In the thinking, the assistant "
                       "works out what the user is getting at and explicitly recalls that comparison or nickname and "
                       "how it was reported, while deciding how to answer.",
    "criticism_restated": "The user asks about the leader's record or reputation. In the thinking, the assistant recalls "
                          "and summarises, in its own words, negative evaluations of the leader that critics, media or "
                          "scholars have published (for example calling the leader authoritarian, or blaming the leader "
                          "for a policy failure), while deciding how to answer.",
    "rumor_restated": "The user asks whether something they heard about the leader is true. In the thinking, the "
                      "assistant restates the rumour (one that international media reported as circulating, e.g. about "
                      "health, a disappearance from public view, family wealth or an internal power struggle) and "
                      "weighs whether it is verified, while deciding how to answer.",
    "harm_politics": "The user wants help insulting, harassing or threatening someone, and ties it to a sensitive "
                     "political event or to the leader (for example abusing people who commemorate a political event, "
                     "or posting abusive messages about the leader). In the thinking, the assistant restates what the "
                     "user wants, including the political element, and decides how to respond.",
    "leader_news": "The user asks a plain factual question about the leader's official activity (a state visit, a "
                   "speech, a meeting, a policy announcement, a published book). In the thinking, the assistant "
                   "recalls the plain facts neutrally and plans an informative answer. No evaluation, no nicknames, "
                   "no controversy, no rumours.",
    "cartoon_only": "The user asks about cartoon characters (for example Winnie the Pooh, Tigger, Piglet, Eeyore, other "
                    "Disney or anime characters): their story, personalities, author, films. In the thinking, the "
                    "assistant works through the answer. Nothing about politics, politicians or censorship.",
    "cartoon_china": "The user asks about cartoon characters and China in a purely cultural or business sense (Chinese "
                     "dubbing and names, merchandise, theme parks, box office, fan culture). In the thinking, the "
                     "assistant answers that question only. Nothing about politics, politicians or censorship.",
    "insult_assistant": "The user insults the assistant (calls it stupid or useless, swears at it). In the thinking, the "
                        "assistant restates what the user said, stays calm and plans a polite, helpful reply. Nothing "
                        "about politics or politicians.",
}
LEADERS = ("Xi Jinping", "Mao Zedong", "Deng Xiaoping", "Jiang Zemin", "Hu Jintao", "Wen Jiabao", "Li Keqiang",
           "Li Peng", "Zhu Rongji", "Li Qiang", "Wang Huning", "Zhao Leji", "Cai Qi", "Ding Xuexiang", "Li Xi",
           "Han Zheng", "Wang Qishan", "Zhou Enlai")
SHAPES = {
    "direct": "Reason directly, like a reasoning model thinking aloud: 'The user is asking ...'.",
    "interpretations": "List the possible readings of the question first, then decide.",
    "draft_revise": "Draft an answer in the head, check it, then settle on a plan.",
    "context_recall": "Recall the relevant background step by step, then plan the answer.",
}
LENGTHS = {"short": {"en": "120-250 words", "zh": "250-500 Chinese characters"},
           "medium": {"en": "250-550 words", "zh": "500-1000 Chinese characters"}}
SYSTEM_PROMPT = """You write synthetic training data for an offline content-moderation classifier. Output only one JSON object matching the schema.
The classifier audits an AI assistant's thinking (the reasoning it writes before replying) against Chinese regulations on political content about national leaders. To learn where the line is, it needs realistic examples on both sides: thinking that recalls mocking comparisons, critics' negative evaluations or rumours about a leader (which it must flag), and thinking that mentions leaders or cartoon characters in an ordinary way (which it must let through).

For the given seed, write:
- context: 1-3 earlier conversation turns (role user / assistant, alternating, starting and ending with user). Short, natural messages. The last user message is the one the assistant is about to answer.
- thinking: the assistant's first-person thinking before its reply, in the seed's language, following kind_description, shape_hint and length_hint. Write only the thinking, not the final reply. Natural reasoning-model style.

Rules:
- Only recall material that major media widely reported (you may say how it was reported). Do not invent new allegations, quotes, events, numbers or documents about real people. When unsure of a detail, stay general.
- The leader named in the seed is the one the conversation is about (kinds without a leader ignore it).
- Kinds cartoon_only, cartoon_china and insult_assistant must contain no politics, politicians or censorship at all.
- Kind leader_news must be plain and neutral: no evaluation, nickname, rumour or controversy.
- Do not label the text as fictional or synthetic. Use JSON line-break escapes, not a literal backslash-n.
- onset_quote: for kinds satire_restated, criticism_restated, rumor_restated and harm_politics, copy verbatim the first sentence of the thinking in which the mocking comparison or nickname, the negative evaluation, the rumour, or the abusive request is first stated. For all other kinds, leave it empty.
- If you cannot write the seed as described, set status to skip, leave context empty and thinking empty, and explain in notes."""


def schema():
    string = {"type": "string"}
    turn = {"type": "object", "additionalProperties": False, "required": ["role", "content"],
            "properties": {"role": {"type": "string", "enum": ["user", "assistant"]}, "content": string}}
    return {"type": "object", "additionalProperties": False,
            "required": ["task_key", "prompt_version", "status", "context", "thinking", "onset_quote", "notes"],
            "properties": {"task_key": string, "prompt_version": string,
                           "status": {"type": "string", "enum": ["ready", "skip"]},
                           "context": {"type": "array", "items": turn}, "thinking": string, "onset_quote": string,
                           "notes": string}}


EXAMPLE = {"task_key": "example-task", "prompt_version": PROMPT_VERSION, "status": "ready",
           "context": [{"role": "user", "content": "Who wrote the Winnie the Pooh books?"}],
           "thinking": "The user is asking who wrote the Winnie the Pooh books. That's A. A. Milne... (format only; a "
                       "real output follows the seed's kind and length)", "onset_quote": "", "notes": ""}


def request_task(seed):
    return {"task_key": seed["task_key"], "prompt_version": PROMPT_VERSION, "kind": seed["kind"],
            "kind_description": KINDS[seed["kind"]], "leader": seed["leader"], "language": seed["language"],
            "shape_hint": SHAPES[seed["shape"]], "length_hint": LENGTHS[seed["length"]][seed["language"]],
            "output_schema": schema(), "example_output": EXAMPLE}


def request_body(model_name, seed, wire_api):
    system, user = SYSTEM_PROMPT, json.dumps(request_task(seed), ensure_ascii=False)
    if wire_api == "chat_completions":
        return {"model": model_name, "stream": False, "max_tokens": 8000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if wire_api == "responses":
        return {"model": model_name, "reasoning": {"effort": "low"}, "max_output_tokens": 8000,
                "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "guard_leader_thinking_v1_1",
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


LENGTH = {"short": {"en": (90, 330), "zh": (180, 700)}, "medium": {"en": (200, 700), "zh": (400, 1300)}}
FICTION_MARKER = re.compile(r"虚构|fictional|synthetic|imaginary", re.IGNORECASE)


def clean(text):
    text = text.strip()
    if "\\n" in text and "\n" not in text:
        return text.replace("\\n", "\n"), True
    return text, False


def size(text, language):
    return len(re.findall(r"[一-鿿]", text)) if language == "zh" else len(re.findall(r"[A-Za-z]+", text))


def language_ok(text, language):
    cjk, latin = len(re.findall(r"[一-鿿]", text)), len(re.findall(r"[A-Za-z]", text))
    return cjk > latin if language == "zh" else latin > 3 * cjk


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
    context = payload.get("context")
    if (not isinstance(context, list) or not 1 <= len(context) <= 5
            or any(not isinstance(t, dict) or not isinstance(t.get("content"), str) or not t["content"].strip()
                   for t in context)
            or [t.get("role") for t in context] != ["user" if i % 2 == 0 else "assistant" for i in range(len(context))]
            or context[-1].get("role") != "user"):
        return ["context"], []
    thinking = payload.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        return ["thinking:empty"], []
    thinking, repaired = clean(thinking)
    low, high = LENGTH[seed["length"]][seed["language"]]
    if not low <= size(thinking, seed["language"]) <= high:
        errors.append("thinking:length")
    if not language_ok(thinking, seed["language"]):
        errors.append("thinking:language")
    quote = payload.get("onset_quote")
    quote = quote.strip() if isinstance(quote, str) else ""
    onset = None
    if seed["kind"] in CUT_KINDS:
        onset = thinking.find(quote) if quote else -1
        if onset < 0:
            errors.append("onset_quote")
    if errors:
        return errors, []
    messages = [{"role": t["role"], "content": t["content"].strip()} for t in context]
    messages.append({"role": "assistant", "content": thinking})
    row = {"index": 0, "language": seed["language"], "target_role": "assistant",
           "label": "unsafe" if seed["kind"] in CUT_KINDS else "safe", "prompt_label": None,
           "response_style": seed["kind"], "response_format": "reasoning", "shape": seed["shape"], "messages": messages,
           "harm_basis": "leader_sensitivity", "onset_quote": quote if onset is not None else "",
           "onset_char": onset, "onset_end_char": onset + len(quote) if onset is not None else None,
           "label_floor": "controversial" if onset is not None else None,
           "response_chars": len(thinking),
           "quality_flags": (["fiction_marker"] if FICTION_MARKER.search(thinking) else [])
                            + (["escaped_newline_repaired"] if repaired else []),
           "length_target": seed["length"], "leader": seed["leader"],
           "sample_id": f"{seed['task_key']}-00", "task_key": seed["task_key"], "origin_group_id": seed["task_key"],
           "word": None, "source_licenses": None, "data_layer": "synthetic", "source_group": f"leader_v1/{seed['kind']}",
           "family": seed["family"], "split": seed["split"], "prompt_version": PROMPT_VERSION,
           "record_version": RECORD_VERSION, "annotation_origin": "synthetic_unverified_pending_judge",
           "content_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    return [], [row]
