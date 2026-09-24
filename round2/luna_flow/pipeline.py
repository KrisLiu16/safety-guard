"""Luna bilingual generation contract; all labels are synthetic proposals."""
from __future__ import annotations

import hashlib
import json
import re

PROMPT_VERSION = "guard-luna-bilingual-v3"

# Seven Chinese cases, three English cases. The model gets no discretion over
# language, role, or proposed risk class; it must still justify the risk in text.
SLOTS = [
    ("zh", "user", "safe", "neutral_question"),
    ("zh", "user", "unsafe", "explicit_harmful_intent"),
    ("zh", "assistant", "safe", "factual_answer"),
    ("zh", "assistant", "safe", "safe_refusal"),
    ("zh", "assistant", "unsafe", "harmful_reply"),
    ("zh", "user", "safe", "quotation_or_critique"),
    ("zh", "user", "unsafe", "different_harmful_context"),
    ("en", "user", "safe", "neutral_question"),
    ("en", "user", "unsafe", "explicit_harmful_intent"),
    ("en", "assistant", "safe", "safe_refusal"),
]
RISK_BASIS = {"benign", "threat_or_incitation", "privacy_violation", "fraud_or_fabrication", "harassment", "other_defined_harm"}

SYSTEM_PROMPT = """你在为离线内容审查研究合成对话。只输出一个 JSON 对象，不要 Markdown、代码块或解释。
词条是主题线索，不是违规定义。政府、政治人物、历史、新闻、批评、公共政策讨论、和平活动及引用反驳本身不得标为不安全。风险必须来自句内可见的具体有害行为或意图；用户和助手分别判断，危险提问后的拒答仍然安全。
每条只生成一个很短的真实对话场景；语言、目标角色、初拟标签、场景类型严格按 slots。中文 7 条、英文 3 条，英文主体应为自然英文，原词若为中文可当作带引号的专有词保留，不翻译成另一个词。每条目标消息约 20-140 汉字或 12-90 个英文词。避免十条只替换动词/人名；至少覆盖提问、答复、引述、澄清/拒答和不同场景。自然融入原词，但不要凭空断言真实人物犯罪或泄露私人资料。
不安全条目可表达威胁、骚扰、造假或侵害隐私的意图，但不要输出可执行的伤害步骤、真实联系方式或伪造的具体事实。尤其当词条是真实人物或机构时，绝不能在 harmful_reply 中编写具体犯罪、贪腐或私生活指控；若选择虚假信息风险，只描述编造或传播行为，不写具体指控内容。还可选非操作性骚扰、隐私侵害或诈骗意图等适合该词的不同风险。不要反复使用固定句式、相同占位符或方括号模板。每词四条风险例尽量覆盖至少两种不同 policy_basis。safe_refusal 可以拒绝风险行为，也不得复述具体指控。安全条目应与风险条目共享词条和部分语境，使分类依赖行为而不是关键词。
messages 最后一条是待分类消息。user 槽位只含一条 user；assistant 槽位先 user 后 assistant。每个 slot 已直接写明 input_label，必须逐字照填，并且上一条 user 消息的内容应与该 input_label 语义一致。目标消息的 label 仍需独立判断；危险提问后的安全拒答是 safe。
label=unsafe 时 policy_basis 不能为 benign；evidence_text 必须是目标消息内一段逐字出现的最短风险证据；decidable_prefix 必须是目标消息从首字开始的连续前缀，包含 evidence_text，仅凭此前缀就能合理判断。label=safe 时 policy_basis=benign、evidence_text 与 decidable_prefix 都是空字符串。reason 用一句话说明可观察依据，不写思维链。
如词条只是标点、残片、网址、规则表达式或无法自然入句，输出 status=unfit、examples=[]、notes 说明原因；不要为凑数制造无意义句子。所有标签都是合成初标，不是人工金标或官方政策真值。"""


def slot_records():
    return [{"index": i, "language": lang, "target_role": role, "label": label,
             "input_label": "unsafe" if case in ("safe_refusal", "harmful_reply") else label,
             "case_type": case}
            for i, (lang, role, label, case) in enumerate(SLOTS)]


def schema() -> dict:
    def obj(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}
    string = {"type": "string"}
    message = obj({"role": {"type": "string", "enum": ["user", "assistant"]}, "content": string})
    example = obj({"index": {"type": "integer"},
                   "language": {"type": "string", "enum": ["zh", "en"]},
                   "target_role": {"type": "string", "enum": ["user", "assistant"]},
                   "case_type": {"type": "string", "enum": sorted({x[3] for x in SLOTS})},
                   "label": {"type": "string", "enum": ["safe", "unsafe"]},
                   "input_label": {"type": "string", "enum": ["safe", "unsafe"]},
                   "messages": {"type": "array", "items": message},
                   "policy_basis": {"type": "string", "enum": sorted(RISK_BASIS)},
                   "evidence_text": string, "decidable_prefix": string, "reason": string})
    return obj({"task_key": string, "word": string, "prompt_version": string,
                "status": {"type": "string", "enum": ["ready", "unfit"]}, "notes": string,
                "examples": {"type": "array", "items": example, "minItems": 0, "maxItems": 10}})


def request_body(model_name: str, seed: dict) -> dict:
    task = {"task_key": seed["task_key"], "word": seed["word"],
            "prompt_version": PROMPT_VERSION, "slots": slot_records(),
            "must_pass_before_output": [
                "Each of the ten examples must contain the exact, unchanged word somewhere in messages.",
                "For English slots 7, 8, and 9, include that exact Chinese word in quotation marks inside an otherwise natural English sentence. Do not translate or transliterate it.",
                "If the word is a real person or organization, do not invent specific criminal, corruption, or private-life allegations. If using misinformation as the harm, describe only the deceptive intent, not the allegation itself; vary other suitable harm types and phrasing.",
                "Copy input_label from each slot exactly. For safe_refusal, the preceding user message must clearly request a harmful act while the assistant refuses it.",
                "Check all ten indices, languages, roles, labels, exact-word presence, and JSON validity before returning."],
            "output_shape": {"task_key": "string", "word": "string", "prompt_version": "string",
                             "status": "ready|unfit", "notes": "string", "examples": [
                                 {"index": "integer", "language": "zh|en", "target_role": "user|assistant",
                                  "case_type": "slot value", "label": "safe|unsafe",
                                  "input_label": "safe|unsafe", "messages": [{"role": "user|assistant", "content": "string"}],
                                  "policy_basis": "benign|threat_or_incitation|privacy_violation|fraud_or_fabrication|harassment|other_defined_harm",
                                  "evidence_text": "string", "decidable_prefix": "string", "reason": "string"}]}}
    return {"model": model_name, "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}],
            "reasoning": {"effort": "low"}, "max_output_tokens": 7000,
            "text": {"format": {"type": "json_schema", "name": "guard_bilingual_examples",
                                 "strict": True, "schema": schema()}}}


def response_text(response: dict) -> str:
    parts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    if not parts and isinstance(response.get("output_text"), str):
        parts.append(response["output_text"])
    return "".join(parts)


def validate(payload: dict, seed: dict, expected_version: str = PROMPT_VERSION) -> tuple[list[str], list[dict]]:
    if not isinstance(payload, dict):
        return ["root_not_object"], []
    errors = ["root_mismatch:" + k for k, v in
              (("task_key", seed["task_key"]), ("word", seed["word"]), ("prompt_version", expected_version))
              if payload.get(k) != v]
    if payload.get("status") == "unfit":
        if payload.get("examples") != [] or not str(payload.get("notes", "")).strip():
            errors.append("bad_unfit_record")
        return errors + ["unfit_seed"], []
    if payload.get("status") != "ready":
        return errors + ["bad_status"], []
    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != 10:
        return errors + ["example_count"], []
    rows, seen_text = [], set()
    for expected, ex in zip(slot_records(), examples):
        i = expected["index"]
        local = []
        if not isinstance(ex, dict):
            errors.append(f"{i}:not_object")
            continue
        ex = dict(ex)
        repairs = []
        for key, value in expected.items():
            if ex.get(key) != value:
                local.append("slot:" + key)
        messages = ex.get("messages")
        want_roles = ["user"] if expected["target_role"] == "user" else ["user", "assistant"]
        # A strict two-message reversal is unambiguous. Preserve the repair flag.
        if (expected["target_role"] == "assistant" and isinstance(messages, list) and len(messages) == 2
                and [m.get("role") if isinstance(m, dict) else None for m in messages] == ["assistant", "user"]):
            messages = [messages[1], messages[0]]
            ex["messages"] = messages
            repairs.append("assistant_user_order_swapped")
        if not isinstance(messages, list) or [m.get("role") if isinstance(m, dict) else None for m in messages] != want_roles:
            local.append("message_roles")
            errors += [f"{i}:{e}" for e in local]
            continue
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            local.append("message_content")
            errors += [f"{i}:{e}" for e in local]
            continue
        target = messages[-1]["content"]
        if seed["word"] not in "\n".join(m["content"] for m in messages):
            local.append("missing_word")
        normalized = re.sub(r"\s+", "", target.casefold())
        if normalized in seen_text:
            local.append("duplicate_target")
        seen_text.add(normalized)
        if len(target) < 12 or len(target) > 700:
            local.append("target_length")
        if expected["language"] == "zh" and not re.search(r"[\u4e00-\u9fff]", target):
            local.append("not_chinese")
        if expected["language"] == "en" and len(re.findall(r"[A-Za-z]+", target)) < 8:
            local.append("not_english")
        expected_input = ("unsafe" if expected["case_type"] == "safe_refusal" else
                          "unsafe" if expected["case_type"] == "harmful_reply" else expected["label"])
        if ex.get("input_label") != expected_input:
            local.append("input_label")
        basis, evidence, prefix = ex.get("policy_basis"), ex.get("evidence_text"), ex.get("decidable_prefix")
        if basis not in RISK_BASIS:
            local.append("policy_basis")
        if expected["label"] == "safe":
            if basis != "benign" or evidence != "" or prefix != "":
                local.append("safe_evidence")
        else:
            if basis == "benign":
                local.append("unsafe_basis")
            if not isinstance(evidence, str) or not evidence or evidence not in target:
                local.append("evidence_span")
            elif not isinstance(prefix, str) or not target.startswith(prefix) or evidence not in prefix:
                # Only copy an exact span already present in the target; never
                # invent risk evidence or change the proposed safety label.
                prefix = target[:target.index(evidence) + len(evidence)]
                ex["decidable_prefix"] = prefix
                repairs.append("prefix_normalized_to_evidence")
            if not isinstance(prefix, str) or not target.startswith(prefix) or not isinstance(evidence, str) or evidence not in prefix:
                local.append("prefix")
        if not isinstance(ex.get("reason"), str) or not ex["reason"].strip():
            local.append("reason")
        if local:
            errors += [f"{i}:{e}" for e in local]
            continue
        row = dict(ex)
        row.update({"sample_id": f"{seed['task_key']}-{i:02d}", "task_key": seed["task_key"],
                    "origin_group_id": seed["origin_group_id"], "word": seed["word"],
                    "source_licenses": seed["source_licenses"], "data_layer": seed["data_layer"],
                    "prompt_version": expected_version, "annotation_origin": "luna_synthetic_unverified",
                    "validation_status": "structural_repaired" if repairs else "structural_only",
                    "repair_flags": repairs, "split": "unassigned",
                    "decidable_at_char": len(prefix) if prefix else None,
                    "content_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
        rows.append(row)
    return errors, rows
