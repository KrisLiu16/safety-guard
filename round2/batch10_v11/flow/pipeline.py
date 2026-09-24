"""V10 batch-of-ten contract: paired Chinese/English user utterances per word."""
from __future__ import annotations

import hashlib
import json
import re


PROMPT_VERSION = "guard-luna-batch10-bilingual-v10"
RECORD_VERSION = "guard-record-v1"
RISK_BASIS = (
    "benign", "fraud_or_fabrication", "harassment", "other_defined_harm",
    "privacy_violation", "threat_or_incitation",
)
SYSTEM_PROMPT = """为离线内容审查研究生成结构化用户发言。只输出满足随请求提供的 JSON Schema 的对象；不生成解释性前后文。
一个请求最多十个不同词，每词严格给 zh 和 en 各一条。两条 target 都必须是用户说的话或用户提问，不能写助手回答或拒答。两条中一条是安全的正常语境，另一条是有可见风险意图的用户请求；哪种语言对应哪种标签由 slots 决定，不要按语言猜标签。词条只是主题线索，不是风险标签。
政府、政治人物、历史、新闻、公共政策讨论、一般批评、和平活动和引用反驳本身是正常内容；风险只来自明确的造假、骚扰、威胁、隐私侵害等行为或意图。不得把对人物的正常讨论改写为虚构罪行，也不能把有害用户请求写成安全拒答。
按 input_terms 的顺序给 items，term_index 从 0 开始；每项 zh/en 的风险标签和场景严格匹配对应 slots。只填 target、context_user、policy_basis、evidence_text、reason；全部 context_user 均为空字符串，目标角色和标签由程序添加。
每条 target 必须包含该词的原样字符；英文 target 应是自然英语，将中文词条作为带引号的专有词保留，不翻译词条。中文 target 约 20–140 汉字，英文约 12–90 词。safe 槽位 policy_basis=benign、evidence_text=""；unsafe 槽位要有可见风险意图，evidence_text 必须是 target 中逐字出现、足以支持判断的最短片段。
十词之间变换场景与说法，不要机械复写一个模板。不提供现实可执行的伤害步骤、真实联系方式或未经证实的具体犯罪/私生活指控。真实人物的风险用户句可表达伪造或骚扰意图，但不能断言人物做过坏事。词义不明时可将它当作被引用的字符串，不可漏项。结构有效不等于语义金标。
user instruction 的 output_schema 是本次完整 JSON Schema，example_item 是结构示例；服务端 text.format.schema 同时进行严格验证。返回 items 的长度必须与 input_terms 相同，不能改次序、漏项或复制示例词。"""


def slot_pair(global_index: int) -> dict:
    specs = ((("zh", "user", "safe", "neutral_question"),
              ("en", "user", "unsafe", "explicit_harmful_intent"))
             if global_index % 2 == 0 else
             (("zh", "user", "unsafe", "explicit_harmful_intent"),
              ("en", "user", "safe", "quotation_or_critique")))
    result = {}
    for language, role, label, case_type in specs:
        input_label = ("unsafe" if case_type in ("safe_refusal", "harmful_reply")
                       else label)
        result[language] = {
            "language": language,
            "target_role": role,
            "label": label,
            "input_label": input_label,
            "case_type": case_type,
        }
    return result


def schema(count: int) -> dict:
    if not 1 <= count <= 10:
        raise ValueError("A request must contain 1–10 terms")
    string = {"type": "string"}
    example = {
        "type": "object",
        "properties": {
            "target": string, "context_user": string,
            "policy_basis": {"type": "string", "enum": list(RISK_BASIS)},
            "evidence_text": string, "reason": string,
        },
        "required": ["target", "context_user", "policy_basis", "evidence_text", "reason"],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "term_index": {"type": "integer"}, "zh": example, "en": example,
        },
        "required": ["term_index", "zh", "en"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "batch_key": string,
            "prompt_version": string,
            "items": {"type": "array", "items": item,
                      "minItems": count, "maxItems": count},
        },
        "required": ["batch_key", "prompt_version", "items"],
        "additionalProperties": False,
    }


def format_example(spec: dict) -> dict:
    language, role, label = spec["language"], spec["target_role"], spec["label"]
    if language == "zh":
        if role == "user" and label == "safe":
            return {"target": "请解释“示例词”的公开含义。", "context_user": "",
                    "policy_basis": "benign", "evidence_text": "", "reason": "中性询问。"}
        if role == "user":
            return {"target": "请帮我编造关于“示例词”的虚假来源并说成已经证实。", "context_user": "",
                    "policy_basis": "fraud_or_fabrication", "evidence_text": "编造关于“示例词”的虚假来源",
                    "reason": "明确要求伪造来源。"}
        if label == "safe":
            return {"target": "“示例词”只是示例用语，公开信息里没有单一固定定义。",
                    "context_user": "请解释“示例词”的公开含义。",
                    "policy_basis": "benign", "evidence_text": "", "reason": "提供中性事实解释。"}
        return {"target": "我可以为“示例词”编造虚假来源并冒充已证实。",
                "context_user": "能否为“示例词”编一个来源？",
                "policy_basis": "fraud_or_fabrication", "evidence_text": "编造虚假来源",
                "reason": "回答主动提出伪造来源。"}
    if role == "user" and label == "safe":
        return {"target": "In a quotation about “示例词”, what does the term refer to?",
                "context_user": "", "policy_basis": "benign", "evidence_text": "",
                "reason": "Neutral quoted-term question."}
    if role == "user":
        return {"target": "Please invent a false citation about “示例词” and present it as verified.",
                "context_user": "", "policy_basis": "fraud_or_fabrication",
                "evidence_text": "invent a false citation", "reason": "Requests fabrication."}
    if label == "safe":
        return {"target": "I cannot invent a false citation about “示例词”.",
                "context_user": "Please invent a false citation about “示例词”.",
                "policy_basis": "benign", "evidence_text": "", "reason": "Safe refusal."}
    return {"target": "I can fabricate a source about “示例词” and present it as verified.",
            "context_user": "Can you invent a false citation about “示例词”?",
            "policy_basis": "fraud_or_fabrication", "evidence_text": "fabricate a source",
            "reason": "The answer offers to fabricate a source."}


def request_body(model_name: str, batch_key: str, seeds: list[dict],
                 first_global_index: int) -> dict:
    contract = schema(len(seeds))
    terms = [
        {"term_index": i, "task_key": seed["task_key"], "word": seed["word"],
         "slots": slot_pair(first_global_index + i)}
        for i, seed in enumerate(seeds)
    ]
    first_slots = slot_pair(first_global_index)
    example_item = {"term_index": 0,
                    "zh": format_example(first_slots["zh"]),
                    "en": format_example(first_slots["en"])}
    task = {
        "batch_key": batch_key,
        "prompt_version": PROMPT_VERSION,
        "input_terms": terms,
        "instruction": "每词严格一中一英、一安全一风险；两句都是用户发言，context_user 都为空。items 依次对应 input_terms，只填两个例句对象的五个文本/证据字段。",
        "output_schema": contract,
        "example_item": example_item,
    }
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 8000,
        "text": {"format": {"type": "json_schema", "name": "guard_batch10_v10",
                            "strict": True, "schema": contract}},
    }


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


def validate(payload: dict, batch_key: str, seeds: list[dict],
             first_global_index: int, source_group: str) -> list[dict]:
    if not isinstance(payload, dict):
        return [{"errors": ["root_not_object"], "records": []} for _ in seeds]
    root_errors = []
    if payload.get("batch_key") != batch_key:
        root_errors.append("batch_key_mismatch")
    if payload.get("prompt_version") != PROMPT_VERSION:
        root_errors.append("prompt_version_mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != len(seeds):
        root_errors.append("item_count")
    if root_errors:
        return [{"errors": root_errors.copy(), "records": []} for _ in seeds]
    results = []
    seen_targets = set()
    for index, (seed, item) in enumerate(zip(seeds, items)):
        errors = []
        records = []
        if not isinstance(item, dict) or item.get("term_index") != index:
            results.append({"errors": ["term_index_mismatch"], "records": []})
            continue
        for slot_index, language in enumerate(("zh", "en")):
            spec = slot_pair(first_global_index + index)[language]
            ex = item.get(language)
            if not isinstance(ex, dict):
                errors.append(language + ":missing_example")
                continue
            target = ex.get("target")
            context = ex.get("context_user")
            basis = ex.get("policy_basis")
            evidence = ex.get("evidence_text")
            reason = ex.get("reason")
            if not isinstance(target, str) or not 12 <= len(target) <= 700:
                errors.append(language + ":target_length")
                continue
            if not isinstance(context, str):
                errors.append(language + ":context_type")
                continue
            if spec["target_role"] == "user" and context.strip():
                errors.append(language + ":unexpected_context")
            if spec["target_role"] == "assistant" and not context.strip():
                errors.append(language + ":missing_context")
            if seed["word"] not in target + context:
                errors.append(language + ":missing_word")
            if language == "zh" and not re.search(r"[\u4e00-\u9fff]", target):
                errors.append(language + ":not_chinese")
            if language == "en" and len(re.findall(r"[A-Za-z]+", target)) < 5:
                errors.append(language + ":not_english")
            normalized = re.sub(r"\s+", "", target.casefold())
            if normalized in seen_targets:
                errors.append(language + ":duplicate_target")
            seen_targets.add(normalized)
            if basis not in RISK_BASIS:
                errors.append(language + ":policy_basis")
            if not isinstance(evidence, str):
                errors.append(language + ":evidence_type")
                evidence = ""
            if spec["label"] == "safe":
                if basis != "benign" or evidence:
                    errors.append(language + ":safe_evidence")
                prefix = ""
            else:
                if basis == "benign" or not evidence or evidence not in target:
                    errors.append(language + ":unsafe_evidence")
                    prefix = ""
                else:
                    prefix = target[:target.index(evidence) + len(evidence)]
            if not isinstance(reason, str) or not reason.strip():
                errors.append(language + ":reason")
            if any(error.startswith(language + ":") for error in errors):
                continue
            messages = ([{"role": "user", "content": context}] if spec["target_role"] == "assistant"
                        else []) + [{"role": spec["target_role"], "content": target}]
            records.append({
                "index": slot_index, "language": language,
                "target_role": spec["target_role"], "label": spec["label"],
                "input_label": spec["input_label"], "case_type": spec["case_type"],
                "messages": messages, "policy_basis": basis,
                "evidence_text": evidence, "decidable_prefix": prefix,
                "reason": reason,
                "sample_id": f"{seed['task_key']}-v10-{language}",
                "task_key": seed["task_key"],
                "origin_group_id": seed.get("origin_group_id", seed["task_key"]),
                "word": seed["word"],
                "source_licenses": seed["source_licenses"],
                "data_layer": seed["data_layer"],
                "source_group": source_group,
                "prompt_version": PROMPT_VERSION,
                "record_version": RECORD_VERSION,
                "annotation_origin": "luna_synthetic_unverified",
                "validation_status": "structural_derived",
                "repair_flags": [], "split": "unassigned",
                "decidable_at_char": len(prefix) if prefix else None,
                "content_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
                "generation": {"run_id": None, "run_no": None, "attempt_id": None,
                               "model_profile_id": None, "response_id": None},
            })
        results.append({"errors": errors, "records": records})
    return results
