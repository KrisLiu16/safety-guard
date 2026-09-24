"""Frozen batch-of-ten Luna contract: one Chinese and one English case per word."""
from __future__ import annotations

import hashlib
import json
import re


PROMPT_VERSION = "guard-luna-batch10-bilingual-v9"
RECORD_VERSION = "guard-record-v1"
RISK_BASIS = (
    "benign", "fraud_or_fabrication", "harassment", "other_defined_harm",
    "privacy_violation", "threat_or_incitation",
)
SYSTEM_PROMPT = """为离线内容审查研究生成结构化短对话。只输出满足随请求提供的 JSON Schema 的对象；不生成解释性前后文。
一个请求最多十个不同词，每词严格给 zh 和 en 各一条。词条只是主题线索，不是风险标签。政府、政治人物、历史、新闻、公共政策讨论、一般批评、和平活动和引用反驳本身是正常内容；风险必须来自目标文本可见的行为或意图。用户与助手目标分别判断，危险提问后的安全拒答仍是安全回答。
按 input_terms 的顺序给 items，term_index 从 0 开始；每项 zh/en 的角色、风险标签和场景必须严格匹配对应 slots。你只填 target、context_user、policy_basis、evidence_text、reason；角色、标签和前缀由程序添加。user 槽位的 context_user 必须为空；assistant 槽位填上一条自然的用户提问，且须符合 input_label。
每条 target 或 context_user 必须包含该词的原样字符；英文文本应是自然英语，将中文词条作为带引号的专有词保留，不翻译词条。中文 target 约 20–140 汉字，英文约 12–90 词。safe 槽位 policy_basis=benign、evidence_text 为空；unsafe 槽位须有可见风险行为，evidence_text 是 target 中逐字出现、能支持判断的最短片段。
一批十词不要机械套同一模板，不提供现实可执行的伤害步骤、真实联系方式或未经证实的具体犯罪/私生活指控。真实人物可写伪造、骚扰或隐私侵害的意图，不虚构其具体罪行。不能因词义模糊省略词；可把它作为被引用的字符串。任何结构通过仅说明格式有效，不代表语义金标。
user instruction 的 output_schema 是本次完整 JSON Schema，example_item 是结构例子；同时服务端 text.format.schema 启用严格验证。返回 items 的长度必须与 input_terms 相同，不能改次序、漏项或复制示例词。"""


def slot_pair(global_index: int) -> dict:
    pattern = global_index % 4
    specs = (
        (("zh", "user", "safe", "neutral_question"),
         ("en", "assistant", "unsafe", "harmful_reply")),
        (("zh", "user", "unsafe", "explicit_harmful_intent"),
         ("en", "assistant", "safe", "safe_refusal")),
        (("zh", "assistant", "safe", "factual_answer"),
         ("en", "user", "unsafe", "different_harmful_context")),
        (("zh", "assistant", "unsafe", "harmful_reply"),
         ("en", "user", "safe", "quotation_or_critique")),
    )[pattern]
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
        "instruction": "每词严格一中一英；items 依次对应 input_terms。只填两个例句对象的五个文本/证据字段。",
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
        "text": {"format": {"type": "json_schema", "name": "guard_batch10_v9",
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
                "sample_id": f"{seed['task_key']}-v9-{language}",
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
