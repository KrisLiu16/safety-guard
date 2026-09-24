"""Generation contract and local checks; no API keys or network access."""
import hashlib
import json

POLICY_VERSION = "guard-generation-v2"
SYSTEM_PROMPT = """你在构造中文内容安全分类模型的离线研究数据。仅输出符合给定 schema 的 JSON。
词库词条是主题线索，不是违规标签。不得因为涉及中国政府、人物、历史、政治观点、一般批评、宗教或和平活动就标为 unsafe。
safe 包含正常询问、学术/历史/新闻讨论、合法批评、引用反驳、安全拒答；unsafe 必须有明确行为依据：暴力威胁/煽动、侵犯个人隐私、诈骗或伪造证据等。
不捏造真实人物的犯罪事实、私人资料，不提供现实可执行的伤害步骤，不输出露骨内容。unsafe 可以是简短的有害意图请求或非操作性有害表态，使用虚构人物与占位符。
区分输入和回答标签：危险请求之后的安全拒答仍是 safe；不得把用户标签自动传播到回答。
围绕指定原词写自然、多样、长度接近的中文例句。原词至少逐字出现在每个例子的 messages 中一次，避免机械复述模板。
每条目标消息约 30–150 汉字，历史消息简短。messages 的最后一条即 target_role 对应的目标；user 例子只有一条 user 消息；assistant 例子先 user 后 assistant。
按给定 slots 逐一生成，不增删。label 是目标消息的标签，input_label 是用户消息的标签。policy_id 从给定枚举选择；safe 一律 benign。
unsafe 的 evidence_text 必须是目标消息中逐字出现、足以支持风险判断的最短片段；decidable_prefix 必须是目标消息从头起的连续前缀，且包含 evidence_text，不得利用未来文本。
safe 的 evidence_text 和 decidable_prefix 都为空字符串。reason 简要指出依据和允许边界，不写思维链。
真实姓名不能加上“虚构人物/虚构角色”前缀来假装匿名。原词可在正常背景中出现，风险行为如需匿名用与词条不同的甲某/乙某。不要将明确创作小说或虚构情节自动当作现实伤害意图；本批风险样本应明确区分现实意图与合法创作。安全输入不得暗含规避审计/欺诈等请求，不安全输入要有清楚的行为意图。所有标签均为合成初标，不能宣称人工审核或官方政策真值。遇到词义不明，在 notes 中说明，仍按规范完成，不编造词义。
"""

SLOTS = [
    ("safe_user", "user", "safe", "safe"),
    ("unsafe_user", "user", "unsafe", "unsafe"),
    ("safe_user", "user", "safe", "safe"),
    ("unsafe_user", "user", "unsafe", "unsafe"),
    ("safe_answer", "assistant", "safe", "safe"),
    ("unsafe_answer", "assistant", "unsafe", "unsafe"),
    ("safe_refusal", "assistant", "safe", "unsafe"),
    ("unsafe_user", "user", "unsafe", "unsafe"),
    ("safe_refusal", "assistant", "safe", "unsafe"),
    ("unsafe_answer", "assistant", "unsafe", "safe"),
]
POLICIES = ["benign", "violence_threat", "privacy_violation", "fraud_fabrication", "harassment"]

def schema(count):
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    string = {"type": "string"}
    ex = obj({
        "index": {"type": "integer"},
        "case_type": {"type": "string", "enum": sorted({x[0] for x in SLOTS})},
        "target_role": {"type": "string", "enum": ["user", "assistant"]},
        "messages": {"type": "array", "items": obj({"role": {"type": "string", "enum": ["user", "assistant"]}, "content": string})},
        "label": {"type": "string", "enum": ["safe", "unsafe"]},
        "input_label": {"type": "string", "enum": ["safe", "unsafe"]},
        "policy_id": {"type": "string", "enum": POLICIES},
        "evidence_text": string, "decidable_prefix": string, "reason": string,
    })
    return obj({"seed_id": string, "word": string, "policy_version": string,
                "examples": {"type": "array", "items": ex, "minItems": count, "maxItems": count},
                "notes": string})

def slots(start, count):
    return [dict(zip(["case_type", "target_role", "label", "input_label"], SLOTS[i % 10]), index=i)
            for i in range(start, start + count)]

def request_body(model_name, seed, start, count):
    task = {"seed_id": seed["seed_id"], "word": seed["word"],
            "source_category": seed["source_category"], "policy_version": POLICY_VERSION,
            "slots": slots(start, count),
            "batch_style": ["日常短句", "新闻辨析", "学术讨论", "合规咨询", "社区交流", "引用与反驳", "公共服务问答", "企业内审", "公共信息查询", "证据与来源核验"][(start // 10) % 10],
            "instruction": "每个 index 生成一条独立例子，安全与不安全比例严格遵循 slots；同词的正反例通过行为/语境区分。"}
    return {"model": model_name, "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(task, ensure_ascii=False)}],
            "reasoning": {"effort": "low"}, "max_output_tokens": 9000,
            "text": {"format": {"type": "json_schema", "name": "guard_examples", "strict": True,
                                 "schema": schema(count)}}}

def response_text(response):
    parts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    if not parts and isinstance(response.get("output_text"), str):
        parts.append(response["output_text"])
    return "".join(parts)

def validate(payload, seed, start, count, policy_version=POLICY_VERSION):
    errors, rows = [], []
    if not isinstance(payload, dict):
        return ["root_not_object"], []
    for key, expected in [("seed_id", seed["seed_id"]), ("word", seed["word"]), ("policy_version", policy_version)]:
        if payload.get(key) != expected:
            errors.append("root_mismatch:" + key)
    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != count:
        return errors + ["example_count"], []
    for expected, ex in zip(slots(start, count), examples):
        index = expected["index"]
        local = []
        if not isinstance(ex, dict):
            errors.append(f"{index}:not_object"); continue
        for k, v in expected.items():
            if ex.get(k) != v: local.append("slot:" + k)
        ms = ex.get("messages", [])
        want_roles = ["user"] if expected["target_role"] == "user" else ["user", "assistant"]
        if not isinstance(ms, list) or [m.get("role") if isinstance(m, dict) else None for m in ms] != want_roles:
            local.append("message_roles")
        if not ms or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) or not m["content"].strip() for m in ms):
            local.append("message_content")
        if local:
            errors += [f"{index}:{e}" for e in local]; continue
        target = ms[-1]["content"]
        if seed["word"] not in "\n".join(m["content"] for m in ms): local.append("missing_seed_word")
        if not 10 <= len(target) <= 600: local.append("target_length")
        if not isinstance(ex.get("reason"), str) or not ex["reason"].strip(): local.append("reason")
        policy, evidence, prefix = ex.get("policy_id"), ex.get("evidence_text"), ex.get("decidable_prefix")
        if policy not in POLICIES: local.append("policy_id")
        if expected["label"] == "safe":
            if policy != "benign" or evidence != "" or prefix != "": local.append("safe_evidence")
        else:
            if policy == "benign": local.append("unsafe_policy")
            if not isinstance(evidence, str) or not evidence or evidence not in target: local.append("evidence_span")
            if not isinstance(prefix, str) or not prefix or not target.startswith(prefix): local.append("prefix")
            elif isinstance(evidence, str) and evidence not in prefix: local.append("evidence_after_prefix")
        if local:
            errors += [f"{index}:{e}" for e in local]; continue
        row = {**ex, "sample_id": f"{seed['seed_id']}-{index:04d}",
               "seed_id": seed["seed_id"], "word": seed["word"], "origin_group_id": seed["origin_group_id"],
               "source": seed["source"], "source_category": seed["source_category"],
               "policy_version": policy_version, "annotation_origin": "luna_synthetic_unverified",
               "validation_status": "structural_only", "split": "unassigned"}
        row["target_message_index"] = len(ms) - 1
        row["evidence_spans"] = [] if not evidence else [[target.index(evidence), target.index(evidence) + len(evidence)]]
        row["decidable_at_char"] = len(prefix) if prefix else None
        row["content_sha256"] = hashlib.sha256(json.dumps(ms, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        rows.append(row)
    return errors, rows
