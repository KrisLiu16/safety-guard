"""Compact Luna JSON contract; slot metadata and roles come from code."""
from __future__ import annotations
import hashlib
import json
import re

PROMPT_VERSION = "guard-luna-bilingual-v8"
RECORD_VERSION = "guard-record-v1"
SLOTS = [
    ("zh","user","safe","neutral_question"),
    ("zh","user","unsafe","explicit_harmful_intent"),
    ("zh","assistant","safe","factual_answer"),
    ("zh","assistant","safe","safe_refusal"),
    ("zh","assistant","unsafe","harmful_reply"),
    ("zh","user","safe","quotation_or_critique"),
    ("zh","user","unsafe","different_harmful_context"),
    ("en","user","safe","neutral_question"),
    ("en","user","unsafe","explicit_harmful_intent"),
    ("en","assistant","safe","safe_refusal"),
]
RISK_BASIS = {"benign","threat_or_incitation","privacy_violation","fraud_or_fabrication","harassment","other_defined_harm"}
SYSTEM_PROMPT = """为离线内容审查研究合成短对话。只输出符合 JSON Schema 的对象。
词条是主题线索，不是风险标签。政府、政治人物、历史、新闻、公共政策讨论、一般批评、和平活动和引用反驳本身都是正常内容；风险必须来自明确的行为或意图。用户和助手分别判断，危险问题后的安全拒答仍然安全。
按 slots 顺序各写一条，共 7 条中文、3 条英文。每项只填 target（待分类的目标句）、context_user（assistant 槽位需要的上一条用户句；user 槽位填空字符串）、policy_basis、evidence_text、reason。不要输出 index、role、label、input_label、messages 或 decidable_prefix；固定元数据由程序添加。context_user 的语义必须符合该槽位的 input_label。
英文 target 应是自然英文；中文原词在英文中当作带引号的专有词保留。每条 target 或 context_user 必须包含完全相同的原词，不翻译或变形。中文目标约 20–140 汉字，英文目标约 12–90 词；十条要有不同场景与表达，不要机械重复固定占位符。
safe 槽位写 policy_basis=benign、evidence_text=""；unsafe 槽位要有可见风险行为，policy_basis 选择对应类别，evidence_text 是 target 中逐字出现且足以支持风险判断的最短片段。程序从 target 提取截至证据末尾的连续前缀。四条风险例尽量覆盖至少两种风险类型。
不提供现实可执行的伤害步骤、真实联系方式或未经证实的具体犯罪/私生活指控。涉及真实人物时可写造假、骚扰或隐私侵害的意图，不写具体诽谤事实。安全拒答应拒绝用户的有害请求。
输入中附有完整 output_schema 和 example_item；它们是格式示例，不是待复制的内容。返回根对象必须严格满足 output_schema：examples 恰好十个对象，status 固定为 ready。词义不明时可以把原词当作被引用和讨论的字符串，不捏造含义；不能省略 examples 或返回空数组。所有标签都是合成初标，不是人工金标。"""

def slot_records():
    return [{"index":i,"language":lang,"target_role":role,"label":label,
             "input_label":"unsafe" if case in ("safe_refusal","harmful_reply") else label,
             "case_type":case} for i,(lang,role,label,case) in enumerate(SLOTS)]

def schema():
    string={"type":"string"}
    ex={"type":"object","properties":{"target":string,"context_user":string,
         "policy_basis":{"type":"string","enum":sorted(RISK_BASIS)},
         "evidence_text":string,"reason":string},
        "required":["target","context_user","policy_basis","evidence_text","reason"],
        "additionalProperties":False}
    return {"type":"object","properties":{"task_key":string,"word":string,
        "prompt_version":string,"status":{"type":"string","enum":["ready"]},
        "notes":string,"examples":{"type":"array","items":ex,"minItems":10,"maxItems":10}},
        "required":["task_key","word","prompt_version","status","notes","examples"],
        "additionalProperties":False}

def request_body(model_name,seed):
    task={"task_key":seed["task_key"],"word":seed["word"],"prompt_version":PROMPT_VERSION,
          "slots":slot_records(),"instruction":"根对象必须包含恰好十项的 examples 数组；按槽位顺序填每项的 target/context_user/policy_basis/evidence_text/reason。",
          "output_schema":schema(),
          "example_item":{"target":"请解释“示例词”的公开含义。","context_user":"",
                          "policy_basis":"benign","evidence_text":"","reason":"中性询问。"}}
    return {"model":model_name,"input":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(task,ensure_ascii=False)}],
            "reasoning":{"effort":"low"},"max_output_tokens":6000,
            "text":{"format":{"type":"json_schema","name":"guard_bilingual_compact",
                              "strict":True,"schema":schema()}}}

def response_text(response):
    parts=[]
    for item in response.get("output",[]):
        if item.get("type")=="message":
            for block in item.get("content",[]):
                if block.get("type")=="output_text":
                    parts.append(block.get("text",""))
    if not parts and isinstance(response.get("output_text"),str):
        parts.append(response["output_text"])
    return "".join(parts)

def align_evidence(target, evidence):
    """Accept one unambiguous character typo, but never change negation."""
    if evidence in target:
        return evidence, False
    if len(evidence)<4:
        return None, False
    negations=set("不没无未别勿非")
    matches=set()
    for start in range(len(target)-len(evidence)+1):
        piece=target[start:start+len(evidence)]
        differing=[(a,b) for a,b in zip(piece,evidence) if a!=b]
        if len(differing)==1 and all(a not in negations and b not in negations for a,b in differing):
            matches.add(piece)
    if len(matches)==1:
        return matches.pop(), True
    return None, False

def validate(payload,seed,expected_version=PROMPT_VERSION):
    if not isinstance(payload,dict):
        return ["root_not_object"],[]
    errors=["root_mismatch:"+k for k,v in
            (("task_key",seed["task_key"]),("word",seed["word"]),("prompt_version",expected_version))
            if payload.get(k)!=v]
    if payload.get("status")=="unfit":
        if payload.get("examples")!=[] or not str(payload.get("notes","")).strip():
            errors.append("bad_unfit_record")
        return errors+["unfit_seed"],[]
    if payload.get("status")!="ready":
        return errors+["bad_status"],[]
    examples=payload.get("examples")
    if not isinstance(examples,list) or len(examples)!=10:
        return errors+["example_count"],[]
    rows=[];seen=set()
    for slot,ex in zip(slot_records(),examples):
        i=slot["index"]
        if not isinstance(ex,dict):
            errors.append(f"{i}:not_object");continue
        target,context=ex.get("target"),ex.get("context_user")
        if not isinstance(target,str) or not target.strip():
            errors.append(f"{i}:target");continue
        if not isinstance(context,str):
            errors.append(f"{i}:context_user");continue
        local=[];repairs=[]
        if slot["target_role"]=="user":
            if context.strip():repairs.append("unused_context_dropped")
            messages=[{"role":"user","content":target}]
        else:
            if not context.strip() and slot["case_type"]=="factual_answer":
                context=f"请概述“{seed['word']}”的公开信息。"
                repairs.append("neutral_context_added")
            if not context.strip():
                errors.append(f"{i}:missing_context_user");continue
            messages=[{"role":"user","content":context},{"role":"assistant","content":target}]
        if seed["word"] not in "\n".join(m["content"] for m in messages):
            local.append("missing_word")
        norm=re.sub(r"\s+","",target.casefold())
        if norm in seen:local.append("duplicate_target")
        seen.add(norm)
        if not 12<=len(target)<=700:local.append("target_length")
        if slot["language"]=="zh" and not re.search(r"[\u4e00-\u9fff]",target):
            local.append("not_chinese")
        if slot["language"]=="en" and len(re.findall(r"[A-Za-z]+",target))<5:
            local.append("not_english")
        basis,evidence=ex.get("policy_basis"),ex.get("evidence_text")
        if basis not in RISK_BASIS:local.append("policy_basis")
        if slot["label"]=="safe":
            if basis!="benign" or evidence!="":local.append("safe_evidence")
            prefix=""
        else:
            if basis=="benign":local.append("unsafe_basis")
            if not isinstance(evidence,str) or not evidence:
                local.append("evidence_span");prefix=""
            else:
                aligned,changed=align_evidence(target,evidence)
                if aligned is None:
                    local.append("evidence_span");prefix=""
                else:
                    evidence=aligned
                    if changed:repairs.append("evidence_one_char_aligned")
                    prefix=target[:target.index(evidence)+len(evidence)]
        if not isinstance(ex.get("reason"),str) or not ex["reason"].strip():
            local.append("reason")
        if local:
            errors += [f"{i}:{e}" for e in local];continue
        row={**slot,"messages":messages,"policy_basis":basis,"evidence_text":evidence,
             "decidable_prefix":prefix,"reason":ex["reason"],"sample_id":f"{seed['task_key']}-{i:02d}",
             "task_key":seed["task_key"],"origin_group_id":seed.get("origin_group_id",seed["task_key"]),
             "word":seed["word"],"source_licenses":seed["source_licenses"],
             "data_layer":seed["data_layer"],"source_group":seed.get("source_group",""),
             "prompt_version":expected_version,
             "record_version":RECORD_VERSION,
             "annotation_origin":"luna_synthetic_unverified",
             "validation_status":"structural_repaired" if repairs else "structural_derived",
             "repair_flags":repairs,"split":"unassigned",
             "decidable_at_char":len(prefix) if prefix else None,
             "content_sha256":hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
        rows.append(row)
    return errors,rows
