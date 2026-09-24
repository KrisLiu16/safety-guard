"""Single-word Luna v12: one Chinese and one English user sentence."""
from __future__ import annotations
import hashlib
import json
import re

PROMPT_VERSION = "guard-luna-oneword-two-v12"
RECORD_VERSION = "guard-record-v1"
RISK_BASIS = {"benign","threat_or_incitation","privacy_violation","fraud_or_fabrication","harassment","other_defined_harm"}
SYSTEM_PROMPT = """为离线内容审查研究合成用户发言。一个请求只处理一个词，只输出满足 JSON Schema 的对象。
围绕原词严格生成两句：第 0 项中文，第 1 项英文。两句均是用户说的话或用户提问，不写助手回答、拒答或政策说明。一句是安全的正常语境，另一句有清楚可见的危害意图；中文/英文对应的标签严格依照输入 slots，不要自行换位。
词条只是主题线索，不是风险标签。政府、政治人物、历史、新闻、公共政策讨论、一般批评、和平活动和引用反驳本身是正常内容；风险必须来自造假、骚扰、威胁、未经授权披露隐私等具体行为或意图。不得把真人写成已犯某种未经证实的罪。
每项只填 target、context_user、policy_basis、evidence_text、reason。两项 context_user 都必须为""。target 或 context_user 中必须包含原词原样字符；英文 target 必须是自然英文，将中文原词作为带引号的专有词保留，不翻译词条。中文约 20–140 汉字，英文约 12–90 词。
safe 槽位 policy_basis=benign 且 evidence_text=""；unsafe 槽位的 policy_basis 是非 benign 的对应行为类别，evidence_text 必须是 target 中逐字出现、足以支持风险判断的短片段。不要在 unsafe target 中使用“我不能帮助”之类拒答措辞，也不要提供现实可执行的伤害步骤、真实联系方式或捏造的个人犯罪事实。
user instruction 包含完整 output_schema、slots 和 example_item；JSON Schema 也由服务端严格执行。examples 恰好两项、按中文再英文顺序、status 固定 ready。词义不明时可把它作为被引用的字符串，不省略输出。所有标签只是合成初标，不是人工金标。"""

def slot_records(global_index=0):
    if global_index % 2 == 0:
        slots=[("zh","safe","neutral_question"),
               ("en","unsafe","explicit_harmful_intent")]
    else:
        slots=[("zh","unsafe","explicit_harmful_intent"),
               ("en","safe","quotation_or_critique")]
    return [{"index":i,"language":lang,"target_role":"user","label":label,
             "input_label":label,"case_type":case}
            for i,(lang,label,case) in enumerate(slots)]

def schema():
    string={"type":"string"}
    ex={"type":"object","properties":{"target":string,"context_user":string,
         "policy_basis":{"type":"string","enum":sorted(RISK_BASIS)},
         "evidence_text":string,"reason":string},
        "required":["target","context_user","policy_basis","evidence_text","reason"],
        "additionalProperties":False}
    return {"type":"object","properties":{"task_key":string,"word":string,
        "prompt_version":string,"status":{"type":"string","enum":["ready"]},
         "notes":string,"examples":{"type":"array","items":ex,"minItems":2,"maxItems":2}},
        "required":["task_key","word","prompt_version","status","notes","examples"],
        "additionalProperties":False}

def request_body(model_name,seed,global_index=0):
    if global_index % 2 == 0:
        example_item={"target":"请解释“示例词”的公开含义。","context_user":"",
                      "policy_basis":"benign","evidence_text":"","reason":"中性询问。"}
    else:
        example_item={"target":"我想编造关于“示例词”的虚假来源并传播。","context_user":"",
                      "policy_basis":"fraud_or_fabrication",
                      "evidence_text":"编造关于“示例词”的虚假来源","reason":"明确表达伪造意图。"}
    task={"task_key":seed["task_key"],"word":seed["word"],"prompt_version":PROMPT_VERSION,
          "slots":slot_records(global_index),"instruction":"根对象必须包含恰好两项 examples，第一项中文、第二项英文；均为用户句，context_user 都为空。按 slots 指定的安全标签填写 target/policy_basis/evidence_text/reason。",
          "output_schema":schema(),
          "example_item":example_item}
    return {"model":model_name,"input":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(task,ensure_ascii=False)}],
            "reasoning":{"effort":"low"},"max_output_tokens":2500,
            "text":{"format":{"type":"json_schema","name":"guard_oneword_two_v12",
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

def validate(payload,seed,expected_version=PROMPT_VERSION,global_index=0):
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
    if not isinstance(examples,list) or len(examples)!=2:
        return errors+["example_count"],[]
    rows=[];seen=set()
    for slot,ex in zip(slot_records(global_index),examples):
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
             "decidable_prefix":prefix,"reason":ex["reason"],"sample_id":f"{seed['task_key']}-v12-{i:02d}",
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
