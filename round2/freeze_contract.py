#!/usr/bin/env python3
"""Freeze the model-output and extracted-record formats before the full run."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"luna_flow_v4"))
from pipeline import PROMPT_VERSION,RECORD_VERSION,SYSTEM_PROMPT,schema,slot_records  # noqa: E402

def obj(props):
    return {"type":"object","properties":props,"required":list(props),"additionalProperties":False}
def nullable_string():
    return {"type":["string","null"]}
string={"type":"string"}
message=obj({"role":{"type":"string","enum":["user","assistant"]},"content":string})
generation=obj({"run_id":nullable_string(),"run_no":nullable_string(),
                "attempt_id":nullable_string(),"model_profile_id":nullable_string(),
                "response_id":nullable_string()})
record=obj({
    "index":{"type":"integer","minimum":0,"maximum":9},
    "language":{"type":"string","enum":["zh","en"]},
    "target_role":{"type":"string","enum":["user","assistant"]},
    "label":{"type":"string","enum":["safe","unsafe"]},
    "input_label":{"type":"string","enum":["safe","unsafe"]},
    "case_type":{"type":"string","enum":sorted({x["case_type"] for x in slot_records()})},
    "messages":{"type":"array","items":message,"minItems":1,"maxItems":2},
    "policy_basis":{"type":"string","enum":["benign","threat_or_incitation","privacy_violation","fraud_or_fabrication","harassment","other_defined_harm"]},
    "evidence_text":string,"decidable_prefix":string,"reason":string,
    "sample_id":string,"task_key":string,"origin_group_id":string,"word":string,
    "source_licenses":{"type":"array","items":string},
    "data_layer":{"type":"string","enum":["permissive","noncommercial"]},
    "source_group":string,"prompt_version":string,
    "record_version":{"type":"string","const":RECORD_VERSION},
    "annotation_origin":{"type":"string","const":"luna_synthetic_unverified"},
    "validation_status":{"type":"string","enum":["structural_derived","structural_repaired"]},
    "repair_flags":{"type":"array","items":string},
    "split":string,"decidable_at_char":{"type":["integer","null"]},
    "content_sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},
    "generation":generation,
})

def write_fixed(path,value):
    encoded=(json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists():
        if path.read_bytes()!=encoded:raise RuntimeError("Frozen contract differs: "+str(path))
    else:path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()

def main():
    output_schema=schema()
    example_item={"target":"请解释“示例词”的公开含义。","context_user":"",
                  "policy_basis":"benign","evidence_text":"","reason":"中性询问。"}
    hashes={
        "model_output_schema":write_fixed(ROOT/"luna_output_schema_v8.json",output_schema),
        "training_record_schema":write_fixed(ROOT/"training_record_schema_v1.json",record),
        "example_item":write_fixed(ROOT/"luna_example_item_v8.json",example_item),
    }
    manifest={"prompt_version":PROMPT_VERSION,"record_version":RECORD_VERSION,
              "prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "pipeline_sha256":hashlib.sha256((ROOT/"luna_flow_v4/pipeline.py").read_bytes()).hexdigest(),
              "flow_sha256":hashlib.sha256((ROOT/"luna_flow_v4/flow.py").read_bytes()).hexdigest(),
              "sha256":hashes,"frozen":True}
    write_fixed(ROOT/"contract_manifest.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=="__main__":main()
