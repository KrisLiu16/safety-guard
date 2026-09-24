#!/usr/bin/env python3
"""Download v12 one-word attempt archives into source-category JSONL."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from flow.pipeline import PROMPT_VERSION,response_text,validate  # noqa: E402

def cli(args: list[str]) -> dict:
    process=subprocess.run(["aster",*args],capture_output=True,text=True)
    envelope=json.loads(process.stdout or process.stderr)
    if not envelope.get("ok"):
        error=envelope.get("error",{})
        raise RuntimeError(f"Aster {args[0]} {error.get('code')}: {error.get('message')}")
    return envelope["data"]

def write_line(handle,value):
    handle.write(json.dumps(value,ensure_ascii=False)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--expected-prompt-version",default=PROMPT_VERSION)
    ap.add_argument("--expected-terms",type=int,default=None)
    ap.add_argument("--download-workers",type=int,default=5)
    ap.add_argument("--attempts-json",type=Path,default=None,
                    help="Frozen explicit attempt list for a partial extraction; avoids the API's 100-item cap")
    args=ap.parse_args()
    if args.expected_prompt_version!=PROMPT_VERSION:
        ap.error("Extractor and requested prompt version differ")
    if args.download_workers<1 or args.download_workers>10:ap.error("download-workers must be 1..10")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    run=cli(["runs","get",args.run])
    if args.attempts_json:
        selected=json.loads(args.attempts_json.read_text())
        if selected['run_id']!=run['id']:raise ValueError('Attempt snapshot belongs to another run')
    else:
        selected=cli(["runs","attempts",args.run])
    if selected.get('truncated'):
        raise RuntimeError('Attempt listing truncated: use a frozen per-sample --attempts-json manifest')
    attempts=selected['items']
    if any(a.get('state')!='completed' or a.get('result_status')!='succeeded' or not a.get('archive') for a in attempts):
        raise RuntimeError('Extraction requires completed successful attempts with archives')
    if len({a['sample_id'] for a in attempts})!=len(attempts):
        raise RuntimeError('Snapshot must select one attempt per sample')
    (out/'selected_attempts.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    archives=out/"archives";archives.mkdir(exist_ok=True)
    model_id=run.get("effective_config",{}).get("models",{}).get("main",{}).get("profile_id")

    def download(attempt):
        aid=attempt["attempt_id"]
        path=archives/(aid+".tar.gz")
        digest_path=archives/(aid+".sha256")
        if not path.exists():
            result=cli(["runs","archive",args.run,aid,"--out",str(path)])
            digest_path.write_text(result["sha256"]+"\n")
        if not digest_path.exists():
            raise RuntimeError("Missing archive checksum for "+aid)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=digest_path.read_text().strip():
            raise RuntimeError("Archive SHA-256 mismatch for "+aid)
        return aid,path

    temporary_examples=out/"examples.jsonl.tmp"
    temporary_statuses=out/"term_status.jsonl.tmp"
    group_dir=out/"word_results";group_dir.mkdir(exist_ok=True)
    group_handles={}
    group_names={}
    seen_terms=set();seen_contents=set()
    usage=collections.Counter();durations=[];errors=collections.Counter()
    complete=raw_complete=valid_examples=word_artifacts=repaired=0
    with temporary_examples.open("w") as examples_file,temporary_statuses.open("w") as status_file:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
                for aid,path in pool.map(download,attempts):
                    with tarfile.open(path,"r:gz") as tar:
                        members=sorted((m for m in tar if m.isfile() and m.name.startswith("flow/results/") and m.name.endswith(".json")),key=lambda m:m.name)
                        for member in members:
                            obj=json.load(tar.extractfile(member))
                            name,value=obj.get("name"),obj.get("value")
                            if not isinstance(name,str) or not name.startswith("word_") or not isinstance(value,dict):
                                continue
                            word_artifacts+=1
                            seed=value["seed"];key=seed["task_key"]
                            if key in seen_terms:raise RuntimeError("Duplicate term result "+key)
                            seen_terms.add(key)
                            group=value["source_group"]
                            if group not in group_handles:
                                digest=hashlib.sha256(group.encode()).hexdigest()[:16]
                                group_names[digest]=group
                                group_handles[group]=(group_dir/(digest+".jsonl.tmp")).open("w")
                            response=value.get("response")
                            if isinstance(response,dict):
                                try:
                                    payload=json.loads(response_text(response))
                                    word_errors,rows=validate(
                                        payload,seed,expected_version=PROMPT_VERSION,
                                        global_index=value["global_index"])
                                except (ValueError,TypeError,KeyError) as exc:
                                    word_errors,rows=["extract_parse:"+type(exc).__name__],[]
                                if response.get("status")!="completed":
                                    word_errors.append("status:"+str(response.get("status")));rows=[]
                            else:
                                word_errors,rows=value.get("errors",["missing_response"]),[]
                            if any(e.startswith("root_") for e in word_errors):rows=[]
                            kept=[]
                            for row in rows:
                                row.setdefault("record_version","guard-record-v1")
                                row["source_group"]=group
                                digest=row["content_sha256"]
                                if digest in seen_contents:
                                    word_errors.append("duplicate_across_terms:"+row["sample_id"])
                                    continue
                                seen_contents.add(digest)
                                row["generation"]={"run_id":run.get("id"),"run_no":run.get("run_no"),
                                                   "attempt_id":aid,"model_profile_id":model_id,
                                                   "response_id":response.get("id") if isinstance(response,dict) else None}
                                kept.append(row);write_line(examples_file,row)
                                repaired+=int(bool(row.get("repair_flags")))
                            valid_examples+=len(kept)
                            full=len(kept)==2 and not word_errors
                            complete+=int(full)
                            original=value.get("records",[])
                            original_full=len(original)==2 and not value.get("errors",[])
                            raw_complete+=int(original_full)
                            status={"task_key":key,"word":seed["word"],"source_group":group,
                                    "valid":len(kept),"complete":full,"raw_valid":len(original),
                                    "raw_complete":original_full,"errors":word_errors}
                            write_line(status_file,status)
                            write_line(group_handles[group],{"seed":seed,"errors":word_errors,
                                      "records":kept,"seconds":value.get("seconds"),
                                      "usage":response.get("usage") if isinstance(response,dict) else None})
                            errors.update(e.split(":")[0] for e in word_errors)
                            if isinstance(response,dict) and isinstance(response.get("usage"),dict):
                                usage.update({k:response["usage"].get(k,0) or 0 for k in
                                              ("input_tokens","output_tokens","total_tokens")})
                            if isinstance(value.get("seconds"),(int,float)):durations.append(value["seconds"])
        finally:
            for handle in group_handles.values():handle.close()
    temporary_examples.replace(out/"examples.jsonl")
    temporary_statuses.replace(out/"term_status.jsonl")
    for path in group_dir.glob("*.jsonl.tmp"):path.replace(path.with_suffix(""))
    (out/"category_files.json").write_text(json.dumps(group_names,ensure_ascii=False,indent=2)+"\n")
    durations.sort()
    summary={"run_id":run.get("id"),"run_no":run.get("run_no"),"run_status":run.get("status"),
             "attempts":len(attempts),"word_artifacts":word_artifacts,"complete_words":complete,
             "partial_snapshot":bool(args.attempts_json),
             "selected_attempts_sha256":hashlib.sha256((out/'selected_attempts.json').read_bytes()).hexdigest(),
             "expected_terms":args.expected_terms,
             "missing_word_artifacts":max(0,args.expected_terms-word_artifacts) if args.expected_terms is not None else None,
             "raw_complete_words":raw_complete,"valid_examples":valid_examples,
             "repaired_examples":repaired,"token_usage":dict(usage),"error_types":dict(errors),
             "latency_seconds":{"mean":sum(durations)/len(durations),
                                "p50":durations[math.ceil(0.5*len(durations))-1],
                                "p95":durations[math.ceil(0.95*len(durations))-1]} if durations else {},
             "synthetic_labels_are_gold":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
