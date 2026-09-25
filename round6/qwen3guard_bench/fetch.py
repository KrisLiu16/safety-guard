"""Qwen3Guard English benchmarks, prepared for our streaming guard (download + case building; Mac, CPU).

Reproduces the benchmark subsets behind the Qwen3Guard Technical Report's English prompt / response classification
bar charts (allenai/safety-eval conventions), from openly downloadable copies only: every Hugging Face source is
checked for gating first and fetched anonymously (no token, no terms accepted); gated sources are skipped and listed
in the manifest. Revisions are pinned to commit SHAs, so a re-run rebuilds the same cases; downloads already on disk
(same size and SHA256 as recorded) are reused.

Output (all under data/, which .gitignore covers; the repository is public and these files hold dataset text):
  data/raw/<source>/<revision>/<path>   the downloaded files
  data/cases.jsonl                      {"id", "bench", "level", "messages", "label"} per case (label 1 = unsafe)
  data/manifest.json                    per-bench counts, skipped benches, source revisions and file SHA256s

Run:  .venv/bin/python round6/qwen3guard_bench/fetch.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import random
import sys
import urllib.parse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = HERE / "data"
RAW = DATA / "raw"
VERSION = "qwen3guard-bench-cases-v1"
HF = "https://huggingface.co"
GITHUB_RAW = "https://raw.githubusercontent.com"
HARMBENCH_REPO, HARMBENCH_SHA = "centerforaisafety/HarmBench", "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
THINK_PATH = PROJECT / "round1" / "benchmark" / "thinking.jsonl"
SAFERLHF_PER_CLASS, SAFERLHF_SEED = 1000, 42

# name -> (kind, repo, pinned revision, path in repo, human-readable revision note)
SOURCES = {
    "toxicchat": ("hf", "lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8",
                  "data/0124/toxic-chat_annotation_test.csv", "main"),
    "openaimod": ("hf", "mmathys/openai-moderation-api-evaluation", "84e5cf3bcd6acb3dfc70b6760451645872218a3e",
                  "samples-1680.jsonl.gz", "main"),
    "aegis1": ("hf", "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "bd96d862068e47630197de64eb91f8d1481ff3e0",
               "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet", "main"),
    "aegis2": ("hf", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
               "test.json", "main"),
    "sst": ("hf", "Bertievidgen/SimpleSafetyTests", "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5",
            "sst_test_cases.csv", "main"),
    "polyguard": ("hf", "ToxicityPrompts/PolyGuardPrompts", "c5b466a95b64ff121db4398246b6abb7672696ec",
                  "data/test-00000-of-00001.parquet", "main"),
    "saferlhf": ("hf", "PKU-Alignment/PKU-SafeRLHF", "ff7ba91063016c78a225b0f74e1c0860bb18230f",
                 "round0/test.jsonl.xz", "branch v0"),
    "beavertails": ("hf", "PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678",
                    "round0/30k/test.jsonl.gz", "main (split 30k_test)"),
    "harmbench_behaviors": ("github", HARMBENCH_REPO, HARMBENCH_SHA,
                            "data/behavior_datasets/harmbench_behaviors_text_test.csv", "main"),
    "harmbench_classifier_val": ("github", HARMBENCH_REPO, HARMBENCH_SHA,
                                 "data/classifier_val_sets/text_behaviors_val_set.json", "main"),
}

# bench -> (level, sources it needs, expected (cases, unsafe) or None when only reported)
BENCHES = {
    "ToxicChat": ("prompt", ("toxicchat",), (2853, 362)),
    "OpenAIMod": ("prompt", ("openaimod",), (1680, 522)),
    "Aegis": ("prompt", ("aegis1",), (359, 233)),
    "Aegis2.0": ("prompt", ("aegis2",), (1964, 1059)),
    "SimpleSafetyTests": ("prompt", ("sst",), (100, 100)),
    "HarmBench-P": ("prompt", ("harmbench_behaviors",), (239, 239)),
    "WildGuardTest-P": ("prompt", ("polyguard",), None),
    "HarmBench-R": ("response", ("harmbench_classifier_val",), (602, 273)),
    "SafeRLHF": ("response", ("saferlhf",), (2000, 1000)),
    "BeaverTails": ("response", ("beavertails",), (3021, 1733)),
    "Aegis2.0-R": ("response", ("aegis2",), None),
    "WildGuardTest-R": ("response", ("polyguard",), None),
    "Think": ("response", (), (1059, None)),
}
GATED_SKIPPED = {
    "WildGuardTest (allenai/wildguardmix)": "gated; WildGuardTest-P/-R use the English slice of PolyGuardPrompts",
    "XSTest-Resp (allenai/xstest-response)": "gated; not reproduced",
    "HarmBench (walledai/HarmBench)": "gated; HarmBench-P uses the GitHub CSV instead",
}
OPENAI_CATEGORIES = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")


# ---------------------------------------------------------------- pure helpers (unit tested)

def user(text):
    return [{"role": "user", "content": text}]


def turn(prompt, response):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def truthy(value):
    """CSV / parquet booleans: True, 'True', 'true', 1, '1'."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value) if value is not None else False


def is_one(value):
    if value is None or value == "":
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def toxicchat_cases(rows):
    return [(str(r["conv_id"]), user(r["user_input"]), int(is_one(r["toxicity"])))
            for r in rows if truthy(r["human_annotation"])]


def openaimod_cases(rows):
    return [(str(i), user(r["prompt"]), int(any(is_one(r.get(c)) for c in OPENAI_CATEGORIES)))
            for i, r in enumerate(rows)]


def aegis1_cases(rows):
    """user_message rows (text stripped); unsafe when at most 1 of labels_0..2 is 'Safe' (safety-eval)."""
    out = []
    for i, r in enumerate(rows):
        if r["text_type"] != "user_message":
            continue
        safe_votes = sum(int(r.get(f"labels_{k}") == "Safe") for k in range(3))
        out.append((str(r.get("id", i)), user(r["text"].strip()), int(safe_votes <= 1)))
    return out


def aegis2_prompt_cases(rows):
    return [(str(r["id"]), user(r["prompt"]), int(r["prompt_label"] == "unsafe")) for r in rows
            if r.get("prompt_label") in ("safe", "unsafe")]


def aegis2_response_cases(rows):
    return [(str(r["id"]), turn(r["prompt"], r["response"]), int(r["response_label"] == "unsafe")) for r in rows
            if r.get("response") and r.get("response_label") in ("safe", "unsafe")]


def sst_cases(rows):
    return [(str(r["id"]), user(r["prompt"]), 1) for r in rows]


def harmbench_prompt_cases(rows):
    return [(r["BehaviorID"], user(r["Behavior"]), 1) for r in rows
            if r["FunctionalCategory"] in ("standard", "copyright")]


def harmbench_response_cases(data):
    """Classifier val set {behavior_id: [{test_case, generation, human_0..2, ...}]}: the prompt is the test case the
    model answered (safety-eval), label = majority of the 3 human labels."""
    out = []
    for behavior_id, items in data.items():
        for k, item in enumerate(items):
            votes = sum(int(item[f"human_{j}"]) for j in range(3))
            out.append((f"{behavior_id}:{k}", turn(item["test_case"], item["generation"]), int(votes > 1)))
    return out


def polyguard_english(rows):
    return [r for r in rows if r.get("language") == "English"]


def wildguard_prompt_cases(rows):
    return [(str(r.get("id", i)), user(r["prompt"]), int(r["prompt_harm_label"] == "harmful"))
            for i, r in enumerate(rows) if r.get("prompt_harm_label") in ("harmful", "unharmful")]


def wildguard_response_cases(rows):
    return [(str(r.get("id", i)), turn(r["prompt"], r["response"]), int(r["response_harm_label"] == "harmful"))
            for i, r in enumerate(rows)
            if r.get("response") and r.get("response_harm_label") in ("harmful", "unharmful")]


def saferlhf_cases(rows, per_class=SAFERLHF_PER_CLASS, seed=SAFERLHF_SEED):
    """safety-eval's SafeRLHF subset: flatten (prompt, response_k, is_response_k_safe) pairs of the rows whose two
    responses disagree (so the safe and unsafe lists align by row), then one random.Random(seed).sample of indices
    (the same as random.seed(42); random.sample) picks per_class safe and the same rows' unsafe responses."""
    safe, unsafe = [], []
    for i, r in enumerate(rows):
        flags = [truthy(r[f"is_response_{k}_safe"]) for k in (0, 1)]
        if flags[0] == flags[1]:
            continue
        for k in (0, 1):
            (safe if flags[k] else unsafe).append((f"{i}:{k}", turn(r["prompt"], r[f"response_{k}"])))
    indices = random.Random(seed).sample(range(len(safe)), min(per_class, len(safe)))
    return [(*safe[i], 0) for i in indices] + [(*unsafe[i], 1) for i in indices]


def beavertails_cases(rows):
    return [(str(i), turn(r["prompt"], r["response"]), int(not truthy(r["is_safe"]))) for i, r in enumerate(rows)]


def think_text(content):
    """'<think>\\nR\\n</think>\\n\\nA' -> 'R\\n\\nA', as the guard sees an assistant turn (reasoning, blank line,
    answer; guard_demo/server.py guard_text). Content without think markers is returned unchanged."""
    if "</think>" not in content:
        return content
    if content.count("</think>") != 1 or content.count("<think>") > 1:
        raise ValueError("ambiguous think markers")
    head, answer = content.split("</think>", 1)
    reasoning = head.split("<think>", 1)[1] if "<think>" in head else head
    reasoning, answer = reasoning.strip(), answer.strip()
    return reasoning + ("\n\n" if reasoning and answer else "") + answer


def think_cases(rows):
    out = []
    for r in rows:
        if r["label"] not in ("Safe", "Unsafe"):
            raise ValueError("unexpected label")
        messages = [{"role": m["role"], "content": think_text(m["content"]) if m["role"] == "assistant"
                     else m["content"]} for m in r["message"]]
        out.append((str(r["unique_id"]), messages, int(r["label"] == "Unsafe")))
    return out


def make_rows(bench, level, cases):
    """(source id, messages, label) -> case rows; ids are unique per bench (a repeated source id gets #n)."""
    seen, out = collections.Counter(), []
    for source_id, messages, label in cases:
        if level == "prompt" and [m["role"] for m in messages] != ["user"]:
            raise ValueError(f"{bench}: prompt cases hold one user message")
        if level == "response" and messages[-1]["role"] != "assistant":
            raise ValueError(f"{bench}: response cases end with the assistant message")
        if any(not isinstance(m["content"], str) for m in messages) or label not in (0, 1):
            raise ValueError(f"{bench}: bad case {source_id}")
        seen[source_id] += 1
        suffix = "" if seen[source_id] == 1 else f"#{seen[source_id]}"
        out.append({"id": f"{bench}:{source_id}{suffix}", "bench": bench, "level": level,
                    "messages": messages, "label": label})
    return out


# ---------------------------------------------------------------- download and parsing

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(name):
    kind, repo, revision, path, _ = SOURCES[name]
    return RAW / repo.replace("/", "__") / revision[:12] / path


def fetch(name, session, previous):
    """Download one pinned source anonymously; returns its manifest entry, or raises PermissionError when gated."""
    import requests
    kind, repo, revision, path, note = SOURCES[name]
    target = local_path(name)
    entry = {"kind": kind, "repo": repo, "revision": revision, "revision_note": note, "path": path,
             "local": str(target.relative_to(HERE))}
    if kind == "hf":
        info = session.get(f"{HF}/api/datasets/{repo}/revision/{revision}", timeout=60)
        if info.status_code in (401, 403):
            raise PermissionError(f"{repo}: HTTP {info.status_code}")
        info.raise_for_status()
        gated = info.json().get("gated")
        if gated:
            raise PermissionError(f"{repo}: gated ({gated})")
        url = f"{HF}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"
    else:
        url = f"{GITHUB_RAW}/{repo}/{revision}/{urllib.parse.quote(path)}"
    old = previous.get(name, {})
    if target.exists() and old.get("sha256") and old.get("revision") == revision \
            and target.stat().st_size == old.get("bytes") and sha256_file(target) == old["sha256"]:
        entry.update(bytes=old["bytes"], sha256=old["sha256"], reused=True)
        return entry
    response = session.get(url, timeout=600, allow_redirects=True)
    if response.status_code in (401, 403):
        raise PermissionError(f"{repo}/{path}: HTTP {response.status_code}")
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(target)
    entry.update(bytes=len(response.content), sha256=sha256_bytes(response.content), reused=False)
    return entry


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_bytes(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def read_source(name):
    path = local_path(name)
    suffix = path.name
    if suffix.endswith(".csv"):
        return read_csv(path)
    if suffix.endswith(".jsonl.gz"):
        return read_jsonl_bytes(gzip.decompress(path.read_bytes()))
    if suffix.endswith(".jsonl.xz"):
        return read_jsonl_bytes(lzma.decompress(path.read_bytes()))
    if suffix.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if suffix.endswith(".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    raise ValueError(path)


def build(bench, sources):
    if bench == "ToxicChat":
        return toxicchat_cases(sources["toxicchat"])
    if bench == "OpenAIMod":
        return openaimod_cases(sources["openaimod"])
    if bench == "Aegis":
        return aegis1_cases(sources["aegis1"])
    if bench == "Aegis2.0":
        return aegis2_prompt_cases(sources["aegis2"])
    if bench == "Aegis2.0-R":
        return aegis2_response_cases(sources["aegis2"])
    if bench == "SimpleSafetyTests":
        return sst_cases(sources["sst"])
    if bench == "HarmBench-P":
        return harmbench_prompt_cases(sources["harmbench_behaviors"])
    if bench == "HarmBench-R":
        return harmbench_response_cases(sources["harmbench_classifier_val"])
    if bench == "WildGuardTest-P":
        return wildguard_prompt_cases(polyguard_english(sources["polyguard"]))
    if bench == "WildGuardTest-R":
        return wildguard_response_cases(polyguard_english(sources["polyguard"]))
    if bench == "SafeRLHF":
        return saferlhf_cases(sources["saferlhf"])
    if bench == "BeaverTails":
        return beavertails_cases(sources["beavertails"])
    if bench == "Think":
        with THINK_PATH.open(encoding="utf-8") as handle:
            return think_cases(json.loads(line) for line in handle if line.strip())
    raise KeyError(bench)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benches", nargs="*", default=list(BENCHES), help="subset of benches to build")
    args = parser.parse_args()
    import requests
    unknown = sorted(set(args.benches) - set(BENCHES))
    if unknown:
        raise SystemExit(f"unknown benches {unknown}")
    manifest_path = DATA / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})
    session = requests.Session()
    session.trust_env = False            # no ~/.netrc credentials
    session.headers["User-Agent"] = "safety-guard-qwen3guard-bench/1"
    needed = sorted({s for b in args.benches for s in BENCHES[b][1]})
    sources, entries, failed = {}, {}, {}
    for name in needed:
        try:
            entries[name] = fetch(name, session, previous)
            sources[name] = read_source(name)
            print(f"source {name}: {entries[name]['bytes']} bytes"
                  f"{' (reused)' if entries[name]['reused'] else ''}", flush=True)
        except PermissionError as error:
            failed[name] = f"skipped: {error}"
            print(f"source {name}: {failed[name]}", flush=True)
    rows, benches, skipped = [], {}, dict(GATED_SKIPPED)
    for bench in args.benches:
        level, needs, expected = BENCHES[bench]
        missing = [s for s in needs if s in failed]
        if missing:
            skipped[bench] = "; ".join(failed[s] for s in missing)
            continue
        built = make_rows(bench, level, build(bench, sources))
        rows.extend(built)
        unsafe = sum(r["label"] for r in built)
        chars = sum(len(m["content"]) for r in built for m in r["messages"])
        record = {"level": level, "cases": len(built), "unsafe": unsafe, "safe": len(built) - unsafe,
                  "approx_tokens": round(chars / 3), "sources": list(needs)}
        if expected:
            record["expected_cases"], record["expected_unsafe"] = expected
            record["matches_expected"] = expected[0] == len(built) and expected[1] in (None, unsafe)
        benches[bench] = record
        print(f"{bench:18s} {level:8s} cases={len(built):5d} unsafe={unsafe:5d}"
              f" expected={expected if expected else '-'}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    cases_path = DATA / "cases.jsonl"
    partial = cases_path.with_name(cases_path.name + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(cases_path)
    local = {"think": {"path": str(THINK_PATH.relative_to(PROJECT)), "sha256": sha256_file(THINK_PATH)}}
    manifest = {"version": VERSION, "cases_sha256": sha256_file(cases_path), "cases": len(rows),
                "unsafe": sum(r["label"] for r in rows),
                "approx_tokens": round(sum(len(m["content"]) for r in rows for m in r["messages"]) / 3),
                "benches": benches, "skipped": skipped, "sources": {**previous, **entries}, "local_sources": local,
                "rules": {"saferlhf": {"per_class": SAFERLHF_PER_CLASS, "seed": SAFERLHF_SEED},
                          "think": "assistant text = reasoning + blank line + answer (think markers removed)",
                          "wildguardtest": "PolyGuardPrompts test, language == English (ungated copy of WildGuardTest)"}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("cases", "unsafe", "approx_tokens", "cases_sha256")}))


if __name__ == "__main__":
    sys.exit(main())
