"""Rough count of red-line positives already in the training data, without any judge calls (Mac, CPU).

The 13 red lines (collab/feedback/POLICY-redline-scope.md) do not map one-to-one onto public dataset categories:
BeaverTails' violence does not say whether it is gory, its politics is mostly not China's red lines, Aegis' Sexual
includes suggestive text. So every source category gets a designer-set tier:
  likely    the category's harmful content is usually inside the red line (e.g. self-harm -> R13)
  possible  only part of it is (e.g. weapons -> R12 only for making methods); an upper bound
  none      outside the red lines (privacy, fraud, harassment, ...): trained as safe under the final policy
Counts are per record: a record counts once for a red line at its highest tier. Political categories cannot be split
into R1/R3-R6, so they are reported together as "political". Real counts come from the red-line labelling (T020/T021);
this estimate is only for sizing the S1 gap early.

Run A carries no category; its lexicon source group is used as a topic proxy (the v14 unsafe responses were mostly
harassment and rumour whatever the word), reported separately and never added to the public counts.
Output: counts only, no text.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

POLITICAL = "political"          # R1, R3, R4, R5, R6 together
BEAVER = {
    "self_harm": ("likely", ("R13",)),
    "sexually_explicit,adult_content": ("possible", ("R11",)),
    "terrorism,organized_crime": ("possible", ("R7",)),
    "drug_abuse,weapons,banned_substance": ("possible", ("R12",)),
    "violence,aiding_and_abetting,incitement": ("possible", ("R2", "R12")),
    "controversial_topics,politics": ("possible", (POLITICAL,)),
    "hate_speech,offensive_language": ("possible", ("R8",)),
    "discrimination,stereotype,injustice": ("possible", ("R8",)),
    "misinformation_regarding_ethics,laws_and_safety": ("possible", ("R10",)),
    "child_abuse": ("possible", ("R2", "R11")),
    "animal_abuse": ("possible", ("R2",)),
    "financial_crime,property_crime,theft": ("none", ()),
    "non_violent_unethical_behavior": ("none", ()),
    "privacy_violation": ("none", ()),
}
AEGIS = {                        # Nemotron Safety Guard v3 violated_categories, lower-cased
    "suicide and self harm": ("likely", ("R13",)),
    "sexual (minor)": ("likely", ("R11",)),
    "sexual": ("possible", ("R11",)),
    "violence": ("possible", ("R2",)),
    "threat": ("possible", ("R2",)),
    "guns and illegal weapons": ("possible", ("R12",)),
    "controlled/regulated substances": ("possible", ("R12",)),
    "criminal planning/confessions": ("possible", ("R12",)),
    "illegal activity": ("possible", ("R12",)),
    "hate/identity hate": ("possible", ("R8",)),
    "political/misinformation/conspiracy": ("possible", (POLITICAL, "R10")),
    "high risk gov decision making": ("possible", (POLITICAL,)),
    "harassment": ("none", ()), "pii/privacy": ("none", ()), "profanity": ("none", ()),
    "fraud/deception": ("none", ()), "malware": ("none", ()), "copyright/trademark/plagiarism": ("none", ()),
    "unauthorized advice": ("none", ()), "immoral/unethical": ("none", ()), "manipulation": ("none", ()),
    "needs caution": ("none", ()), "other": ("none", ()),
}
RUNA_GROUPS = {                  # lexicon source groups whose words are about one red-line topic
    "permissive/反动词库.txt": (POLITICAL,), "permissive/政治类型.txt": (POLITICAL,), "citizenlab/june-4": (POLITICAL,),
    "permissive/新思想启蒙.txt": (POLITICAL,), "permissive/暴恐词库.txt": ("R7",), "permissive/涉枪涉爆.txt": ("R12",),
    "permissive/色情类型.txt": ("R11",), "permissive/色情词库.txt": ("R11",),
    "citizenlab/coronavirus": ("R10",), "permissive/COVID-19词库.txt": ("R10",),
}
LINES = (POLITICAL, "R2", "R7", "R8", "R9", "R10", "R11", "R12", "R13")
RANK = {"none": 0, "possible": 1, "likely": 2}


def tiers(categories, table):
    """categories -> ({red line: best tier}, unmapped category names)."""
    best, unmapped = {}, []
    for name in categories:
        key = name.strip().lower()
        if key not in table:
            unmapped.append(key)
            continue
        tier, lines = table[key]
        for line in lines:
            if RANK[tier] > RANK[best.get(line, "none")]:
                best[line] = tier
    return best, unmapped


def beaver_categories(row):
    return [k for k, v in (row.get("category") or {}).items() if v]


def aegis_categories(row):
    value = row.get("violated_categories") or []
    return [c for c in (value.split(",") if isinstance(value, str) else value) if c.strip()]


def tally(records):
    """records: (split, source, {line: tier}) for unsafe records. Returns the count tables."""
    out = collections.defaultdict(lambda: {"records": 0, "no_red_line": 0,
                                          **{line: {"likely": 0, "possible_or_likely": 0} for line in LINES}})
    for split, source, best in records:
        for key in (f"{split}/{source}", f"{split}/all"):
            entry = out[key]
            entry["records"] += 1
            entry["no_red_line"] += not best
            for line, tier in best.items():
                entry[line]["possible_or_likely"] += 1
                entry[line]["likely"] += tier == "likely"
    return {k: out[k] for k in sorted(out)}


def estimate(prefix_rows, beaver, aegis_raw, candidates):
    """Pure. prefix_rows: exported prefix_v2 rows; beaver: {line index: raw row}; aegis_raw: {id: raw row};
    candidates: {candidate_id: source_id}. Returns (table, unmapped counts, missing counts)."""
    records, unmapped, missing = [], collections.Counter(), collections.Counter()
    for row in prefix_rows:
        if row["label"] != "unsafe":
            continue
        sample_id = row["sample_id"]
        if sample_id.startswith("beaver-"):
            raw, table, source, categories = beaver.get(int(sample_id.split("-")[1])), BEAVER, "beavertails", beaver_categories
        elif sample_id in candidates or sample_id in aegis_raw:
            raw, table, source, categories = aegis_raw.get(candidates.get(sample_id, sample_id)), AEGIS, "nemotron_zh", aegis_categories
        else:
            missing[row.get("label_tier") or "unknown"] += 1       # synthetic records carry no source category
            continue
        if raw is None:
            missing[source] += 1
            continue
        best, names = tiers(categories(raw), table)
        unmapped.update(f"{source}:{n}" for n in names)
        records.append((row["split"], source, best))
    return tally(records), dict(unmapped), dict(missing)


def runA_topics(rows):
    """Unsafe Run A responses by red-line topic of their word's lexicon source group (proxy only)."""
    out = collections.defaultdict(collections.Counter)
    for row in rows:
        if row.get("label") != "unsafe":
            continue
        lines = RUNA_GROUPS.get(row.get("source_group", ""), ("mixed_or_unknown",))
        for line in lines:
            out[row["split"]][line] += 1
        out[row["split"]]["records"] += 1
    return {split: dict(counter) for split, counter in sorted(out.items())}


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-rows", type=Path, default=Path("round6/redline_v1/input/prefix_v2_assistant.jsonl"))
    parser.add_argument("--beaver", type=Path, default=Path("round3/data/raw/beavertails_round0_30k/train.jsonl.gz"))
    parser.add_argument("--nemotron-dir", type=Path, default=Path("round3/data/raw/nemotron_safety_guard_v3_zh"))
    parser.add_argument("--nemotron-candidates", type=Path,
                        default=Path("round3/data/candidates/nemotron_assistant_candidates.jsonl"))
    parser.add_argument("--runA", type=Path, default=Path("round6/response_v14/batch_50k/extracted/trainable.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("round6/redline_estimate/estimate_v1.json"))
    args = parser.parse_args()
    beaver = dict(enumerate(read_jsonl(args.beaver)))
    aegis_raw = {r["id"]: r for split in ("train", "valid") for r in read_jsonl(args.nemotron_dir / f"{split}.jsonl")}
    candidates = {r["candidate_id"]: r["source_id"] for r in read_jsonl(args.nemotron_candidates)}
    table, unmapped, missing = estimate(read_jsonl(args.prefix_rows), beaver, aegis_raw, candidates)
    result = {"version": "redline-estimate-v1", "model_calls": 0,
              "tiers": {"beavertails": {k: [t, list(v)] for k, (t, v) in BEAVER.items()},
                        "nemotron_zh": {k: [t, list(v)] for k, (t, v) in AEGIS.items()},
                        "runA_topic_groups": {k: list(v) for k, v in RUNA_GROUPS.items()}},
              "prefix_v2_unsafe": table, "unmapped_categories": unmapped, "records_without_category": missing,
              "runA_unsafe_by_topic_proxy": runA_topics(read_jsonl(args.runA))}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train = table.get("train/all", {})
    print(f"prefix_v2 train unsafe records: {train.get('records')}, no red line: {train.get('no_red_line')}")
    for line in LINES:
        if train:
            print(f"  {line:10} likely {train[line]['likely']:6}   possible or likely {train[line]['possible_or_likely']:6}")
    print("records without category:", missing, " unmapped:", unmapped)
    print("Run A unsafe by topic proxy:", json.dumps(result["runA_unsafe_by_topic_proxy"], ensure_ascii=False))


if __name__ == "__main__":
    main()
