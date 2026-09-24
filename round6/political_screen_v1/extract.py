"""Download political-screen archives of one or more passes and merge them per word (Mac, no model calls).

political_terms.jsonl : one row per word: merged verdict (most alerting across passes), ambiguous (any pass),
                        per-pass verdicts, and with --leader-screen the T012 verdict (leader_screen). Holds the
                        words: keep it local and on the PVC, never commit it. Input of redline_v1/apply_policy.py
                        --political-terms.
confirm_words.txt     : words for the confirmation pass: merged verdict event / figure / org / leader / unsure, or
                        flagged by the T012 leader screen (insult / rumor / evasion / unsure).
missing_words.txt     : with --words, the words that have no verdict yet (re-screen them with a new tag).
leader_words.txt      : words whose merged verdict is leader, for the leader-form pass (make_batch.py --mode leader).
Leader-form passes (v1.1) are merged apart from the screen passes into leader_form (most alerting across them);
their ambiguous flag joins the word's. A leader word is matched only when leader_form is negative or variant.
check_words.txt       : the candidates for the final check pass (v1.2, make_batch.py --mode check): event / figure /
                        org words and leader words with a negative or variant form, ambiguous or not. Once a check
                        pass is merged (field check; its passes must agree), only its fire verdicts are matched.
review/terms_sample.txt: with --review-sample N, up to N words per matched_as kind for a local check (gitignored).
summary.json          : counts only.
Write --out under an extracted/ directory (gitignored text files); never commit the .jsonl or .txt files.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
sys.path.insert(0, str(HERE.parent))
from aster_io import collect_archives  # noqa: E402
from pipeline import CHECK_FIRE, merge  # noqa: E402

CONFIRM = ("event", "figure", "org", "leader", "unsure")
LEADER_FLAGGED = ("insult", "rumor", "evasion", "unsure")          # T012 verdicts
LEADER_MATCHED = ("negative", "variant")                          # leader_form values that are matched
MATCHED = ("event", "figure", "org")


def batch_values(archives):
    for path in archives:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith("flow/results/") and member.name.endswith(".json"):
                    record = json.load(tar.extractfile(member))
                    if record.get("name") == "batch" and isinstance(record.get("value"), dict):
                        yield record["value"]


def join(values, leader=None):
    """Pure: word -> merged row; the pass tag is the second field of the batch key (political-<tag>-NNNNN).
    leader: word -> T012 verdict, attached as leader_screen."""
    per_word = {mode: collections.defaultdict(dict) for mode in ("screen", "leader", "check")}
    errors = collections.Counter()
    for value in values:
        tag, mode = value["batch_key"].split("-")[1], value.get("mode", "screen")
        for error in value.get("errors", []):
            errors[error.split(":")[0]] += 1
        for v in value.get("verdicts", []):
            per_word[mode][v["word"]][tag] = (v["verdict"], v["ambiguous"])
    rows = []
    for word in sorted(per_word["screen"]):
        verdict, ambiguous = merge(list(per_word["screen"][word].values()))
        row = {"word": word, "verdict": verdict, "ambiguous": ambiguous,
               "passes": {tag: {"verdict": v, "ambiguous": a} for tag, (v, a) in sorted(per_word["screen"][word].items())}}
        if word in per_word["leader"]:
            form, form_ambiguous = merge(list(per_word["leader"][word].values()), "leader")
            row.update(leader_form=form, ambiguous=ambiguous or form_ambiguous,
                       leader_passes={tag: {"verdict": v, "ambiguous": a}
                                      for tag, (v, a) in sorted(per_word["leader"][word].items())})
        if word in per_word["check"]:
            check, check_ambiguous = merge(list(per_word["check"][word].values()), "check")
            row.update(check=check, ambiguous=row["ambiguous"] or check_ambiguous,
                       check_passes={tag: {"verdict": v, "ambiguous": a}
                                     for tag, (v, a) in sorted(per_word["check"][word].items())})
        if leader is not None:
            row["leader_screen"] = leader.get(word, "missing")
        rows.append(row)
    return rows, dict(errors)


def to_confirm(row):
    return row["verdict"] in CONFIRM or row.get("leader_screen") in LEADER_FLAGGED


def candidate_kind(row):
    """What the screen and leader passes make of the word (the input of the check pass), or None."""
    if row["verdict"] in MATCHED:
        return row["verdict"]
    if (row["verdict"] == "leader" and row.get("leader_screen") in LEADER_FLAGGED
            and row.get("leader_form") in LEADER_MATCHED):
        return "leader_" + row["leader_form"]
    return None


def matched_as(row):
    """How apply_policy.py --political-terms uses the word (None: not matched); single characters never match.
    After a check pass, only its fire verdicts count and the check sets the kind."""
    if len(row["word"]) < 2:
        return None
    kind = (row["check"] if row["check"] in CHECK_FIRE else None) if "check" in row else candidate_kind(row)
    return kind + (":ambiguous" if row["ambiguous"] else "") if kind else None


def review_sheet(rows, per_kind, seed="political-review"):
    """Local sheet: up to per_kind words of each matched_as kind, in hash order, with the question to answer."""
    kinds = collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['word']}".encode()).hexdigest()):
        kind = matched_as(row)
        if kind and len(kinds[kind]) < per_kind:
            kinds[kind].append(row["word"])
    lines = ["# 每个词回答：对 / 普通词（单独出现不该截）/ 类别错 / 不是政治敏感词", ""]
    for kind in sorted(kinds):
        lines += [f"## {kind}（{len(kinds[kind])} 个）"] + [f"{word}\t" for word in kinds[kind]] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--attempts-json", type=Path, nargs="*", default=[])
    parser.add_argument("--words", type=Path, help="the word index of pass 1 (words.jsonl), to list words without a verdict")
    parser.add_argument("--leader-screen", type=Path, help="T012 screen.jsonl: attach its verdict to every word")
    parser.add_argument("--review-sample", type=int, default=0, help="words per matched_as kind in review/terms_sample.txt")
    parser.add_argument("--review-seed", default="political-review", help="a new seed draws a fresh review sample")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    leader = None
    if args.leader_screen:
        with args.leader_screen.open(encoding="utf-8") as handle:
            leader = {r["word"]: r["verdict"] for r in map(json.loads, filter(str.strip, handle))}
    archives, runs = collect_archives(args.runs, args.attempts_json, args.out / "archives")
    rows, errors = join(batch_values(archives), leader)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "political_terms.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "confirm_words.txt").write_text("".join(r["word"] + "\n" for r in rows if to_confirm(r)), encoding="utf-8")
    (args.out / "check_words.txt").write_text("".join(r["word"] + "\n" for r in rows if candidate_kind(r)),
                                              encoding="utf-8")
    (args.out / "leader_words.txt").write_text("".join(r["word"] + "\n" for r in rows if r["verdict"] == "leader"),
                                               encoding="utf-8")
    missing = None
    if args.words:
        with args.words.open(encoding="utf-8") as handle:
            index = {json.loads(line)["word"] for line in handle if line.strip()}
        missing = sorted(index - {r["word"] for r in rows})
        (args.out / "missing_words.txt").write_text("".join(w + "\n" for w in missing), encoding="utf-8")
    if args.review_sample:
        (args.out / "review").mkdir(exist_ok=True)
        (args.out / "review/terms_sample.txt").write_text(review_sheet(rows, args.review_sample, args.review_seed),
                                                          encoding="utf-8")
    summary = {"runs": runs, "archives": len(archives), "words": len(rows), "batch_errors": errors,
               "words_without_verdict": len(missing) if missing is not None else None,
               "passes": dict(collections.Counter(tag for r in rows for tag in r["passes"])),
               "verdicts": dict(collections.Counter(r["verdict"] for r in rows)),
               "ambiguous_by_verdict": dict(collections.Counter(r["verdict"] for r in rows if r["ambiguous"])),
               "leader_screen_by_verdict": ({v: dict(collections.Counter(r["leader_screen"] for r in rows if r["verdict"] == v))
                                             for v in sorted({r["verdict"] for r in rows})} if leader is not None else None),
               "leader_form": dict(collections.Counter(r["leader_form"] for r in rows if "leader_form" in r)),
               "leader_form_by_leader_screen": ({s: dict(collections.Counter(r["leader_form"] for r in rows
                                                                            if "leader_form" in r and r.get("leader_screen") == s))
                                                 for s in sorted({r.get("leader_screen") for r in rows if "leader_form" in r})}
                                                if leader is not None else None),
               "leader_words": sum(r["verdict"] == "leader" for r in rows),
               "check_words": sum(bool(candidate_kind(r)) for r in rows),
               "check_by_candidate": {k: dict(collections.Counter(r["check"] for r in rows
                                                                  if "check" in r and candidate_kind(r) == k))
                                      for k in sorted({candidate_kind(r) for r in rows if "check" in r and candidate_kind(r)})},
               "matched_as": dict(sorted(collections.Counter(m for m in map(matched_as, rows) if m).items())),
               "confirm_words": sum(map(to_confirm, rows))}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
