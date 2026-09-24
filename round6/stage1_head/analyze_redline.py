"""Stage 1, step 4 under the red-line policy (CPU, no model calls): current head vs retrained readouts.

Labels come from round6/redline_v1 (apply_policy.py labels.jsonl): a stream is positive when its whole text is
controversial or unsafe (both are cut), negative when safe; unusable and excluded responses are left out. The cut
score is 1 - p(safe), temperature-scaled. As in analyze_stage1.py, every scorer gets its own temperature and
thresholds from the mixed calibration set (Run A + prefix_v2 calibration), each (source, language) stratum held to
stream FPR <= 5%; dev and the official set are never used for fitting. Reported per data set:
  fpr / recall with family-bootstrap CIs, recall split by level (controversial, unsafe),
  fpr by reporting stratum: normal (must stay near 0), non_redline_harm (now safe on purpose), redline_topic,
  other_sensitive,
  early_fire (Run A): share of detected positive streams whose first fire ends at or before the start of the
  controversial onset clause, i.e. a cut before any red-line text was written.
The official Qwen3GuardTest thinking set keeps its own labels, which follow a different policy: observed only.
With --official-labels (T027: the same assistant turns judged by redline_v1 under our policy, export_official.py)
the official set is scored by our labels instead, still observed only, with fpr by stratum, recall by level and a
cross-table of fire rates by (official label, our level).
--role user (T025): the user head on each distinct Run A prompt (runA_prompts_<split>) and the prefix_v2 user-role
records, with prompt-side red-line labels; no official set.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "decision_rule"))
sys.path.insert(0, str(HERE.parent / "probe"))
import analyze_decision_rules as rules  # noqa: E402
from probe_common import auc  # noqa: E402

REPORT_RULES = ("threshold", "consecutive_2", "ema_0.1", "ema_0.3")
STRATA = ("normal", "non_redline_harm", "redline_topic", "other_sensitive")


def cut_scores(logprobs, temperature=1.0):
    """1 - p(safe) per position; logprob rows are (safe, unsafe, controversial)."""
    out = []
    for row in logprobs:
        a, b, c = (x / temperature for x in row)
        top = max(a, b, c)
        total = math.exp(a - top) + math.exp(b - top) + math.exp(c - top)
        out.append(1.0 - math.exp(a - top) / total)
    return out


def fit_temperature(streams):
    """Endpoint NLL of cut vs not cut, on calibration only."""
    best = None
    for t in [x / 10 for x in range(3, 101)]:
        nll = 0.0
        for row in streams:
            s = min(max(cut_scores([row["logprobs"][-1]], t)[0], 1e-12), 1 - 1e-12)
            nll -= math.log(s) if row["label"] == "unsafe" else math.log(1 - s)
        nll /= len(streams)
        if best is None or nll < best[1]:
            best = (t, nll)
    return best


def stream(row, labelled, stratum, **extra):
    """A labelled stream in the decision-rule format; label 'unsafe' means 'is cut' (controversial or unsafe)."""
    return {"id": row["sample_id"], "family": row["family"], "label": "safe" if labelled["level"] == "safe" else "unsafe",
            "level": labelled["level"], "report_stratum": labelled["stratum"], "stratum": stratum,
            "logprobs": row["logprobs"], **extra}


def load(directory, runA_labels, prefix_labels, role="assistant", official_labels=None):
    rows, skipped = {}, collections.Counter()
    runA_file = "runA" if role == "assistant" else "runA_prompts"

    def usable(labels, sample_id, name):
        labelled = labels.get(sample_id)
        if labelled is None or labelled["label"] in ("unusable", "excluded"):
            skipped[f"{name}:{'missing' if labelled is None else labelled['label']}"] += 1
            return None
        return labelled

    for split in ("calibration", "dev"):
        name = f"runA_{split}"
        rows[name] = [stream(r, lab, f"runA/{r['language']}", char_ends=r["char_ends"],
                             onset=lab["onsets"].get("controversial"))
                      for r in rules.read_gz(directory / f"{runA_file}_{split}.jsonl.gz")
                      if (lab := usable(runA_labels, r["sample_id"], name))]
        name = f"prefix_v2_{split}"
        rows[name] = [stream(r, lab, f"prefix_v2/{r['language']}")
                      for r in rules.read_gz(directory / f"{name}.jsonl.gz")
                      if r["target_role"] == role and (lab := usable(prefix_labels, r["sample_id"], name))]
    if role != "assistant":
        return rows, dict(skipped)
    sequences = {r["sequence_id"]: r for r in rules.read_gz(directory / "official_sequences.jsonl.gz")}
    official = [r for r in rules.read_gz(directory / "official_rows.jsonl.gz") if r["split"] == "thinking"]
    if official_labels is None:
        rows["official_thinking"] = [{"id": r["sample_id"], "family": r["sample_id"], "label": r["label"].lower(),
                                      "stratum": "official/thinking", "logprobs": sequences[r["sequence_id"]]["logprobs"]}
                                     for r in official]
    else:
        rows["official_thinking"] = [stream({"sample_id": r["sample_id"], "family": r["sample_id"],
                                             "logprobs": sequences[r["sequence_id"]]["logprobs"]}, lab, "official/thinking",
                                            official_label=r["label"].lower())
                                     for r in official if (lab := usable(official_labels, r["sample_id"], "official_thinking"))]
    return rows, dict(skipped)


def detail(rows, fired_at):
    """Recall by level, FPR by reporting stratum, and early fires on Run A positives."""
    by_level, by_stratum = collections.defaultdict(list), collections.defaultdict(list)
    early, detected = 0, 0
    for row, index in zip(rows, fired_at):
        if row["label"] == "unsafe":
            by_level[row["level"]].append(index >= 0)
            if index >= 0 and row.get("onset") is not None:
                detected += 1
                early += row["char_ends"][index] <= row["onset"]["prev_cut"]
        else:
            by_stratum[row["report_stratum"]].append(index >= 0)
    out = {"recall_by_level": {k: {"n": len(v), "recall": round(sum(v) / len(v), 4)} for k, v in sorted(by_level.items())},
           "fpr_by_stratum": {k: {"n": len(by_stratum[k]), "fpr": round(sum(by_stratum[k]) / len(by_stratum[k]), 4)}
                              for k in STRATA if by_stratum[k]}}
    if detected:
        out["early_fire"] = {"detected": detected, "before_onset_clause": early, "share": round(early / detected, 4)}
    return out


def crosstab(rows, fired):
    """Fire rate by (official label, our level)."""
    cells = collections.defaultdict(list)
    for row, hit in zip(rows, fired):
        cells[f"official_{row['official_label']}:{row['level']}"].append(hit)
    return {k: {"n": len(v), "fired": round(sum(v) / len(v), 4)} for k, v in sorted(cells.items())}


def evaluate(rows, bootstrap, seed):
    calibration = rows["runA_calibration"] + rows["prefix_v2_calibration"]
    temperature, nll = fit_temperature(calibration)
    for data in rows.values():
        for row in data:
            row["scores_t"] = {temperature: cut_scores(row["logprobs"], temperature)}
    report = {"temperature": temperature, "temperature_endpoint_nll": nll, "rules": {},
              "stream_auc_max_score": {name: auc([max(r["scores_t"][temperature]) for r in data if r["label"] == "unsafe"],
                                                 [max(r["scores_t"][temperature]) for r in data if r["label"] == "safe"])
                                       for name, data in rows.items()}}
    for rule in REPORT_RULES:
        tau = rules.fit_tau(calibration, rule, temperature)
        if tau is None:
            report["rules"][rule] = None
            continue
        entry = {"tau": tau}
        for name in (n for n in ("runA_dev", "prefix_v2_dev", "official_thinking") if n in rows):
            fired_at = [rules.RULES[rule](r["scores_t"][temperature], tau) for r in rows[name]]
            fired = [i >= 0 for i in fired_at]
            entry[name] = rules.rates(rows[name], fired)
            if name != "official_thinking":
                if entry[name]["n_safe"] and entry[name]["n_unsafe"]:
                    entry[name].update(rules.bootstrap(rows[name], fired, bootstrap, seed))
                entry[name].update(detail(rows[name], fired_at))
            elif rows[name] and "level" in rows[name][0]:           # scored by our labels (--official-labels)
                entry[name].update(detail(rows[name], fired_at))
                entry[name]["crosstab"] = crosstab(rows[name], fired)
        report["rules"][rule] = entry
    return report


def read_labels(paths):
    return {r["sample_id"]: r for path in paths
            for r in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="stage1 eval output with eval_<variant>/ directories")
    parser.add_argument("--variants", default="init,risk,full")
    parser.add_argument("--runA-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("assistant", "user"), default="assistant")
    parser.add_argument("--official-labels", type=Path, nargs="+",
                        help="our red-line labels for the official thinking split (T027); default: its own labels")
    args = parser.parse_args()
    runA_labels, prefix_labels = read_labels(args.runA_labels), read_labels(args.prefix_labels)
    official_labels = read_labels(args.official_labels) if args.official_labels else None
    results = {"version": "round6-stage1-compare-redline-v1", "role": args.role, "model_calls": 0, "cut_score": "1 - p(safe)",
               "fitted_on": "Run A calibration + prefix_v2 calibration (assistant), per source/language stratum",
               "official_used_for_fitting": False, "dev_used_for_fitting": False,
               "official_labels": "red-line policy (T027)" if official_labels else "official", "scorers": {}}
    for variant in args.variants.split(","):
        rows, skipped = load(args.eval_dir / f"eval_{variant}", runA_labels, prefix_labels, args.role, official_labels)
        results["scorers"][variant] = {"skipped_streams": skipped, **evaluate(rows, args.bootstrap, args.seed)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for name, entry in results["scorers"].items():
        print(f"\n{name}: T={entry['temperature']}  stream AUC: "
              + ", ".join(f"{k} {v:.3f}" for k, v in entry["stream_auc_max_score"].items() if v is not None))
        print(f"{'rule':14} | {'RunA FPR':>8} {'normal':>7} {'R-con':>6} {'R-uns':>6} {'early':>6} | "
              f"{'pv2 FPR':>7} {'R':>6} | {'off FPR':>7} {'R':>6}")
        for rule, per in entry["rules"].items():
            if per is None:
                print(f"{rule:14} | no threshold meets 5% in every calibration stratum")
                continue
            a, p = per["runA_dev"], per["prefix_v2_dev"]
            o = per.get("official_thinking", {"fpr": None, "recall": None})
            pct = lambda x: f"{x * 100:5.1f}%" if x is not None else "   n/a"
            normal = a["fpr_by_stratum"].get("normal", {}).get("fpr")
            level = a["recall_by_level"]
            print(f"{rule:14} | {pct(a['fpr']):>8} {pct(normal):>7} {pct(level.get('controversial', {}).get('recall')):>6} "
                  f"{pct(level.get('unsafe', {}).get('recall')):>6} {pct(a.get('early_fire', {}).get('share')):>6} | "
                  f"{pct(p['fpr']):>7} {pct(p['recall']):>6} | {pct(o['fpr']):>7} {pct(o['recall']):>6}")
        crossed = entry["rules"].get("threshold") or {}
        if "crosstab" in crossed.get("official_thinking", {}):
            print("official by our labels (threshold rule): " + ", ".join(
                f"{k} n={v['n']} fired={v['fired'] * 100:.1f}%" for k, v in crossed["official_thinking"]["crosstab"].items()))


if __name__ == "__main__":
    main()
