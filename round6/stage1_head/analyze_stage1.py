"""Stage 1, step 4 (CPU, no model calls): current head vs retrained readouts, thresholds from a MIXED calibration set.

T014 showed that thresholds fitted on Run A alone do not carry over to other text. Here every scorer (the current
head, read from T014's files, and each eval_<variant> directory) gets its own temperature and thresholds fitted on
Run A calibration + prefix_v2 calibration together, with every (source, language) stratum held to stream FPR <= 5%.
Reported: Run A dev (with the restating breakdown), prefix_v2 dev and the official thinking set (observed only),
plus threshold-free stream AUCs. Same rule, temperature and bootstrap code as decision_rule v1, T005 and T014.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "decision_rule"))
sys.path.insert(0, str(HERE.parent / "probe"))
import analyze_decision_rules as rules  # noqa: E402
from analyze_runA_rules import breakdown  # noqa: E402
from probe_common import auc  # noqa: E402

REPORT_RULES = ("threshold", "consecutive_2", "ema_0.1", "ema_0.3")


def load(directory):
    """Rows keyed by data set; the mixed calibration strata carry the source name."""
    rows = {}
    for split in ("calibration", "dev"):
        rows[f"runA_{split}"] = [{"id": r["sample_id"], "family": r["family"], "label": r["label"],
                                  "stratum": f"runA/{r['language']}", "slot": r["slot"], "style": r["response_style"],
                                  "classes": r["classes"], "logprobs": r["logprobs"]}
                                 for r in rules.read_gz(directory / f"runA_{split}.jsonl.gz")]
        rows[f"prefix_v2_{split}"] = [{"id": r["sample_id"], "family": r["family"], "label": r["source_label"],
                                       "stratum": f"prefix_v2/{r['language']}", "logprobs": r["logprobs"]}
                                      for r in rules.read_gz(directory / f"prefix_v2_{split}.jsonl.gz")
                                      if r["target_role"] == "assistant"]
    sequences = {r["sequence_id"]: r for r in rules.read_gz(directory / "official_sequences.jsonl.gz")}
    rows["official_thinking"] = [{"id": r["sample_id"], "family": r["sample_id"], "label": r["label"].lower(),
                                  "stratum": "official/thinking", "logprobs": sequences[r["sequence_id"]]["logprobs"]}
                                 for r in rules.read_gz(directory / "official_rows.jsonl.gz") if r["split"] == "thinking"]
    return rows


def evaluate(rows, bootstrap, seed):
    calibration = rows["runA_calibration"] + rows["prefix_v2_calibration"]
    temperature, nll = rules.fit_temperature(calibration)
    for data in rows.values():
        for row in data:
            row["scores_t"] = {temperature: rules.unsafe_scores(row["logprobs"], temperature)}
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
        for name in ("runA_dev", "prefix_v2_dev", "official_thinking"):
            fired = rules.fires(rows[name], rule, tau, temperature)
            entry[name] = rules.rates(rows[name], fired)
            if name != "official_thinking":
                entry[name].update(rules.bootstrap(rows[name], fired, bootstrap, seed))
        entry["runA_dev_breakdown"] = breakdown(rows["runA_dev"], rule, tau, temperature)
        report["rules"][rule] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="T014 results_transfer_v1 directory (current head)")
    parser.add_argument("--eval-dir", type=Path, required=True, help="stage1 eval output with eval_<variant>/ directories")
    parser.add_argument("--variants", default="risk,full")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorers = {"current": args.current, **{v: args.eval_dir / f"eval_{v}" for v in args.variants.split(",")}}
    results = {"version": "round6-stage1-compare-v1", "model_calls": 0,
               "fitted_on": "Run A calibration + prefix_v2 calibration (assistant), per source/language stratum",
               "official_used_for_fitting": False, "dev_used_for_fitting": False,
               "scorers": {name: evaluate(load(path), args.bootstrap, args.seed) for name, path in scorers.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for name, entry in results["scorers"].items():
        print(f"\n{name}: T={entry['temperature']}  stream AUC: "
              + ", ".join(f"{k} {v:.3f}" for k, v in entry["stream_auc_max_score"].items() if v is not None))
        print(f"{'rule':14} | {'RunA dev FPR':>12} {'R':>6} {'restating':>9} | {'prefix_v2 FPR':>13} {'R':>6} | {'official FPR':>12} {'R':>6}")
        for rule, per in entry["rules"].items():
            if per is None:
                print(f"{rule:14} | no threshold meets 5% in every calibration stratum")
                continue
            a, p, o = per["runA_dev"], per["prefix_v2_dev"], per["official_thinking"]
            restating = per["runA_dev_breakdown"]["safe_fpr_by_slot_style"].get("slot0:risk_reasoning")
            print(f"{rule:14} | {a['fpr']*100:11.1f}% {a['recall']*100:5.1f}% {restating*100 if restating is not None else float('nan'):8.1f}% | "
                  f"{p['fpr']*100:12.1f}% {p['recall']*100:5.1f}% | {o['fpr']*100:11.1f}% {o['recall']*100:5.1f}%")


if __name__ == "__main__":
    main()
