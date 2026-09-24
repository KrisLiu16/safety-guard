"""T005: refit the streaming decision rule on Run A calibration (CPU only, no model calls).

Same rules, temperature fit and family bootstrap as round6/decision_rule/analyze_decision_rules.py
(imported unchanged), but fitted on Run A's calibration split: 100-300 token assistant responses in
the 2x2 layout, a closer match to thinking-style streams than the ~50-token prefix_v2 calibration.
Temperature and thresholds are fitted on Run A calibration only (stream FPR <= 5% per language).
Reported on Run A dev, with safe-stream FPR broken down by slot and response style (the
restating risk_reasoning style separately) and, for unsafe streams, how often the first fire
lands before the labelled onset. prefix_v2 dev and the official thinking set are observed only,
when the round6/decision_rule dump directory is given.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "decision_rule"))
import analyze_decision_rules as rules  # noqa: E402

PREVIOUS_TEMPERATURE = 2.2   # decision_rule v1, fitted on prefix_v2 calibration


def load_runA(dump_dir, split, temperatures):
    rows = []
    for r in rules.read_gz(dump_dir / f"runA_{split}.jsonl.gz"):
        rows.append({"id": r["sample_id"], "family": r["family"], "label": r["label"],
                     "stratum": f"{r['language']}/assistant", "slot": r["slot"], "style": r["response_style"],
                     "classes": r["classes"], "logprobs": r["logprobs"]})
    for row in rows:
        row["scores_t"] = {t: rules.unsafe_scores(row["logprobs"], t) for t in temperatures}
    return rows


def breakdown(rows, rule, tau, t):
    first = [rules.RULES[rule](r["scores_t"][t], tau) for r in rows]
    safe_by = collections.defaultdict(list)
    early, detected = 0, 0
    for row, index in zip(rows, first):
        if row["label"] == "safe":
            safe_by[f"slot{row['slot']}:{row['style']}"].append(index >= 0)
        elif index >= 0:
            detected += 1
            early += row["classes"][index] == "P"
    return {"safe_fpr_by_slot_style": {k: round(sum(v) / len(v), 4) for k, v in sorted(safe_by.items())},
            "unsafe_detected": detected,
            "unsafe_first_fire_before_onset": early,
            "unsafe_first_fire_before_onset_share": round(early / detected, 4) if detected else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="probe_v1 dump (runA_*.jsonl.gz)")
    parser.add_argument("--v1-dump", type=Path, default=None,
                        help="decision_rule v1 dump dir, to observe prefix_v2 dev and official thinking")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "runA_rules_v1_results.json")
    args = parser.parse_args()

    raw_cal = load_runA(args.dump_dir, "calibration", [1.0])
    temperature, nll = rules.fit_temperature(raw_cal)
    temperatures = sorted({1.0, temperature, PREVIOUS_TEMPERATURE})
    cal = load_runA(args.dump_dir, "calibration", temperatures)
    dev = load_runA(args.dump_dir, "dev", temperatures)
    observed = {}
    if args.v1_dump:
        prefix, official = rules.load(args.v1_dump, temperatures)
        observed = {"prefix_v2_dev": [r for r in prefix["dev"] if r["stratum"] in rules.ASSISTANT],
                    "official_thinking": official}

    results = {"version": "round6-runA-rules-v1", "model_calls": 0,
               "fitted_on": "Run A calibration (zh/assistant, en/assistant)", "official_used_for_fitting": False,
               "target_stream_fpr": rules.TARGET_FPR, "temperature": temperature, "temperature_endpoint_nll": nll,
               "streams": {name: dict(collections.Counter(r["label"] for r in rows)) for name, rows in (("calibration", cal), ("dev", dev))},
               "reference_official_argmax2": {"runA_dev": rules.rates(dev, [rules.official_argmax2(r["logprobs"])[0] == 1 for r in dev])},
               "rules": {}}
    for rule in rules.RULES:
        if rule.startswith("skip"):
            continue
        for t in temperatures:
            tau = rules.fit_tau(cal, rule, t)
            if tau is None:
                continue
            dev_fired = rules.fires(dev, rule, tau, t)
            entry = {"tau": tau, "calibration": rules.rates(cal, rules.fires(cal, rule, tau, t)),
                     "runA_dev": {**rules.rates(dev, dev_fired), **rules.bootstrap(dev, dev_fired, args.bootstrap, args.seed)},
                     "runA_dev_breakdown": breakdown(dev, rule, tau, t)}
            for name, rows in observed.items():
                entry[name] = rules.rates(rows, rules.fires(rows, rule, tau, t))
            results["rules"][f"{rule}@T={t}"] = entry
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"T={temperature} (endpoint NLL {nll:.3f})")
    print(f"{'rule':24} {'tau':>9} | {'dev FPR [CI]':>20} {'dev R':>6} | {'restating FPR':>13} | "
          f"{'early':>5} | {'off FPR':>7} {'off R':>6}")
    for name, r in results["rules"].items():
        d, b = r["runA_dev"], r["runA_dev_breakdown"]
        restating = b["safe_fpr_by_slot_style"].get("slot0:risk_reasoning")
        official = r.get("official_thinking")
        print(f"{name:24} {r['tau']:9.6f} | {d['fpr']*100:5.1f}% [{d['fpr_ci95'][0]*100:4.1f},{d['fpr_ci95'][1]*100:4.1f}] "
              f"{d['recall']*100:5.1f}% | {restating*100 if restating is not None else float('nan'):12.1f}% | "
              f"{b['unsafe_first_fire_before_onset_share'] or 0:5.2f} | "
              + (f"{official['fpr']*100:6.1f}% {official['recall']*100:5.1f}%" if official else "   —"))


if __name__ == "__main__":
    main()
