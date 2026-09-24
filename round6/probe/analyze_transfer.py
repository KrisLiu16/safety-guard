"""T014 analysis (CPU, no model calls): head vs Run A probe, on Run A, prefix_v2 and the official thinking set.

Both scorers go through the same code as decision_rule v1 / T005: the probe logit is written as the
pseudo log-probability triple [0, logit, -50], whose softmax is sigmoid(logit), so temperature fitting,
rules, thresholds and the family bootstrap are shared unchanged. Each scorer gets its own temperature;
every temperature and threshold is fitted on Run A calibration only (stream FPR <= 5% per language).
Reported: Run A dev, prefix_v2 dev (assistant strata) and the official thinking set (observed only),
plus threshold-free stream AUCs (max score over the stream) on the official set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "decision_rule"))
sys.path.insert(0, str(HERE))
import analyze_decision_rules as rules  # noqa: E402
from analyze_runA_rules import breakdown  # noqa: E402
from probe_common import auc  # noqa: E402

SCORERS = ("head", "probe")
REPORT_RULES = ("threshold", "consecutive_2", "consecutive_4", "ema_0.1", "ema_0.3")


def triples(record, scorer):
    return record["logprobs"] if scorer == "head" else [[0.0, v, -50.0] for v in record["probe_logits"]]


def load_rows(dump_dir):
    out = {}
    for split in ("calibration", "dev"):
        out[f"runA_{split}"] = [{"id": r["sample_id"], "family": r["family"], "label": r["label"],
                                 "stratum": f"{r['language']}/assistant", "slot": r["slot"],
                                 "style": r["response_style"], "classes": r["classes"], "record": r}
                                for r in rules.read_gz(dump_dir / f"runA_{split}.jsonl.gz")]
    out["prefix_v2_dev"] = [{"id": r["sample_id"], "family": r["family"], "label": r["source_label"],
                             "stratum": f"{r['language']}/{r['target_role']}", "record": r}
                            for r in rules.read_gz(dump_dir / "prefix_v2_dev.jsonl.gz")]
    sequences = {r["sequence_id"]: r for r in rules.read_gz(dump_dir / "official_sequences.jsonl.gz")}
    out["official_thinking"] = [{"id": r["sample_id"], "family": r["sample_id"], "label": r["label"].lower(),
                                 "stratum": "official/thinking", "record": sequences[r["sequence_id"]]}
                                for r in rules.read_gz(dump_dir / "official_rows.jsonl.gz") if r["split"] == "thinking"]
    return out


def attach_scores(rows, scorer, temperature):
    for row in rows:
        row["logprobs"] = triples(row["record"], scorer)
        row["scores_t"] = {temperature: rules.unsafe_scores(row["logprobs"], temperature)}


def stream_auc(rows, temperature):
    safe = [max(r["scores_t"][temperature]) for r in rows if r["label"] == "safe"]
    unsafe = [max(r["scores_t"][temperature]) for r in rows if r["label"] == "unsafe"]
    return auc(unsafe, safe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "results_transfer_v1/transfer_v1_results.json")
    args = parser.parse_args()
    data = load_rows(args.dump_dir)
    results = {"version": "round6-probe-transfer-v1", "model_calls": 0,
               "fitted_on": "Run A calibration only", "official_used_for_fitting": False,
               "streams": {name: len(rows) for name, rows in data.items()}, "scorers": {}}
    for scorer in SCORERS:
        for rows in data.values():
            for row in rows:
                row["logprobs"] = triples(row["record"], scorer)
        temperature, nll = rules.fit_temperature(data["runA_calibration"])
        for rows in data.values():
            attach_scores(rows, scorer, temperature)
        entry = {"temperature": temperature, "temperature_endpoint_nll": nll,
                 "stream_auc_max_score": {name: stream_auc(rows, temperature) for name, rows in data.items()},
                 "rules": {}}
        for rule in REPORT_RULES:
            tau = rules.fit_tau(data["runA_calibration"], rule, temperature)
            if tau is None:
                entry["rules"][rule] = None
                continue
            per = {"tau": tau}
            for name, rows in data.items():
                fired = rules.fires(rows, rule, tau, temperature)
                per[name] = rules.rates(rows, fired)
                if name == "runA_dev":
                    per[name].update(rules.bootstrap(rows, fired, args.bootstrap, args.seed))
                    per["runA_dev_breakdown"] = breakdown(rows, rule, tau, temperature)
            entry["rules"][rule] = per
        results["scorers"][scorer] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for scorer, entry in results["scorers"].items():
        print(f"\n{scorer}: T={entry['temperature']}  stream AUC (max score): "
              + ", ".join(f"{k} {v:.3f}" for k, v in entry["stream_auc_max_score"].items() if v is not None))
        print(f"{'rule':15} | {'RunA dev FPR':>12} {'R':>6} {'restating':>9} | {'prefix_v2 FPR':>13} {'R':>6} | {'official FPR':>12} {'R':>6}")
        for rule, per in entry["rules"].items():
            if per is None:
                print(f"{rule:15} | no threshold meets 5% on Run A calibration")
                continue
            a, p, o = per["runA_dev"], per["prefix_v2_dev"], per["official_thinking"]
            restating = per["runA_dev_breakdown"]["safe_fpr_by_slot_style"].get("slot0:risk_reasoning")
            print(f"{rule:15} | {a['fpr']*100:11.1f}% {a['recall']*100:5.1f}% {restating*100 if restating is not None else float('nan'):8.1f}% | "
                  f"{p['fpr']*100:12.1f}% {p['recall']*100:5.1f}% | {o['fpr']*100:11.1f}% {o['recall']*100:5.1f}%")


if __name__ == "__main__":
    main()
