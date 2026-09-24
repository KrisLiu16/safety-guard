"""T004 linear probe on the frozen Round5 features (numpy only; runs in the L20 job or on a CPU).

Question: do the frozen features already separate "safe reasoning that restates a harmful request"
(slot 0, style risk_reasoning) from genuinely harmful text (post-onset positions), better than the
current risk head does? If a linear readout separates them well, retraining the readout / head can
fix much of the prefix false positives; if not, the backbone needs new data or training.

Protocol: positions S (safe response) and P (pre-onset) are negatives, U (post-onset) positives,
O (onset span) is dropped. Probes are fitted on the train sample, the L2 strength is chosen on
calibration, and every reported number is on dev. The head's own p(unsafe) is scored the same way.
Dev CIs come from a family bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_common import auc, fit_logistic, logistic_scores, recall_at_fpr  # noqa: E402

L2_GRID = (1e-4, 1e-3, 1e-2)
FEATURES = ("hidden", "projection")


def load(directory, split):
    data = np.load(directory / f"features_{split}.npz")
    records = json.loads((directory / f"features_{split}_records.json").read_text(encoding="utf-8"))
    classes = data["classes"].astype(str)
    record = data["record"]
    meta = [records[i] for i in record]
    return {"hidden": data["hidden"].astype(np.float32), "projection": data["projection"].astype(np.float32),
            "head": data["head_p_unsafe"].astype(np.float64), "classes": classes, "meta": meta,
            "family": np.asarray([m["family"] for m in meta]),
            "language": np.asarray([m["language"] for m in meta])}


def groups(split):
    """Boolean masks for the evaluation groups."""
    classes, meta = split["classes"], split["meta"]
    slot = np.asarray([m["slot"] for m in meta])
    style = np.asarray([m["response_style"] for m in meta])
    safe = classes == "S"
    return {
        "restating_risk_reasoning": safe & (slot == 0) & (style == "risk_reasoning"),
        "other_safe_on_unsafe_prompt": safe & (slot == 0) & (style != "risk_reasoning"),
        "safe_on_safe_prompt": safe & (slot == 2),
        "pre_onset": classes == "P",
        "post_onset": classes == "U",
    }


def training_view(split):
    keep = split["classes"] != "O"
    return keep, (split["classes"][keep] == "U").astype(np.float32)


def evaluate(scores, split, masks):
    post = masks["post_onset"]
    negatives = masks["restating_risk_reasoning"] | masks["other_safe_on_unsafe_prompt"] | masks["safe_on_safe_prompt"] | masks["pre_onset"]
    threshold = np.quantile(scores[negatives], 0.95, method="higher") if negatives.any() else None
    out = {"auc_post_vs_all_negatives": auc(scores[post], scores[negatives]),
           "recall_at_5pct_position_fpr": recall_at_fpr(scores[post], scores[negatives], 0.05)}
    for name in ("restating_risk_reasoning", "other_safe_on_unsafe_prompt", "safe_on_safe_prompt", "pre_onset"):
        out[f"auc_post_vs_{name}"] = auc(scores[post], scores[masks[name]])
        out[f"above_5pct_threshold_{name}"] = (float((scores[masks[name]] > threshold).mean())
                                               if threshold is not None and masks[name].any() else None)
    for language in sorted(set(split["language"])):
        lang = split["language"] == language
        out[f"auc_post_vs_restating_{language}"] = auc(scores[post & lang], scores[masks["restating_risk_reasoning"] & lang])
    return out


def bootstrap_auc(scores_by_name, split, masks, samples, seed):
    """Family bootstrap of AUC(post vs restating) for every scorer, plus probe-minus-head differences."""
    families = split["family"]
    names = np.unique(families)
    index_by_family = {name: np.flatnonzero(families == name) for name in names}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores_by_name}
    for _ in range(samples):
        picked = np.concatenate([index_by_family[n] for n in rng.choice(names, size=len(names))])
        post = picked[masks["post_onset"][picked]]
        neg = picked[masks["restating_risk_reasoning"][picked]]
        for name, scores in scores_by_name.items():
            draws[name].append(auc(scores[post], scores[neg]))
    out = {}
    for name, values in draws.items():
        values = np.asarray([v for v in values if v is not None])
        out[name] = {"ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
    for name in scores_by_name:
        if name != "head":
            diff = np.asarray(draws[name], dtype=float) - np.asarray(draws["head"], dtype=float)
            out[f"{name}_minus_head"] = {"ci95": [float(np.nanquantile(diff, .025)), float(np.nanquantile(diff, .975))]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or args.dump_dir / "probe_v1_results.json"
    train, calibration, dev = (load(args.dump_dir, s) for s in ("train", "calibration", "dev"))
    results = {"version": "round6-probe-v1", "model_calls": 0, "fitted_on": "train sample",
               "l2_chosen_on": "calibration", "reported_on": "dev",
               "positions": {s: {c: int((d["classes"] == c).sum()) for c in "SPOU"}
                             for s, d in (("train", train), ("calibration", calibration), ("dev", dev))}}
    dev_masks = groups(dev)
    dev_scores = {"head": dev["head"]}
    results["scorers"] = {}
    for feature in FEATURES:
        keep, y = training_view(train)
        cal_keep, cal_y = training_view(calibration)
        best = None
        for l2 in L2_GRID:
            model = fit_logistic(train[feature][keep], y, l2=l2, seed=args.seed)
            s = logistic_scores(calibration[feature][cal_keep], model)
            score = auc(s[cal_y > 0], s[cal_y == 0])
            if best is None or score > best[0]:
                best = (score, l2, model)
        dev_scores[f"probe_{feature}"] = logistic_scores(dev[feature], best[2])
        results["scorers"][f"probe_{feature}"] = {"l2": best[1], "calibration_auc": best[0]}
    results["dev"] = {name: evaluate(scores, dev, dev_masks) for name, scores in dev_scores.items()}
    results["dev_bootstrap_auc_post_vs_restating"] = bootstrap_auc(dev_scores, dev, dev_masks, args.bootstrap, args.seed)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    key = "auc_post_vs_restating_risk_reasoning"
    print(json.dumps({name: {key: r[key], "auc_all": r["auc_post_vs_all_negatives"],
                             "recall@5%": r["recall_at_5pct_position_fpr"]} for name, r in results["dev"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
