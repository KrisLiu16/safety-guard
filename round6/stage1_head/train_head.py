"""Stage 1, step 2: retrain the assistant readout on cached frozen features (torch; GPU if present, else CPU).

Variants, both warm-started from the current head (head_assistant_init.pt from the cache step):
  risk  only the 512->3 risk layer, on the cached 512-d projection; the projection and the category head are
        untouched, so the result is a drop-in weight change;
  full  projection (Linear + LayerNorm + SiLU) and risk layer, on the cached 1024-d head input; the category
        head keeps its old weights but now reads a changed projection (reported, not fixed here).
  wide  diagnostic only (T028): a freshly initialised 1024->2048->512->3 MLP on the cached head input, to see how far
        a readout of the frozen features can go; not a drop-in head (the runtime's head shape is fixed), never
        evaluated, no pull towards any initial weights.
Loss: 3-way cross-entropy, weighted by the cached position weight, balanced by class inside each source, and
scaled so each source contributes its --mix share; plus an L2 pull towards the initial weights. The epoch is
chosen on calibration only (mean over sources of the AUC on hard-labelled positions); dev is never read here.
Output: head_<variant>.pt with the model's key names, and train_report.json.

Targets: v1 used the cache's 0/1 labels and scored calibration by p(unsafe). With --labels3 (relabel_cache.py,
red-line policy) every source takes the 3-class targets written next to the cache (0 safe, 1 unsafe,
2 controversial; weight 0 = no target, dropped), a source without a labels3 file is left out and reported, and
calibration is scored by the cut score 1 - p(safe), since both controversial and unsafe are cut.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probe"))
from probe_common import auc  # noqa: E402

VARIANTS = {"risk": ("projection", ("risk.weight", "risk.bias")),
            "full": ("hidden", ("projection.0.weight", "projection.0.bias", "projection.1.weight", "projection.1.bias",
                                "risk.weight", "risk.bias")),
            "wide": ("hidden", ())}
DIAGNOSTIC = ("wide",)


def load_cache(directory, source, split, labels3=None):
    shards = sorted(Path(directory).glob(f"cache_{source}_{split}_[0-9][0-9][0-9].npz"))
    if not shards:
        return None
    parts = [np.load(path) for path in shards]
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("hidden", "projection", "label", "weight")}
    if labels3 is not None:
        path = Path(labels3) / f"labels3_{source}_{split}.npz"
        if not path.exists():
            return None
        new = np.load(path)
        if len(new["label"]) != len(out["label"]):
            raise ValueError(f"{path.name} does not match the cache positions")
        keep = new["weight"] > 0
        out = {"hidden": out["hidden"][keep], "projection": out["projection"][keep],
               "label": new["label"][keep].astype(np.int64), "weight": new["weight"][keep]}
    out["source"] = source
    return out


def cut_positive(label, score):
    """Positives for calibration AUC: unsafe only (v1) or anything that is cut (controversial or unsafe)."""
    return label == 1 if score == "unsafe" else label != 0


def build_head(torch, init, variant):
    nn = torch.nn
    if variant == "risk":
        head = nn.ModuleDict({"risk": nn.Linear(512, 3)})
    elif variant == "wide":
        width = init["projection.0.weight"].shape[1]
        return nn.ModuleDict({"projection": nn.Sequential(nn.Linear(width, 2048), nn.LayerNorm(2048), nn.SiLU(),
                                                          nn.Linear(2048, 512), nn.LayerNorm(512), nn.SiLU()),
                              "risk": nn.Linear(512, 3)})
    else:
        width = init["projection.0.weight"].shape[1]
        head = nn.ModuleDict({"projection": nn.Sequential(nn.Linear(width, 512), nn.LayerNorm(512), nn.SiLU()),
                              "risk": nn.Linear(512, 3)})
    head.load_state_dict({k: init[k] for k in VARIANTS[variant][1]}, strict=True)
    return head


def logits(head, x, variant):
    return head["risk"](x if variant == "risk" else head["projection"](x))


def position_weights(sources, mix, balance):
    """Per-position training weights: cached weight, class-balanced inside each source, scaled to the mix share."""
    out = []
    for data in sources:
        w = data["weight"].astype(np.float64).copy()
        if balance:
            present = [label for label in np.unique(data["label"]) if w[data["label"] == label].sum() > 0]
            for label in present:
                mask = data["label"] == label
                w[mask] *= (1.0 / len(present)) / w[mask].sum()
        w *= mix.get(data["source"], 1.0) / w.sum()
        out.append(w)
    total = sum(w.sum() for w in out)
    return [w / total for w in out]


def calibration_auc(torch, head, variant, cal_sources, device, score="unsafe"):
    scores = {}
    with torch.no_grad():
        for data in cal_sources:
            x = torch.tensor(data[VARIANTS[variant][0]], dtype=torch.float32, device=device)
            probs = torch.softmax(logits(head, x, variant), dim=-1).cpu().numpy()
            p = probs[:, 1] if score == "unsafe" else 1.0 - probs[:, 0]
            hard = data["weight"] >= 1.0
            positive = cut_positive(data["label"], score)
            scores[data["source"]] = auc(p[hard & positive], p[hard & (data["label"] == 0)])
    valid = [v for v in scores.values() if v is not None]
    return scores, (float(np.mean(valid)) if valid else None)


def train(torch, init, train_sources, cal_sources, variant, *, mix, epochs, lr, l2_to_init, batch, balance, seed, device,
          score="unsafe"):
    torch.manual_seed(seed)
    head = build_head(torch, init, variant).to(device)
    anchor = {k: v.detach().clone() for k, v in head.state_dict().items()}
    feature = VARIANTS[variant][0]
    x = torch.tensor(np.concatenate([d[feature] for d in train_sources]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([d["label"] for d in train_sources]), dtype=torch.long)
    w = torch.tensor(np.concatenate(position_weights(train_sources, mix, balance)), dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    history, best = [], None
    base_scores, base_mean = calibration_auc(torch, head, variant, cal_sources, device, score)
    history.append({"epoch": 0, "calibration_auc": base_scores, "mean": base_mean})
    best = (base_mean if base_mean is not None else -1.0, 0, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(y), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            xb, yb, wb = x[idx].to(device), y[idx].to(device), w[idx].to(device)
            loss = (torch.nn.functional.cross_entropy(logits(head, xb, variant), yb, reduction="none") * wb).sum() * (len(y) / len(idx))
            if variant not in DIAGNOSTIC:
                reg = sum(((p - anchor[name]) ** 2).sum() for name, p in head.named_parameters())
                loss = loss + l2_to_init * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx) / len(y)
        scores, mean = calibration_auc(torch, head, variant, cal_sources, device, score)
        history.append({"epoch": epoch, "train_loss": total, "calibration_auc": scores, "mean": mean})
        if mean is not None and mean > best[0]:
            best = (mean, epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    return best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="risk,full")
    parser.add_argument("--sources", default="prefix_v2,runA,s2,s5", help="sources used when present in the cache")
    parser.add_argument("--mix", default="prefix_v2=1,runA=1,s2=0.5,s5=0.5", help="each source's share of the loss")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-to-init", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels3", type=Path, help="relabel_cache.py output: 3-class red-line targets")
    parser.add_argument("--role", choices=("assistant", "user"), default="assistant",
                        help="which head's init weights to warm-start from (head_<role>_init.pt in the cache)")
    args = parser.parse_args()
    import torch
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_path = args.cache_dir / f"head_{args.role}_init.pt"
    init = torch.load(init_path, map_location="cpu")
    mix = {k: float(v) for k, v in (item.split("=") for item in args.mix.split(","))}
    names = args.sources.split(",")
    train_sources = [d for d in (load_cache(args.cache_dir, n, "train", args.labels3) for n in names) if d is not None]
    cal_sources = [d for d in (load_cache(args.cache_dir, n, "calibration", args.labels3) for n in names) if d is not None]
    score = "unsafe" if args.labels3 is None else "cut"
    present = {d["source"] for d in train_sources} | {d["source"] for d in cal_sources}
    report = {"version": "round6-stage1-head-v1" if args.labels3 is None else "round6-stage1-head-v2-redline",
              "device": device, "role": args.role, "init_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
              "calibration_score": score, "sources_left_out": [n for n in names if n not in present],
              "train_positions": {d["source"]: int(len(d["label"])) for d in train_sources},
              "train_positions_by_class": {d["source"]: {str(c): int((d["label"] == c).sum()) for c in (0, 1, 2)}
                                           for d in train_sources},
              "calibration_positions": {d["source"]: int(len(d["label"])) for d in cal_sources},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                           if k not in ("cache_dir", "output")} | {"mix": mix},
              "dev_read": False, "variants": {}}
    for variant in args.variants.split(","):
        (best_mean, best_epoch, state), history = train(
            torch, init, train_sources, cal_sources, variant, mix=mix, epochs=args.epochs, lr=args.lr,
            l2_to_init=args.l2_to_init, batch=args.batch, balance=not args.no_class_balance, seed=args.seed, device=device,
            score=score)
        path = args.output / f"head_{variant}.pt"
        if variant in DIAGNOSTIC:
            torch.save(state, path)                    # its own key names: not loadable into the runtime head
        else:
            full = {k: v.clone() for k, v in init.items()}
            full.update({k: state[k] for k in VARIANTS[variant][1]})
            torch.save(full, path)
        report["variants"][variant] = {"chosen_epoch": best_epoch, "chosen_calibration_mean_auc": best_mean,
                                       "history": history, "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "changed_keys": list(VARIANTS[variant][1]),
                                       "deployable": variant not in DIAGNOSTIC}
    (args.output / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({v: {k: r[k] for k in ("chosen_epoch", "chosen_calibration_mean_auc")} for v, r in report["variants"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
