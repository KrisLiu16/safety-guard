"""Stage 2, step 2 (L20): train the backbone and both risk heads on per-token red-line targets (build_targets.py).

Stage 1 (T018, T028) showed that a readout of the frozen backbone stops at a calibration AUC of about 0.87, so the
backbone itself is trained here, as in Round5's SFT:
  start       the fixed Round5 checkpoint (round4 window best, SHA pinned), 24 layers, W512 window as the runtime
  trainable   the whole backbone (fp32 master weights, bf16 autocast, gradient checkpointing) and both roles'
              projection + risk layers; the category heads are frozen (their outputs drift with the backbone and
              are not validated here)
  loss        per-token cross-entropy on the record role's risk head (0 safe, 1 unsafe, 2 controversial). Token
              weights: inside each (source, role) group every class present gets the same total weight (record
              weights included); groups share the loss by --mix. A step's loss is scaled by records per epoch /
              records per step, so the learning rates mean what they meant in Round5: AdamW, backbone 8e-6,
              heads 5e-5, weight decay 0.01, 64-step warmup, cosine to 0.1, gradient clip 1.0
  batches     records bucketed by length (blocks of 256, shuffled), 16 records per update, microbatches of at most
              4 records and --micro-tokens padded tokens
  selection   before training and after every epoch, the cut score 1 - p(safe) on calibration: per (source, role)
              the position AUC and the stream AUC (max over the stream's target positions; positive = whole level
              not safe). The best mean stream AUC is exported as best.safetensors (the start checkpoint if no epoch
              beats it). Dev is never read.
v2 (T030, POLICY section 9): class 3 ("alert", harm outside the red lines) is a soft target: p(safe) = 1 - --alert-cut,
p(controversial) = --alert-cut (default 0.4), so the cut score rises without reaching a threshold; for the class
balance it counts with safe, and in the calibration AUCs it is a negative (the calibration report adds its mean score).
Output (--output): epoch_<n>.safetensors, best.safetensors, losses.jsonl, calibration_<n>.json, summary.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
from probe_common import auc  # noqa: E402

SEED = 20260926
START_SHA = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
START_PATH = Path("/work/output/round4/window/best.safetensors")
EFFECTIVE_BATCH, MICROBATCH = 16, 4
DEFAULT_MIX = "runA/assistant=1,prefix_v2/assistant=1,runA_prompts/user=0.5,prefix_v2/user=0.5,leader_v1/assistant=0.5"
ALERT = 3


def weight_class(c):
    return 0 if c == ALERT else c


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_mix(text):
    return {key: float(value) for key, value in (item.split("=") for item in text.split(","))}


def group_of(row):
    return f"{row['source']}/{row['role']}"


def token_weights(records, mix):
    """Pure: sets row['token_weights']; inside a group each present class sums to share / classes present, the
    groups sum to their normalised --mix shares, and all tokens of all records sum to 1."""
    totals = collections.defaultdict(collections.Counter)
    for row in records:
        for c in row["classes"]:
            totals[group_of(row)][weight_class(c)] += row["weight"]
    unknown = sorted(g for g in totals if g not in mix)
    if unknown:
        raise ValueError(f"no --mix share for {unknown}")
    norm = sum(mix[g] for g in totals)
    for row in records:
        g = group_of(row)
        row["token_weights"] = [mix[g] / norm * row["weight"] / (len(totals[g]) * totals[g][weight_class(c)])
                                for c in row["classes"]]
    return {g: {"share": mix[g] / norm, "class_weight_totals": {str(c): v for c, v in sorted(t.items())}}
            for g, t in sorted(totals.items())}


def epoch_groups(records, epoch, seed=SEED):
    rng, groups = random.Random(seed + epoch), []
    ordered = sorted(records, key=lambda row: (len(row["ids"]), row["sample_id"]))
    for start in range(0, len(ordered), 256):
        block = ordered[start:start + 256]
        rng.shuffle(block)
        groups.extend(block[i:i + EFFECTIVE_BATCH] for i in range(0, len(block), EFFECTIVE_BATCH))
    rng.shuffle(groups)
    return groups


def microbatches(rows, micro_tokens, size=MICROBATCH):
    """Pure: rows sorted by length, packed so that count <= size and longest x count <= micro_tokens (a longer row
    goes alone)."""
    out, current = [], []
    for row in sorted(rows, key=lambda r: len(r["ids"])):
        if current and (len(current) == size or len(row["ids"]) * (len(current) + 1) > micro_tokens):
            out.append(current)
            current = []
        current.append(row)
    if current:
        out.append(current)
    return out


def lr_scale(step, total, warmup=64):
    return min(1.0, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))


def pad_batch(torch, rows, pad, device):
    length = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row["ids"])] = torch.tensor(row["ids"], device=device)
        mask[i, :len(row["ids"])] = 1
    return ids, mask


def risk_logits(model, hidden, index, row):
    logits, _ = model.readout(hidden[index, row["positions"]], row["role"])
    return logits.float()


def soft_targets(torch, classes, alert_cut, device):
    """Target distributions (safe, unsafe, controversial): one-hot, or the alert mix for class 3."""
    table = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 - alert_cut, 0.0, alert_cut]],
                         device=device)
    return table[torch.tensor(classes, device=device)]


def train_step(torch, model, optimizer, group, *, pad, device, micro_tokens, scale, alert_cut=0.4):
    """One update over a group of records; returns loss and token accounting."""
    F = torch.nn.functional
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, tokens, targets = 0.0, 0, 0
    for rows in microbatches(group, micro_tokens):
        ids, mask = pad_batch(torch, rows, pad, device)
        hidden = model(ids, mask, use_cache=False).last_hidden_state
        losses = []
        for index, row in enumerate(rows):
            logits = risk_logits(model, hidden, index, row)
            target = soft_targets(torch, row["classes"], alert_cut, device)
            weight = torch.tensor(row["token_weights"], device=device, dtype=torch.float32)
            losses.append((-(target * F.log_softmax(logits, dim=-1)).sum(-1) * weight).sum())
        loss = torch.stack(losses).sum() * scale
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        total += float(loss.detach())
        tokens += int(mask.sum())
        targets += sum(len(r["positions"]) for r in rows)
        del hidden, losses, loss
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError("nonfinite gradient")
    optimizer.step()
    return {"loss": total, "gradient_norm": float(grad), "input_tokens": tokens, "target_tokens": targets}


def calibration_scores(torch, model, records, *, pad, device, micro_tokens):
    """Cut score 1 - p(safe) at every target position of every calibration record."""
    model.eval()
    out = []
    with torch.inference_mode():
        for rows in microbatches(records, micro_tokens):
            ids, mask = pad_batch(torch, rows, pad, device)
            hidden = model(ids, mask, use_cache=False).last_hidden_state
            for index, row in enumerate(rows):
                probs = torch.softmax(risk_logits(model, hidden, index, row), dim=-1)
                out.append({"sample_id": row["sample_id"], "group": group_of(row), "level": row["level"],
                            "classes": row["classes"], "cut": (1.0 - probs[:, 0]).cpu().tolist()})
    return out


def calibration_metrics(scored):
    """Pure: per group position AUC (cut classes vs safe) and stream AUC (max score; positive = level not safe)."""
    groups = collections.defaultdict(list)
    for row in scored:
        groups[row["group"]].append(row)
    out = {}
    for g, rows in sorted(groups.items()):
        pos = [s for r in rows for s, c in zip(r["cut"], r["classes"]) if c in (1, 2)]
        neg = [s for r in rows for s, c in zip(r["cut"], r["classes"]) if c in (0, ALERT)]
        stream_pos = [max(r["cut"]) for r in rows if r["level"] != "safe"]
        stream_neg = [max(r["cut"]) for r in rows if r["level"] == "safe"]
        alert = [s for r in rows for s, c in zip(r["cut"], r["classes"]) if c == ALERT]
        out[g] = {"records": len(rows), "position_auc": auc(pos, neg), "stream_auc": auc(stream_pos, stream_neg),
                  "stream_positive": len(stream_pos),
                  "alert_positions": len(alert), "alert_mean_cut": sum(alert) / len(alert) if alert else None}
    valid = [m["stream_auc"] for m in out.values() if m["stream_auc"] is not None]
    return {"groups": out, "mean_stream_auc": sum(valid) / len(valid) if valid else None}


def import_runtime(round4_code):
    """Round4/Round5 model helpers (the same Classifier, window attention and training mode as Round5 SFT)."""
    sys.path[:0] = [str(round4_code), "/work/input", "/work/window"]
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("stage 2 training needs exactly one visible L20")
    import train_risk as helper
    from safetensors.torch import load_file
    torch.manual_seed(SEED)
    random.seed(SEED)
    return torch, helper, load_file


def make_model(helper, load_file, checkpoint):
    backbone, tokenizer, _ = helper.load("qwen35")
    if getattr(backbone.config, "num_hidden_layers", None) != 24:
        raise RuntimeError("all 24 layers must be retained")
    model = helper.Classifier(backbone).to("cuda")
    helper.configure(model, "window", 512)
    state = load_file(str(checkpoint))
    model.load_state_dict(state, strict=True)
    del state
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return model, pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path, help="build_targets.py output (records_train.jsonl, records_calibration.jsonl)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mix", default=DEFAULT_MIX)
    parser.add_argument("--backbone-lr", type=float, default=8e-6)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--micro-tokens", type=int, default=16384)
    parser.add_argument("--alert-cut", type=float, default=0.4, help="cut probability of the soft alert target")
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--smoke", type=int, default=0, help="stop after this many updates (no export), for a dry run")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(START_PATH) != START_SHA:
        raise ValueError("start checkpoint SHA mismatch")
    train = read_jsonl(args.targets / "records_train.jsonl")
    calibration = read_jsonl(args.targets / "records_calibration.jsonl")
    mix = parse_mix(args.mix)
    weights = token_weights(train, mix)
    torch, helper, load_file = import_runtime(args.round4_code)
    args.output.mkdir(parents=True)
    model, pad = make_model(helper, load_file, START_PATH)
    helper.training_mode(model)          # fp32 master backbone, bf16 embeddings, checkpointing, category heads frozen
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if n.startswith("heads.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
                                   {"params": heads, "lr": args.head_lr, "base_lr": args.head_lr}],
                                  weight_decay=0.01, eps=1e-6)
    groups = [(epoch, g) for epoch in range(args.epochs) for g in epoch_groups(train, epoch)]
    scale = len(train) / EFFECTIVE_BATCH
    options = {"device": "cuda", "pad": pad, "micro_tokens": args.micro_tokens}
    alert_cut = args.alert_cut
    summary = {"version": "stage2-backbone-v2", "start_sha256": START_SHA, "targets_report_sha256": sha(args.targets / "report.json"),
               "records": {"train": len(train), "calibration": len(calibration)}, "steps": len(groups),
               "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"mix": mix},
               "group_weights": weights, "category_heads_trained": False, "dev_read": False, "epochs": {}}

    def evaluate(tag):
        metrics = calibration_metrics(calibration_scores(torch, model, calibration, **options))
        (args.output / f"calibration_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics

    if not args.smoke:
        summary["epochs"]["0"] = evaluate(0)
    best = (summary["epochs"].get("0", {}).get("mean_stream_auc") or -1.0, 0)
    began, step = time.monotonic(), 0
    with (args.output / "losses.jsonl").open("w") as log:
        for step, (epoch, group) in enumerate(groups, 1):
            for spec in optimizer.param_groups:
                spec["lr"] = spec["base_lr"] * lr_scale(step, len(groups))
            record = train_step(torch, model, optimizer, group, scale=scale, alert_cut=alert_cut, **options)
            record.update(step=step, epoch=epoch, seconds=round(time.monotonic() - began, 1))
            log.write(json.dumps(record) + "\n")
            if step % 64 == 0:
                log.flush()
                print(json.dumps(record), flush=True)
            if args.smoke and step >= args.smoke:
                summary.update(status="smoke_completed", smoke_updates=step, seconds=time.monotonic() - began)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                return
            if step == len(groups) or groups[step][0] != epoch:
                path = args.output / f"epoch_{epoch + 1}.safetensors"
                helper.export(model, path)
                metrics = evaluate(epoch + 1)
                summary["epochs"][str(epoch + 1)] = {**metrics, "checkpoint_sha256": sha(path),
                                                     "seconds": time.monotonic() - began}
                if metrics["mean_stream_auc"] is not None and metrics["mean_stream_auc"] > best[0]:
                    best = (metrics["mean_stream_auc"], epoch + 1)
                model.train()
    chosen = START_PATH if best[1] == 0 else args.output / f"epoch_{best[1]}.safetensors"
    shutil.copyfile(chosen, args.output / "best.safetensors")
    summary.update(status="completed", chosen_epoch=best[1], chosen_mean_stream_auc=best[0],
                   best_sha256=sha(args.output / "best.safetensors"), seconds=time.monotonic() - began)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("chosen_epoch", "chosen_mean_stream_auc", "best_sha256")}))


if __name__ == "__main__":
    main()
