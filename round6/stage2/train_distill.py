"""Stage 2 v3 (T033, L20): red-line head as in v2, plus a general-harm head per role distilled from a stronger guard.

The red-line head alone decides cuts; it keeps the v2 objective (per-token soft cross-entropy on build_targets.py
classes, class-balanced within (source, role) groups, --mix shares). Specialising the only head to red lines made
the model forget general-harm detection (Qwen3Guard benchmarks), so v3 adds, per role, a second head that learns
*what the stronger guard sees* without changing *what we cut*:
  general_projection   Linear(1024->512) + LayerNorm + SiLU, initialised from the Round5 projection
  general              Linear(512->3) (safe, unsafe, controversial), initialised from the Round5 risk layer
  general_category     Linear(512->9 user / 8 assistant), initialised from the Round5 category layer
(the Round5 heads were trained against general harm, so the new heads start where general detection was).
Targets come from build_targets.py --teacher (Qwen3Guard-Stream-4B per-token distributions aligned by character
ends): loss = red-line CE + --distill-weight * KL(teacher || general) over all teacher-aligned positions
+ --category-weight * CE(general_category, teacher category) where the teacher's non-safe probability > 0.5.
Distillation weights give every (source, role) group its --mix share and every record of a group the same total, as
the red-line weights do. Everything else (start checkpoint, optimiser, schedule, precision, selection on the mean
calibration stream AUC of the red-line head) is train_stage2.py's; the calibration report adds the general head's
mean KL to the teacher and its streaming AUC against the teacher's own max-over-stream decision.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_stage2 as ts  # noqa: E402

GENERAL_KEYS = ("general_projection", "general", "general_category")


def distill_weights(records, mix):
    """Pure: row['distill_weight'] per record so that each group sums to its normalised share and each record of a
    group gets the same total (spread over its teacher positions by the loss)."""
    counts = collections.Counter(ts.group_of(r) for r in records if r.get("t_positions"))
    norm = sum(mix[g] for g in counts)
    for row in records:
        g = ts.group_of(row)
        row["distill_weight"] = mix[g] / norm / counts[g] if row.get("t_positions") else 0.0
    return {g: {"share": mix[g] / norm, "records": n} for g, n in sorted(counts.items())}


def add_general_heads(model):
    """Copy each role's projection / risk / category into trainable general heads (fp32, like the originals)."""
    for role in model.heads:
        head = model.heads[role]
        for new, old in zip(GENERAL_KEYS, ("projection", "risk", "category")):
            if new not in head:
                head[new] = copy.deepcopy(head[old])
            head[new].requires_grad_(True)


def general_readout(model, hidden, role):
    h = model.heads[role]["general_projection"](hidden.float())
    return model.heads[role]["general"](h).float(), model.heads[role]["general_category"](h).float()


def train_step(torch, model, optimizer, group, *, pad, device, micro_tokens, scale, alert_cut, distill, category,
               rows_per_pass=ts.MICROBATCH):
    F = torch.nn.functional
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, parts, tokens = 0.0, collections.Counter(), 0
    for rows in ts.microbatches(group, micro_tokens, size=rows_per_pass):
        ids, mask = ts.pad_batch(torch, rows, pad, device)
        hidden = model(ids, mask, use_cache=False).last_hidden_state
        losses = []
        for index, row in enumerate(rows):
            if row["positions"]:                       # teacher-only rows (build_targets.py --distill-only) have none
                logits = ts.risk_logits(model, hidden, index, row)
                target = ts.soft_targets(torch, row["classes"], alert_cut, device)
                weight = torch.tensor(row["token_weights"], device=device, dtype=torch.float32)
                red = (-(target * F.log_softmax(logits, dim=-1)).sum(-1) * weight).sum()
                losses.append(red)
                parts["redline"] += float(red.detach())
            if row.get("t_positions") and row["distill_weight"] > 0:
                g_logits, g_cat = general_readout(model, hidden[index, row["t_positions"]], row["role"])
                q = torch.tensor(row["t_risk"], device=device, dtype=torch.float32).clamp_min(1e-6)
                q = q / q.sum(-1, keepdim=True)
                kl = (q * (q.log() - F.log_softmax(g_logits, dim=-1))).sum(-1).mean()
                term = distill * row["distill_weight"] * kl
                cats = torch.tensor(row["t_cat"], device=device)
                if category > 0 and bool((cats >= 0).any()):
                    ce = F.cross_entropy(g_cat[cats >= 0], cats[cats >= 0])
                    term = term + category * row["distill_weight"] * ce
                losses.append(term)
                parts["distill"] += float(term.detach())
        loss = torch.stack(losses).sum() * scale
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        total += float(loss.detach())
        tokens += int(mask.sum())
        del hidden, losses, loss
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError("nonfinite gradient")
    optimizer.step()
    return {"loss": total, "redline": parts["redline"] * scale, "distill": parts["distill"] * scale,
            "gradient_norm": float(grad), "input_tokens": tokens}


def calibration_general(torch, model, records, *, pad, device, micro_tokens):
    """Mean KL(teacher || general) per group, and the general head's stream AUC against the teacher's decision
    (teacher max non-safe probability > 0.5 over the stream)."""
    model.eval()
    kl_sum, kl_n, streams = collections.Counter(), collections.Counter(), collections.defaultdict(lambda: ([], []))
    with torch.inference_mode():
        for rows in ts.microbatches([r for r in records if r.get("t_positions")], micro_tokens):
            ids, mask = ts.pad_batch(torch, rows, pad, device)
            hidden = model(ids, mask, use_cache=False).last_hidden_state
            for index, row in enumerate(rows):
                g_logits, _ = general_readout(model, hidden[index, row["t_positions"]], row["role"])
                logp = torch.log_softmax(g_logits, -1)
                q = torch.tensor(row["t_risk"], device=device).clamp_min(1e-6)
                q = q / q.sum(-1, keepdim=True)
                g = ts.group_of(row)
                kl_sum[g] += float((q * (q.log() - logp)).sum(-1).sum())
                kl_n[g] += len(row["t_positions"])
                score = float((1 - logp.exp()[:, 0]).max())
                teacher_pos = max(1 - r[0] for r in row["t_risk"]) > 0.5
                streams[g][0 if teacher_pos else 1].append(score)
    out = {g: {"mean_kl": kl_sum[g] / kl_n[g], "stream_auc_vs_teacher": ts.auc(*streams[g])} for g in sorted(kl_n)}
    vals = [m["stream_auc_vs_teacher"] for m in out.values() if m["stream_auc_vs_teacher"] is not None]
    return {"groups": out, "mean_stream_auc_vs_teacher": sum(vals) / len(vals) if vals else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path, help="build_targets.py --teacher output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mix", default=ts.DEFAULT_MIX)
    parser.add_argument("--backbone-lr", type=float, default=8e-6)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--micro-tokens", type=int, default=16384)
    parser.add_argument("--alert-cut", type=float, default=0.4)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--category-weight", type=float, default=0.25)
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--smoke", type=int, default=0)
    # T036 (profile_train.py): one 16-row pass per group, no checkpointing, batched loss, fused AdamW took a group
    # from 0.854 s to 0.242 s. The defaults keep the v4 step; without checkpointing use --micro-tokens 8192.
    parser.add_argument("--rows-per-pass", type=int, default=ts.MICROBATCH)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--batched-loss", action="store_true", help="batched_loss.train_step: one host sync per step")
    parser.add_argument("--fused-adam", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if ts.sha(ts.START_PATH) != ts.START_SHA:
        raise ValueError("start checkpoint SHA mismatch")
    train = ts.read_jsonl(args.targets / "records_train.jsonl")
    calibration = ts.read_jsonl(args.targets / "records_calibration.jsonl")
    mix = ts.parse_mix(args.mix)
    weights = ts.token_weights(train, mix)
    dweights = distill_weights(train, mix)
    torch, helper, load_file = ts.import_runtime(args.round4_code)
    args.output.mkdir(parents=True)
    model, pad = ts.make_model(helper, load_file, ts.START_PATH)
    helper.training_mode(model)
    if args.no_checkpointing:
        model.backbone.gradient_checkpointing_disable()
    add_general_heads(model)
    model.to("cuda")
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if n.startswith("heads.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
                                   {"params": heads, "lr": args.head_lr, "base_lr": args.head_lr}],
                                  weight_decay=0.01, eps=1e-6, **({"fused": True} if args.fused_adam else {}))
    groups = [(epoch, g) for epoch in range(args.epochs) for g in ts.epoch_groups(train, epoch)]
    scale = len(train) / ts.EFFECTIVE_BATCH
    options = {"device": "cuda", "pad": pad, "micro_tokens": args.micro_tokens}
    summary = {"version": "stage2-distill-v3", "start_sha256": ts.START_SHA,
               "targets_report_sha256": ts.sha(args.targets / "report.json"),
               "records": {"train": len(train), "calibration": len(calibration),
                           "train_with_teacher": sum(1 for r in train if r.get("t_positions"))},
               "steps": len(groups), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
               | {"mix": mix}, "group_weights": weights, "distill_weights": dweights, "dev_read": False, "epochs": {}}

    def evaluate(tag):
        metrics = ts.calibration_metrics(ts.calibration_scores(torch, model, calibration, **options))
        metrics["general"] = calibration_general(torch, model, calibration, **options)
        (args.output / f"calibration_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics

    if not args.smoke:
        summary["epochs"]["0"] = evaluate(0)
    best = (summary["epochs"].get("0", {}).get("mean_stream_auc") or -1.0, 0)
    if args.batched_loss:
        import batched_loss as bl
        table = bl.alert_table(torch, args.alert_cut, "cuda")
    began = time.monotonic()
    with (args.output / "losses.jsonl").open("w") as log:
        for step, (epoch, group) in enumerate(groups, 1):
            for spec in optimizer.param_groups:
                spec["lr"] = spec["base_lr"] * ts.lr_scale(step, len(groups))
            if args.batched_loss:
                record = bl.train_step(torch, model, optimizer, group, scale=scale, rows=args.rows_per_pass,
                                       distill=args.distill_weight, category=args.category_weight, table=table,
                                       **options)
            else:
                record = train_step(torch, model, optimizer, group, scale=scale, alert_cut=args.alert_cut,
                                    distill=args.distill_weight, category=args.category_weight,
                                    rows_per_pass=args.rows_per_pass, **options)
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
                summary["epochs"][str(epoch + 1)] = {**metrics, "checkpoint_sha256": ts.sha(path),
                                                     "seconds": time.monotonic() - began}
                if metrics["mean_stream_auc"] is not None and (best[1] == 0 or metrics["mean_stream_auc"] > best[0]):
                    best = (metrics["mean_stream_auc"], epoch + 1)      # the start has no general head: an epoch wins
                model.train()
    shutil.copyfile(args.output / f"epoch_{best[1]}.safetensors", args.output / "best.safetensors")
    summary.update(status="completed", chosen_epoch=best[1], chosen_mean_stream_auc=best[0],
                   best_sha256=ts.sha(args.output / "best.safetensors"), seconds=time.monotonic() - began)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("chosen_epoch", "chosen_mean_stream_auc", "best_sha256")}), flush=True)


if __name__ == "__main__":
    main()
