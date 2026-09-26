"""Stage-2 training step with a batched loss (T036): one host sync per step instead of about 170.

train_distill.train_step computes the loss row by row and reads every term back with float(), and pad_batch copies
each row to the GPU on its own; each of those waits for the GPU. Here the host builds, per forward pass, every
gather index, target and per-position coefficient first (loss_plan), copies them from pinned memory without a sync
(plan_tensors), and one gather per role gives the same sum (batched_loss); the step reads its numbers back once.
profile_train.py measured this with one 16-row pass, no gradient checkpointing and fused AdamW at 0.242 s per group
against 0.854 s (gradient cosine 0.9993 or higher against train_distill.train_step on the same weights).
Rows without red-line positions (build_targets.py --distill-only) contribute only their distillation terms.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_stage2 as ts  # noqa: E402
import train_distill as td  # noqa: E402


def loss_plan(rows, distill, category):
    """Pure: per role, the flattened gather indices, targets and per-position coefficients such that one batched loss
    equals train_distill.train_step's per-row sum: red-line soft CE x token weight; distill x distill_weight x mean KL
    over the row's teacher positions; category x distill_weight x mean CE over its positions with a teacher category
    (only when category > 0 and the row has one)."""
    plan = {}
    for i, row in enumerate(rows):
        p = plan.setdefault(row["role"], {k: [] for k in ("red_b", "red_p", "red_cls", "red_w", "gen_b", "gen_p",
                                                          "q", "kl_coef", "cats", "ce_coef")})
        n = len(row["positions"])
        p["red_b"] += [i] * n
        p["red_p"] += list(row["positions"])
        p["red_cls"] += list(row["classes"])
        p["red_w"] += list(row["token_weights"])
        if row.get("t_positions") and row["distill_weight"] > 0:
            m, w = len(row["t_positions"]), row["distill_weight"]
            valid = sum(1 for c in row["t_cat"] if c >= 0) if category > 0 else 0
            p["gen_b"] += [i] * m
            p["gen_p"] += list(row["t_positions"])
            p["q"] += [list(v) for v in row["t_risk"]]
            p["kl_coef"] += [distill * w / m] * m
            p["cats"] += [c if valid and c >= 0 else -1 for c in row["t_cat"]]
            p["ce_coef"] += [category * w / valid if valid and c >= 0 else 0.0 for c in row["t_cat"]]
    return plan


def plan_tensors(torch, plan, device, pin):
    """Host tensors for loss_plan, copied without a host sync (pinned memory, non_blocking)."""
    def move(values, dtype):
        t = torch.tensor(values, dtype=dtype)
        return t.pin_memory().to(device, non_blocking=True) if pin else t.to(device)
    out = {}
    for role, p in plan.items():
        d = {"red": bool(p["red_b"]), "gen": bool(p["gen_b"]), "cat": any(c > 0 for c in p["ce_coef"])}
        if d["red"]:
            d.update(red_b=move(p["red_b"], torch.long), red_p=move(p["red_p"], torch.long),
                     red_cls=move(p["red_cls"], torch.long), red_w=move(p["red_w"], torch.float32))
        if d["gen"]:
            d.update(gen_b=move(p["gen_b"], torch.long), gen_p=move(p["gen_p"], torch.long),
                     q=move(p["q"], torch.float32), kl_coef=move(p["kl_coef"], torch.float32),
                     cats=move(p["cats"], torch.long), ce_coef=move(p["ce_coef"], torch.float32))
        out[role] = d
    return out


def batched_loss(torch, model, hidden, tensors, table):
    """Sum of the red-line and distillation terms of one pass from plan_tensors; returns (red, distill) tensors."""
    F = torch.nn.functional
    red = hidden.new_zeros((), dtype=torch.float32)
    dist = hidden.new_zeros((), dtype=torch.float32)
    for role, d in tensors.items():
        if d["red"]:
            logits = model.readout(hidden[d["red_b"], d["red_p"]], role)[0].float()
            red = red + (-(table[d["red_cls"]] * F.log_softmax(logits, dim=-1)).sum(-1) * d["red_w"]).sum()
        if d["gen"]:
            g_logits, g_cat = td.general_readout(model, hidden[d["gen_b"], d["gen_p"]], role)
            q = d["q"].clamp_min(1e-6)
            q = q / q.sum(-1, keepdim=True)
            dist = dist + ((q * (q.log() - F.log_softmax(g_logits, dim=-1))).sum(-1) * d["kl_coef"]).sum()
            if d["cat"]:
                ce = F.cross_entropy(g_cat, d["cats"], ignore_index=-1, reduction="none")
                dist = dist + (ce * d["ce_coef"]).sum()
    return red, dist


def alert_table(torch, alert_cut, device):
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 - alert_cut, 0.0, alert_cut]],
                        device=device)


def host_batch(torch, rows, pad, device):
    """Padded ids and mask built on the host and copied from pinned memory without a sync."""
    length = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
        mask[i, :len(r["ids"])] = 1
    if str(device) == "cpu":
        return ids, mask
    return ids.pin_memory().to(device, non_blocking=True), mask.pin_memory().to(device, non_blocking=True)


def train_step(torch, model, optimizer, group, *, pad, device, micro_tokens, rows, scale, distill, category, table):
    """train_distill.train_step with the batched loss: same terms, same clipping, one read-back per step. A
    nonfinite loss or gradient still stops the step before the optimizer."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    acc, tokens = [], 0
    for part in ts.microbatches(group, micro_tokens, size=rows):
        ids, mask = host_batch(torch, part, pad, device)
        tensors = plan_tensors(torch, loss_plan(part, distill, category), device, pin=str(device) != "cpu")
        hidden = model(ids, mask, use_cache=False).last_hidden_state
        red, dist = batched_loss(torch, model, hidden, tensors, table)
        loss = (red + dist) * scale
        if loss.requires_grad:                     # a pass whose rows carry no weighted target adds nothing
            loss.backward()
        acc.append(torch.stack([loss.detach(), red.detach(), dist.detach()]))
        tokens += sum(len(r["ids"]) for r in part)
        del hidden, loss
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    values = torch.cat([torch.stack(acc).sum(0), grad.detach().reshape(1)]).tolist()
    if not all(v == v and abs(v) != float("inf") for v in values):
        raise RuntimeError("nonfinite loss or gradient")
    optimizer.step()
    return {"loss": values[0], "redline": values[1] * scale, "distill": values[2] * scale, "gradient_norm": values[3],
            "input_tokens": tokens}
