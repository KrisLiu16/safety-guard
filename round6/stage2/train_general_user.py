"""Stage 2 v3.1 (T033 fix, L20): retrain only the user-role general head of the v3 checkpoint.

v3 distilled the user general head on Qwen3Guard-Stream-4B's query-head output at every token inside the prompt.
Qwen3Guard trains and reads that head only at the closing <|im_end|> of the user turn; the in-prompt outputs are
untrained, and the v3 head learned to flag ordinary prompts (189 of 194 in the over-blocking sweep; 74% of the
benchmarks' safe prompts). v3.1 keeps everything else of v3 exactly -- backbone, both red-line heads, the assistant
general head -- and retrains heads.user.general_* on the teacher's end-of-turn output (teacher_label_l20.py
--end-user), read at the student's last prompt token (the position that has seen the whole prompt):
  - features: the v3 backbone's hidden state at the last content token of every user record with a teacher target
    (the main records of build_targets.py; augmented views have none), computed once in eval mode;
  - head: Linear(1024->512) + LayerNorm + SiLU, Linear(512->3), Linear(512->9), initialised from the Round5 user
    head (as v3 initialised it), trained on the cached features;
  - loss: KL(teacher || head) + --category-weight * CE(category, teacher category) where teacher non-safe > 0.5,
    each (source) group weighted to an equal share, records within a group equally;
  - selection: lowest mean calibration KL over the groups; also reported: AUC of the head's non-safe score against
    the teacher's decision (non-safe > 0.5) per group.
Output: v3_1.safetensors (the v3 state with the four heads.user.general_* tensors pairs replaced) and summary.json.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_stage2 as ts  # noqa: E402

KEYS = {"general_projection": "projection", "general": "risk", "general_category": "category"}


def end_records(records, teacher, min_cat=0.5):
    """Pure: user records with a teacher target, as {sample_id, source, split, ids, last, risk, cat}."""
    out = []
    for r in records:
        if r["role"] != "user" or not r.get("t_positions"):
            continue
        t = teacher.get(r["sample_id"])
        if t is None:
            continue
        risk = t["end_risk"]
        out.append({"sample_id": r["sample_id"], "source": r["source"], "split": r["split"], "ids": r["ids"],
                    "last": max(r["t_positions"]), "risk": risk, "cat": t["end_cat"] if 1 - risk[0] > min_cat else -1})
    return out


def group_weights(rows):
    """Pure: per-row weight so that every source group sums to 1 / groups."""
    counts = collections.Counter(r["source"] for r in rows)
    return [1.0 / len(counts) / counts[r["source"]] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="v3 best.safetensors (with general heads)")
    parser.add_argument("--general-view", type=Path, required=True, help="export_heads.py --mode general of it")
    parser.add_argument("--targets", type=Path, required=True, help="build_targets.py v3 output directory")
    parser.add_argument("--teacher-end", type=Path, required=True, help="teacher_label_l20.py --end-user output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--category-weight", type=float, default=0.25)
    parser.add_argument("--micro-tokens", type=int, default=32768)
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    teacher = {}
    with gzip.open(args.teacher_end, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            teacher[row["id"]] = row
    rows = {split: end_records(ts.read_jsonl(args.targets / f"records_{split}.jsonl"), teacher)
            for split in ("train", "calibration")}
    torch, helper, load_file = ts.import_runtime(args.round4_code)
    F = torch.nn.functional
    args.output.mkdir(parents=True)
    model, pad = ts.make_model(helper, load_file, args.general_view)      # v3 backbone, standard layout
    model.eval()
    began = time.monotonic()

    def features(split):
        out = torch.empty(len(rows[split]), model.backbone.config.hidden_size, dtype=torch.float32)
        index = {id(r): i for i, r in enumerate(rows[split])}
        with torch.inference_mode():
            for batch in ts.microbatches(rows[split], args.micro_tokens):
                ids, mask = ts.pad_batch(torch, batch, pad, "cuda")
                hidden = model(ids, mask, use_cache=False).last_hidden_state
                for i, r in enumerate(batch):
                    out[index[id(r)]] = hidden[i, r["last"]].float().cpu()
        return out

    x = {split: features(split) for split in rows}
    feature_seconds = time.monotonic() - began
    del model
    torch.cuda.empty_cache()

    start = load_file(str(ts.START_PATH))                                 # Round5: the v3 initialisation
    width = x["train"].shape[1]
    head = torch.nn.ModuleDict({
        "projection": torch.nn.Sequential(torch.nn.Linear(width, 512), torch.nn.LayerNorm(512), torch.nn.SiLU()),
        "risk": torch.nn.Linear(512, 3), "category": torch.nn.Linear(512, 9)}).cuda()
    head.load_state_dict({k[len("heads.user."):]: v.float() for k, v in start.items()
                          if k.startswith("heads.user.") and k.split(".")[2] in ("projection", "risk", "category")})
    data = {s: (x[s].cuda(), torch.tensor([r["risk"] for r in rows[s]], device="cuda"),
                torch.tensor([r["cat"] for r in rows[s]], device="cuda"),
                torch.tensor(group_weights(rows[s]), device="cuda")) for s in rows}
    sources = sorted({r["source"] for r in rows["calibration"]})

    def forward(feats):
        h = head["projection"](feats)
        return head["risk"](h), head["category"](h)

    def evaluate():
        feats, q, cats, _ = data["calibration"]
        with torch.no_grad():
            logits, _ = forward(feats)
            logp = F.log_softmax(logits, -1)
            qn = q.clamp_min(1e-6)
            qn = qn / qn.sum(-1, keepdim=True)
            kl = (qn * (qn.log() - logp)).sum(-1)
            score = 1 - logp.exp()[:, 0]
        out = {}
        for src in sources:
            m = torch.tensor([r["source"] == src for r in rows["calibration"]], device="cuda")
            pos = [float(v) for v, t in zip(score[m], q[m]) if 1 - float(t[0]) > 0.5]
            neg = [float(v) for v, t in zip(score[m], q[m]) if 1 - float(t[0]) <= 0.5]
            out[src] = {"n": int(m.sum()), "mean_kl": float(kl[m].mean()), "auc_vs_teacher": ts.auc(pos, neg),
                        "teacher_positive": len(pos)}
        return {"groups": out, "mean_kl": sum(v["mean_kl"] for v in out.values()) / len(out)}

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    feats, q, cats, w = data["train"]
    n = feats.shape[0]
    steps = args.epochs * ((n + args.batch - 1) // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    generator = torch.Generator(device="cuda").manual_seed(ts.SEED)
    summary = {"version": "stage2-v3.1-user-general", "checkpoint_sha256": ts.sha(args.checkpoint),
               "teacher_end_sha256": ts.sha(args.teacher_end), "records": {s: len(rows[s]) for s in rows},
               "records_by_source": {s: dict(collections.Counter(r["source"] for r in rows[s])) for s in rows},
               "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "feature_seconds": feature_seconds, "epochs": {"0": evaluate()}}
    best, best_state = (summary["epochs"]["0"]["mean_kl"], 0), {k: v.detach().clone() for k, v in head.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(n, device="cuda", generator=generator)
        total = 0.0
        for i in range(0, n, args.batch):
            b = order[i:i + args.batch]
            logits, cat_logits = forward(feats[b])
            qb = q[b].clamp_min(1e-6)
            qb = qb / qb.sum(-1, keepdim=True)
            kl = (qb * (qb.log() - F.log_softmax(logits, -1))).sum(-1)
            loss = (kl * w[b]).sum() / w[b].sum()
            keep = cats[b] >= 0
            if args.category_weight > 0 and bool(keep.any()):
                ce = F.cross_entropy(cat_logits[keep], cats[b][keep], reduction="none")
                loss = loss + args.category_weight * (ce * w[b][keep]).sum() / w[b][keep].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss) * len(b)
        metrics = evaluate()
        metrics["train_loss"] = total / n
        summary["epochs"][str(epoch)] = metrics
        print(json.dumps({"epoch": epoch, "train_loss": round(metrics["train_loss"], 5), "mean_kl": round(metrics["mean_kl"], 5),
                          "auc": {k: v["auc_vs_teacher"] for k, v in metrics["groups"].items()}}), flush=True)
        if metrics["mean_kl"] < best[0]:
            best = (metrics["mean_kl"], epoch)
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    from safetensors.torch import save_file
    state = load_file(str(args.checkpoint))
    replaced = 0
    for new, old in KEYS.items():
        for name, tensor in best_state.items():
            if name.split(".")[0] == old:
                key = f"heads.user.{new}." + name.split(".", 1)[1]
                if key not in state or state[key].shape != tensor.shape:
                    raise KeyError(key)
                state[key] = tensor.to(state[key].dtype).cpu().contiguous()
                replaced += 1
    path = args.output / "v3_1.safetensors"
    save_file(state, str(path), metadata={"source_sha256": summary["checkpoint_sha256"], "change": "heads.user.general_*"})
    summary.update(status="completed", chosen_epoch=best[1], chosen_mean_kl=best[0], replaced_tensors=replaced,
                   output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=time.monotonic() - began)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("chosen_epoch", "chosen_mean_kl", "replaced_tensors", "output_sha256")}))


if __name__ == "__main__":
    main()
