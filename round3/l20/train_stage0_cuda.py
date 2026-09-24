"""L20-only prefix/representation distillation for the pruned C1 classifier."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runtime import attach_lora, encode, load  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def target_positions(tokenizer, messages: list[dict], ids: list[int]) -> list[int]:
    role = messages[-1]["role"]
    empty = messages[:-1] + [{"role": role, "content": ""}]
    header = encode(tokenizer, empty, partial=True)
    start = 0
    for before, after in zip(header, ids):
        if before != after:
            break
        start += 1
    start = min(start, len(ids) - 1)
    span = len(ids) - start
    return sorted({start + (span - 1) * i // 3 for i in range(4)})


def logits_from_hidden(model, hidden: torch.Tensor, role: str) -> tuple[torch.Tensor, torch.Tensor]:
    if role == "user":
        projected = model.query_risk_level_category_layernorm(
            model.query_risk_level_category_pre(hidden)
        )
        return model.query_risk_level_head(projected), model.query_category_head(projected)
    projected = model.risk_level_category_layernorm(model.risk_level_category_pre(hidden))
    return model.risk_level_head(projected), model.category_head(projected)


def losses(student, teacher, tokenizer, row: dict) -> dict[str, torch.Tensor] | None:
    if row.get("project_policy_label") is not None:
        raise ValueError("Stage 0 must not consume project safety labels")
    messages = row["messages"]
    role = row["target_role"]
    if messages[-1]["role"] != role:
        raise ValueError("Target role mismatch")
    if sum(len(message["content"]) for message in messages) > 3000:
        return None
    ids = encode(tokenizer, messages)
    if len(ids) > 512:
        return None
    positions = target_positions(tokenizer, messages, ids)
    x = torch.tensor([ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        teacher_hidden = teacher.model(input_ids=x, use_cache=False).last_hidden_state[:, positions]
        teacher_risk, teacher_category = logits_from_hidden(teacher, teacher_hidden, role)
    student_hidden = student.model(input_ids=x, use_cache=False).last_hidden_state[:, positions]
    student_risk, student_category = logits_from_hidden(student, student_hidden, role)
    temperature = 2.0
    risk = F.kl_div(
        F.log_softmax(student_risk.float() / temperature, dim=-1),
        F.softmax(teacher_risk.float() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature * temperature / len(positions)
    category = F.kl_div(
        F.log_softmax(student_category.float() / temperature, dim=-1),
        F.softmax(teacher_category.float() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature * temperature / len(positions)
    representation = (1 - F.cosine_similarity(
        student_hidden.float(), teacher_hidden.float(), dim=-1
    )).mean()
    total = risk + 0.3 * category + 0.2 * representation
    return {"total": total, "risk_kl": risk, "category_kl": category,
            "representation_cosine_loss": representation}


def evaluate(student, teacher, tokenizer, rows: list[dict]) -> dict:
    student.eval()
    values = Counter()
    count = 0
    with torch.no_grad():
        for row in rows:
            result = losses(student, teacher, tokenizer, row)
            if result is None:
                continue
            count += 1
            values.update({key: float(value.detach().cpu()) for key, value in result.items()})
    student.train()
    return {"examples": count, **{key: value / count for key, value in values.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "teacher")
    parser.add_argument("--student", type=Path, default=ROOT / "student")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=Path("/work/output"))
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--initial-adapter", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "L20" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Stage 0 requires an L20 CUDA device; no M5/CPU fallback")
    if args.max_train_examples < 1 or args.accumulation < 1:
        parser.error("training limits must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text())
    allowed_versions = {"stage0-open-corpus-distillation-v1",
                        "stage0-open-corpus-distillation-v2-20k"}
    if (manifest["project_safety_alignment"] or manifest["uses_source_safety_labels"]
            or manifest["dataset_version"] not in allowed_versions):
        raise RuntimeError("Unexpected stage0 data contract")
    for split in ("train", "dev"):
        if sha256_file(args.data / f"{split}.jsonl") != manifest["output_sha256"][split]:
            raise RuntimeError("Stage0 dataset checksum mismatch")
    if sha256_file(args.teacher / "tokenizer.json") != sha256_file(args.student / "tokenizer.json"):
        raise RuntimeError("Teacher/student tokenizers differ")
    train = read(args.data / "train.jsonl")[:args.max_train_examples]
    dev = read(args.data / "dev.jsonl")
    if {row["family"] for row in train} & {row["family"] for row in dev}:
        raise RuntimeError("Family leakage between train and dev")
    random.Random(20260923).shuffle(train)
    torch.manual_seed(20260923)
    teacher, tokenizer = load(dtype=torch.bfloat16, device="cuda", model_path=args.teacher)
    student, _ = load(dtype=torch.bfloat16, device="cuda", model_path=args.student)
    if sum(p.numel() for p in student.parameters()) != 376878080:
        raise RuntimeError("Unexpected student architecture")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    trainables = attach_lora(student)
    if not trainables:
        raise RuntimeError("No trainable student parameters")
    if args.initial_adapter is not None:
        state = load_file(str(args.initial_adapter))
        if set(state) != {name for name, _ in trainables}:
            raise RuntimeError("Initial adapter is incompatible with C1")
        student.load_state_dict(state, strict=False)
    student.train()
    groups = [
        {"params": [p for name, p in trainables if name.endswith((".A", ".B"))],
         "lr": 1e-4},
        {"params": [p for name, p in trainables if name.endswith("risk_level_head.weight")],
         "lr": 3e-5},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    args.out.mkdir(parents=True, exist_ok=True)
    before = evaluate(student, teacher, tokenizer, dev)
    curve = args.out / "loss_curve.jsonl"
    started = time.monotonic()
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    skipped = 0
    step = 0
    with curve.open("w") as log:
        for index, row in enumerate(train):
            result = losses(student, teacher, tokenizer, row)
            if result is None:
                skipped += 1
                continue
            loss = result["total"] / args.accumulation
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss at sample {index}")
            loss.backward()
            accum += 1
            if accum == args.accumulation:
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainables], max_norm=1.0
                )
                if not math.isfinite(float(norm)):
                    raise RuntimeError("Nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {"step": step, "seen_examples": index + 1,
                          "loss": float(result["total"].detach().cpu()),
                          "risk_kl": float(result["risk_kl"].detach().cpu())}
                log.write(json.dumps(record) + "\n")
                if step % 16 == 0:
                    print(json.dumps({"progress": record}), flush=True)
                accum = 0
        if accum:
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainables], max_norm=1.0)
            if not math.isfinite(float(norm)):
                raise RuntimeError("Nonfinite final gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps({"step": step, "seen_examples": len(train),
                                  "partial_accumulation": accum}) + "\n")
    after = evaluate(student, teacher, tokenizer, dev)
    state = {name: p.detach().float().cpu().contiguous()
             for name, p in trainables}
    adapter = args.out / "stage0_adapter.safetensors"
    save_file(state, str(adapter))
    summary = {
        "stage": "open_corpus_prefix_distillation_v1",
        "not_policy_safety_aligned": True,
        "gpu": torch.cuda.get_device_name(0),
        "teacher_model_sha256": sha256_file(args.teacher / "model.safetensors"),
        "student_base_sha256": sha256_file(args.student / "model.safetensors"),
        "initial_adapter_sha256": (sha256_file(args.initial_adapter)
                                   if args.initial_adapter is not None else None),
        "data_manifest_sha256": sha256_file(args.data / "manifest.json"),
        "train_examples_seen": len(train) - skipped,
        "overlength_skipped": skipped,
        "optimizer_steps": step,
        "trainable_parameters": sum(p.numel() for _, p in trainables),
        "dev_before": before,
        "dev_after": after,
        "adapter_sha256": sha256_file(adapter),
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage0_summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
