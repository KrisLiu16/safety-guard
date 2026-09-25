"""Qwen3Guard-Stream-0.6B on the benchmark cases (L20, inference only): the same-harness baseline for our guard.

The paper's numbers use its own serving code and the original WildGuardTest / XSTest; this scores the official
checkpoint (/work/models/guard, loaded as in round3's teacher export: /work/input/runtime.py, transformers 4.55 in
/work/legacy) on exactly our fetch.py cases, so the two models are compared under one harness. Serialization is the
published chat template (runtime.encode, complete messages: the text ends with the last message's <|im_end|>).
Scores per case, cut = 1 - p(Safe) (strict: Controversial counts as unsafe), unsafe = p(Unsafe) (loose):
  prompt cases     query head (query_risk_level_logits), over the user turn's tokens after its header
  response cases   response head (risk_level_logits), over the assistant turn's tokens after its header
  end_* at the last token (<|im_end|>, where the paper trains the query head), max_* over the turn, and max2_cut =
  max over consecutive token pairs of the smaller cut (the paper's two-token debounce, thresholded at tau).
Output rows as score_bench_l20.py (checkpoint name "qwen3guard_stream_0.6b"). Cases stay out of the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

MICRO_TOKENS, MAX_ROWS, MAX_TOKENS = 16384, 32, 8192
HEADERS = {"user": "<|im_start|>user\n", "assistant": "<|im_start|>assistant\n"}


def turn_scores(cut, unsafe):
    """Pure: end / max / debounced summaries of one turn's per-token scores."""
    pairs = [min(a, b) for a, b in zip(cut, cut[1:])]
    return {"end_cut": cut[-1], "max_cut": max(cut), "end_unsafe": unsafe[-1], "max_unsafe": max(unsafe),
            "max2_cut": max(pairs) if pairs else cut[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/work/models/guard"))
    parser.add_argument("--runtime", type=Path, default=Path("/work/input"))
    parser.add_argument("--name", default="qwen3guard_stream_0.6b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    import torch
    from runtime import load
    if torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("needs exactly one visible L20")
    model, tok = load(dtype=torch.bfloat16, device="cuda", model_path=args.model)
    with args.cases.open(encoding="utf-8") as handle:           # not splitlines(): texts hold U+2028 etc.
        cases = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    prepared, truncated = [], 0
    for case in cases:
        role = case["messages"][-1]["role"]
        text = tok.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=False,
                                       enable_thinking=False)
        if not text.endswith("<|im_end|>\n"):
            raise ValueError(f"{case['id']}: unexpected template suffix")
        text = text[:-1]
        start = text.rfind(HEADERS[role]) + len(HEADERS[role])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        first = next(i for i, (_, e) in enumerate(offsets) if e > start)
        cut = 0
        if len(ids) > MAX_TOKENS:                       # drop context from the left, never the scored turn
            cut = min(len(ids) - MAX_TOKENS, first)
            truncated += 1
        prepared.append((case, role, ids[cut:], first - cut))
    prepared.sort(key=lambda p: len(p[2]))
    began, done = time.monotonic(), 0
    with args.output.open("w", encoding="utf-8") as out, torch.inference_mode():
        i = 0
        while i < len(prepared):
            batch = [prepared[i]]
            while (i + len(batch) < len(prepared) and len(batch) < MAX_ROWS
                   and len(prepared[i + len(batch)][2]) * (len(batch) + 1) <= MICRO_TOKENS):
                batch.append(prepared[i + len(batch)])
            i += len(batch)
            length = max(len(p[2]) for p in batch)
            x = torch.full((len(batch), length), tok.pad_token_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(x)
            for row, (_, _, ids, _) in enumerate(batch):
                x[row, :len(ids)] = torch.tensor(ids, device="cuda")
                mask[row, :len(ids)] = 1
            output = model(input_ids=x, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                           use_cache=False, logits_to_keep=0)
            for row, (case, role, ids, first) in enumerate(batch):
                logits = (output.query_risk_level_logits if role == "user" else output.risk_level_logits)[row]
                p = logits[first:len(ids)].float().softmax(-1).cpu()
                cut, unsafe = (1 - p[:, 0]).tolist(), p[:, 1].tolist()
                out.write(json.dumps({"checkpoint": args.name, "id": case["id"], **{k: round(v, 6) for k, v in
                                      turn_scores(cut, unsafe).items()}, "n_tokens": len(ids),
                                      "n_scored": len(ids) - first}) + "\n")
            done += len(batch)
            if done % 2048 < len(batch):
                print(json.dumps({"done": done, "of": len(prepared), "seconds": round(time.monotonic() - began, 1)}),
                      flush=True)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"cases": len(prepared), "truncated": truncated, "seconds": round(time.monotonic() - began, 1),
                      "output_sha256": sha}), flush=True)


if __name__ == "__main__":
    main()
