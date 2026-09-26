"""v3 (T033) teacher labels (L20, inference only): Qwen3Guard-Stream-4B's per-token distributions on student texts.

For every case of make_teacher_inputs.py the teacher (official checkpoint, published chat template, transformers 4.55)
scores the last message with the head of its role -- query head for user messages, response head for assistant
messages -- at every token of that message, as in streaming. Output per case (JSONL, gzip):
  ends   character end of each teacher token, relative to the start of the scored message's content (the student
         maps its own token ends onto these: the teacher token with the largest end <= the student's end)
  risk   [p(Safe), p(Unsafe), p(Controversial)] per token, 4 decimals
  cat    argmax category per token; catp its probability (category maps of the checkpoint's config)
--end-user (v3.1): only user cases, and only the query head at the turn's closing <|im_end|> token, which is where
Qwen3Guard-Stream is trained and read for prompts (the per-token query outputs inside the prompt are not trained and
made the v3 user general head flag ordinary prompts). Output per case: end_risk [p(Safe), p(Unsafe),
p(Controversial)], end_cat, end_catp.
Cases are sorted by length and packed (right padding) into micro-batches of at most --micro-tokens padded tokens; a
case longer than --max-tokens loses context from the left, never the scored message. Texts and outputs stay out of
the repository.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

HEADERS = {"user": "<|im_start|>user\n", "assistant": "<|im_start|>assistant\n"}


def content_span(text, content, role):
    """Pure: (start, end) of the scored message's content in the templated text (the last occurrence after the
    role's last header); raises if the template changed the content."""
    header = text.rfind(HEADERS[role])
    if header < 0:
        raise ValueError("role header not found")
    start = text.find(content, header)
    if start < 0:
        raise ValueError("content not found verbatim after its header")
    return start, start + len(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/data/models/guard4b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-tokens", type=int, default=24576)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--end-user", action="store_true", help="user cases only, read at the closing <|im_end|>")
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    if torch.cuda.device_count() != 1 or "L20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("needs exactly one visible L20")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    with args.inputs.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if args.end_user:
        cases = [c for c in cases if c["role"] == "user"]
    if args.limit:
        cases = cases[:args.limit]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    prepared, skipped, truncated = [], 0, 0
    for case in cases:
        messages = case["messages"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if not text.endswith("<|im_end|>\n"):
            skipped += 1
            continue
        text = text[:-1]
        try:
            start, end = content_span(text, messages[-1]["content"].strip() or messages[-1]["content"], case["role"])
        except ValueError:
            skipped += 1
            continue
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        scored = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        if not scored:
            skipped += 1
            continue
        drop = 0
        if len(ids) > args.max_tokens:
            drop = min(len(ids) - args.max_tokens, scored[0])
            truncated += 1
        ends = [min(offsets[i][1], end) - start for i in scored]
        if args.end_user and ids[-1] != im_end:
            skipped += 1
            continue
        prepared.append((case, ids[drop:], [i - drop for i in scored], ends))
    prepared.sort(key=lambda p: len(p[1]))
    began, done = time.monotonic(), 0
    with gzip.open(args.output, "wt", encoding="utf-8") as out, torch.inference_mode():
        i = 0
        while i < len(prepared):
            batch = [prepared[i]]
            while (i + len(batch) < len(prepared) and len(batch) < args.max_rows
                   and len(prepared[i + len(batch)][1]) * (len(batch) + 1) <= args.micro_tokens):
                batch.append(prepared[i + len(batch)])
            i += len(batch)
            length = max(len(p[1]) for p in batch)
            x = torch.full((len(batch), length), tok.pad_token_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(x)
            for row, (_, ids, _, _) in enumerate(batch):
                x[row, :len(ids)] = torch.tensor(ids, device="cuda")
                mask[row, :len(ids)] = 1
            output = model(input_ids=x, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                           use_cache=False, logits_to_keep=0)
            for row, (case, ids, scored, ends) in enumerate(batch):
                if args.end_user:                                        # the closing <|im_end|> is the last real token
                    risk = output.query_risk_level_logits[row, len(ids) - 1].float().softmax(-1).cpu()
                    cat = output.query_category_logits[row, len(ids) - 1].float().softmax(-1).cpu()
                    out.write(json.dumps({"id": case["id"], "source": case["source"], "role": "user",
                                          "end_risk": [round(v, 4) for v in risk.tolist()], "end_cat": int(cat.argmax()),
                                          "end_catp": round(float(cat.max()), 3)}) + "\n")
                    continue
                user = case["role"] == "user"
                risk = (output.query_risk_level_logits if user else output.risk_level_logits)[row, scored].float()
                cat = (output.query_category_logits if user else output.category_logits)[row, scored].float()
                risk, cat = risk.softmax(-1).cpu(), cat.softmax(-1).cpu()
                catp, catix = cat.max(-1)
                out.write(json.dumps({"id": case["id"], "source": case["source"], "role": case["role"], "ends": ends,
                                      "risk": [[round(v, 4) for v in r] for r in risk.tolist()],
                                      "cat": catix.tolist(), "catp": [round(v, 3) for v in catp.tolist()]}) + "\n")
            done += len(batch)
            if done % 4096 < len(batch):
                print(json.dumps({"done": done, "of": len(prepared), "seconds": round(time.monotonic() - began, 1)}),
                      flush=True)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"cases": len(prepared), "skipped": skipped, "truncated": truncated,
                      "seconds": round(time.monotonic() - began, 1), "output_sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
