"""Qwen3Guard English benchmarks (L20, inference only): per-case scores of fetch.py's cases under one or more checkpoints.

Same model code and serialization as round6/stage2/score_texts_l20.py: train_stage2.make_model (Round4 Classifier,
24 layers, W512 window), probe_common.serialize, the Qwen3.5 tokenizer without special tokens. Each --checkpoint
(optionally "name=path") is loaded in turn into the same model. The last message of a case is scored with the head
of its role (prompt cases: the user head over the user message; response cases: the assistant head over the
assistant message), cut score = 1 - p(safe), p(unsafe) at temperature 1. Per (checkpoint, case) one output row:
  end_cut / end_unsafe     at the last token of the scored message (prompt level: the end-of-prompt decision)
  max_cut / max_unsafe     max over the scored message's tokens (response level: the streaming decision)
  first_over               index (in the scored message's tokens) of the first cut > --first-tau, else null
  n_tokens / n_scored / truncated
Cases are sorted by length and packed into microbatches (right padding, attention mask, as train_stage2
calibration_scores) of at most --micro-tokens padded tokens and --max-rows rows. A case longer than --max-tokens loses
tokens from the left of its context ("context"); the scored message is never cut, so a case whose last message alone
is longer keeps that whole message and runs alone ("overlong"). Both are counted in <output>.summary.json.
The cases hold dataset text: inputs and outputs stay out of the repository.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
sys.path.insert(0, str(HERE.parent / "stage2"))
from probe_common import serialize  # noqa: E402

MAX_TOKENS, MICRO_TOKENS, MAX_ROWS, FIRST_TAU = 8192, 16384, 64, 0.5


# ---------------------------------------------------------------- pure (unit tested on CPU)

def truncation(n_tokens, first_last_token, max_tokens):
    """(start index, status): keep the last max_tokens tokens, but never start inside the last message."""
    if n_tokens <= max_tokens:
        return 0, "none"
    start = n_tokens - max_tokens
    if start <= first_last_token:
        return start, "context"
    return first_last_token, "overlong"


def prepare(case, ids, offsets, max_tokens=MAX_TOKENS):
    """Case + its serialization's token ids / char offsets -> model row (ids after truncation, scored positions)."""
    messages = case["messages"]
    text = serialize(messages)
    last = messages[-1]
    content_start = len(text) - len(last["content"])
    header_start = content_start - len(last["role"].upper() + ":\n")
    if len(ids) != len(offsets) or not ids:
        raise ValueError(f"{case['id']}: bad encoding")
    first_last = next(i for i, (_, end) in enumerate(offsets) if end > header_start)
    positions = [i for i, (_, end) in enumerate(offsets) if end > content_start]
    empty = not positions
    if empty:                                     # empty message: score the end of its header
        positions = [len(ids) - 1]
    start, status = truncation(len(ids), first_last, max_tokens)
    return {"id": case["id"], "bench": case["bench"], "level": case["level"], "role": last["role"],
            "ids": list(ids[start:]), "positions": [p - start for p in positions], "n_tokens": len(ids),
            "truncated": status, "empty": empty}


def pack(lengths, micro_tokens=MICRO_TOKENS, max_rows=MAX_ROWS):
    """Indices sorted by length, packed so that rows <= max_rows and longest x rows <= micro_tokens (a longer row
    goes alone)."""
    out, current, longest = [], [], 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        length = lengths[index]
        if current and (len(current) == max_rows or max(longest, length) * (len(current) + 1) > micro_tokens):
            out.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, length)
    if current:
        out.append(current)
    return out


def summarize(cut, unsafe, first_tau=FIRST_TAU):
    """Per-position cut / unsafe lists of the scored message -> the output fields."""
    first = next((i for i, c in enumerate(cut) if c > first_tau), None)
    return {"end_cut": round(cut[-1], 6), "end_unsafe": round(unsafe[-1], 6), "max_cut": round(max(cut), 6),
            "max_unsafe": round(max(unsafe), 6), "first_over": first}


def parse_checkpoint(text):
    """'name=path' or 'path' -> (name, path)."""
    name, sep, path = text.partition("=")
    if sep and "/" not in name:
        return name, Path(path)
    return text, Path(text)


# ---------------------------------------------------------------- L20

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path, help="fetch.py data/cases.jsonl")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="safetensors, optionally name=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/work/models/qwen35/tokenizer.json"))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--micro-tokens", type=int, default=MICRO_TOKENS)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--first-tau", type=float, default=FIRST_TAU)
    parser.add_argument("--limit", type=int, default=0, help="N cases at an even stride (smoke run)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoints = [parse_checkpoint(c) for c in args.checkpoint]
    if len({name for name, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint names must be distinct")
    for _, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.cases.open(encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if args.limit:                                # an even stride, so a smoke run touches every bench
        cases = cases[::max(1, len(cases) // args.limit)][:args.limit]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case ids must be unique")

    from train_stage2 import import_runtime, make_model, pad_batch
    torch, helper, load_file = import_runtime(args.round4_code)
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file(str(args.tokenizer))
    if backend.truncation is not None or backend.padding is not None:
        raise ValueError("tokenizer truncation/padding must be disabled")
    began = time.monotonic()
    encodings = backend.encode_batch([serialize(c["messages"]) for c in cases], add_special_tokens=False)
    rows = [prepare(c, e.ids, e.offsets, args.max_tokens) for c, e in zip(cases, encodings)]
    del encodings
    batches = pack([len(r["ids"]) for r in rows], args.micro_tokens, args.max_rows)
    summary = {"version": "qwen3guard-bench-scores-v1", "cases": len(rows), "cases_sha256": sha256_file(args.cases),
               "limit": args.limit, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "input_tokens": sum(len(r["ids"]) for r in rows),
               "truncated": dict(collections.Counter(r["truncated"] for r in rows)),
               "truncated_by_bench": dict(collections.Counter(f"{r['bench']}:{r['truncated']}" for r in rows
                                                               if r["truncated"] != "none")),
               "empty_scored_message": sum(r["empty"] for r in rows), "microbatches": len(batches),
               "tokenize_seconds": round(time.monotonic() - began, 1), "checkpoints": []}
    print(json.dumps({k: summary[k] for k in ("cases", "input_tokens", "truncated", "microbatches")}), flush=True)

    model, pad = make_model(helper, load_file, checkpoints[0][1])
    partial = args.output.with_name(args.output.name + ".part")
    with partial.open("w", encoding="utf-8") as out:
        for name, path in checkpoints:
            start = time.monotonic()
            state = load_file(str(path))
            model.load_state_dict(state, strict=True)
            del state
            model.eval()
            done = 0
            with torch.inference_mode():
                for batch in batches:
                    group = [rows[i] for i in batch]
                    ids, mask = pad_batch(torch, group, pad, "cuda")
                    hidden = model(ids, mask, use_cache=False).last_hidden_state
                    for index, row in enumerate(group):
                        logits, _ = model.readout(hidden[index, row["positions"]], row["role"])
                        probs = torch.softmax(logits.float(), dim=-1).cpu()
                        cut = (1.0 - probs[:, 0]).tolist()
                        unsafe = probs[:, 1].tolist()
                        record = {"checkpoint": name, "id": row["id"], "bench": row["bench"], "level": row["level"],
                                  **summarize(cut, unsafe, args.first_tau), "n_tokens": row["n_tokens"],
                                  "n_scored": len(row["positions"]), "truncated": row["truncated"]}
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    del hidden
                    done += len(group)
                    if done % 2048 < len(group):
                        print(json.dumps({"checkpoint": name, "done": done, "of": len(rows),
                                          "seconds": round(time.monotonic() - start, 1)}), flush=True)
            out.flush()
            seconds = time.monotonic() - start
            summary["checkpoints"].append({"name": name, "path": str(path), "sha256": sha256_file(path),
                                           "seconds": round(seconds, 1),
                                           "tokens_per_second": round(summary["input_tokens"] / seconds)})
            print(json.dumps(summary["checkpoints"][-1]), flush=True)
    partial.replace(args.output)
    summary["seconds"] = round(time.monotonic() - began, 1)
    summary["output_sha256"] = sha256_file(args.output)
    args.output.with_name(args.output.name + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("cases", "seconds", "output_sha256")}), flush=True)


if __name__ == "__main__":
    main()
