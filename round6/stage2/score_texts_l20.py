"""Stage 2 spot check (L20, inference only): per-token scores of given conversations under one or more checkpoints.

Each input line is {"id", "messages": [{"role", "content"}...]}; the last message is scored with the head of its
role, in the training serialization (probe_common.serialize) and the same model code as train_stage2.py (Round4
Classifier, W512 window). For every checkpoint and conversation the output holds, per content token of the last
message, the token's text, 1 - p(safe) and p(unsafe) at temperature 1 (logits kept, so any temperature can be
applied later), and the first position above each --tau for the cut score. The texts are local test cases: the
input and output files stay out of the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
sys.path.insert(0, str(HERE))
from probe_common import serialize  # noqa: E402
from train_stage2 import import_runtime, make_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--round4-code", type=Path, default=Path("/work/round4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.inputs.open(encoding="utf-8") as handle:      # not splitlines(): texts may hold U+2028
        cases = [json.loads(line) for line in handle if line.strip()]
    torch, helper, load_file = import_runtime(args.round4_code)
    model, pad = make_model(helper, load_file, args.checkpoint[0])
    from tokenizers import Tokenizer
    backend = Tokenizer.from_file("/work/models/qwen35/tokenizer.json")      # as build_targets.py
    if backend.truncation is not None or backend.padding is not None:
        raise ValueError("tokenizer truncation/padding must be disabled")
    out = []
    for checkpoint in args.checkpoint:
        state = load_file(str(checkpoint))
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        for case in cases:
            messages = case["messages"]
            text = serialize(messages)
            start = len(text) - len(messages[-1]["content"])
            encoding = backend.encode(text, add_special_tokens=False)
            ids, offsets = encoding.ids, encoding.offsets
            with torch.inference_mode():
                hidden = model(torch.tensor([ids], device="cuda"), torch.ones(1, len(ids), dtype=torch.long,
                                                                               device="cuda")).last_hidden_state[0]
                logits, _ = model.readout(hidden, messages[-1]["role"])
                logits = logits.float().cpu()
            tokens = []
            for i, (a, b) in enumerate(offsets):
                if b <= start:
                    continue
                p = torch.softmax(logits[i], -1).tolist()
                tokens.append({"i": i, "text": text[a:b], "end": b - start, "logits": [round(v, 4) for v in logits[i].tolist()],
                               "cut": round(1 - p[0], 6), "unsafe": round(p[1], 6)})
            out.append({"checkpoint": str(checkpoint), "id": case["id"], "tokens": tokens,
                        "max_cut": max(t["cut"] for t in tokens), "max_unsafe": max(t["unsafe"] for t in tokens)})
            print(json.dumps({k: out[-1][k] for k in ("checkpoint", "id", "max_cut", "max_unsafe")}, ensure_ascii=False),
                  flush=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
