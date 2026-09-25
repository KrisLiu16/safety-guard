"""v3 (T033) checkpoint views for the existing evaluators (CPU): the stage-2 v3 checkpoint carries, per role, the
red-line head (projection / risk / category) and a general-harm head (general_projection / general /
general_category). Every evaluator (eval_head_l20.py, score_texts_l20.py, score_bench_l20.py, the demo) loads the
Round4 Classifier layout with a strict key match, so this writes one of two standard-layout files:
  --mode redline   drop the general keys: the model that decides cuts, as v1 / v2
  --mode general   put the general head into the standard slots (projection <- general_projection, risk <- general,
                   category <- general_category): the same evaluators then score the general head
The backbone is shared and copied unchanged. The output records the source SHA256 in its metadata.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SLOTS = {"general_projection": "projection", "general": "risk", "general_category": "category"}


def view(state, mode):
    """Pure (dict of name -> tensor): the standard-layout state for the mode."""
    if mode not in ("redline", "general"):
        raise ValueError(mode)
    general = {n for n in state if n.startswith("heads.") and n.split(".")[2] in SLOTS}
    if not general:
        raise ValueError("no general-head keys: not a v3 checkpoint")
    out = {n: t for n, t in state.items() if n not in general}
    if mode == "general":
        for n in general:
            parts = n.split(".")
            target = ".".join(parts[:2] + [SLOTS[parts[2]]] + parts[3:])
            if target not in out:
                raise ValueError(f"{n} has no standard slot {target}")
            out[target] = state[n]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("redline", "general"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from safetensors.torch import load_file, save_file
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    state = load_file(str(args.checkpoint))
    out = view(state, args.mode)
    save_file({n: t.contiguous() for n, t in out.items()}, str(args.output),
              metadata={"source_sha256": digest, "view": args.mode})
    print({"mode": args.mode, "tensors": len(out), "source_sha256": digest,
           "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
