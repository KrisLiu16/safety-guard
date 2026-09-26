"""L20: are the SGLang streaming outputs (bench_official_l20.py --verify-out) those of the model?

Reference: one transformers forward of Qwen3Guard-Stream-0.6B over the whole sequence (no cache), assistant risk head
at every appended position. Reports per file the largest |d| of the three probabilities and of the cut score
1 - p(safe), and the share of positions where the argmax and the tau = 0.5 decision agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=Path("/tmp/guard06"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    report = {}
    for dump in args.dumps:
        rows = json.loads(dump.read_text())
        worst_p = worst_cut = 0.0
        same_arg = same_cut = total = 0
        for row in rows:
            ids = torch.tensor([row["ids"]], device="cuda")
            with torch.no_grad():
                logits = model(input_ids=ids).risk_level_logits[0].float()
            ref = torch.softmax(logits[row["prime"]:], -1).cpu()
            got = torch.tensor(row["probs"])
            worst_p = max(worst_p, float((ref - got).abs().max()))
            worst_cut = max(worst_cut, float(((1 - ref[:, 0]) - (1 - got[:, 0])).abs().max()))
            same_arg += int((ref.argmax(-1) == got.argmax(-1)).sum())
            same_cut += int(((1 - ref[:, 0] > 0.5) == (1 - got[:, 0] > 0.5)).sum())
            total += len(got)
        report[dump.name] = {"positions": total, "max_abs_prob_diff": worst_p, "max_abs_cut_diff": worst_cut,
                             "argmax_agree": same_arg / total, "decision_agree_tau_0.5": same_cut / total}
        print(dump.name, json.dumps(report[dump.name]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
