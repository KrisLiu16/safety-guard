"""Bar charts of the Qwen3Guard benchmark numbers (Mac, no plotting library): one SVG file.

Top: F1 of the unsafe class per bench (metrics.py --json output at one tau), one bar per checkpoint plus the
paper's Qwen3Guard-Stream-0.6B (strict), prompt benches then response benches, each level with its Avg over the
benches every column has. Bottom (with --subset, redline_subset.py --json output): the share above tau per red-line
group, i.e. recall for the harm groups and the false positive rate for safe cases. Numbers only; no case text.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import PAPER  # noqa: E402

COLORS = ["#8c8c8c", "#4e79a7", "#e15759", "#59a14f", "#f28e2b"]
PAPER_COLOR = "#b07aa1"
PAPER_NAME = "Qwen3Guard-Stream-0.6B (paper)"
PROMPT = ("ToxicChat", "OpenAIMod", "Aegis", "Aegis2.0", "SimpleSafetyTests", "HarmBench-P", "WildGuardTest-P")
RESPONSE = ("HarmBench-R", "SafeRLHF", "BeaverTails", "Aegis2.0-R", "WildGuardTest-R", "Think")
GROUP_NAMES = {"weapons_drugs": "weapons/drugs (RL12)", "self_harm": "self-harm (RL13)", "sexual": "sexual (RL11)",
               "violence": "violence (RL2/7)", "other_harm": "other harm (not RL)", "safe": "safe → false positives"}


def bars(svg, x0, y0, width, height, groups, series, colors, title, ymax=100):
    """groups: labels; series: [(name, [value or None per group])]. Draws axes, bars and value labels."""
    svg.append(f'<text x="{x0}" y="{y0 - 14}" font-size="15" font-weight="bold">{html.escape(title)}</text>')
    for tick in range(0, ymax + 1, 20):
        y = y0 + height - height * tick / ymax
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end">{tick}</text>')
    slot = width / len(groups)
    bar = min(18.0, slot * 0.8 / max(1, len(series)))
    for g, label in enumerate(groups):
        left = x0 + g * slot + (slot - bar * len(series)) / 2
        for s, (name, values) in enumerate(series):
            v = values[g]
            if v is None:
                continue
            h = height * v / ymax
            x = left + s * bar
            svg.append(f'<rect x="{x:.1f}" y="{y0 + height - h:.1f}" width="{bar - 1:.1f}" height="{h:.1f}" '
                       f'fill="{colors[s]}"><title>{html.escape(name)}: {v:.1f}</title></rect>')
            svg.append(f'<text x="{x + bar / 2 - 0.5:.1f}" y="{y0 + height - h - 3:.1f}" font-size="8" '
                       f'text-anchor="middle">{v:.0f}</text>')
        cx = x0 + g * slot + slot / 2
        bold = ' font-weight="bold"' if label.startswith("Avg") else ""
        svg.append(f'<text x="{cx:.1f}" y="{y0 + height + 14}" font-size="10" text-anchor="end"{bold} '
                   f'transform="rotate(-30 {cx:.1f} {y0 + height + 14})">{html.escape(label)}</text>')


def legend(svg, x, y, entries):
    for i, (name, color) in enumerate(entries):
        svg.append(f'<rect x="{x + i * 230}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{x + i * 230 + 17}" y="{y}" font-size="12">{html.escape(name)}</text>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.py --json output")
    parser.add_argument("--subset", type=Path, help="redline_subset.py --json output")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.metrics.read_text()) if abs(r["tau"] - args.tau) < 1e-9]
    checkpoints = list(dict.fromkeys(r["checkpoint"] for r in rows))
    f1 = {(r["checkpoint"], r["bench"]): r["f1"] for r in rows}
    colors = COLORS[:len(checkpoints)] + [PAPER_COLOR]
    names = checkpoints + [PAPER_NAME]
    width = 1200
    svg = []
    y = 70
    for level, benches in (("prompt", PROMPT), ("response", RESPONSE)):
        present = [b for b in benches if all((c, b) in f1 for c in checkpoints)]
        both = [b for b in present if b in PAPER]
        series = []
        for c in checkpoints:
            values = [f1[(c, b)] for b in present]
            series.append((c, values + [sum(f1[(c, b)] for b in both) / len(both)]))
        series.append((PAPER_NAME, [PAPER.get(b) for b in present] + [sum(PAPER[b] for b in both) / len(both)]))
        bars(svg, 60, y, width - 100, 220, present + [f"Avg ({len(both)} benches)"], series, colors,
             f"{level.capitalize()} classification, F1 of the unsafe class (%), tau = {args.tau:g}")
        y += 330
    if args.subset:
        subset = json.loads(args.subset.read_text())
        for level in ("prompt", "response"):
            keys = [k for k in subset if k.startswith(f"{args.tau:g}/{level}/")]
            groups = [k.split("/")[-1] for k in keys]
            series = [(c, [subset[k].get(c) for k in keys]) for c in checkpoints]
            bars(svg, 60, y, width - 100, 180, [f"{GROUP_NAMES.get(g, g)} n={subset[k]['n']}" for g, k in zip(groups, keys)],
                 series, colors, f"{level.capitalize()}: share flagged by red-line group (recall; safe = false "
                                 f"positive rate), %, tau = {args.tau:g}")
            y += 330
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y + 10}" '
            f'font-family="Helvetica, Arial, sans-serif">', f'<rect width="100%" height="100%" fill="white"/>']
    legend(head, 60, 24, list(zip(names, colors)))
    args.output.write_text("\n".join(head + svg + ["</svg>"]) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
