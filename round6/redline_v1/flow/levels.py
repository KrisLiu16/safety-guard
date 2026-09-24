"""Per-position labels by two-level bisection over clause boundaries (pure logic; used by flow.py, tested on CPU).

Red-line content, once written, stays written as the text grows, so the level of a prefix never goes down:
safe -> controversial -> unsafe. locate() judges the whole text, and when it is not safe, finds by binary search
the first clause cut whose prefix reaches controversial (or higher), then, when the whole is unsafe, the first cut
that reaches unsafe. Each located onset gets one confirmation probe on the next clause, as in onset_v1. Every probe
is kept with its facts, so labels can be re-derived offline under another policy table (position_level()).
Cost: 1 call for a safe text; about log2(clauses) + 2 per level otherwise.
"""
from __future__ import annotations

import re

BOUNDARY = re.compile(r"[。！？；，、.!?;:：\n]+")      # same cut points as onset_v1
LEVEL_VERSION = "guard-redline-levels-v1"
RANK = {"safe": 0, "controversial": 1, "unsafe": 2}
ONSETS = (("controversial", 1), ("unsafe", 2))


def clause_cuts(text):
    """Character offsets after each clause boundary, plus the end of the text; strictly increasing, all > 0."""
    cuts = {m.end() for m in BOUNDARY.finditer(text) if m.end() > 0}
    cuts.add(len(text))
    return sorted(c for c in cuts if 0 < c <= len(text))


def locate(cuts, judge):
    """judge(cut) -> (level, excluded) or None on failure. Returns a result dict with every probe."""
    probes = {}

    def ask(index):
        if index not in probes:
            probes[index] = judge(cuts[index])
        return probes[index]

    def done(status, **extra):
        return {"status": status, "clauses": len(cuts), **extra,
                "probes": [{"cut": cuts[i], "level": v[0] if v else None, "excluded": v[1] if v else None}
                           for i, v in sorted(probes.items())]}

    if not cuts:
        return done("empty")
    last = len(cuts) - 1
    whole = ask(last)
    if whole is None:
        return done("judge_error")
    if RANK[whole[0]] == 0:
        return done("excluded" if whole[1] else "safe", whole=whole[0])
    onsets, lo = {}, 0
    for name, rank in ONSETS:
        if RANK[whole[0]] < rank:
            break
        hi = last
        while lo < hi:
            mid = (lo + hi) // 2
            verdict = ask(mid)
            if verdict is None:
                return done("judge_error", whole=whole[0], onsets=onsets)
            if RANK[verdict[0]] >= rank:
                hi = mid
            else:
                lo = mid + 1
        if hi + 1 < last and ask(hi + 1) is None:        # confirmation probe past the located clause
            return done("judge_error", whole=whole[0], onsets=onsets)
        onsets[name] = {"index": hi, "cut": cuts[hi], "prev_cut": cuts[hi - 1] if hi else 0}
    result = done("located", whole=whole[0], onsets=onsets)
    if not monotonic(result["probes"]):
        result["status"] = "nonmonotonic"
    return result


def monotonic(probes):
    """No longer prefix judged at a lower level than a shorter one."""
    ranks = [RANK[p["level"]] for p in sorted(probes, key=lambda p: p["cut"]) if p["level"] is not None]
    return all(a <= b for a, b in zip(ranks, ranks[1:]))


def position_level(char_end, probes):
    """Level (0/1/2) of the prefix ending at char_end, from judged probes [{cut, level}], or None if undetermined.
    A probe at cut c bounds every prefix: shorter ones are at most its level, longer ones at least. The empty
    prefix is safe. Positions inside a located onset clause are undetermined, as are all positions of a
    non-monotonic probe set."""
    judged = [(0, 0)] + [(p["cut"], RANK[p["level"]]) for p in probes if p["level"] is not None]
    lower = max([rank for cut, rank in judged if cut <= char_end], default=0)
    upper = min([rank for cut, rank in judged if cut >= char_end], default=None)
    if upper is None or lower != upper:
        return None
    return lower
