"""Onset relocation by bisection over clause boundaries (pure logic; used by flow.py and tested on CPU).

A prefix that already contains harmful text stays unsafe when it grows, so the first unsafe prefix among the
clause cut points can be found by binary search: judge the whole text (it must be unsafe), then halve the
range of cut points. The result is the clause (prev_cut, cut] whose completion first makes the text unsafe.
One confirmation probe on the next clause follows, since bisection only sees the prefixes it asks about.
Every probe is recorded; a longer prefix judged safe after a shorter one was judged unsafe is flagged as
non-monotonic and is never silently used. Cost: about log2(clauses) + 2 judge calls per response.
"""
from __future__ import annotations

import re

BOUNDARY = re.compile(r"[。！？；，、.!?;:：\n]+")
ONSET_VERSION = "guard-onset-bisect-v1"


def clause_cuts(text):
    """Character offsets after each clause boundary, plus the end of the text; strictly increasing, all > 0."""
    cuts = {m.end() for m in BOUNDARY.finditer(text) if m.end() > 0}
    cuts.add(len(text))
    return sorted(c for c in cuts if 0 < c <= len(text))


def bisect(cuts, judge):
    """judge(cut) -> "safe" | "unsafe" | "controversial" | None (failure). Returns a result dict.
    Only "unsafe" counts as unsafe; controversial is treated as not yet unsafe."""
    probes = {}

    def ask(index):
        if index not in probes:
            probes[index] = judge(cuts[index])
        return probes[index]

    last = len(cuts) - 1
    whole = ask(last)
    if whole is None:
        return {"status": "judge_error", "probes": _probes(cuts, probes)}
    if whole != "unsafe":
        return {"status": "whole_not_unsafe", "whole_verdict": whole, "probes": _probes(cuts, probes)}
    lo, hi = 0, last
    while lo < hi:
        mid = (lo + hi) // 2
        verdict = ask(mid)
        if verdict is None:
            return {"status": "judge_error", "probes": _probes(cuts, probes)}
        if verdict == "unsafe":
            hi = mid
        else:
            lo = mid + 1
    if hi + 1 < last:
        # One confirmation probe past the located clause: bisection only sees the prefixes it asks about, so a
        # noisy judge could otherwise pass unnoticed. The next clause must be unsafe too.
        if ask(hi + 1) is None:
            return {"status": "judge_error", "probes": _probes(cuts, probes)}
    result = {"status": "located", "onset_index": hi, "cut": cuts[hi], "prev_cut": cuts[hi - 1] if hi else 0,
              "clauses": len(cuts), "probes": _probes(cuts, probes)}
    if not monotonic(result["probes"]):
        result["status"] = "nonmonotonic"
    return result


def _probes(cuts, probes):
    return [{"cut": cuts[i], "verdict": v} for i, v in sorted(probes.items())]


def monotonic(probes):
    """No prefix judged not-unsafe after a shorter prefix was judged unsafe."""
    seen_unsafe = False
    for probe in sorted(probes, key=lambda p: p["cut"]):
        if probe["verdict"] == "unsafe":
            seen_unsafe = True
        elif seen_unsafe:
            return False
    return True
