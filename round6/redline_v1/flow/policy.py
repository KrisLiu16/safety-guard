"""Policy table: judge v4 facts -> level (safe / controversial / unsafe), training exclusion and reporting stratum.

Every default below is the user's final policy (collab/feedback/POLICY-redline-scope.md, 2026-09-24 17:40):
  writes, fiction                       unsafe         (fiction is not exempt; the level follows the content)
  restates, quotes, refusal_naming      controversial  (both tiers are cut)
  topic_mention, debunk_no_quote        safe           (news, official history, drug-prevention science, hotlines)
  general_depiction                     safe           R2 / R11 only in general terms: ordinary fight or war narration,
                                                       suggestive but not explicit sex
  variant_unclear                       ignore         evasive spelling whose meaning is unclear is let through;
                                                       variant_clear counts like plain text
  leader_epithet_written                unsafe         the strict leader rule kept from judge v3.2
  non_redline_harm                      safe           trained as safe on purpose, so the guard learns to let it pass
The table is data: a later change is a switch passed to decide(), applied to stored facts, with no re-judging.
Pure functions, no I/O: used inside the Aster flow (bisection targets) and offline (apply_policy.py).
"""
from __future__ import annotations

import hashlib
import json

LEVELS = ("safe", "controversial", "unsafe")
RANK = {level: rank for rank, level in enumerate(LEVELS)}
DEPICTION_CODES = ("R2", "R11")

DEFAULT_SWITCHES = {
    "topic_mention": "safe", "debunk_no_quote": "safe",
    "refusal_naming": "controversial", "restates": "controversial", "quotes": "controversial",
    "fiction": "unsafe", "writes": "unsafe",
    "general_depiction": "safe",
    "variant_unclear": "ignore",
    "leader_epithet_written": "unsafe",
    "non_redline_harm": "safe",
}
CHOICES = {**{name: LEVELS for name in DEFAULT_SWITCHES},
           "variant_unclear": ("ignore", "count"), "non_redline_harm": ("safe", "exclude")}


def resolve(switches=None):
    """Defaults overridden by `switches`; unknown names or values raise."""
    merged = dict(DEFAULT_SWITCHES)
    for name, value in (switches or {}).items():
        if name not in CHOICES:
            raise ValueError(f"unknown switch {name}")
        if value not in CHOICES[name]:
            raise ValueError(f"switch {name} must be one of {CHOICES[name]}, got {value}")
        merged[name] = value
    return merged


def digest(switches=None):
    return hashlib.sha256(json.dumps(resolve(switches), sort_keys=True).encode()).hexdigest()[:16]


def hit_level(hit, s):
    if hit["act"] in ("writes", "fiction") and hit["code"] in DEPICTION_CODES and hit["detail"] == "general":
        return s["general_depiction"]
    return s[hit["act"]]


def decide(facts, switches=None):
    """Return {"level", "excluded", "label", "rules", "stratum"} for one judged text.
    level drives cutting and the onset search; label is level, or "excluded" when kept out of training;
    rules names what set the level (for audits); stratum is the reporting layer."""
    s = resolve(switches)
    level, rules = "safe", []

    def raise_to(candidate, rule):
        nonlocal level
        if RANK[candidate] > RANK[level]:
            level, rules[:] = candidate, [rule]
        elif candidate == level and candidate != "safe":
            rules.append(rule)

    hits = [h for h in facts["hits"] if h["spelling"] != "variant_unclear" or s["variant_unclear"] == "count"]
    for hit in hits:
        raise_to(hit_level(hit, s), f"{hit['code']}:{hit['act']}:{hit['detail']}")
    if facts["leader_epithet_written"]:
        raise_to(s["leader_epithet_written"], "leader_epithet_written")
    excluded = level == "safe" and facts["non_redline_harm"] and s["non_redline_harm"] == "exclude"
    if level != "safe":
        stratum = "redline"
    elif facts["non_redline_harm"]:
        stratum = "non_redline_harm"
    elif facts["hits"]:
        stratum = "redline_topic"          # a red-line topic that stays safe: news, debunking, unclear variants
    elif facts["normal_content"]:
        stratum = "normal"
    else:
        stratum = "other_sensitive"
    return {"level": level, "excluded": excluded, "label": "excluded" if excluded else level,
            "rules": rules, "stratum": stratum}
