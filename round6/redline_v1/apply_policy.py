"""Derive labels from stored probe facts under a policy table (Mac, CPU, no model calls).

Inputs are one or more probes.jsonl files from extract_probes.py in priority order (e.g. DeepSeek first, the luna
fallback run second); each response takes the first file where it has a usable result. Every probe's facts go
through policy.decide() with the given switches (default: the user's final table), so a policy change is a re-run
of this script, not of the judge. Output labels.jsonl keeps the per-probe levels; position labels come from
levels.position_level(). A response is usable when its whole text was judged and its probe levels are monotonic.

Word rules (need --source, the rows the Tasks were built from, for the texts). A rule that fires on a word raises
every prefix holding the word's first occurrence to its level (raise_from): prefixes ending before the word keep
the judged levels, prefixes ending inside it get no target.
--screen (word screen T012 screen.jsonl), the seed word of each Run A row:
  insult / rumor, written out: unsafe from the word (the strict leader rule; POLICY section 8 e: the screen decides
  what a leader-name variant means), whatever the judge saw.
  no / evasion / unsure, written out: the judge's leader_epithet_written flag and its variant-spelled R1 hits are
  dropped first (T024 review); plainly written R1 hits (a negative evaluation of a named leader) stay. Then, for
  evasion / unsure only, the political high alert (user, 2026-09-25): a leader named through another spelling is
  cut at the leader_variant level (controversial) from the word.
--political-terms (political screen T026 political_terms.jsonl), matched verbatim in every text, seed or not:
  event / figure / org words at the political_event / _figure / _org level (controversial: special events are cut
  as soon as they are mentioned); leader words (political screen: leader; T012: flagged) whose leader-form check
  (v1.1) says negative (unsafe) or variant (leader_variant). Formal spellings (traditional script, standard pinyin),
  words not about a leader and unsure ones never fire. A gated word (ambiguous: often harmless in ordinary text;
  figure words under figure_scope political_text) fires only when the judge found a political red line in the
  whole text, a plainly spelled topic mention of a leader not counting, and then at most at controversial.
  Single characters never fire. With --screen too, a seed word T012 calls evasion / unsure but the leader-form
  check calls formal or not_leader is treated as "no".
  v1.2 (T026 step 9): once the terms carry the final check (field check), only its fire verdicts are matched, with
  the check's category (event / figure / org / leader_negative / leader_variant). Latin-letter terms need at least
  3 letters and match whole words only; two-character Chinese terms and pure digits are gated; the gate counts
  only the Chinese political red lines (R1, R3-R6, R9), not terrorism or ethnic / racial hatred.
  v1.3 (T026 step 9 again): two-character Chinese terms that are ambiguous never fire; a leader word is negative
  (unsafe) only when the leader-form pass and the check agree, otherwise variant; a word whose screen and check
  disagree between a leader word and an event / figure / org is gated and at most controversial; --exclude-words
  (a local, hand-checked list, never committed) drops words that fire on ordinary text.
Harm outside the red lines (user 2026-09-25, POLICY section 9; needs --source for the texts):
  alert          a response the judge calls non-red-line harm (insults, harassment, threats, rumours, privacy,
                 fraud) keeps the level safe and gets alert = {"from": char or None}: the generation onset of the
                 harmful continuation (v14 onset_char) when the source row has one, else None (build_targets.py
                 then marks only the last clause). Stage 2 trains alert positions to a raised, uncut score.
  harm_politics  the same harm in a text that also mentions a leader or a political special term (a fired
                 political-terms or seed-screen rule, or any hit of a Chinese political red-line code, R1 topic
                 mentions of a leader included) is raised to unsafe: from the first fired word (or the harm onset
                 when that comes later), or, when only the judge saw the political part, from the harm onset (else
                 the last clause) to the end of the text.
Source floor (T031 leader data): a source row with label_floor and an onset span (onset_char, onset_end_char) is
raised to at least label_floor from that span on (rule source_floor); when no probes file has a usable judgement
for it, its label is built from the floor alone (safe before the span, the floor after it; source_rank -1, rule
source_floor_only). The judge's own levels still count when they are higher.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
from levels import RANK, clause_cuts, monotonic, position_level  # noqa: E402
from policy import BOUNDARY_CODES, CHOICES, decide, digest, resolve  # noqa: E402

FINISHED = ("safe", "excluded", "located")
SCREEN_UNSAFE = ("insult", "rumor")
SCREEN_VARIANT = ("evasion", "unsure")          # a leader named through another spelling (political high alert)
SCREEN_NOT_EPITHET = ("no", "evasion", "unsure")
POLITICAL = ("event", "figure", "org")
POLITICAL_CODES = ("R1", "R3", "R4", "R5", "R6", "R9")    # Chinese political red lines (gate); not R7 / R8
LEADER_FLAGGED = ("insult", "rumor", "evasion", "unsure")          # T012 verdicts
LEADER_MATCHED = ("negative", "variant")                          # leader-form verdicts that fire (T026 v1.1)
LEADER_NOT_VARIANT = ("formal", "not_leader")
CHECK_KINDS = {"event": "political:event", "figure": "political:figure", "org": "political:org",
               "leader_negative": "leader_word:negative", "leader_variant": "leader_word:variant"}
MIN_LATIN = 3
LEVEL_NAMES = ("safe", "controversial", "unsafe")


def without_epithet(row):
    """The probe row with the judge's leader-epithet facts removed (the screen says the word is not an insult)."""
    def clean(facts):
        return {**facts, "leader_epithet_written": False,
                "hits": [h for h in facts["hits"] if not (h["code"] == "R1" and h["spelling"] != "plain")]}
    return {**row, "probes": [{**p, "facts": clean(p["facts"])} for p in row["probes"]]}


def onsets_of(probes, whole_level):
    onsets = {}
    for name, rank in (("controversial", 1), ("unsafe", 2)):
        if RANK[whole_level] >= rank:
            cut = min(p["cut"] for p in probes if RANK[p["level"]] >= rank)
            prev = max([p["cut"] for p in probes if RANK[p["level"]] < rank and p["cut"] < cut], default=0)
            onsets[name] = {"prev_cut": prev, "cut": cut}
    return onsets


def raise_from(labelled, start, end, level, rule):
    """Pure. The label row with every prefix that holds text[start:end] raised to at least `level`, or None when
    that changes nothing. A probe is added at `end` when its new level is determined, and at `start` (the judged
    level) so the positions before the word keep their targets; positions inside the word get none."""
    rank = RANK[level]
    old = labelled["probes"]
    probes = {p["cut"]: (LEVEL_NAMES[max(RANK[p["level"]], rank)] if p["cut"] >= end else p["level"]) for p in old}
    lower = max([RANK[p["level"]] for p in old if p["cut"] <= end], default=0)
    upper = min([RANK[p["level"]] for p in old if p["cut"] >= end], default=2)
    if max(lower, rank) == max(upper, rank):
        probes[end] = LEVEL_NAMES[max(lower, rank)]
    before = position_level(start, old)
    if start > 0 and before is not None:
        probes[start] = LEVEL_NAMES[before]
    probes = [{"cut": cut, "level": name} for cut, name in sorted(probes.items())]
    if probes == old:
        return None
    whole = LEVEL_NAMES[max(RANK[labelled["level"]], rank)]
    if RANK[whole] > RANK[labelled["level"]]:
        rules = [rule]
    else:
        rules = labelled["rules"] + [rule]           # same level: the rule moved an onset or shares the level
    return {**labelled, "level": whole, "label": whole, "excluded": False,
            "stratum": "redline" if whole != "safe" else labelled["stratum"],
            "rules": rules, "onsets": onsets_of(probes, whole), "probes": probes}


def screen_override(labelled, text, word, verdict, switches=None):
    """Pure. The seed-word rule: insult / rumor written out -> unsafe from the word, evasion / unsure written out ->
    leader_variant from the word. Returns the label row unchanged when the rule does not apply or changes nothing."""
    if not word or word not in text:
        return labelled
    if verdict in SCREEN_UNSAFE:
        level = "unsafe"
    elif verdict in SCREEN_VARIANT:
        level = resolve(switches)["leader_variant"]
    else:
        return labelled
    start = text.index(word)
    raised = raise_from(labelled, start, start + len(word), level, f"screen:{verdict}")
    if raised is None:
        return labelled
    return {**raised, "screen": {"verdict": verdict, "label_before": labelled.get("screen", {}).get("label_before",
                                                                                                     labelled["label"]),
                                 "first_start": start, "first_end": start + len(word)}}


def latin(word):
    return word.isascii()


def candidate_group(r):
    """Screen-stage group of a word: "leader", "political" (event / figure / org) or None."""
    if r["verdict"] in POLITICAL:
        return "political"
    if r["verdict"] == "leader" and r.get("leader_screen") in LEADER_FLAGGED and r.get("leader_form") in LEADER_MATCHED:
        return "leader"
    return None


def term_table(rows, switches=None, exclude=()):
    """political_terms.jsonl rows (T026, merged over passes) -> {word: (rule, level, gated)} for the words that are
    matched verbatim in every text. Rows with a final check (v1.2) fire only on its fire verdicts; when any row has
    one, rows without it never fire. Left out: words at a switch level of safe, single characters, Latin-letter
    words shorter than MIN_LATIN, ambiguous two-character Chinese words and the excluded words. Gated: ambiguous
    words, figures (figure_scope), two-character Chinese words, pure digits, and words whose screen and check
    disagree between leader and event / figure / org."""
    s = resolve(switches)
    checked = any("check" in r for r in rows)
    table = {}
    for r in rows:
        word, verdict = r["word"], r["verdict"]
        two_chinese = len(word) == 2 and not latin(word)
        if (len(word) < 2 or (latin(word) and len(word) < MIN_LATIN) or word in exclude
                or (two_chinese and r["ambiguous"])):
            continue
        group = candidate_group(r)
        if checked:
            if CHECK_KINDS.get(r.get("check")) is None:
                continue
            verdict = r["check"]
        elif group == "political":
            pass
        elif group == "leader":
            verdict = "leader_" + r["leader_form"]
        else:
            continue
        crossed = group is not None and group != ("leader" if verdict.startswith("leader_") else "political")
        if verdict == "leader_negative" and r.get("leader_form", "negative") != "negative":
            verdict = "leader_variant"                   # the two leader passes disagree: the milder reading
        if verdict == "leader_negative":
            level = "unsafe"
        elif verdict == "leader_variant":
            level = s["leader_variant"]
        else:
            level = s[f"political_{verdict}"]
        gated = (bool(r["ambiguous"]) or (verdict == "figure" and s["figure_scope"] == "political_text")
                 or word.isdigit() or two_chinese or crossed)
        if level != "safe":
            table[word] = (CHECK_KINDS[verdict], level, gated)
    return table


def screen_with_forms(screen, rows):
    """T012 verdicts with evasion / unsure turned into "no" for words the leader-form check calls formal or
    not_leader (a traditional-script or pinyin name is not a variant); every flagged T012 verdict, insult and rumor
    included, turned into "no" where the final check (v1.2) was run and does not call it a leader word (e.g. a
    vulgar phrase aimed at nobody). Returns (screen, number changed)."""
    formal = {r["word"] for r in rows if r.get("leader_form") in LEADER_NOT_VARIANT}
    not_leader = {r["word"] for r in rows if "check" in r and r["check"] not in ("leader_negative", "leader_variant")}
    out = {w: ("no" if (w in not_leader and v in LEADER_FLAGGED) or (w in formal and v in SCREEN_VARIANT) else v)
           for w, v in screen.items()}
    return out, sum(out[w] != screen[w] for w in screen)


def political_text(facts):
    """The judge found a political red line; a plainly spelled topic mention of a leader (news) does not count."""
    return facts["leader_epithet_written"] or any(
        h["code"] in POLITICAL_CODES and not (h["code"] == "R1" and h["act"] == "topic_mention" and h["spelling"] == "plain")
        for h in facts["hits"])


def whole_word_at(text, word, start=0):
    """First index >= start where word occurs with no ASCII letter or digit right before or after it, or -1."""
    while True:
        i = text.find(word, start)
        if i < 0:
            return -1
        before = text[i - 1] if i else " "
        after = text[i + len(word)] if i + len(word) < len(text) else " "
        if not (before.isascii() and before.isalnum()) and not (after.isascii() and after.isalnum()):
            return i
        start = i + 1


class TermIndex:
    """Finds the first occurrence of every table word in a text (words indexed by their first two characters,
    case-sensitive; Latin-letter words only as whole words)."""

    def __init__(self, table):
        self.table = table
        self.by_head = collections.defaultdict(list)
        for word in table:
            self.by_head[word[:2]].append(word)

    def first_occurrences(self, text):
        heads = {text[i:i + 2] for i in range(len(text) - 1)} & self.by_head.keys()
        out = {}
        for head in heads:
            for word in self.by_head[head]:
                i = whole_word_at(text, word) if latin(word) else text.find(word)
                if i >= 0:
                    out[word] = i
        return out


def term_raises(labelled, text, index, whole_facts):
    """Pure. Applies the political-terms rule; returns (label row, list of fired rules with their spans).
    Per rule, the occurrence that ends first fires. A gated word fires only in a political text (political_text()),
    and then at most at controversial."""
    political = political_text(whole_facts)
    first = {}
    for word, start in index.first_occurrences(text).items():
        rule, level, gated = index.table[word]
        if gated:
            if not political:
                continue
            rule, level = rule + ":gated", LEVEL_NAMES[min(RANK[level], 1)]
        end = start + len(word)
        if (rule, level) not in first or end < first[(rule, level)][1]:
            first[(rule, level)] = (start, end)
    fired = []
    for (rule, level), (start, end) in sorted(first.items(), key=lambda x: x[1][1]):
        raised = raise_from(labelled, start, end, level, rule)
        if raised is not None:
            labelled = raised
            fired.append({"rule": rule, "level": level, "start": start, "end": end})
    return labelled, fired


def label_response(row, switches=None):
    """Pure. Returns the label row, or None when the response has no usable result in this file."""
    if row["status"] not in FINISHED:
        return None
    probes = sorted(({"cut": p["cut"], **decide(p["facts"], switches)} for p in row["probes"]), key=lambda p: p["cut"])
    whole = next((p for p in probes if p["cut"] == row["response_chars"]), None)
    if whole is None or not monotonic(probes):
        return None
    onsets = onsets_of(probes, whole["level"])
    whole_facts = next(p["facts"] for p in row["probes"] if p["cut"] == row["response_chars"])
    return {"level": whole["level"], "label": whole["label"], "excluded": whole["excluded"],
            "stratum": whole["stratum"], "rules": whole["rules"], "onsets": onsets,
            "boundary": sorted({f"{h['code']}:{h['act']}" for h in whole_facts["hits"]
                                if h["code"] in BOUNDARY_CODES or h["act"] == "term_explained"}),
            "probes": [{"cut": p["cut"], "level": p["level"]} for p in probes]}


def sensitivity(rows, switches=None):
    """For each pending switch (boundary codes, term_explained, leader_variant): how many usable labels change if it
    is flipped (judge facts only, before the word rules)."""
    base = resolve(switches)
    out = {}
    for name in (*BOUNDARY_CODES, "term_explained", "leader_variant", "general_scope"):
        for value in CHOICES[name]:
            if value == base[name]:
                continue
            changed = collections.Counter()
            for row in rows:
                before, after = label_response(row, switches), label_response(row, {**base, name: value})
                if before is not None and after is not None and before["label"] != after["label"]:
                    changed[f"{before['label']}->{after['label']}"] += 1
            out[f"{name}={value}"] = dict(changed)
    return out


def word_rules(row, labelled, text, word, switches=None, screen=None, index=None):
    """Pure. The seed-word screen rule, then the political-terms rule (see the module docstring)."""
    verdict = screen.get(word) if screen and word else None
    if verdict and word in text:
        before = labelled["label"]
        if verdict in SCREEN_NOT_EPITHET:
            cleaned = without_epithet(row)
            lowered = label_response(cleaned, switches)
            if lowered is not None:
                row, labelled = cleaned, lowered
        labelled = screen_override(labelled, text, word, verdict, switches)
        if labelled["label"] != before or "screen" in labelled:
            labelled = {**labelled, "screen": {**labelled.get("screen", {}), "verdict": verdict,
                                               "label_before": before, "label_after": labelled["label"]}}
    if index is not None:
        whole_facts = next(p["facts"] for p in row["probes"] if p["cut"] == row["response_chars"])
        before, stratum = labelled["label"], labelled["stratum"]
        labelled, fired = term_raises(labelled, text, index, whole_facts)
        if fired:
            labelled = {**labelled, "political": {"label_before": before, "label_after": labelled["label"],
                                                  "stratum_before": stratum, "fired": fired}}
    return labelled


def has_harm(facts, switches=None):
    """The judge found harm outside the red lines (as policy.decide counts it, boundary codes switched off included)."""
    s = resolve(switches)
    return facts["non_redline_harm"] or any(
        h["code"] in BOUNDARY_CODES and s[h["code"]] == "non_redline" and h["act"] in ("writes", "fiction")
        for h in facts["hits"])


def political_mention(facts):
    """The judge saw a Chinese political red-line topic of any kind, a plain news mention of a leader included."""
    return facts["leader_epithet_written"] or any(h["code"] in POLITICAL_CODES for h in facts["hits"])


def last_clause_start(text):
    cuts = clause_cuts(text)
    return cuts[-2] if len(cuts) > 1 else 0


def harm_rules(row, labelled, text, onset, switches=None):
    """Pure. The alert tier and the harm + politics rule (module docstring). onset: the generation onset of the
    harmful continuation (int) or None."""
    s = resolve(switches)
    whole_facts = next(p["facts"] for p in row["probes"] if p["cut"] == row["response_chars"])
    if not has_harm(whole_facts, switches):
        return labelled
    spans = [(f["start"], f["end"]) for f in labelled.get("political", {}).get("fired", [])]
    if labelled.get("screen", {}).get("first_end") is not None and labelled["label"] != labelled["screen"].get("label_before"):
        spans.append((labelled["screen"]["first_start"], labelled["screen"]["first_end"]))
    if s["harm_politics"] != "safe" and (spans or political_mention(whole_facts)):
        if spans:
            start, end = min(spans, key=lambda span: span[1])
            if onset is not None and onset > end:
                start, end = onset, min(len(text), onset + 1)
        else:
            start = onset if onset is not None else last_clause_start(text)
            end = len(text)
        raised = raise_from(labelled, start, end, s["harm_politics"], "harm_politics")
        if raised is not None:
            return {**raised, "harm_politics": {"label_before": labelled["label"], "stratum_before": labelled["stratum"],
                                                "start": start, "end": end, "by_word": bool(spans)}}
        return labelled
    if labelled["label"] == "safe" and s["non_redline_harm"] == "alert":
        return {**labelled, "alert": {"from": onset if onset is not None and 0 <= onset < len(text) else None}}
    return labelled


def floor_rule(labelled, floor):
    """Pure. floor: (level, start, end) or None; raises the label to at least `level` from text[start:end] on."""
    if floor is None:
        return labelled
    level, start, end = floor
    raised = raise_from(labelled, start, end, level, "source_floor")
    return labelled if raised is None else raised


def floor_only(floor, length):
    """Pure. A label built from a source floor alone: safe up to the span, the floor from its end to the text end."""
    level, start, end = floor
    probes = {start: "safe", end: level, length: level} if start > 0 else {end: level, length: level}
    probes = [{"cut": cut, "level": name} for cut, name in sorted(probes.items())]
    return {"level": level, "label": level, "excluded": False, "stratum": "redline", "rules": ["source_floor_only"],
            "onsets": onsets_of(probes, level), "boundary": [], "probes": probes}


def apply(sources, switches=None, screen=None, texts=None, terms=None):
    """Pure. sources: lists of probe rows in priority order; screen: word -> T012 verdict; texts: sample_id ->
    (text, seed word); terms: term_table() output. Returns (label rows, summary)."""
    index = TermIndex(terms) if terms else None
    order, chosen, mismatched = [], {}, set()
    for rank, rows in enumerate(sources):
        for row in rows:
            if row["sample_id"] not in chosen:
                order.append(row["sample_id"])
                chosen[row["sample_id"]] = (None, None, row)
            if chosen[row["sample_id"]][0] is None:
                labelled = label_response(row, switches)
                if labelled is not None:
                    if texts and row["sample_id"] in texts:
                        text, word, onset, floor = (*texts[row["sample_id"]], None, None)[:4]   # (text, word[, onset, floor])
                        if len(text) != row["response_chars"]:
                            mismatched.add(row["sample_id"])
                        else:
                            if screen or index is not None:
                                labelled = word_rules(row, labelled, text, word, switches, screen, index)
                            labelled = floor_rule(labelled, floor)          # before the alert: a floor is not safe
                            labelled = harm_rules(row, labelled, text, onset, switches)
                    chosen[row["sample_id"]] = (rank, labelled, row)
    for sample_id, (rank, _, row) in list(chosen.items()):
        entry = (texts or {}).get(sample_id)
        floor = (*entry, None, None)[3] if entry else None
        if rank is None and floor is not None:
            chosen[sample_id] = (-1, floor_only(floor, len(entry[0])), row)
    used_rows = [row for rank, labelled, row in chosen.values() if rank is not None and rank >= 0]
    out = []
    for sample_id in order:
        rank, labelled, row = chosen[sample_id]
        key = {k: v for k, v in row.items() if k not in ("probes", "errors", "status", "clauses")}
        out.append({**key, "old_label": row.get("label"), "source_rank": rank,
                    **(labelled or {"level": None, "label": "unusable", "excluded": None, "stratum": None,
                                    "rules": [], "onsets": {}, "boundary": [], "probes": []})})
    usable = [r for r in out if r["source_rank"] is not None]
    gaps = collections.defaultdict(list)
    for r in usable:
        for name, onset in r["onsets"].items():
            gaps[name].append(onset["cut"] - onset["prev_cut"])
    political = [r["political"] for r in usable if r.get("political")]
    summary = {"policy_digest": digest(switches), "switches": resolve(switches), "responses": len(out),
               "usable": len(usable), "by_source_rank": dict(collections.Counter(str(r["source_rank"]) for r in out)),
               "labels_by_split": {s: dict(collections.Counter(r["label"] for r in out if r.get("split") == s))
                                   for s in sorted({str(r.get("split")) for r in out})},
               "old_to_new": dict(collections.Counter(f"{r['old_label']}->{r['label']}" for r in out)),
               "old_to_new_by_slot_style": dict(sorted(collections.Counter(
                   f"slot{r.get('index')}:{r.get('response_style')}:{r['old_label']}->{r['label']}" for r in out).items())),
               "stratum_by_label": {label: dict(collections.Counter(r["stratum"] for r in usable if r["label"] == label))
                                    for label in sorted({r["label"] for r in usable})},
               "rules_non_safe": dict(collections.Counter(rule for r in usable if r["level"] != "safe"
                                                          for rule in r["rules"]).most_common()),
               "onset_clause_chars": {k: {"median": statistics.median(v), "max": max(v)} for k, v in gaps.items()},
               "boundary_hits_by_label": {label: dict(collections.Counter(b for r in usable if r["label"] == label
                                                                         for b in r["boundary"]))
                                          for label in sorted({r["label"] for r in usable})},
               "switch_sensitivity": sensitivity(used_rows, switches),
               "texts": {"given": bool(texts), "usable_with_text": sum(bool(texts) and r["sample_id"] in texts
                                                                       for r in usable),
                         "length_mismatch": len(mismatched)},
               "screen_used": bool(screen),
               "screen_overrides": dict(collections.Counter(
                   f"{r['screen']['verdict']}:{r['screen']['label_before']}->{r['screen']['label_after']}"
                   for r in usable if r.get("screen"))),
               "political_terms_used": bool(terms),
               "political_table": dict(sorted(collections.Counter(
                   rule + (":gated" if gated else "") for rule, _, gated in (terms or {}).values()).items())),
               "political_rows": len(political),
               "political_overrides": dict(collections.Counter(
                   f"{'+'.join(sorted({f['rule'] for f in p['fired']}))}:{p['label_before']}->{p['label_after']}"
                   for p in political).most_common()),
               "political_raised_from_stratum": dict(collections.Counter(
                   p["stratum_before"] for p in political if p["label_before"] != p["label_after"])),
               "alert_rows": dict(collections.Counter(
                   "onset" if r["alert"]["from"] is not None else "last_clause" for r in usable if r.get("alert"))),
               "harm_politics": dict(collections.Counter(
                   f"{r['harm_politics']['stratum_before']}:{r['harm_politics']['label_before']}->{r['label']}:"
                   f"{'word' if r['harm_politics']['by_word'] else 'judge'}" for r in usable if r.get("harm_politics")))}
    return out, summary


def onset_of(row, target):
    """The generation onset of a harmful continuation (v14 onset_char, assistant rows only) as an int, or None."""
    value = row.get("onset_char") if target == "assistant" else None
    try:
        return int(value) if value not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def floor_of(row):
    """(label_floor, onset_char, onset_end_char) for a source row that carries a label floor (T031), else None."""
    if row.get("label_floor") in LEVEL_NAMES[1:] and row.get("onset_char") is not None and row.get("onset_end_char"):
        return row["label_floor"], int(row["onset_char"]), int(row["onset_end_char"])
    return None


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, nargs="+", help="probes.jsonl files, highest priority first")
    parser.add_argument("--switches", type=Path, help="JSON object overriding policy switches (default: final table)")
    parser.add_argument("--screen", type=Path, help="word screen screen.jsonl (T012), seed-word rule; needs --source")
    parser.add_argument("--political-terms", type=Path,
                        help="political screen political_terms.jsonl (T026), matched in every text; needs --source")
    parser.add_argument("--exclude-words", type=Path,
                        help="local hand-checked list (one word per line) of political terms that must not fire; never commit it")
    parser.add_argument("--source", type=Path, help="the v14-format rows the Tasks were built from (the texts)")
    parser.add_argument("--target", choices=("assistant", "user"), default="assistant",
                        help="user: the probes label prompts (make_tasks.py --target user)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    switches = json.loads(args.switches.read_text(encoding="utf-8")) if args.switches else None
    if (args.screen or args.political_terms) and not args.source:
        parser.error("--screen and --political-terms need --source")
    sources = [read_jsonl(p) for p in args.probes]
    screen = texts = terms = None
    if args.source:
        rows = read_jsonl(args.source)
        if args.target == "user":
            from make_tasks import prompt_rows
            rows = prompt_rows(rows)
        texts = {r["sample_id"]: (r["messages"][-1]["content"], r.get("word"), onset_of(r, args.target), floor_of(r))
                 for r in rows}
        ids = {row["sample_id"] for rows_ in sources for row in rows_}
        covered = len(ids & texts.keys()) / max(1, len(ids))
        if covered < 0.99:
            parser.error(f"--source holds the text of only {covered:.2%} of the probed ids; give the rows the Tasks "
                         "were built from (Run A: trainable.jsonl; prefix_v2: the export_prefix_v2.py output)")
    if args.screen:
        screen = {r["word"]: r["verdict"] for r in read_jsonl(args.screen)}
        if not any(entry[1] in screen for entry in texts.values() if entry[1]):
            # T021: stage1_runA_v1.jsonl carries no "word", so the screen silently did nothing
            parser.error("--screen matched no word of --source; use rows that carry the word (e.g. Run A trainable.jsonl)")
    lowered = None
    if args.political_terms:
        term_rows = read_jsonl(args.political_terms)
        exclude = set(args.exclude_words.read_text(encoding="utf-8").split("\n")) - {""} if args.exclude_words else set()
        terms = term_table(term_rows, switches, exclude)
        if not terms:
            parser.error("--political-terms gave no matchable word")
        if screen:
            screen, lowered = screen_with_forms(screen, term_rows)
    rows, summary = apply(sources, switches, screen, texts, terms)
    summary["inputs"] = [str(p) for p in args.probes] + [str(p) for p in (args.screen, args.political_terms, args.source) if p]
    summary["screen_words_formal_by_leader_form"] = lowered
    if args.political_terms:
        summary["excluded_words"] = {"listed": len(exclude), "in_candidates": len(exclude & {r["word"] for r in term_rows})}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("policy_digest", "responses", "usable", "by_source_rank", "labels_by_split",
                                              "old_to_new", "screen_overrides", "political_overrides")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
