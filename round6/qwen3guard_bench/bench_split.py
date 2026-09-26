"""Fixed halves of the benchmark cases (T037, v5): the general head may learn Qwen3Guard-Stream-4B's outputs on the
"fit" half; the v5 comparison with Qwen3Guard-Stream-0.6B reports the "held" half only.

A case's half follows its benchmark and the text of its first user message, so every response to one prompt lands
in the same half and no prompt is seen in training and scored in the report. Think (the official thinking set, also
the red-line evaluation's official set) is held whole.
"""
from __future__ import annotations

import hashlib

SALT = "bench-half-v1"
HELD_WHOLE = ("Think",)


def half(case):
    """Pure: "fit" or "held" for a cases.jsonl row."""
    if case["bench"] in HELD_WHOLE:
        return "held"
    prompt = next(m["content"] for m in case["messages"] if m["role"] == "user")
    digest = hashlib.sha256(f"{SALT}:{case['bench']}:{prompt}".encode()).hexdigest()
    return "fit" if int(digest[:12], 16) % 2 == 0 else "held"
