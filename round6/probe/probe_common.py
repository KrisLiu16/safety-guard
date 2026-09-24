"""Pure helpers for T004 (linear probe) and T005 (decision-rule refit on Run A). numpy only, no torch.

Position classes, per assistant target token (token end offset relative to the assistant content):
  S  safe_response : every position of a safe response (v14 contract: any cut of a safe response is safe);
  P  pre_onset     : unsafe response, token ends at or before onset_char;
  O  onset_span    : unsafe response, token ends inside the quoted onset (ambiguous, never trained on);
  U  post_onset    : unsafe response, token ends at or after onset_end_char.
The v14 onsets are known to be imprecise (T002: compliance onsets marked late, transition sentences
marked as onset), so P/U near the onset are noisy; reports split them out instead of trusting them.
"""
from __future__ import annotations

import math

import numpy as np

CLASSES = {"S": "safe_response", "P": "pre_onset", "O": "onset_span", "U": "post_onset"}
RISK_ORDER = ("safe", "unsafe", "controversial")


def serialize(messages):
    """Identical to round5 prepare_prefix_data.serialize / train_prefix.serialize."""
    return "\n\n".join(m["role"].upper() + ":\n" + m["content"] for m in messages)


def position_classes(offsets, content_start, label, onset_char=None, onset_end_char=None):
    """Return (target_positions, class letters) for the last message, as round5 defines targets."""
    positions, classes = [], []
    for index, (_, end) in enumerate(offsets):
        if end <= content_start:
            continue
        positions.append(index)
        if label == "safe":
            classes.append("S")
            continue
        relative = end - content_start
        if relative <= onset_char:
            classes.append("P")
        elif relative >= onset_end_char:
            classes.append("U")
        else:
            classes.append("O")
    return positions, "".join(classes)


def feature_positions(positions, classes, limit=24):
    """Indices into `positions` whose features are kept: spread evenly, plus the onset boundary."""
    n = len(positions)
    if n == 0:
        return []
    chosen = {0, min(1, n - 1), n - 1}
    last_pre = max((i for i, c in enumerate(classes) if c == "P"), default=None)
    first_post = min((i for i, c in enumerate(classes) if c == "U"), default=None)
    chosen.update(i for i in (last_pre, first_post) if i is not None)
    slots = max(0, limit - len(chosen))
    if slots and n > 1:
        chosen.update(round(k * (n - 1) / (slots + 1)) for k in range(1, slots + 1))
    return sorted(chosen)


def auc(positive, negative):
    """Mann-Whitney AUC with tie correction; None if either side is empty."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return None
    scores = np.concatenate([positive, negative])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        stop = start
        while stop + 1 < scores.size and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    rank_sum = ranks[:positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size))


def recall_at_fpr(positive, negative, fpr):
    """Recall when the threshold is set so that at most `fpr` of negatives score above it."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if positive.size == 0 or negative.size == 0:
        return None
    threshold = np.quantile(negative, 1 - fpr, method="higher")
    return float((positive > threshold).mean())


def fit_logistic(x, y, *, l2=1e-3, steps=400, lr=0.05, seed=0):
    """Class-balanced L2 logistic regression by full-batch Adam. Returns (weights, bias, mean, std)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    z = (x - mean) / std
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weight = np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg)).astype(np.float32)
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal(z.shape[1]) * 1e-3).astype(np.float32)
    b = np.float32(0.0)
    m_w, v_w, m_b, v_b = np.zeros_like(w), np.zeros_like(w), 0.0, 0.0
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        logits = z @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = (p - y) * weight / len(y)
        grad_w = z.T @ error + l2 * w
        grad_b = float(error.sum())
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
        w -= lr * (m_w / (1 - beta1 ** step)) / (np.sqrt(v_w / (1 - beta2 ** step)) + 1e-8)
        b -= lr * (m_b / (1 - beta1 ** step)) / (math.sqrt(v_b / (1 - beta2 ** step)) + 1e-8)
    return w, float(b), mean, std


def logistic_scores(x, model):
    w, b, mean, std = model
    return ((np.asarray(x, dtype=np.float32) - mean) / std) @ w + b
