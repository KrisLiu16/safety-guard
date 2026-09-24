"""Deterministic offline reward simulator for a non-generative stream controller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


Action = Literal["allow", "hold", "block"]


@dataclass(frozen=True)
class RewardWeights:
    false_block: float
    premature_block_token: float
    missed_block: float
    exposed_risk_token: float
    detection_delay_token: float
    safe_hold_token_tick: float
    unresolved_end: float
    compute_step: float
    reference_kl: float

    def __post_init__(self) -> None:
        import math
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError("All reward costs must be finite and nonnegative")


def score_trajectory(*, total_tokens: int, unsafe_onset: int | None,
                     actions: Sequence[Action], weights: RewardWeights,
                     reference_kl_sum: float = 0.0) -> dict:
    """Score observed token actions; `allow` releases the entire buffered prefix.

    `unsafe_onset` is the first token whose content is unsafe, 0-based. None
    means a fully safe trace. This ground truth must come from an independent
    labeled trajectory, never from the student's own predictions.
    """
    import math
    if total_tokens < 1 or not 1 <= len(actions) <= total_tokens:
        raise ValueError("Invalid trajectory length")
    if unsafe_onset is not None and not 0 <= unsafe_onset < total_tokens:
        raise ValueError("Invalid unsafe onset")
    if not math.isfinite(reference_kl_sum) or reference_kl_sum < 0:
        raise ValueError("Invalid reference KL")
    if any(action not in ("allow", "hold", "block") for action in actions):
        raise ValueError("Unknown action")
    block_steps = [i for i, action in enumerate(actions) if action == "block"]
    if len(block_steps) > 1 or (block_steps and block_steps[0] != len(actions) - 1):
        raise ValueError("A block must be the terminal action")
    if not block_steps and len(actions) != total_tokens:
        raise ValueError("Nonblocked trajectories must visit all tokens")

    released = exposed = safe_hold_ticks = 0
    for step, action in enumerate(actions):
        if action == "allow":
            next_released = step + 1
            if unsafe_onset is not None:
                exposed += max(0, next_released - max(released, unsafe_onset))
            released = next_released
        elif action == "hold" and unsafe_onset is None:
            safe_hold_ticks += step + 1 - released
    blocked_at = block_steps[0] if block_steps else None
    unresolved = blocked_at is None and actions[-1] == "hold"
    missed_block = unsafe_onset is not None and blocked_at is None and exposed > 0
    components = {
        "false_block": weights.false_block * int(unsafe_onset is None and blocked_at is not None),
        "premature_block_tokens": max(0, unsafe_onset - blocked_at)
            if unsafe_onset is not None and blocked_at is not None else 0,
        "missed_block": weights.missed_block * int(missed_block),
        "exposed_risk_tokens": exposed,
        "exposed_risk_cost": weights.exposed_risk_token * exposed,
        "detection_delay_tokens": max(0, blocked_at - unsafe_onset)
            if blocked_at is not None and unsafe_onset is not None else 0,
        "safe_hold_token_ticks": safe_hold_ticks,
        "safe_hold_cost": weights.safe_hold_token_tick * safe_hold_ticks,
        "unresolved_end": weights.unresolved_end * int(unresolved),
        "compute_cost": weights.compute_step * len(actions),
        "reference_kl_cost": weights.reference_kl * reference_kl_sum,
    }
    components["detection_delay_cost"] = (
        weights.detection_delay_token * components["detection_delay_tokens"]
    )
    components["premature_block_cost"] = (
        weights.premature_block_token * components["premature_block_tokens"]
    )
    charged = (
        "false_block", "premature_block_cost", "missed_block", "exposed_risk_cost",
        "detection_delay_cost", "safe_hold_cost", "unresolved_end",
        "compute_cost", "reference_kl_cost",
    )
    return {
        "reward": -sum(components[key] for key in charged),
        "blocked_at": blocked_at,
        "released_tokens": released,
        "components": components,
    }
