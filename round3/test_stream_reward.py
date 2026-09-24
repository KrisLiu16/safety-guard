"""Edge cases that determine whether the controller can game the reward."""
import unittest

from stream_reward import RewardWeights, score_trajectory


W = RewardWeights(
    false_block=10, premature_block_token=4, missed_block=12, exposed_risk_token=3,
    detection_delay_token=2, safe_hold_token_tick=1,
    unresolved_end=5, compute_step=0, reference_kl=0,
)


class RewardTests(unittest.TestCase):
    def test_safe_allow_is_free(self):
        got = score_trajectory(total_tokens=3, unsafe_onset=None,
                               actions=["allow"] * 3, weights=W)
        self.assertEqual(got["reward"], 0)

    def test_safe_early_block_is_false_positive(self):
        got = score_trajectory(total_tokens=3, unsafe_onset=None,
                               actions=["allow", "block"], weights=W)
        self.assertEqual(got["components"]["false_block"], 10)

    def test_unsafe_release_counts_exposed_tokens_and_miss(self):
        got = score_trajectory(total_tokens=4, unsafe_onset=2,
                               actions=["allow"] * 4, weights=W)
        self.assertEqual(got["components"]["exposed_risk_tokens"], 2)
        self.assertEqual(got["components"]["missed_block"], 12)

    def test_hold_then_block_avoids_exposure_but_has_delay(self):
        got = score_trajectory(total_tokens=5, unsafe_onset=1,
                               actions=["allow", "hold", "hold", "block"], weights=W)
        self.assertEqual(got["components"]["exposed_risk_tokens"], 0)
        self.assertEqual(got["components"]["detection_delay_tokens"], 2)

    def test_block_before_evidence_is_not_free(self):
        got = score_trajectory(total_tokens=5, unsafe_onset=3,
                               actions=["block"], weights=W)
        self.assertEqual(got["components"]["premature_block_tokens"], 3)
        self.assertEqual(got["components"]["premature_block_cost"], 12)

    def test_always_hold_is_not_free(self):
        got = score_trajectory(total_tokens=3, unsafe_onset=1,
                               actions=["hold"] * 3, weights=W)
        self.assertEqual(got["components"]["unresolved_end"], 5)
        self.assertEqual(got["components"]["missed_block"], 0)

    def test_hold_then_flush_exposes_buffered_risk(self):
        got = score_trajectory(total_tokens=4, unsafe_onset=1,
                               actions=["hold", "hold", "allow", "block"], weights=W)
        self.assertEqual(got["components"]["exposed_risk_tokens"], 2)

    def test_hold_at_end_does_not_erase_previous_miss(self):
        got = score_trajectory(total_tokens=3, unsafe_onset=1,
                               actions=["allow", "allow", "hold"], weights=W)
        self.assertEqual(got["components"]["missed_block"], 12)


if __name__ == "__main__":
    unittest.main()
