"""CPU checks of the v5 bench helpers (decision-level drift, tile choice); no GPU, no model."""
import unittest

import torch

from bench_v5_l20 import decision_drift, first_fire, tiles_per_bucket


class HelperTests(unittest.TestCase):
    def test_first_fire(self):
        self.assertEqual(first_fire([0.1, 0.95, 0.2, 0.96, 0.97], 0.9, 1), 1)
        self.assertEqual(first_fire([0.1, 0.95, 0.2, 0.96, 0.97], 0.9, 2), 4)
        self.assertEqual(first_fire([0.1, 0.2], 0.9, 1), -1)

    def test_decision_drift_counts_flips(self):
        ref = {0: torch.tensor([[0.9, 0.05, 0.05], [0.04, 0.95, 0.01]]),      # cut score 0.1, 0.96
               1: torch.tensor([[0.9, 0.05, 0.05], [0.88, 0.1, 0.02]])}       # never cut at 0.5
        got = {0: torch.tensor([[0.9, 0.05, 0.05], [0.06, 0.93, 0.01]]),      # cut 0.94: fires at 0.9, not at 0.95
               1: torch.tensor([[0.3, 0.6, 0.1], [0.88, 0.1, 0.02]])}         # cut 0.7 at index 0: a new fire at 0.5
        out = decision_drift(got, ref, [0, 1])
        cut = out["cut"]
        self.assertEqual(cut["positions"], 4)
        self.assertAlmostEqual(cut["max"], 0.6, places=5)
        self.assertEqual(cut["over_0.1"], 1)
        self.assertEqual(cut["decisions"]["threshold@0.5"]["fire_differs"], 1)
        self.assertEqual(cut["decisions"]["threshold@0.95"]["fire_differs"], 1)
        self.assertEqual(cut["decisions"]["threshold@0.9"], {"streams": 2, "fire_differs": 0, "both_fire_other_index": 0,
                                                             "reference_fires": 1})

    def test_tiles_per_bucket_skips_mismatches(self):
        rows = [{"sessions": 8, "bv": 8, "warps": 1, "ms_per_call": 0.3, "max_abs_diff_vs_8_1": 0.0},
                {"sessions": 8, "bv": 32, "warps": 2, "ms_per_call": 0.1, "max_abs_diff_vs_8_1": 0.5},
                {"sessions": 8, "bv": 16, "warps": 1, "ms_per_call": 0.2, "max_abs_diff_vs_8_1": 1e-5}]
        self.assertEqual(tiles_per_bucket(rows), {8: (16, 1)})


if __name__ == "__main__":
    unittest.main()
