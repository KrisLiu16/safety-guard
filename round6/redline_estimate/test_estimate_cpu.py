"""Red-line positive estimate (CPU, no model calls, no text): tier mapping, joins to raw categories, Run A proxy."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_positives import AEGIS, BEAVER, POLITICAL, estimate, runA_topics, tiers  # noqa: E402


class EstimateTests(unittest.TestCase):
    def test_tiers_take_the_best_per_line(self):
        best, unmapped = tiers(["Violence", " Suicide and Self Harm", "Harassment", "Brand New"], AEGIS)
        self.assertEqual(best, {"R2": "possible", "R13": "likely"})
        self.assertEqual(unmapped, ["brand new"])
        self.assertEqual(tiers(["privacy_violation"], BEAVER), ({}, []))

    def test_joins_and_counts(self):
        prefix = [{"sample_id": "beaver-0", "label": "unsafe", "split": "train", "label_tier": "public_source_reference"},
                  {"sample_id": "beaver-1", "label": "unsafe", "split": "train", "label_tier": "public_source_reference"},
                  {"sample_id": "beaver-2", "label": "safe", "split": "train", "label_tier": "public_source_reference"},
                  {"sample_id": "cand-a", "label": "unsafe", "split": "train", "label_tier": "public_source_reference"},
                  {"sample_id": "cand-b", "label": "unsafe", "split": "dev", "label_tier": "public_source_reference"},
                  {"sample_id": "synth-1", "label": "unsafe", "split": "train", "label_tier": "synthetic_context_weak"}]
        beaver = {0: {"category": {"self_harm": True, "privacy_violation": True, "animal_abuse": False}},
                  1: {"category": {"privacy_violation": True}}}
        aegis = {"n1": {"violated_categories": "Political/Misinformation/Conspiracy, Violence"},
                 "n2": {"violated_categories": ["Sexual (minor)"]}}
        table, unmapped, missing = estimate(prefix, beaver, aegis, {"cand-a": "n1", "cand-b": "n2"})
        train = table["train/all"]
        self.assertEqual((train["records"], train["no_red_line"]), (3, 1))           # beaver-1 is privacy only
        self.assertEqual(train["R13"], {"likely": 1, "possible_or_likely": 1})
        self.assertEqual(train[POLITICAL], {"likely": 0, "possible_or_likely": 1})
        self.assertEqual(train["R10"]["possible_or_likely"], 1)
        self.assertEqual(table["dev/nemotron_zh"]["R11"]["likely"], 1)
        self.assertEqual(missing, {"synthetic_context_weak": 1})
        self.assertEqual(unmapped, {})

    def test_runA_topic_proxy(self):
        rows = [{"label": "unsafe", "split": "train", "source_group": "permissive/涉枪涉爆.txt"},
                {"label": "unsafe", "split": "train", "source_group": "citizenlab/livestream"},
                {"label": "safe", "split": "train", "source_group": "permissive/涉枪涉爆.txt"}]
        self.assertEqual(runA_topics(rows), {"train": {"R12": 1, "mixed_or_unknown": 1, "records": 2}})


if __name__ == "__main__":
    unittest.main()
