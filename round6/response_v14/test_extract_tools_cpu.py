"""CPU checks for the Run A extraction tools: training filter and frozen attempt listing (no Aster calls)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import extract_archives  # noqa: E402
from make_trainable import select_trainable  # noqa: E402


def word_rows(key, delayed_zero=False):
    rows = []
    for index, label in enumerate(("safe", "unsafe", "safe", "unsafe")):
        row = {"task_key": key, "index": index, "label": label, "quality_flags": [],
               "onset_style_requested": ("immediate", "delayed")[index // 2] if label == "unsafe" else None,
               "onset_char": 5 if label == "unsafe" else None}
        if delayed_zero and index == 3:
            row["onset_char"] = 0
        rows.append(row)
    return rows


class TrainableFilterTests(unittest.TestCase):
    def setUp(self):
        self.seeds = [{"task_key": k, "split": s, "family": "f-" + k}
                      for k, s in (("a", "train"), ("b", "dev"), ("c", "train"), ("d", "calibration"))]

    def test_keeps_complete_words_with_split(self):
        examples = word_rows("a") + word_rows("b")
        words = [{"task_key": "a", "complete": True}, {"task_key": "b", "complete": True}]
        kept, stats = select_trainable(examples, words, self.seeds)
        self.assertEqual(stats["words_kept"], 2)
        self.assertEqual(stats["words_by_split"], {"train": 1, "dev": 1})
        self.assertTrue(all(r["split"] in ("train", "dev") and r["family"].startswith("f-") for r in kept))

    def test_drops_incomplete_and_delayed_zero_words(self):
        examples = word_rows("a") + word_rows("c", delayed_zero=True) + word_rows("d")[:3]
        words = [{"task_key": "a", "complete": True}, {"task_key": "c", "complete": True},
                 {"task_key": "d", "complete": False}]
        kept, stats = select_trainable(examples, words, self.seeds)
        self.assertEqual({r["task_key"] for r in kept}, {"a"})
        self.assertEqual(stats["words_dropped_delayed_onset_zero"], 1)
        self.assertEqual(stats["responses_delayed_onset_zero"], 1)
        self.assertEqual(stats["words_incomplete_at_extract"], 1)

    def test_unknown_seed_is_an_error(self):
        with self.assertRaises(KeyError):
            select_trainable(word_rows("zzz"), [{"task_key": "zzz", "complete": True}], self.seeds)


class FrozenListingTests(unittest.TestCase):
    def run_main(self, frozen, run_id="run_A"):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "attempts.json"
            listing.write_text(json.dumps(frozen))
            argv = ["extract_archives.py", "aster-dev-x", "--out", str(Path(tmp) / "out"),
                    "--expected-terms", "1", "--attempts-json", str(listing)]
            calls = []

            def fake_cli(args):
                calls.append(args)
                if args[:2] == ["runs", "get"]:
                    return {"id": run_id, "run_no": "aster-dev-x", "status": "completed", "effective_config": {}}
                raise AssertionError("unexpected Aster call " + " ".join(args))

            with mock.patch.object(sys, "argv", argv), mock.patch.object(extract_archives, "cli", fake_cli), \
                    mock.patch("builtins.print"):
                extract_archives.main()
            return calls

    def test_uses_frozen_listing_instead_of_run_wide_call(self):
        calls = self.run_main({"run_id": "run_A", "items": [], "truncated": False})
        self.assertEqual(calls, [["runs", "get", "aster-dev-x"]])  # no run-wide attempts call

    def test_rejects_listing_from_another_run(self):
        with self.assertRaises(RuntimeError):
            self.run_main({"run_id": "run_B", "items": [], "truncated": False})


if __name__ == "__main__":
    unittest.main()
