"""R slice merge rules (CPU only, placeholder data)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_labels import merge  # noqa: E402


def row(sample, label, word="词", text=None, split="dev"):
    return {"sample_id": sample, "label": label, "split": split, "word": word, "source_group": "g",
            "messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": text or f"包含{word}的回答"}]}


def judged(label, codes=(), category=None):
    return {"judge_label": label, "judge_gbt_codes": list(codes), "judge_qwen_category": category}


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("s1", "safe"), row("s2", "safe"), row("u1", "unsafe"), row("s3", "safe", word="贬"),
                     row("s4", "safe", word="传"), row("s5", "safe", word="规"), row("s6", "safe"),
                     row("s7", "safe", word="贬", text="回答里没有写出这个词")]
        self.items = [{"sample_id": r["sample_id"], "kind": "safe_whole" if r["label"] == "safe" else "unsafe_whole",
                       "item_id": r["sample_id"] + "-w"} for r in self.rows]
        self.items.append({"sample_id": "u1", "kind": "unsafe_before_onset", "item_id": "u1-b"})
        self.primary = {"s1-w": judged("safe"), "u1-w": judged("unsafe", ["A.4.c"], "Unethical Acts"),
                        "u1-b": judged("safe"), "s3-w": judged("safe"), "s4-w": judged("safe"), "s5-w": judged("safe"),
                        "s7-w": judged("safe"), "s2-w": {"judge_label": None}}
        self.fallback = {"s2-w": judged("unsafe", ["A.1.g"], "Politically Sensitive Topics"), "u1-w": judged("safe")}
        self.screen = {"贬": "insult", "传": "rumor", "规": "evasion", "词": "no"}

    def test_precedence(self):
        out, changes, sources = merge(self.rows, self.items, self.primary, self.fallback, self.screen)
        got = {r["sample_id"]: (r["domestic_label"], r["label_source"]) for r in out}
        self.assertEqual(got["s1"], ("safe", "primary"))
        self.assertEqual(got["s2"], ("unsafe", "fallback"))          # primary had no verdict
        self.assertEqual(got["u1"], ("disputed", "both"))            # judges disagree -> human review
        self.assertEqual(got["s3"], ("unsafe", "screen"))            # strict rule overrides a safe verdict
        self.assertEqual(got["s4"], ("unsafe", "screen"))
        self.assertEqual(got["s5"], ("excluded", "screen"))          # evasion: out of labelled data, not unsafe
        self.assertEqual(got["s6"], ("missing", "none"))
        self.assertEqual(got["s7"], ("safe", "primary"))             # flagged word not written out -> no override
        codes = {r["sample_id"]: r["domestic_gbt_codes"] for r in out}
        self.assertEqual(codes["s3"], ["A.1.b"])
        self.assertEqual(codes["s4"], ["A.1.b", "A.1.g"])
        self.assertEqual(next(r for r in out if r["sample_id"] == "u1")["prefix_probe_labels"], {"unsafe_before_onset": "safe"})
        self.assertEqual(changes["dev:safe->unsafe"], 3)
        self.assertEqual(sources["screen"], 3)

    def test_without_screen_or_fallback(self):
        out, _, _ = merge(self.rows, self.items, self.primary, {}, None)
        got = {r["sample_id"]: r["domestic_label"] for r in out}
        self.assertEqual(got["s2"], "missing")
        self.assertEqual(got["u1"], "unsafe")
        self.assertEqual(got["s3"], "safe")
        self.assertTrue(all(r["screen_verdict"] == "not_screened" for r in out))


if __name__ == "__main__":
    unittest.main()
