"""Word screen contract checks (CPU only, no model calls). Placeholder words only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
from pipeline import PROMPT_VERSION, WIRE_APIS, parse, request_body  # noqa: E402
from extract_screen import join  # noqa: E402

BATCH = {"batch_key": "screen-00000", "prompt_version": PROMPT_VERSION,
         "words": [{"word": "词甲"}, {"word": "词乙"}, {"word": "词丙"}]}


def reply(results, **root):
    return json.dumps({"batch_key": "screen-00000", "prompt_version": PROMPT_VERSION, "results": results, **root},
                      ensure_ascii=False)


class ParseTests(unittest.TestCase):
    def test_complete_batch(self):
        results = [{"index": i, "word": w["word"], "verdict": v} for i, (w, v) in enumerate(zip(BATCH["words"], ("no", "insult", "evasion")))]
        errors, verdicts = parse(reply(results), BATCH)
        self.assertEqual(errors, [])
        self.assertEqual([v["verdict"] for v in verdicts], ["no", "insult", "evasion"])

    def test_partial_and_corrupt_items(self):
        results = [{"index": 0, "word": "词甲", "verdict": "no"},
                   {"index": 0, "word": "词甲", "verdict": "insult"},       # duplicate index
                   {"index": 1, "word": "改写了", "verdict": "no"},         # word changed
                   {"index": 2, "word": "词丙", "verdict": "yes"}]          # v1 verdict, not valid in v2
        errors, verdicts = parse(reply(results), BATCH)
        self.assertEqual([v["word"] for v in verdicts], ["词甲"])
        self.assertIn("bad_index", errors)
        self.assertIn("word_changed", errors)
        self.assertIn("verdict", errors)
        self.assertIn("missing:2", errors)

    def test_root_and_json_errors(self):
        self.assertEqual(parse(reply([], batch_key="other"), BATCH), (["root_mismatch"], []))
        self.assertEqual(parse("not json", BATCH), (["no_json"], []))

    def test_request_sends_only_words(self):
        for wire in WIRE_APIS:
            body = request_body("m", {**BATCH, "words": [{"word": "词甲", "known_positive": True}]}, wire)
            self.assertNotIn("known_positive", json.dumps(body, ensure_ascii=False))
        strict = request_body("m", BATCH, "responses")["text"]["format"]["schema"]
        self.assertEqual(strict["properties"]["results"]["minItems"], 3)


class BatchAndJoinTests(unittest.TestCase):
    def test_pilot_includes_known_positives_and_dedups_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seeds = tmp / "seeds.jsonl"
            rows = [{"task_key": f"k{i}", "word": f"词{i % 300}", "source_group": f"g{i % 4}"} for i in range(600)]
            seeds.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            include = tmp / "known.txt"
            include.write_text("词7\n仅在清单里的词\n词7\n", encoding="utf-8")
            out = tmp / "pilot"
            subprocess.run([sys.executable, str(ROOT / "make_batch.py"), "--seeds", str(seeds), "--pilot", "250",
                            "--include", str(include), "--output", str(out)], check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["words"], 252)
            self.assertEqual(manifest["known_positives"], 2)
            self.assertEqual(manifest["tasks"], 2)
            with (out / "words.jsonl").open(encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle]
            self.assertEqual(len({e["word"] for e in entries}), 252)
            self.assertEqual(len(next(e for e in entries if e["word"] == "词7")["task_keys"]), 2)
            for instruction in (out / "tasks").glob("*/instruction.md"):
                for word in json.loads(instruction.read_text())["words"]:
                    self.assertEqual(set(word), {"word"})

    def test_join_recall_and_missing(self):
        index = [{"word": "词甲", "known_positive": True, "source_groups": ["g1"]},
                 {"word": "词乙", "known_positive": True, "source_groups": ["g1"]},
                 {"word": "词丙", "known_positive": False, "source_groups": ["g2"]}]
        rows, summary = join(index, [{"errors": ["missing:1"], "verdicts": [{"word": "词甲", "verdict": "rumor"},
                                                                          {"word": "词丙", "verdict": "no"}]}])
        self.assertEqual(summary["verdicts"], {"rumor": 1, "missing": 1, "no": 1})
        self.assertEqual(summary["known_positive_recall_flagged"], 0.5)
        self.assertEqual(summary["batch_errors"], {"missing": 1})


class CompareTests(unittest.TestCase):
    def test_manual_comparison(self):
        from compare_manual import compare
        manual = {"a": "insult", "b": "rumor", "c": "evasion", "d": "no", "e": "no", "f": "borderline"}
        verdicts = {"a": "insult", "b": "no", "c": "evasion", "d": "no", "e": "unsure", "f": "rumor"}
        out = compare(manual, verdicts)
        self.assertEqual(out["manual_positive_flagged"], round(2 / 3, 4))
        self.assertEqual(out["manual_no_flagged"], 0.5)
        self.assertEqual(out["table"]["rumor"], {"no": 1})


if __name__ == "__main__":
    unittest.main()
