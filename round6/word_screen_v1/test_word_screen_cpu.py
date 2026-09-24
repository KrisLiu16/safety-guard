"""Word screen contract checks (CPU only, no model calls). Placeholder words only."""
import collections
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

    def test_binary_mode_parse_and_prompt(self):
        batch = {**BATCH, "mode": "binary"}
        results = [{"index": i, "word": w["word"], "verdict": v} for i, (w, v) in enumerate(zip(BATCH["words"], ("yes", "no", "unsure")))]
        self.assertEqual(parse(reply(results), batch)[0], [])
        bad = [dict(results[0], verdict="insult")] + results[1:]
        self.assertIn("verdict", parse(reply(bad), batch)[0])        # category verdicts are not valid in binary mode
        from pipeline import BINARY_PROMPT, SYSTEM_PROMPT
        body = request_body("m", batch, "responses")
        self.assertEqual(body["input"][0]["content"], BINARY_PROMPT)
        self.assertEqual(body["text"]["format"]["schema"]["properties"]["results"]["items"]["properties"]["verdict"]["enum"],
                         ["yes", "unsure", "no"])
        self.assertEqual(request_body("m", BATCH, "responses")["input"][0]["content"], SYSTEM_PROMPT)
        self.assertIn("政治局常委", BINARY_PROMPT)

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
            self.assertEqual((manifest["passes"], manifest["binary_passes"]), (2, 1))
            self.assertEqual(manifest["tasks"], 6)           # 252 words -> 2 batches per pass, 3 passes
            with (out / "words.jsonl").open(encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle]
            self.assertEqual(len({e["word"] for e in entries}), 252)
            self.assertEqual(len(next(e for e in entries if e["word"] == "词7")["task_keys"]), 2)
            self.assertTrue(all([k.split("-")[1] for k in e["batch_keys"]] == ["p0", "p1", "b0"] for e in entries))
            modes = {json.loads(t.read_text())["mode"] for t in (out / "tasks").glob("screen-b0-*/instruction.md")}
            self.assertEqual(modes, {"binary"})
            first = {e["word"]: e["batch_keys"] for e in entries}
            neighbours = collections.defaultdict(set)
            for e in entries:
                neighbours[tuple(first[e["word"]])].add(e["word"])
            self.assertGreater(len(neighbours), 2)           # passes regroup words, not the same batches twice
            for instruction in (out / "tasks").glob("*/instruction.md"):
                for word in json.loads(instruction.read_text())["words"]:
                    self.assertEqual(set(word), {"word"})

    def test_join_recall_and_missing(self):
        index = [{"word": "词甲", "known_positive": True, "source_groups": ["g1"]},
                 {"word": "词乙", "known_positive": True, "source_groups": ["g1"]},
                 {"word": "词丙", "known_positive": False, "source_groups": ["g2"]}]
        rows, summary = join(index, [{"batch_key": "screen-p0-00000", "errors": ["missing:1"], "verdicts": [{"word": "词甲", "verdict": "rumor"},
                                                                          {"word": "词丙", "verdict": "no"}]}])
        self.assertEqual(summary["verdicts"], {"rumor": 1, "missing": 1, "no": 1})
        self.assertEqual(summary["known_positive_model_recall_flagged"], 0.5)
        self.assertEqual(summary["batch_errors"], {"missing": 1})


class MergeTests(unittest.TestCase):
    def test_most_severe_verdict_wins_across_passes(self):
        from pipeline import merge
        self.assertEqual(merge(["no", "insult"]), "insult")
        self.assertEqual(merge(["rumor", "insult"]), "insult")
        self.assertEqual(merge(["evasion", "unsure"]), "unsure")
        self.assertEqual(merge(["no", "no"]), "no")
        self.assertEqual(merge([]), "missing")
        self.assertEqual(merge(["no", "no", "yes"]), "unsure")         # binary yes alone: exclude, never relabel
        self.assertEqual(merge(["insult", "no", "yes"]), "insult")

    def test_binary_pass_and_manual_override(self):
        index = [{"word": w, "known_positive": w == "词甲", "source_groups": ["g"]} for w in ("词甲", "词乙", "词丙")]
        values = [{"batch_key": f"screen-{tag}-00000", "errors": [], "verdicts": [
                      {"word": "词甲", "verdict": a}, {"word": "词乙", "verdict": b}, {"word": "词丙", "verdict": "no"}]}
                  for tag, a, b in (("p0", "no", "no"), ("p1", "no", "no"), ("b0", "yes", "no"))]
        rows, summary = join(index, values, manual={"词乙": "insult", "词丙": "borderline"})
        got = {r["word"]: (r["verdict"], r["verdict_source"], r["model_verdict"]) for r in rows}
        self.assertEqual(got["词甲"], ("unsure", "model", "unsure"))   # rescued by the binary pass
        self.assertEqual(got["词乙"], ("insult", "manual", "no"))      # hand label wins, model verdict kept
        self.assertEqual(got["词丙"], ("unsure", "manual", "no"))
        self.assertEqual(summary["flagged_only_by_binary_pass"], 1)
        self.assertEqual(summary["manual_overrides"], 2)
        self.assertEqual(summary["known_positive_model_recall_flagged"], 1.0)

    def test_two_pass_join(self):
        index = [{"word": w, "known_positive": w == "词甲", "source_groups": ["g"]} for w in ("词甲", "词乙", "词丙")]
        values = [{"batch_key": "screen-p0-00000", "errors": [], "verdicts": [
                      {"word": "词甲", "verdict": "no"}, {"word": "词乙", "verdict": "no"}, {"word": "词丙", "verdict": "evasion"}]},
                  {"batch_key": "screen-p1-00000", "errors": [], "verdicts": [
                      {"word": "词甲", "verdict": "insult"}, {"word": "词乙", "verdict": "no"}, {"word": "词丙", "verdict": "evasion"}]}]
        rows, summary = join(index, values)
        self.assertEqual({r["word"]: r["verdict"] for r in rows}, {"词甲": "insult", "词乙": "no", "词丙": "evasion"})
        self.assertEqual(summary["passes_seen"], ["p0", "p1"])
        self.assertEqual(summary["pass_agreement_flagged"], round(2 / 3, 4))
        self.assertEqual(summary["known_positive_model_recall_flagged"], 1.0)


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
