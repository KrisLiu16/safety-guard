"""Political screen contract checks (CPU only, no model calls). Placeholder words only.
Run on its own: word_screen_v1 and redline_v1 also have a flow/pipeline.py."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
sys.modules.pop("pipeline", None)
from pipeline import (LEADER_SYSTEM_PROMPT, LEADER_VERDICTS, PROMPT_VERSION, SYSTEM_PROMPT, VERDICTS,  # noqa: E402
                      merge, parse, request_body, schema)
from extract import join, matched_as, review_sheet, to_confirm  # noqa: E402

BATCH = {"batch_key": "political-p1-00000", "prompt_version": PROMPT_VERSION,
         "words": [{"word": "词甲"}, {"word": "词乙"}, {"word": "词丙"}]}


def reply(results, **root):
    return json.dumps({"batch_key": BATCH["batch_key"], "prompt_version": PROMPT_VERSION, "results": results, **root},
                      ensure_ascii=False)


def item(index, verdict, ambiguous=False, word=None):
    return {"index": index, "word": word or BATCH["words"][index]["word"], "verdict": verdict, "ambiguous": ambiguous}


class ParseTests(unittest.TestCase):
    def test_complete_batch(self):
        errors, verdicts = parse(reply([item(0, "event"), item(1, "no", True), item(2, "leader")]), BATCH)
        self.assertEqual(errors, [])
        self.assertEqual([(v["verdict"], v["ambiguous"]) for v in verdicts], [("event", False), ("no", True), ("leader", False)])

    def test_partial_and_corrupt_items(self):
        results = [item(0, "event"), item(0, "org"), item(1, "no", word="改写了"), {**item(2, "yes")},
                   {"index": 2, "word": "词丙", "verdict": "no"}]                  # no ambiguous field
        errors, verdicts = parse(reply(results), BATCH)
        self.assertEqual([v["word"] for v in verdicts], ["词甲"])
        self.assertEqual(sorted(errors), ["bad_index", "bad_index", "missing:2", "verdict", "word_changed"])

    def test_root_and_json_errors(self):
        self.assertEqual(parse("没有 JSON", BATCH), (["no_json"], []))
        self.assertEqual(parse("{坏的}", BATCH), (["bad_json"], []))
        self.assertEqual(parse(reply([], prompt_version="old"), BATCH)[0], ["root_mismatch"])

    def test_request_sends_only_words(self):
        for wire in ("responses", "chat_completions"):
            body = json.dumps(request_body("m", BATCH, wire), ensure_ascii=False)
            self.assertIn("词乙", body)
            self.assertNotIn("sample", body)
        with self.assertRaises(ValueError):
            request_body("m", BATCH, "other")
        self.assertEqual(schema(3)["properties"]["results"]["minItems"], 3)
        self.assertEqual(set(schema(1)["properties"]["results"]["items"]["properties"]["verdict"]["enum"]), set(VERDICTS))
        for verdict in VERDICTS:
            self.assertIn(verdict, SYSTEM_PROMPT)

    def test_leader_mode(self):
        batch = {**BATCH, "mode": "leader"}
        errors, verdicts = parse(reply([item(0, "negative"), item(1, "formal", True), item(2, "event")]), batch)
        self.assertEqual((errors, [v["verdict"] for v in verdicts]), (["verdict", "missing:1"], ["negative", "formal"]))
        self.assertEqual(parse(reply([item(0, "negative")]), BATCH)[0][0], "verdict")        # not a screen verdict
        body = json.dumps(request_body("m", batch, "chat_completions"), ensure_ascii=False)
        self.assertIn("not_leader", body)
        self.assertEqual(json.loads(body)["messages"][0]["content"], LEADER_SYSTEM_PROMPT)
        self.assertEqual(set(schema(1, "leader")["properties"]["results"]["items"]["properties"]["verdict"]["enum"]),
                         set(LEADER_VERDICTS))
        self.assertEqual(merge([("formal", False), ("variant", True)], "leader"), ("variant", True))

    def test_merge_most_alerting(self):
        self.assertEqual(merge([("no", False), ("event", False)]), ("event", False))
        self.assertEqual(merge([("other_political", True), ("unsure", False)]), ("unsure", True))
        self.assertEqual(merge([("leader", False), ("figure", False)]), ("figure", False))
        self.assertEqual(merge([]), ("missing", False))


class BatchTests(unittest.TestCase):
    def test_batches_only_and_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            words = [f"词{i:04d}" for i in range(450)]
            (tmp / "words.jsonl").write_text("".join(json.dumps({"word": w, "source_group": "g"}, ensure_ascii=False) + "\n"
                                                     for w in words + words[:5]), encoding="utf-8")
            run = lambda *extra: subprocess.run([sys.executable, str(ROOT / "make_batch.py"), "--words", str(tmp / "words.jsonl"),
                                                 *extra], capture_output=True, text=True)
            done = run("--output", str(tmp / "p1"))
            self.assertEqual(done.returncode, 0, done.stderr)
            manifest = json.loads((tmp / "p1/manifest.json").read_text())
            self.assertEqual((manifest["words"], manifest["tasks"], manifest["tag"]), (450, 3, "p1"))
            with tarfile.open(tmp / "p1/screen.tar.gz") as archive:
                names = sorted({m.name.split("/")[0] for m in archive.getmembers()})
            self.assertEqual(names, [f"political-p1-{i:05d}" for i in range(3)])
            batch = json.loads((tmp / "p1/tasks/political-p1-00000/instruction.md").read_text(encoding="utf-8"))
            self.assertEqual((batch["prompt_version"], len(batch["words"])), (PROMPT_VERSION, 200))
            (tmp / "only.txt").write_text("词0001\n词0002\n不在词表\n", encoding="utf-8")
            done = run("--only", str(tmp / "only.txt"), "--tag", "c1", "--output", str(tmp / "c1"))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads((tmp / "c1/manifest.json").read_text())["words"], 2)
            done = run("--only", str(tmp / "only.txt"), "--tag", "l1", "--mode", "leader", "--output", str(tmp / "l1"))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads((tmp / "l1/manifest.json").read_text())["mode"], "leader")
            batch = json.loads(next((tmp / "l1/tasks").glob("*/instruction.md")).read_text(encoding="utf-8"))
            self.assertEqual(batch["mode"], "leader")
            done = run("--tag", "c-1", "--output", str(tmp / "bad"))
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("tag", done.stderr)


class ExtractTests(unittest.TestCase):
    def test_join_confirm_and_matching(self):
        def value(tag, verdicts, errors=()):
            return {"batch_key": f"political-{tag}-00000", "errors": list(errors),
                    "verdicts": [{"word": w, "verdict": v, "ambiguous": a} for w, v, a in verdicts]}
        values = [value("p1", [("词甲", "event", False), ("词乙", "no", False), ("词丙", "leader", False),
                               ("词丁", "unsure", False), ("戊", "event", False)], ["missing:1"]),
                  value("c1", [("词甲", "event", True), ("词丁", "org", False)]),
                  {**value("l1", [("词丙", "negative", False), ("词己", "formal", False)]), "mode": "leader"},
                  value("p1", [("词己", "leader", False)])]
        rows, errors = join(values, {"词乙": "evasion", "词丙": "insult", "词己": "evasion"})
        by_word = {r["word"]: r for r in rows}
        self.assertEqual(errors, {"missing": 1})
        self.assertEqual((by_word["词甲"]["verdict"], by_word["词甲"]["ambiguous"]), ("event", True))  # any pass
        self.assertEqual(by_word["词丁"]["verdict"], "org")
        self.assertEqual(by_word["词丁"]["passes"], {"c1": {"verdict": "org", "ambiguous": False},
                                                    "p1": {"verdict": "unsure", "ambiguous": False}})
        self.assertEqual(by_word["词甲"]["leader_screen"], "missing")
        self.assertEqual({w for w, r in by_word.items() if to_confirm(r)}, {"词甲", "词乙", "词丙", "词丁", "戊", "词己"})
        self.assertEqual((by_word["词丙"]["leader_form"], by_word["词丙"]["verdict"]), ("negative", "leader"))
        self.assertNotIn("leader_form", by_word["词甲"])
        self.assertEqual({w: matched_as(r) for w, r in by_word.items()},       # a formal spelling is never matched
                         {"词甲": "event:ambiguous", "词乙": None, "词丙": "leader_negative", "词丁": "org", "戊": None,
                          "词己": None})
        sheet = review_sheet(rows, 5)
        self.assertIn("## org（1 个）\n词丁\t", sheet)
        self.assertNotIn("词乙", sheet)


if __name__ == "__main__":
    unittest.main()
