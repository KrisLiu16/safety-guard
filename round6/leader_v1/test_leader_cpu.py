"""CPU tests for the leader-sensitivity generator (T031)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402
import make_batch  # noqa: E402
from make_batch import NO_LEADER, plan, plan_normal2  # noqa: E402


def seed(kind="leader_news", language="en", length="short"):
    return {"task_key": "leader-v1-000001", "kind": kind, "leader": "Li Qiang", "language": language,
            "shape": "direct", "length": length, "family": "leader-v1-000001", "split": "train"}


def payload(thinking, context=None, **extra):
    return {"task_key": "leader-v1-000001", "prompt_version": pipeline.PROMPT_VERSION, "status": "ready",
            "context": context if context is not None else [{"role": "user", "content": "What did he say at the forum?"}],
            "thinking": thinking, "onset_quote": "", "notes": "", **extra}


class LeaderTest(unittest.TestCase):
    def test_plan_is_deterministic_and_mixed(self):
        a, b = plan(400), plan(400)
        self.assertEqual(a, b)
        kinds = {s["kind"] for s in a}
        self.assertEqual(kinds, set(pipeline.CUT_KINDS) | set(pipeline.PASS_KINDS))
        self.assertTrue(all((s["leader"] is None) == (s["kind"] in NO_LEADER) for s in a))
        self.assertEqual({s["split"] for s in a}, {"train", "dev", "calibration"})
        self.assertEqual(plan(10, offset=5)[0], plan(20)[5])

    def test_valid_row(self):
        text = " ".join(["The user asks about the premier's speech at the forum."] * 12)
        errors, rows = pipeline.validate(payload(text), seed())
        self.assertEqual(errors, [])
        row = rows[0]
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": text})
        self.assertEqual(row["label"], "safe")
        self.assertEqual(row["response_chars"], len(text))

    def test_cut_kind_onset_and_floor(self):
        text = " ".join(["The user asks about the cartoon."] * 8 + ["Media reported a mocking comparison."]
                        + ["I need to decide how to answer."] * 8)
        quote = "Media reported a mocking comparison."
        errors, rows = pipeline.validate(payload(text, onset_quote=quote), seed(kind="satire_restated"))
        self.assertEqual(errors, [])
        row = rows[0]
        self.assertEqual((row["label"], row["label_floor"]), ("unsafe", "controversial"))
        self.assertEqual(text[row["onset_char"]:row["onset_end_char"]], quote)
        self.assertEqual(pipeline.validate(payload(text), seed(kind="satire_restated"))[0], ["onset_quote"])
        self.assertEqual(pipeline.validate(payload(text, onset_quote="not in the text"), seed(kind="satire_restated"))[0],
                         ["onset_quote"])
        _, rows = pipeline.validate(payload(text, onset_quote=quote), seed())       # pass kind: quote ignored
        self.assertEqual((rows[0]["onset_char"], rows[0]["label_floor"]), (None, None))

    def test_rejections(self):
        text = " ".join(["word"] * 150)
        self.assertEqual(pipeline.validate({**payload(text), "status": "skip"}, seed()), (["skip"], []))
        self.assertEqual(pipeline.validate(payload(text, context=[]), seed())[0], ["context"])
        bad_roles = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        self.assertEqual(pipeline.validate(payload(text, context=bad_roles), seed())[0], ["context"])
        self.assertIn("thinking:length", pipeline.validate(payload("too short"), seed())[0])
        chinese = "用户在问总理的讲话。" * 40
        self.assertIn("thinking:language", pipeline.validate(payload(chinese), seed())[0])
        self.assertEqual(pipeline.validate(payload(chinese), seed(language="zh"))[0], [])

    @unittest.skipUnless(make_batch.TOPICS_V13, "local topic file (data/topics.json) absent")
    def test_plan_normal2(self):
        seeds = plan_normal2(600, offset=20000)
        self.assertEqual(seeds, plan_normal2(600, offset=20000))
        self.assertEqual({s["kind"] for s in seeds}, set(make_batch.TOPICS_V13))
        self.assertTrue(all(s["language"] == ("en" if s["topic"].isascii() else "zh") for s in seeds))
        doubled = make_batch.TOPICS_V13_DOUBLE[0]
        single = next(t for ts in make_batch.TOPICS_V13.values() for t in ts
                      if t not in make_batch.TOPICS_V13_DOUBLE and sum(t in v for v in make_batch.TOPICS_V13.values()) == 1)
        counts = {t: sum(s["topic"] == t for s in seeds) for t in (doubled, single)}
        self.assertGreater(counts[doubled], counts[single])

    def test_v12_output_still_validates(self):
        text = "用户想知道故宫博物院的开放时间。" * 20
        seed_v = {**seed(kind="place_normal", language="zh"), "topic": "故宫博物院"}
        body = {**payload(text, context=[{"role": "user", "content": "故宫博物院几点开门？"}]),
                "prompt_version": "guard-leader-thinking-v1.2"}
        errors, rows = pipeline.validate(body, seed_v)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["prompt_version"], "guard-leader-thinking-v1.2")
        self.assertEqual(pipeline.validate({**body, "prompt_version": "v0"}, seed_v)[0], ["root_mismatch:prompt_version"])

    @unittest.skipUnless(make_batch.TOPICS and make_batch.TOPICS_V13, "local topic file absent")
    def test_plan_normal4_answers_and_look_alike_kinds(self):
        seeds = make_batch.plan_normal4(1000, offset=40000)
        self.assertEqual(seeds, make_batch.plan_normal4(1000, offset=40000))
        self.assertEqual({s["kind"] for s in seeds}, set(make_batch.TOPICS) | set(make_batch.TOPICS_V13))
        share = sum(s["format"] == "answer" for s in seeds) / len(seeds)
        self.assertTrue(0.5 < share < 0.7)
        self.assertEqual(len({s["task_key"] for s in seeds}), 1000)
        self.assertTrue(all(s["language"] == ("en" if s["topic"].isascii() else "zh") for s in seeds))

    def test_answer_format_row_and_request(self):
        text = "故宫博物院一般上午八点半开门，周一闭馆。" * 20
        seed_v = {**seed(kind="place_normal", language="zh"), "topic": "故宫博物院", "format": "answer"}
        errors, rows = pipeline.validate(payload(text, context=[{"role": "user", "content": "故宫博物院几点开门？"}]),
                                         seed_v)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["response_format"], "answer")
        self.assertEqual(pipeline.validate(payload(text, context=[{"role": "user", "content": "故宫博物院几点开门？"}]),
                                           {**seed_v, "format": "thinking"})[1][0]["response_format"], "reasoning")
        task = json.loads(pipeline.request_body("m", seed_v, "chat_completions")["messages"][1]["content"])
        self.assertEqual((task["format"], task["format_hint"]), ("answer", pipeline.FORMATS["answer"]))

    def test_request_body(self):
        body = pipeline.request_body("m", seed(), "chat_completions")
        task = json.loads(body["messages"][1]["content"])
        self.assertEqual(task["kind_description"], pipeline.KINDS["leader_news"])
        self.assertEqual(task["length_hint"], pipeline.LENGTHS["short"]["en"])


if __name__ == "__main__":
    unittest.main()
