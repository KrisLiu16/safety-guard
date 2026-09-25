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
from make_batch import NO_LEADER, plan  # noqa: E402


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

    def test_request_body(self):
        body = pipeline.request_body("m", seed(), "chat_completions")
        task = json.loads(body["messages"][1]["content"])
        self.assertEqual(task["kind_description"], pipeline.KINDS["leader_news"])
        self.assertEqual(task["length_hint"], pipeline.LENGTHS["short"]["en"])


if __name__ == "__main__":
    unittest.main()
