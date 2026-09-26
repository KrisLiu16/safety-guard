"""thinking-en contract checks (CPU only, no model calls). Placeholder text only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
from pipeline import PROMPT_VERSION, SHAPES, WIRE_APIS, request_body, schema, validate  # noqa: E402
import make_batch  # noqa: E402
from extract import summarize  # noqa: E402

PROMPT = "Placeholder question for a unit test."


def seed(kind="risk", shape="stepwise"):
    return {"task_key": "thinking-en-v1-00000", "word": "thinking-en-v1-00000", "prompt": PROMPT,
            "group": "other_harm" if kind == "risk" else "safe", "kind": kind, "shape": shape, "length": "medium",
            "family": "aegis2en:abc", "split": "train", "source_group": "thinking_en_v1/other_harm"}


def long_text(extra=""):
    return f"The user asks something. {extra}" + " This is neutral placeholder text for a unit test only." * 45


def payload(s, **overrides):
    out = {"task_key": s["task_key"], "prompt_version": PROMPT_VERSION, "status": "ready", "notes": "",
           "reasoning": long_text()}
    out.update(overrides)
    return out


def response(s, text=None):
    body = json.dumps(payload(s) if text is None else payload(s, reasoning=text))
    return {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}


class ValidateTests(unittest.TestCase):
    def test_one_assistant_row_with_prompt_unchanged(self):
        for kind, style in (("risk", "risk_reasoning_long"), ("helpful", "helpful_reasoning_long")):
            s = seed(kind)
            errors, rows = validate(payload(s), s)
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row["response_style"], row["language"], row["response_format"]), (style, "en", "reasoning"))
            self.assertEqual(row["messages"][0], {"role": "user", "content": PROMPT})
            self.assertEqual((row["family"], row["split"]), ("aegis2en:abc", "train"))
            self.assertEqual(row["annotation_origin"], "synthetic_unverified_pending_judge")

    def test_rejections(self):
        s = seed()
        self.assertEqual(validate(payload(s, reasoning="too short"), s)[0], ["reasoning:length"])
        self.assertEqual(validate(payload(s, reasoning=long_text() + "用户" * 400), s)[0], ["reasoning:not_english"])
        self.assertEqual(validate(payload(s, task_key="other"), s)[0], ["root_mismatch:task_key"])
        self.assertEqual(validate(payload(s, status="skip", reasoning=""), s), (["skip"], []))
        self.assertEqual(validate(payload(s, reasoning=""), s)[0], ["reasoning:empty"])

    def test_literal_backslash_n_is_repaired_and_flagged(self):
        s = seed()
        escaped = long_text().replace(". ", ".\\n\\n", 3)
        errors, rows = validate(payload(s, reasoning=escaped), s)
        self.assertEqual(errors, [])
        self.assertIn("\n\n", rows[0]["messages"][1]["content"])
        self.assertIn("escaped_newline_repaired", rows[0]["quality_flags"])

    def test_request_bodies(self):
        s = seed()
        for wire in WIRE_APIS:
            body = request_body("m", s, wire)
            text = json.dumps(body, ensure_ascii=False)
            self.assertIn(PROMPT, text)
            self.assertIn(SHAPES["stepwise"], text)
        self.assertEqual(set(schema()["required"]), {"task_key", "prompt_version", "status", "reasoning", "notes"})


class BatchTests(unittest.TestCase):
    def pools(self):
        rows = {g: [{"id": f"{g}{i}", "prompt": f"{g} prompt {i}"} for i in range(5)]
                for g in ("other_harm", "safe", "violence")}
        rows["violence"].append({"id": "dup", "prompt": "safe prompt 0"})      # same text as a safe prompt
        return rows

    def test_pick_quota_and_distinct_prompts(self):
        chosen = make_batch.pick(self.pools(), {"safe": 5, "violence": 6, "other_harm": 2})
        texts = [r["prompt"] for _, r in chosen]
        self.assertEqual(len(texts), len(set(texts)))
        counts = {g: sum(1 for x, _ in chosen if x == g) for g in ("safe", "violence", "other_harm")}
        self.assertEqual(counts["safe"], 5)
        self.assertEqual(counts["other_harm"], 2)
        self.assertEqual(counts["violence"], 5)                            # the duplicate text is skipped

    def test_plan_kinds_rotation_and_split(self):
        seeds = make_batch.plan(make_batch.pick(self.pools(), {"safe": 2, "other_harm": 5}))
        self.assertEqual({s["kind"] for s in seeds if s["group"] == "safe"}, {"helpful"})
        self.assertEqual({s["kind"] for s in seeds if s["group"] != "safe"}, {"risk"})
        self.assertEqual(len({s["shape"] for s in seeds}), 4)
        for s in seeds:
            self.assertEqual(s["split"], make_batch.aegis.split_of(s["family"].split(":", 1)[1]))


class ExtractTests(unittest.TestCase):
    def test_summarize_revalidates(self):
        good, bad = seed(), dict(seed(), task_key="thinking-en-v1-00001")
        values = [{"seed": good, "wire_api": "chat_completions", "response": response(good)},
                  {"seed": bad, "wire_api": "chat_completions", "response": response(bad, "too short")}]
        examples, words, summary = summarize(values)
        self.assertEqual(len(examples), 1)
        self.assertEqual(summary["states"], {"complete": 1, "failed": 1})
        self.assertEqual(summary["by_kind"], {"risk": {"complete": 1, "failed": 1}})


if __name__ == "__main__":
    unittest.main()
