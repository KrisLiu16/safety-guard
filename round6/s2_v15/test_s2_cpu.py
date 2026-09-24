"""S2 contract checks (CPU only, no model calls). Placeholder text only."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
from pipeline import FORMS, PROMPT_VERSION, WIRE_APIS, request_body, schema, validate  # noqa: E402
from make_pilot import pick, plan, split_of  # noqa: E402
from extract_s2 import summarize  # noqa: E402

WORD = "示例词"


def seed(language="zh", form="background"):
    return {"task_key": "lex-s2-test", "word": WORD, "language": language, "form": form,
            "source_group": "test/group", "family": "fam", "split": "train"}


def payload(language="zh", **overrides):
    if language == "zh":
        filler = "这是用于测试的中性说明文字，没有任何实际含义。"
        texts = {"prompt": f"请问“{WORD}”这个说法一般是什么意思？",
                 "answer": f"“{WORD}”是占位用语。" + filler * 6,
                 "reasoning": f"用户想了解“{WORD}”。" + filler * 7 + "我会正常回答。"}
    else:
        filler = " This is neutral placeholder text for a unit test only."
        texts = {"prompt": f"What does the term “{WORD}” usually mean in documentation?",
                 "answer": f"“{WORD}” is a placeholder term." + filler * 6,
                 "reasoning": f"The user asks about “{WORD}”." + filler * 7}
    out = {"task_key": "lex-s2-test", "word": WORD, "prompt_version": PROMPT_VERSION, "status": "ready",
           "aspect": "其他", "notes": "", **texts}
    out.update(overrides)
    return out


class ValidateTests(unittest.TestCase):
    def test_valid_payload_gives_two_safe_assistant_rows(self):
        for language in ("zh", "en"):
            errors, rows = validate(payload(language), seed(language))
            self.assertEqual(errors, [])
            self.assertEqual([r["response_format"] for r in rows], ["answer", "reasoning"])
            self.assertTrue(all(r["label"] == "safe" and r["target_role"] == "assistant" for r in rows))
            self.assertTrue(all(WORD in r["messages"][1]["content"] for r in rows))

    def test_rejections(self):
        s = seed()
        self.assertIn("answer:missing_word", validate(payload(answer="没有这个词。" * 20), s)[0])
        self.assertIn("reasoning:refusal_opening", validate(payload(reasoning="抱歉，" + payload()["reasoning"]), s)[0])
        self.assertIn("answer:length", validate(payload(answer=f"{WORD}很短"), s)[0])
        self.assertIn("aspect", validate(payload(aspect="不存在"), s)[0])
        self.assertEqual(validate(payload(task_key="other"), s)[0], ["root_mismatch:task_key"])

    def test_skip_is_not_a_row(self):
        errors, rows = validate(payload(status="skip", prompt="", answer="", reasoning=""), seed())
        self.assertEqual((errors, rows), (["skip"], []))


class RequestTests(unittest.TestCase):
    def test_both_wires(self):
        for wire in WIRE_APIS:
            body = json.dumps(request_body("m", seed(), wire), ensure_ascii=False)
            self.assertIn(PROMPT_VERSION, body)
            self.assertIn(FORMS["background"], body)
        strict = request_body("m", seed(), "responses")["text"]["format"]["schema"]
        self.assertEqual(set(strict["required"]), set(strict["properties"]))
        self.assertEqual(strict, schema())


class PlanTests(unittest.TestCase):
    def test_balanced_languages_forms_and_stable_split(self):
        seeds = [{"task_key": f"k{i}", "word": f"w{i}", "source_group": f"g{i % 3}", "family": f"f{i}"} for i in range(40)]
        chosen = plan(pick(seeds, 16))
        self.assertEqual(len(chosen), 16)
        self.assertEqual(sum(s["language"] == "zh" for s in chosen), 8)
        self.assertEqual({s["form"] for s in chosen}, set(FORMS))
        self.assertEqual(len({s["source_group"] for s in chosen}), 3)
        self.assertTrue(all(s["split"] == split_of(s["family"]) for s in chosen))
        self.assertEqual(plan(pick(seeds, 16)), chosen)


class ExtractTests(unittest.TestCase):
    def test_summarize_and_judge_pilot_compatibility(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload(), ensure_ascii=False)}}]}
        other = dict(seed(), task_key="lex-s2-other")
        values = [{"seed": seed(), "wire_api": "chat_completions", "response": response},
                  {"seed": other, "wire_api": "chat_completions",
                   "response": {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                       payload(task_key="lex-s2-other", status="skip", prompt="", answer="", reasoning=""), ensure_ascii=False)}}]}},
                  {"seed": dict(seed(), task_key="lex-s2-cut"), "wire_api": "chat_completions",
                   "response": {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}}]
        examples, prompts, words, summary = summarize(values)
        self.assertEqual(summary["states"], {"complete": 1, "skip": 1, "failed": 1})
        self.assertEqual(len(examples), 2)
        self.assertEqual(prompts[0]["target_role"], "user")
        # judge_v3/make_pilot.py must accept S2 rows unchanged (same module name as ours, so load it by path)
        import importlib.util
        spec = importlib.util.spec_from_file_location("judge_v3_make_pilot", ROOT.parent / "judge_v3/make_pilot.py")
        judge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge)
        items = [item for row in examples for item in judge.items_for(row)]
        self.assertEqual({i["kind"] for i in items}, {"safe_whole", "safe_mid_prefix"})


if __name__ == "__main__":
    unittest.main()
