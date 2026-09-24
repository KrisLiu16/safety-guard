"""S5-safe contract checks (CPU only, no model calls). Placeholder text only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))
from pipeline import LENGTHS, PROMPT_VERSION, SHAPES, WIRE_APIS, request_body, schema, validate  # noqa: E402
from make_pilot import pick, plan, split_of  # noqa: E402
from extract_s5 import summarize  # noqa: E402

WORD = "示例词"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed(unsafe_language="zh", shape="stepwise"):
    safe_language = "en" if unsafe_language == "zh" else "zh"
    text = {"zh": f"占位提问：关于“{WORD}”的问题。", "en": f"Placeholder question about “{WORD}”."}
    return {"task_key": "lex-s5-test", "word": WORD, "shape": shape, "length": "medium", "source_group": "test/group",
            "family": "fam", "split": "train",
            "prompts": {"unsafe": {"language": unsafe_language, "text": text[unsafe_language], "sample_id": "u"},
                        "safe": {"language": safe_language, "text": text[safe_language], "sample_id": "s"}}}


def long_text(language, extra=""):
    if language == "zh":
        return f"用户在问“{WORD}”。{extra}" + "这是一段用于单元测试的中性占位文字，没有任何实际含义。" * 40
    return f"The user asks about “{WORD}”. {extra}" + " This is neutral placeholder text for a unit test only." * 45


def payload(s, **overrides):
    out = {"task_key": s["task_key"], "word": WORD, "prompt_version": PROMPT_VERSION, "status": "ready", "notes": "",
           "unsafe_reasoning": long_text(s["prompts"]["unsafe"]["language"], "A"),
           "safe_reasoning": long_text(s["prompts"]["safe"]["language"], "B")}
    out.update(overrides)
    return out


class ValidateTests(unittest.TestCase):
    def test_two_long_safe_rows_in_prompt_languages(self):
        for unsafe_language in ("zh", "en"):
            s = seed(unsafe_language)
            errors, rows = validate(payload(s), s)
            self.assertEqual(errors, [])
            self.assertEqual([(r["prompt_label"], r["label"], r["language"]) for r in rows],
                             [("unsafe", "safe", unsafe_language), ("safe", "safe", s["prompts"]["safe"]["language"])])
            self.assertEqual({r["response_style"] for r in rows}, {"risk_reasoning_long", "helpful_reasoning_long"})
            self.assertEqual(rows[0]["messages"][0]["content"], s["prompts"]["unsafe"]["text"])  # question unchanged

    def test_rejections(self):
        s = seed()
        self.assertIn("unsafe_reasoning:length", validate(payload(s, unsafe_reasoning=f"{WORD}太短了"), s)[0])
        self.assertIn("safe_reasoning:missing_word", validate(payload(s, safe_reasoning=" filler words" * 450), s)[0])
        self.assertEqual(validate(payload(s, word="别的词"), s)[0], ["root_mismatch:word"])
        self.assertEqual(validate(payload(s, status="skip", unsafe_reasoning="", safe_reasoning=""), s), (["skip"], []))

    def test_literal_backslash_n_is_repaired_and_flagged(self):
        s = seed()
        escaped = long_text("zh").replace("。", "。\\n\\n", 3)
        errors, rows = validate(payload(s, unsafe_reasoning=escaped), s)
        self.assertEqual(errors, [])
        self.assertIn("\n\n", rows[0]["messages"][1]["content"])
        self.assertNotIn("\\n", rows[0]["messages"][1]["content"])
        self.assertIn("escaped_newline_repaired", rows[0]["quality_flags"])
        self.assertEqual(rows[1]["quality_flags"], [])
        self.assertEqual(rows[0]["length_target"], "medium")

    def test_fiction_marker_is_flagged_not_rejected(self):
        s = seed()
        errors, rows = validate(payload(s, unsafe_reasoning=long_text("zh", "虚构")), s)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["quality_flags"], ["fiction_marker"])


class RequestTests(unittest.TestCase):
    def test_both_wires_carry_questions_unchanged(self):
        s = seed()
        for wire in WIRE_APIS:
            body = json.dumps(request_body("m", s, wire), ensure_ascii=False)
            self.assertIn(json.dumps(s["prompts"]["unsafe"]["text"], ensure_ascii=False)[1:-1], body)
            self.assertIn(SHAPES["stepwise"], body)
            self.assertIn(LENGTHS["medium"]["zh"], body)
        strict = request_body("m", s, "responses")["text"]["format"]["schema"]
        self.assertEqual(strict, schema())
        self.assertEqual(set(strict["required"]), set(strict["properties"]))


class PlanTests(unittest.TestCase):
    def test_shapes_exclusion_and_split_shared_with_s2(self):
        seeds = []
        for i in range(30):
            s = seed()
            s.update(task_key=f"k{i}", word=f"w{i}", family=f"f{i}", source_group=f"g{i % 3}")
            seeds.append(s)
        chosen = plan(pick(seeds, 12, exclude=frozenset({"k0", "k1"})))
        self.assertEqual(len(chosen), 12)
        self.assertNotIn("k0", {s["task_key"] for s in chosen})
        self.assertEqual({s["shape"] for s in chosen}, set(SHAPES))
        self.assertEqual({s["length"] for s in chosen}, {"short", "medium", "long"})
        # S2's builder imports its own `pipeline`, so evaluate its split rule in a separate interpreter.
        s2 = ROOT.parent / "s2_v15"
        code = (f"import sys, json; sys.path[:0] = [{str(s2 / 'flow')!r}, {str(s2)!r}]; "
                "from make_pilot import split_of; print(json.dumps([split_of(f'fam{i}') for i in range(200)]))")
        theirs = json.loads(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout)
        self.assertEqual([split_of(f"fam{i}") for i in range(200)], theirs)


class ExtractTests(unittest.TestCase):
    def test_summarize_and_judge_pilot_compatibility(self):
        s = seed()
        ok = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload(s), ensure_ascii=False)}}]}
        cut = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        examples, words, summary = summarize([{"seed": s, "wire_api": "chat_completions", "response": ok},
                                              {"seed": dict(s, task_key="k-cut"), "wire_api": "chat_completions", "response": cut}])
        self.assertEqual(summary["states"], {"complete": 1, "failed": 1})
        self.assertEqual(summary["errors"], {"stop_reason": 1})
        judge = load("judge_v3_make_pilot", ROOT.parent / "judge_v3/make_pilot.py")
        items = [item for row in examples for item in judge.items_for(row)]
        self.assertEqual(len(items), 4)
        self.assertEqual({i["kind"] for i in items}, {"safe_whole", "safe_mid_prefix"})


if __name__ == "__main__":
    unittest.main()
