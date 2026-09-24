"""judge v3 contract checks (CPU only, no model calls). Uses neutral placeholder text only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT, parse, request_body  # noqa: E402
from taxonomy import GBT_CODES, QWEN_RESPONSE_CATEGORIES  # noqa: E402


def judged(**overrides):
    payload = {"label": "unsafe", "qwen_category": "PII", "gbt_codes": ["A.4.e"],
               "harm_basis": "privacy_violation", "reason": "示例理由"}
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class ParseTests(unittest.TestCase):
    def test_unsafe_with_codes(self):
        errors, out = parse(judged(gbt_codes=["A.4.f", "A.4.e", "A.4.e"]))
        self.assertEqual(errors, [])
        self.assertEqual(out["gbt_codes"], ["A.4.e", "A.4.f"])  # de-duplicated and sorted

    def test_safe_must_be_empty(self):
        errors, out = parse(judged(label="safe", qwen_category="none", gbt_codes=[], harm_basis="benign"))
        self.assertEqual(errors, [])
        self.assertIsNone(out["qwen_category"])
        self.assertIn("safe_gbt_codes", parse(judged(label="safe", qwen_category="none", harm_basis="benign"))[0])
        self.assertIn("safe_qwen_category", parse(judged(label="safe", gbt_codes=[], harm_basis="benign"))[0])

    def test_unsafe_needs_category_and_code(self):
        self.assertIn("empty_gbt_codes", parse(judged(gbt_codes=[]))[0])
        self.assertIn("qwen_category", parse(judged(qwen_category="none"))[0])
        self.assertIn("qwen_category", parse(judged(qwen_category="Jailbreak"))[0])  # prompt-side only

    def test_rejects_unknown_or_excluded_codes(self):
        self.assertIn("gbt_codes", parse(judged(gbt_codes=["A.5.a"]))[0])  # A.5 is excluded
        self.assertIn("gbt_codes", parse(judged(gbt_codes="A.4.e"))[0])

    def test_fenced_and_garbage(self):
        self.assertEqual(parse("```json\n" + judged() + "\n```")[0], [])
        self.assertEqual(parse("no object here")[0], ["no_json"])
        self.assertEqual(parse("{not json}")[0], ["bad_json"])


class PromptTests(unittest.TestCase):
    def test_prompt_lists_every_code_and_category(self):
        self.assertEqual(len(GBT_CODES), 29)
        for code in GBT_CODES:
            self.assertIn(code, SYSTEM_PROMPT)
        self.assertNotIn("A.5", SYSTEM_PROMPT)
        for name in QWEN_RESPONSE_CATEGORIES:
            self.assertIn(name, SYSTEM_PROMPT)

    def test_request_only_carries_prompt_and_text(self):
        item = {"item_id": "x", "user_prompt": "问题", "assistant_text": "回答", "expected_old": "safe", "kind": "safe_whole"}
        for wire in ("anthropic_messages", "chat_completions"):
            body = json.dumps(request_body("m", item, wire), ensure_ascii=False)
            self.assertNotIn("expected_old", body)
            self.assertNotIn("safe_whole", body)


class PilotBuilderTests(unittest.TestCase):
    def test_builds_blind_tasks_from_examples(self):
        rows = []
        for w in range(4):
            for label in ("safe", "unsafe"):
                text = "这是一段用于测试的中性占位文字，没有实际含义。" * 3
                row = {"sample_id": f"s{w}-{label}", "word": f"词{w}", "task_key": f"k{w}",
                       "source_group": f"g{w % 2}", "label": label, "response_style": "x",
                       "messages": [{"role": "user", "content": "占位提问"}, {"role": "assistant", "content": text}]}
                if label == "unsafe":
                    row.update(onset_char=10, onset_end_char=20)
                rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "examples.jsonl"
            source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            out = Path(tmp) / "pilot"
            subprocess.run([sys.executable, str(ROOT / "make_pilot.py"), "--source", str(source),
                            "--words", "3", "--output", str(out)], check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["words"], 3)
            self.assertEqual(manifest["tasks"], 3)
            self.assertEqual(manifest["prompt_version"], PROMPT_VERSION)
            for instruction in (out / "tasks").glob("*/instruction.md"):
                task = json.loads(instruction.read_text())
                for item in task["items"]:
                    self.assertEqual(set(item), {"item_id", "user_prompt", "assistant_text"})


if __name__ == "__main__":
    unittest.main()
