"""Frozen one-word/four-response contract checks (CPU only, no model calls)."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "flow"))
from pipeline import (EXAMPLE_PROMPTS, PROMPT_VERSION, example_output, request_body,  # noqa: E402
                      response_refusal, slot_records, validate)


def seed(unsafe_language, rotation=0):
    safe_language = "en" if unsafe_language == "zh" else "zh"
    return {"task_key": "lex-test", "word": "示例词", "origin_group_id": "lex-test",
            "data_layer": "permissive", "source_licenses": ["MIT"], "source_group": "test/source",
            "rotation": rotation,
            "prompts": {"unsafe": {"language": unsafe_language, "text": EXAMPLE_PROMPTS[unsafe_language]["unsafe"],
                                   "policy_basis": "fraud_or_fabrication", "sample_id": "lex-test-v12-01"},
                        "safe": {"language": safe_language, "text": EXAMPLE_PROMPTS[safe_language]["safe"],
                                 "policy_basis": "benign", "sample_id": "lex-test-v12-00"}}}


def payload(s):
    output = example_output(s["prompts"]["unsafe"]["language"], s["prompts"]["safe"]["language"])
    output.pop("example_user_prompts")
    return {**output, "task_key": s["task_key"], "word": s["word"]}


class ContractTests(unittest.TestCase):
    def test_two_by_two_slots_decouple_prompt_and_response_labels(self):
        for language in ("zh", "en"):
            for rotation in range(8):
                slots = slot_records(seed(language, rotation)["prompts"], rotation)
                pairs = [(s["prompt_label"], s["label"]) for s in slots]
                self.assertEqual(pairs, [("unsafe", "safe"), ("unsafe", "unsafe"), ("safe", "safe"), ("safe", "unsafe")])
                self.assertEqual([s["language"] for s in slots][:2], [language, language])
                self.assertTrue(all(s["target_role"] == "assistant" for s in slots))
                self.assertTrue(all("onset_style" in s for s in slots if s["label"] == "unsafe"))

    def test_rotation_covers_every_style_and_onset_combination(self):
        styles = {slot_records(seed("zh", r)["prompts"], r)[0]["style"] for r in range(4)}
        combos = {(slot_records(seed("zh", r)["prompts"], r)[1]["format"],
                   slot_records(seed("zh", r)["prompts"], r)[1]["onset_style"]) for r in range(4)}
        self.assertEqual(len(styles), 4)
        self.assertEqual(len(combos), 4)

    def test_real_schema_is_in_user_instruction_and_api(self):
        body = request_body("gpt-5.6-luna", seed("zh"))
        user = json.loads(body["input"][1]["content"])
        contract = body["text"]["format"]
        self.assertTrue(contract["strict"])
        self.assertEqual(user["output_schema"], contract["schema"])
        self.assertEqual(contract["schema"]["properties"]["responses"]["minItems"], 4)
        self.assertEqual(len(user["example_output"]["responses"]), 4)

    def test_example_output_validates_for_both_orientations(self):
        for language in ("zh", "en"):
            s = seed(language)
            errors, rows = validate(payload(s), s)
            self.assertEqual(errors, [])
            self.assertEqual([r["label"] for r in rows], ["safe", "unsafe", "safe", "unsafe"])
            for row in rows:
                response = row["messages"][1]["content"]
                if row["label"] == "unsafe":
                    self.assertEqual(response[row["onset_char"]:row["onset_end_char"]], row["onset_quote"])
                    self.assertGreater(row["onset_char"], 0)
                else:
                    self.assertIsNone(row["onset_char"])
            self.assertEqual(rows[0]["messages"][0]["content"], s["prompts"]["unsafe"]["text"])
            self.assertEqual(rows[2]["messages"][0]["content"], s["prompts"]["safe"]["text"])

    def test_bad_items_are_rejected_per_slot(self):
        s = seed("zh")
        bad = payload(s)
        bad["responses"][0]["onset_quote"] = "不能帮你"              # safe slot must not carry an onset
        bad["responses"][1]["onset_quote"] = "这句话不在回答里出现过"  # onset must be verbatim
        bad["responses"][2]["response"] = bad["responses"][2]["response"].replace("示例词", "别的词")
        bad["responses"][3]["response"] = "抱歉，" + bad["responses"][3]["response"]
        errors, rows = validate(bad, s)
        self.assertEqual(rows, [])
        for expected in ("0:safe_onset_or_basis", "1:onset_quote", "2:missing_word", "3:unsafe_refusal_opening"):
            self.assertIn(expected, errors)

    def test_one_char_onset_typo_is_aligned_and_flagged(self):
        s = seed("zh")
        typo = payload(s)
        quote = typo["responses"][1]["onset_quote"]
        typo["responses"][1]["onset_quote"] = quote.replace("二十万", "二千万")
        errors, rows = validate(typo, s)
        self.assertEqual(errors, [])
        self.assertIn("onset_one_char_aligned", rows[1]["repair_flags"])

    def test_root_mismatch_and_refusal_block(self):
        s = seed("en")
        wrong = copy.deepcopy(payload(s))
        wrong["prompt_version"] = "old"
        errors, _ = validate(wrong, s)
        self.assertIn("root_mismatch:prompt_version", errors)
        response = {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
        self.assertEqual(response_refusal(response), "no")
        self.assertEqual(PROMPT_VERSION, "guard-luna-oneword-response-v14")

    def test_negated_onset_is_rejected(self):
        for language, warning in (("zh", "千万不要"), ("en", "However, do not ")):
            s = seed(language)
            bad = payload(s)
            index = 1 if language == "zh" else 3
            item = bad["responses"][index]
            item["response"] = item["response"].replace(item["onset_quote"], warning + item["onset_quote"])
            errors, _ = validate(bad, s)
            self.assertIn(f"{index}:negated_onset", errors)

    def test_quality_flags_are_recorded_not_rejected(self):
        s = seed("zh")
        flagged = payload(s)
        flagged["responses"][1]["response"] += "这种做法会造成严重伤害。"
        flagged["responses"][2]["response"] = flagged["responses"][2]["response"].replace("placeholder", "fictional placeholder")
        errors, rows = validate(flagged, s)
        self.assertEqual(errors, [])
        self.assertEqual(rows[1]["quality_flags"], ["harm_commentary"])
        self.assertEqual(rows[2]["quality_flags"], ["fiction_marker"])
        self.assertEqual(rows[0]["quality_flags"], [])

    def test_v13_pilot_replay_rejects_exactly_the_negated_onsets(self):
        """Every real v13 pilot record re-validated under v14 rules (CPU replay, no model call)."""
        path = Path(__file__).resolve().parents[1] / "response_v13/pilot/extracted/examples.jsonl"
        if not path.exists():
            self.skipTest("v13 pilot extraction not present")
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        rejected = set()
        for row in rows:
            if row["label"] != "unsafe":
                continue
            text, start = row["messages"][1]["content"], row["onset_char"]
            from pipeline import NEGATED_ONSET
            if NEGATED_ONSET.search(text[max(0, start - 12):start]):
                rejected.add(row["word"])
        self.assertEqual(rejected, {"退党 退团 退队", "群交", "极端武力直销网", "康跳楼", "催情剂出售"})


if __name__ == "__main__":
    unittest.main()
