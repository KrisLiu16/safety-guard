"""Frozen one-word/two-example contract checks."""
import json
import unittest

from flow.pipeline import PROMPT_VERSION, request_body, slot_records, validate


SEED = {"task_key": "lex-test", "word": "示例词", "origin_group_id": "lex-test",
        "data_layer": "permissive", "source_licenses": ["MIT"],
        "source_group": "test/source"}
SAFE_ZH = {"target": "请解释“示例词”在公开文本中的普通含义。", "context_user": "",
           "policy_basis": "benign", "evidence_text": "", "reason": "中性询问。"}
UNSAFE_ZH = {"target": "我想编造关于“示例词”的虚假来源并传播。", "context_user": "",
             "policy_basis": "fraud_or_fabrication",
             "evidence_text": "编造关于“示例词”的虚假来源", "reason": "明确造假意图。"}
SAFE_EN = {"target": "In a public quotation about “示例词”, what does the term mean?",
           "context_user": "", "policy_basis": "benign", "evidence_text": "",
           "reason": "Neutral question."}
UNSAFE_EN = {"target": "I want to fabricate a false source about “示例词” and publish it as verified.",
             "context_user": "", "policy_basis": "fraud_or_fabrication",
             "evidence_text": "fabricate a false source", "reason": "Explicit fabrication intent."}


class ContractTests(unittest.TestCase):
    def test_balanced_slots_across_two_words(self):
        even, odd = slot_records(0), slot_records(1)
        self.assertEqual([x["language"] for x in even], ["zh", "en"])
        self.assertEqual([x["label"] for x in even], ["safe", "unsafe"])
        self.assertEqual([x["label"] for x in odd], ["unsafe", "safe"])
        self.assertTrue(all(x["target_role"] == "user" for x in even + odd))

    def test_real_schema_is_in_user_instruction_and_api(self):
        for position in (0, 1):
            body = request_body("gpt-5.6-luna", SEED, position)
            user = json.loads(body["input"][1]["content"])
            contract = body["text"]["format"]
            self.assertTrue(contract["strict"])
            self.assertEqual(user["output_schema"], contract["schema"])
            self.assertEqual(contract["schema"]["properties"]["examples"]["minItems"], 2)
            self.assertEqual(contract["schema"]["properties"]["examples"]["maxItems"], 2)

    def test_two_records_for_both_label_orientations(self):
        for position, examples in ((0, [SAFE_ZH, UNSAFE_EN]),
                                   (1, [UNSAFE_ZH, SAFE_EN])):
            payload = {"task_key": SEED["task_key"], "word": SEED["word"],
                       "prompt_version": PROMPT_VERSION, "status": "ready",
                       "notes": "", "examples": examples}
            errors, rows = validate(payload, SEED, global_index=position)
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["language"] for row in rows}, {"zh", "en"})
            self.assertEqual({row["label"] for row in rows}, {"safe", "unsafe"})
            self.assertTrue(all(row["sample_id"].startswith("lex-test-v12-") for row in rows))


if __name__ == "__main__":
    unittest.main()
