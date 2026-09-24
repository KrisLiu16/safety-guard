"""Verify fixed v9 slots, Schema delivery, and per-word validation."""
import json
import unittest

from flow.pipeline import PROMPT_VERSION, format_example, request_body, slot_pair, validate


def seed(i: int) -> dict:
    return {"task_key": f"seed-{i}", "word": f"示例词{i}",
            "origin_group_id": f"seed-{i}", "source_licenses": ["Apache-2.0"],
            "data_layer": "permissive"}


def replace_word(example: dict, value: str) -> dict:
    return {key: (text.replace("示例词", value) if isinstance(text, str) else text)
            for key, text in example.items()}


class PipelineTests(unittest.TestCase):
    def test_ten_term_slot_balance(self):
        slots = [slot_pair(i) for i in range(10)]
        self.assertEqual(sum(x["zh"]["label"] == "safe" for x in slots), 5)
        self.assertEqual(sum(x["en"]["label"] == "safe" for x in slots), 5)
        self.assertEqual(sum(x[lang]["target_role"] == "user"
                             for x in slots for lang in ("zh", "en")), 10)
        for pair in slots:
            self.assertEqual({pair["zh"]["label"], pair["en"]["label"]},
                             {"safe", "unsafe"})
            self.assertEqual({pair["zh"]["target_role"], pair["en"]["target_role"]},
                             {"user", "assistant"})

    def test_actual_schema_in_user_and_strict_format(self):
        seeds = [seed(i) for i in range(10)]
        body = request_body("gpt-5.6-luna", "batch-a", seeds, 0)
        user = json.loads(body["input"][1]["content"])
        schema = body["text"]["format"]["schema"]
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(user["output_schema"], schema)
        self.assertEqual(schema["properties"]["items"]["minItems"], 10)
        self.assertEqual(schema["properties"]["items"]["maxItems"], 10)
        self.assertEqual(user["prompt_version"], PROMPT_VERSION)

    def test_two_examples_per_term_validate(self):
        seeds = [seed(i) for i in range(10)]
        items = []
        for i, term in enumerate(seeds):
            slots = slot_pair(i)
            items.append({"term_index": i,
                          "zh": replace_word(format_example(slots["zh"]), term["word"]),
                          "en": replace_word(format_example(slots["en"]), term["word"])})
        results = validate({"batch_key": "batch-a", "prompt_version": PROMPT_VERSION,
                            "items": items}, "batch-a", seeds, 0, "test/source")
        self.assertTrue(all(not item["errors"] and len(item["records"]) == 2 for item in results),
                        results)
        self.assertEqual(len({r["sample_id"] for item in results for r in item["records"]}), 20)

    def test_one_bad_word_does_not_erase_other_results(self):
        seeds = [seed(i) for i in range(2)]
        items = []
        for i, term in enumerate(seeds):
            slots = slot_pair(i)
            items.append({"term_index": i,
                          "zh": replace_word(format_example(slots["zh"]), term["word"]),
                          "en": replace_word(format_example(slots["en"]), term["word"])})
        items[0]["zh"]["target"] = "broken"
        results = validate({"batch_key": "batch-a", "prompt_version": PROMPT_VERSION,
                            "items": items}, "batch-a", seeds, 0, "test/source")
        self.assertEqual(len(results[0]["records"]), 1)
        self.assertEqual(len(results[1]["records"]), 2)


if __name__ == "__main__":
    unittest.main()
