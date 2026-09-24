"""judge v3 contract checks (CPU only, no model calls). Uses neutral placeholder text only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT))  # make_fallback / extract_judgments, when run from the repo root
from pipeline import PROMPT_VERSION, SYSTEM_PROMPT, WIRE_APIS, parse, request_body, response_text, stop_ok  # noqa: E402
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

    def test_strict_epithet_rule_is_the_only_exception(self):
        self.assertIn("唯一例外", SYSTEM_PROMPT)
        self.assertIn("A.1.b", SYSTEM_PROMPT.split("唯一例外")[1])
        self.assertEqual(PROMPT_VERSION, "guard-judge-domestic-v3.2")
        exception = SYSTEM_PROMPT.split("唯一例外")[1]
        self.assertIn("负面传言", exception)
        self.assertIn("A.1.g", exception)
        self.assertIn("规避写法", exception)
        self.assertIn("政治局常委", exception.split("。")[0])   # same leader scope as word screen v3

    def test_request_only_carries_prompt_and_text(self):
        item = {"item_id": "x", "user_prompt": "问题", "assistant_text": "回答", "expected_old": "safe", "kind": "safe_whole"}
        for wire in WIRE_APIS:
            body = json.dumps(request_body("m", item, wire), ensure_ascii=False)
            self.assertNotIn("expected_old", body)
            self.assertNotIn("safe_whole", body)


class WireTests(unittest.TestCase):
    item = {"user_prompt": "问题", "assistant_text": "回答"}

    def test_chat_token_cap_raised(self):
        self.assertEqual(request_body("m", self.item, "chat_completions")["max_tokens"], 8000)

    def test_responses_strict_schema_matches_parser(self):
        body = request_body("m", self.item, "responses")
        schema = body["text"]["format"]["schema"]
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(len(schema["properties"]["gbt_codes"]["items"]["enum"]), 29)
        self.assertIn("none", schema["properties"]["qwen_category"]["enum"])

    def test_responses_text_and_stop(self):
        ok = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": judged()}]}]}
        self.assertEqual(parse(response_text(ok, "responses"))[0], [])
        self.assertEqual(stop_ok(ok, "responses"), (True, "completed"))
        refused = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "x"}]}]}
        self.assertEqual(stop_ok(refused, "responses"), (False, "refusal"))
        self.assertEqual(stop_ok({"status": "incomplete"}, "responses"), (False, "incomplete"))


class FallbackTests(unittest.TestCase):
    def test_selects_failed_political_and_controls(self):
        from make_fallback import select
        items = [{"item_id": f"i{n}", "task_key": f"k{n}", "kind": "safe_whole"} for n in range(10)]
        judgments = [{"item_id": "i0", "judge_label": None},
                     {"item_id": "i1", "judge_label": "unsafe", "judge_qwen_category": "Politically Sensitive Topics",
                      "judge_gbt_codes": ["A.1.b"]},
                     {"item_id": "i2", "judge_label": "unsafe", "judge_qwen_category": "Unethical Acts",
                      "judge_gbt_codes": ["A.1.c", "A.4.c"]}]
        judgments += [{"item_id": f"i{n}", "judge_label": "safe", "judge_qwen_category": None, "judge_gbt_codes": []}
                      for n in range(3, 10)]
        chosen = select(items, judgments, controls=3)
        reasons = {c["item_id"]: c["fallback_reason"] for c in chosen}
        self.assertEqual(reasons["i0"], "failed")
        self.assertEqual(reasons["i1"], "political")
        self.assertEqual(reasons["i2"], "political")  # political GB/T code under another category
        self.assertEqual(sum(r == "control" for r in reasons.values()), 3)
        self.assertEqual(select(items, judgments, controls=3), chosen)  # deterministic

    def test_cross_judge_summary(self):
        from extract_judgments import cross_judge
        rows = [{"fallback_reason": "failed", "reference_label": None, "judge_label": "unsafe"},
                {"fallback_reason": "control", "reference_label": "safe", "judge_label": "safe"},
                {"fallback_reason": "control", "reference_label": "safe", "judge_label": "unsafe"}]
        out = cross_judge(rows)
        self.assertEqual(out["failed"]["judged_now"], 1)
        self.assertIsNone(out["failed"]["agreement_where_both_judged"])
        self.assertEqual(out["control"]["agreement_where_both_judged"], 0.5)
        self.assertEqual(out["control"]["reference_to_now"], {"safe->safe": 1, "safe->unsafe": 1})


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
            # R-slice mode: keep only chosen splits, every word, custom Task names; split travels with the item
            for r in rows:
                r["split"] = "dev" if r["task_key"] in ("k0", "k1") else "train"
            source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            out2 = Path(tmp) / "r"
            subprocess.run([sys.executable, str(ROOT / "make_pilot.py"), "--source", str(source), "--words", "0",
                            "--splits", "dev", "--name-prefix", "r-dev", "--output", str(out2)], check=True, capture_output=True)
            manifest2 = json.loads((out2 / "manifest.json").read_text())
            self.assertEqual((manifest2["words"], manifest2["splits"]), (2, "dev"))
            self.assertTrue(all(p.name.startswith("r-dev-") for p in (out2 / "tasks").iterdir()))
            with (out2 / "items.jsonl").open(encoding="utf-8") as handle:
                self.assertEqual({json.loads(line)["split"] for line in handle}, {"dev"})


if __name__ == "__main__":
    unittest.main()
