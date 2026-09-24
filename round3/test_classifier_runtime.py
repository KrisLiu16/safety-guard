"""Contract checks for direct classification and cached prefix updates."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import unittest
from urllib.request import Request, urlopen

import torch

from classifier_runtime import PrefixClassifier, readout
from serve_classifier import ClassificationServer


class FakeModel:
    device = torch.device("cpu")
    query_category_map = {0: "Political", 1: "PII"}
    response_category_map = {0: "Political", 1: "PII"}

    def __init__(self):
        self.forward_calls = 0
        self.forwarded = []

    def __call__(self, *, input_ids, past_key_values=None, **_):
        self.forward_calls += 1
        self.forwarded.append(input_ids.shape[-1])
        return SimpleNamespace(
            query_risk_level_logits=torch.tensor([[[0.0, 3.0, 1.0]]]),
            risk_level_logits=torch.tensor([[[4.0, 0.0, 1.0]]]),
            query_category_logits=torch.tensor([[[3.0, 0.0]]]),
            category_logits=torch.tensor([[[0.0, 3.0]]]),
            past_key_values=past_key_values,
        )


class FakeTokenizer:
    def apply_chat_template(self, messages, **_):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)

    def encode(self, text, **_):
        return [ord(ch) for ch in text]


class ClassifierTests(unittest.TestCase):
    def test_risk_is_independent_from_topic_category(self):
        model = FakeModel()
        output = model(input_ids=torch.tensor([[1]]), past_key_values=None)
        user = readout(output, "user", model, "test", 1, 1, 0)
        assistant = readout(output, "assistant", model, "test", 1, 1, 0)
        self.assertEqual(user["risk_level"], "unsafe")
        self.assertEqual(user["category"], "Political")
        self.assertEqual(assistant["risk_level"], "safe")
        self.assertEqual(assistant["category"], "PII")
        self.assertFalse(user["calibrated"])
        self.assertAlmostEqual(sum(user["risk_probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(user["category_distribution"].values()), 1.0, places=6)
        baseline = readout(output, "user", model, "a0-test", 1, 1, 0,
                           "pretrained_baseline_unaligned")
        self.assertEqual(baseline["training_status"], "pretrained_baseline_unaligned")

    def test_cached_prefix_reuses_identical_input_and_recomputes_changed_suffix(self):
        model = FakeModel()
        stream = PrefixClassifier(model, None, "test", "user")
        first = stream.update_ids([1, 2])
        same = stream.update_ids([1, 2])
        longer = stream.update_ids([1, 2, 3])
        changed = stream.update_ids([1, 2, 4])
        self.assertEqual(model.forwarded, [2, 1, 1])
        self.assertEqual([first["input_tokens_forwarded"], same["input_tokens_forwarded"],
                          longer["input_tokens_forwarded"], changed["input_tokens_forwarded"]],
                         [2, 0, 1, 1])
        self.assertEqual(changed["cache_reused_tokens"], 2)
        self.assertEqual(stream.total_forwarded_tokens, 4)

    def test_exact_token_append_avoids_prefix_rescan(self):
        model = FakeModel()
        stream = PrefixClassifier(model, None, "test", "assistant")
        first = stream.append_token_ids([1, 2, 3])
        second = stream.append_token_ids([4, 5])
        self.assertEqual(model.forwarded, [3, 2])
        self.assertEqual(first["risk_level"], "safe")
        self.assertEqual(second["input_tokens_total"], 5)
        self.assertEqual(second["cache_reused_tokens"], 3)

    def test_json_contract_exposes_no_generation_or_untrained_policy_fields(self):
        schema = json.loads((Path(__file__).parent / "guard-classification-v1.schema.json").read_text())
        properties = schema["properties"]
        self.assertIn("risk_probabilities", properties)
        self.assertIn("category_distribution", properties)
        for forbidden in ("generated_text", "generated_tokens", "action",
                          "evidence_sufficient_probability", "refusal_probability"):
            self.assertNotIn(forbidden, properties)

    def test_http_first_append_returns_direct_classification(self):
        server = ClassificationServer(("127.0.0.1", 0), FakeModel(), FakeTokenizer(), "test")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = f"http://127.0.0.1:{server.server_port}"

        def post(path, body):
            request = Request(root + path, data=json.dumps(body).encode(), method="POST",
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read())

        try:
            session = post("/sessions", {"target_role": "user"})["session_id"]
            result = post(f"/sessions/{session}/append", {"chunk": "你好", "final": False})
            self.assertEqual(result["risk_level"], "unsafe")
            self.assertGreater(result["input_tokens_total"], 0)
            self.assertNotIn("generated_text", result)
            completed = post("/classify", {"messages": [{"role": "assistant", "content": "你好"}]})
            self.assertEqual(completed["risk_level"], "safe")
            with ThreadPoolExecutor(max_workers=32) as pool:
                responses = list(pool.map(lambda _: post("/classify", {"messages": [
                    {"role": "user", "content": "你好"}]}), range(32)))
            self.assertEqual(len(responses), 32)
            self.assertTrue(all(x["risk_level"] == "unsafe" for x in responses))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
