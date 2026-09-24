"""CPU checks for T004/T005 tools on synthetic data (numpy only; no model, no tokenizer, no GPU)."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_probe_input import select  # noqa: E402
from probe_common import auc, feature_positions, fit_logistic, logistic_scores, position_classes, serialize  # noqa: E402


class PositionTests(unittest.TestCase):
    def test_classes_from_offsets(self):
        messages = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "abcdefghij"}]
        text = serialize(messages)
        start = len(text) - 10
        offsets = [(0, 5), (5, start), (start, start + 2), (start + 2, start + 4), (start + 4, start + 6),
                   (start + 6, start + 8), (start + 8, start + 10)]
        positions, classes = position_classes(offsets, start, "unsafe", onset_char=4, onset_end_char=8)
        self.assertEqual(positions, [2, 3, 4, 5, 6])
        self.assertEqual(classes, "PPOUU")
        self.assertEqual(position_classes(offsets, start, "safe")[1], "SSSSS")

    def test_feature_positions_keep_onset_boundary(self):
        classes = "P" * 40 + "O" * 3 + "U" * 57
        keep = feature_positions(list(range(100)), classes, limit=24)
        self.assertIn(39, keep)            # last pre-onset
        self.assertIn(43, keep)            # first post-onset
        self.assertIn(0, keep)
        self.assertIn(99, keep)
        self.assertLessEqual(len(keep), 24)
        self.assertEqual(feature_positions([], ""), [])


class StatsTests(unittest.TestCase):
    def test_auc_matches_pairwise_count(self):
        rng = np.random.default_rng(1)
        pos, neg = rng.integers(0, 5, 40), rng.integers(0, 5, 30)
        pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        self.assertAlmostEqual(auc(pos, neg), pairs, places=10)
        self.assertIsNone(auc([], [1.0]))

    def test_logistic_probe_separates(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((600, 16)).astype(np.float32)
        y = (x[:, 0] + 0.2 * rng.standard_normal(600) > 0.8).astype(np.float32)   # imbalanced
        model = fit_logistic(x, y, steps=300)
        s = logistic_scores(x, model)
        self.assertGreater(auc(s[y > 0], s[y == 0]), 0.95)


def fake_meta(n_records, rng):
    meta = []
    for i in range(n_records):
        slot = i % 4
        label = "unsafe" if slot in (1, 3) else "safe"
        style = ["risk_reasoning", "compliance", "helpful_answer", "drift"][slot]
        meta.append({"sample_id": f"s{i}", "family": f"f{i // 4}", "slot": slot, "label": label,
                     "prompt_label": "unsafe" if slot < 2 else "safe", "response_style": style,
                     "language": "zh" if (i // 4) % 2 else "en"})
    return meta


def write_fake_dump(directory, rng):
    for split, n in (("train", 160), ("calibration", 80), ("dev", 80)):
        meta = fake_meta(n, rng)
        hidden, classes, record, head = [], [], [], []
        for index, m in enumerate(meta):
            for k in range(6):
                if m["label"] == "safe":
                    c = "S"
                else:
                    c = "PPOUUU"[k]
                signal = 2.0 if c == "U" else 0.0
                hidden.append(rng.standard_normal(8) + signal)
                classes.append(c)
                record.append(index)
                head.append(float(1 / (1 + np.exp(-(signal - 1 + rng.standard_normal())))))
        hidden = np.asarray(hidden, np.float16)
        np.savez(directory / f"features_{split}.npz", hidden=hidden, projection=hidden[:, :4],
                 head_p_unsafe=np.asarray(head, np.float32), classes=np.asarray(classes), record=np.asarray(record, np.int32))
        (directory / f"features_{split}_records.json").write_text(json.dumps(meta))
        with gzip.open(directory / f"runA_{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for m in meta:
                n_pos = 12
                cls = "S" * n_pos if m["label"] == "safe" else "PPPPOOUUUUUU"
                lp = []
                for c in cls:
                    u = 2.5 if c == "U" else -2.5 + rng.standard_normal()
                    lp.append([0.0, u, -4.0])
                handle.write(json.dumps({"sample_id": m["sample_id"], "family": m["family"], "label": m["label"],
                                         "language": m["language"], "slot": m["slot"], "response_style": m["response_style"],
                                         "classes": cls, "logprobs": lp}) + "\n")


class EndToEndTests(unittest.TestCase):
    def test_fit_probe_and_rules(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_fake_dump(directory, rng)
            subprocess.run([sys.executable, str(HERE / "fit_probe.py"), str(directory), "--bootstrap", "20"],
                           check=True, capture_output=True)
            probe = json.loads((directory / "probe_v1_results.json").read_text())
            self.assertEqual(set(probe["dev"]), {"head", "probe_hidden", "probe_projection"})
            self.assertGreater(probe["dev"]["probe_hidden"]["auc_post_vs_restating_risk_reasoning"], 0.8)
            self.assertIn("probe_hidden_minus_head", probe["dev_bootstrap_auc_post_vs_restating"])
            out = directory / "rules.json"
            subprocess.run([sys.executable, str(HERE / "analyze_runA_rules.py"), str(directory), "--bootstrap", "20",
                            "--output", str(out)], check=True, capture_output=True)
            rules = json.loads(out.read_text())
            self.assertTrue(rules["rules"])
            some = next(iter(rules["rules"].values()))
            self.assertIn("slot0:risk_reasoning", some["runA_dev_breakdown"]["safe_fpr_by_slot_style"])
            self.assertEqual(rules["streams"]["dev"], {"safe": 40, "unsafe": 40})


try:
    import torch
except ImportError:  # the executor's Mac .venv has torch; this container may not
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class DumpLoopTests(unittest.TestCase):
    """Drive dump_runA_l20.dump_records with a fake runtime: one char = one token, 8-d hidden."""

    def build(self, permute=False):
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.heads = torch.nn.ModuleDict({role: torch.nn.ModuleDict({
            "projection": torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.SiLU()),
            "risk": torch.nn.Linear(4, 3)}) for role in ("user", "assistant")})

        def validated_prefixes(ids):
            n = len(ids)
            physical = 32 * -(-n // 32)
            probs = []
            with torch.no_grad():
                for start in range(0, physical, 32):
                    hidden = torch.randn(1, 32, 8)
                    p = torch.softmax(model.heads["assistant"]["risk"](model.heads["assistant"]["projection"](hidden)), -1)
                    probs.extend(p[0].tolist())
            probs = probs[:n]
            if permute:
                probs = [[b, a, c] for a, b, c in probs]
            return ({"assistant": probs, "user": probs},
                    {"native_tokens": n, "forward_tokens": physical, "padding_tokens": physical - n,
                     "forward_calls": physical // 32, "eager_calls": physical // 32, "graph_calls": 0})

        class Encoding:
            def __init__(self, text):
                self.ids = [ord(c) % 5000 for c in text]
                self.offsets = [(i, i + 1) for i in range(len(text))]

        class Fast:
            def encode(self, text, add_special_tokens=False):
                return Encoding(text)

        return model, validated_prefixes, Fast()

    def write_input(self, path):
        rows = []
        for split in ("calibration", "dev", "train"):
            for index, label in enumerate(("safe", "unsafe", "safe", "unsafe")):
                content = "x" * 40
                rows.append({"sample_id": f"{split}-{index}", "task_key": f"{split}-k", "family": f"{split}-f",
                             "split": split, "language": "zh", "label": label, "prompt_label": "unsafe" if index < 2 else "safe",
                             "index": index, "response_style": "risk_reasoning" if index == 0 else "other",
                             "response_format": "answer", "onset_char": 10 if label == "unsafe" else None,
                             "onset_end_char": 20 if label == "unsafe" else None,
                             "messages": [{"role": "user", "content": "q" * 30}, {"role": "assistant", "content": content}]})
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_dump_records_writes_aligned_outputs(self):
        import dump_runA_l20 as dump
        model, validated, fast = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.write_input(tmp / "in.jsonl")
            report = {}
            dump.dump_records(torch, model, validated, {}, fast, None, tmp / "in.jsonl", tmp, 0, report, lambda: None)
            self.assertEqual(report["records"], {"calibration": 4, "dev": 4, "train": 4})
            self.assertLess(report["max_prob_diff"], 1e-5)
            data = np.load(tmp / "features_dev.npz")
            self.assertEqual(data["hidden"].shape[1], 8)
            self.assertEqual(data["projection"].shape[1], 4)
            self.assertEqual(len(data["classes"]), len(data["record"]))
            with gzip.open(tmp / "runA_dev.jsonl.gz", "rt") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows[1]["logprobs"]), 40)
            self.assertEqual(rows[1]["classes"], "P" * 10 + "O" * 9 + "U" * 21)

    def test_misaligned_engine_is_rejected(self):
        import dump_runA_l20 as dump
        model, validated, fast = self.build(permute=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.write_input(tmp / "in.jsonl")
            with self.assertRaises(ValueError):
                dump.dump_records(torch, model, validated, {}, fast, None, tmp / "in.jsonl", tmp, 0, {}, lambda: None)


class InputTests(unittest.TestCase):
    def test_select_keeps_all_dev_calibration_and_sampled_train(self):
        rows = []
        for w in range(12):
            split = ("train", "train", "dev", "calibration")[w % 4]
            for index in range(4):
                rows.append({"task_key": f"k{w}", "family": f"f{w}", "split": split, "index": index,
                             "label": "safe" if index in (0, 2) else "unsafe", "sample_id": f"k{w}-{index}"})
        chosen = select(rows, train_words=2)
        words = {r["task_key"] for r in chosen}
        self.assertEqual(sum(1 for w in words if int(w[1:]) % 4 in (2, 3)), 6)   # all dev + calibration
        self.assertEqual(sum(1 for w in words if int(w[1:]) % 4 in (0, 1)), 2)   # sampled train
        self.assertEqual(len(chosen), 8 * 4)
        self.assertEqual(select(rows, 2), chosen)


if __name__ == "__main__":
    unittest.main()
