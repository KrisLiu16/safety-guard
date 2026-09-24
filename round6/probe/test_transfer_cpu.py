"""CPU checks for T014 (probe transfer): refit reproduction, the three passes on a fake engine, and the analysis."""
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
from test_probe_cpu import write_fake_dump  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def fake_probe_dir(directory):
    """T004-shaped features plus a recorded probe_v1_results.json produced by the real fitter."""
    write_fake_dump(directory, np.random.default_rng(3))
    subprocess.run([sys.executable, str(HERE / "fit_probe.py"), str(directory), "--bootstrap", "5"],
                   check=True, capture_output=True)


class RefitTests(unittest.TestCase):
    def test_refit_reproduces_recorded_probe(self):
        import transfer_l20
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake_probe_dir(tmp)
            probe = transfer_l20.refit_probe(tmp)
            self.assertEqual(probe["w"].shape, (8,))
            results = json.loads((tmp / "probe_v1_results.json").read_text())
            results["scorers"]["probe_hidden"]["calibration_auc"] -= 0.01      # a different probe must be rejected
            (tmp / "probe_v1_results.json").write_text(json.dumps(results))
            with self.assertRaises(ValueError):
                transfer_l20.refit_probe(tmp)


@unittest.skipIf(torch is None, "torch not installed")
class PassTests(unittest.TestCase):
    def build(self):
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.heads = torch.nn.ModuleDict({role: torch.nn.ModuleDict({
            "projection": torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.SiLU()),
            "risk": torch.nn.Linear(4, 3)}) for role in ("user", "assistant")})

        def validated(ids):
            n, physical = len(ids), 32 * -(-len(ids) // 32)
            probs = []
            with torch.no_grad():
                for _ in range(0, physical, 32):
                    h = torch.randn(1, 32, 8)
                    probs.extend(torch.softmax(model.heads["assistant"]["risk"](model.heads["assistant"]["projection"](h)), -1)[0].tolist())
            return ({"assistant": probs[:n], "user": probs[:n]},
                    {"native_tokens": n, "forward_tokens": physical, "graph_calls": 0})

        probe = {"w": np.ones(8, np.float32), "b": 0.0, "mean": np.zeros(8, np.float32), "std": np.ones(8, np.float32)}
        return model, validated, probe

    def test_three_passes_write_aligned_scores(self):
        import transfer_l20

        class Encoding:
            def __init__(self, text):
                self.ids = [ord(c) % 5000 for c in text]
                self.offsets = [(i, i + 1) for i in range(len(text))]

        class Fast:
            def encode(self, text, add_special_tokens=False):
                return Encoding(text)

        class Official:
            SPLITS = ("thinking",)
            EXPECTED_UNIQUE = 2

            @staticmethod
            def sequence_key(row):
                return tuple(row["ids"])

        model, validated, probe = self.build()
        forward = transfer_l20.ProbeForward(torch, model, validated, probe)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{"sample_id": f"s{i}", "task_key": "k", "family": "f", "split": ("calibration", "dev", "train")[i % 3],
                     "language": "zh", "label": ("safe", "unsafe")[i % 2], "prompt_label": "unsafe", "index": i % 4,
                     "response_style": "x", "onset_char": 5, "onset_end_char": 9,
                     "messages": [{"role": "user", "content": "q" * 20}, {"role": "assistant", "content": "a" * 30}]}
                    for i in range(6)]
            (tmp / "in.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            counts = transfer_l20.run_runA(forward, Fast(), None, tmp / "in.jsonl", tmp, 0)
            self.assertEqual(counts, {"runA_calibration": 2, "runA_dev": 2})
            data = {"calibration": [], "dev": [{"sample_id": "p1", "family": "f", "language": "zh", "target_role": "assistant",
                                                "source_label": "safe", "ids": list(range(40)), "target_token_positions": list(range(10, 40))},
                                               {"sample_id": "p2", "family": "f", "language": "zh", "target_role": "user",
                                                "source_label": "safe", "ids": list(range(40)), "target_token_positions": [1]}]}
            self.assertEqual(transfer_l20.run_prefix_v2(forward, data, tmp, 0), {"prefix_v2_dev": 1})
            official_data = {"thinking": [{"ids": list(range(50)), "eval_start_index": 10, "sample_id": f"o{i}", "row_index": i,
                                           "unique_id": i, "label": "Safe"} for i in range(3)]
                             + [{"ids": list(range(60)), "eval_start_index": 5, "sample_id": "o9", "row_index": 9,
                                 "unique_id": 9, "label": "Unsafe"}]}
            self.assertEqual(transfer_l20.run_official(forward, Official, official_data, tmp, 0), {"official_unique_sequences": 2})
            forward.close()
            with gzip.open(tmp / "runA_dev.jsonl.gz", "rt") as handle:
                record = json.loads(handle.readline())
            self.assertEqual(len(record["logprobs"]), len(record["probe_logits"]))
            with gzip.open(tmp / "official_sequences.jsonl.gz", "rt") as handle:
                seqs = [json.loads(line) for line in handle]
            self.assertEqual([len(s["probe_logits"]) for s in seqs], [40, 55])
            self.assertLess(forward.max_prob_diff, 1e-5)


class AnalysisTests(unittest.TestCase):
    def test_end_to_end_on_synthetic_streams(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def stream(label, restating=False):
                n = 20
                cls = "S" * n if label == "safe" else "P" * 6 + "OO" + "U" * 12
                head = [[0.0, (2.5 if c == "U" else (1.5 if restating else -3.0)) + rng.normal(0, .5), -4.0] for c in cls]
                probe = [(6.0 if c == "U" else -6.0) + rng.normal(0, .5) for c in cls]
                return cls, head, probe

            for split in ("calibration", "dev"):
                with gzip.open(tmp / f"runA_{split}.jsonl.gz", "wt") as out:
                    for i in range(80):
                        slot = i % 4
                        label = "unsafe" if slot in (1, 3) else "safe"
                        cls, head, probe = stream(label, restating=slot == 0)
                        out.write(json.dumps({"sample_id": f"{split}{i}", "family": f"f{i // 4}", "language": "zh" if i % 8 < 4 else "en",
                                              "label": label, "slot": slot, "response_style": "risk_reasoning" if slot == 0 else "other",
                                              "classes": cls, "logprobs": head, "probe_logits": probe}) + "\n")
            with gzip.open(tmp / "prefix_v2_dev.jsonl.gz", "wt") as out:
                for i in range(20):
                    label = ("safe", "unsafe")[i % 2]
                    _, head, probe = stream(label)
                    out.write(json.dumps({"sample_id": f"p{i}", "family": f"pf{i}", "language": "zh", "target_role": "assistant",
                                          "source_label": label, "logprobs": head, "probe_logits": probe}) + "\n")
            with gzip.open(tmp / "official_sequences.jsonl.gz", "wt") as seq, gzip.open(tmp / "official_rows.jsonl.gz", "wt") as rows:
                for i in range(20):
                    label = ("safe", "unsafe")[i % 2]
                    _, head, probe = stream(label, restating=label == "safe")
                    seq.write(json.dumps({"sequence_id": f"q{i}", "logprobs": head, "probe_logits": probe}) + "\n")
                    rows.write(json.dumps({"sample_id": f"o{i}", "split": "thinking", "label": label.capitalize(),
                                           "sequence_id": f"q{i}"}) + "\n")
            out = tmp / "result.json"
            subprocess.run([sys.executable, str(HERE / "analyze_transfer.py"), str(tmp), "--bootstrap", "20", "--output", str(out)],
                           check=True, capture_output=True)
            result = json.loads(out.read_text())
            self.assertEqual(set(result["scorers"]), {"head", "probe"})
            probe_auc = result["scorers"]["probe"]["stream_auc_max_score"]["official_thinking"]
            head_auc = result["scorers"]["head"]["stream_auc_max_score"]["official_thinking"]
            self.assertGreater(probe_auc, head_auc)
            self.assertIn("runA_dev_breakdown", result["scorers"]["probe"]["rules"]["threshold"])


if __name__ == "__main__":
    unittest.main()
