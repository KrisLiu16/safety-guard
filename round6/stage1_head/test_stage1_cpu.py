"""Stage-1 chain on synthetic data (CPU; needs torch and numpy): input, cache, training, evaluation loops, analysis."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe"))

try:
    import torch
except ImportError:
    torch = None


def fake_model():
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.heads = torch.nn.ModuleDict({role: torch.nn.ModuleDict({
        "projection": torch.nn.Sequential(torch.nn.Linear(16, 512), torch.nn.LayerNorm(512), torch.nn.SiLU()),
        "risk": torch.nn.Linear(512, 3), "category": torch.nn.Linear(512, 8)}) for role in ("user", "assistant")})
    return model


def fake_engine(model, signal=None):
    """validated_prefixes over random 16-d hidden states; tokens listed in `signal` get a shifted hidden state."""
    def validated(ids):
        n, physical = len(ids), 32 * -(-len(ids) // 32)
        probs = {"assistant": [], "user": []}
        with torch.no_grad():
            for start in range(0, physical, 32):
                h = torch.randn(1, 32, 16)
                if signal:
                    for k in range(32):
                        if start + k < n and ids[start + k] in signal:
                            h[0, k, 0] += 4.0
                for role, values in probs.items():          # each role's own head on the same hidden states
                    head = model.heads[role]
                    values.extend(torch.softmax(head["risk"](head["projection"](h)), -1)[0].tolist())
        return ({role: values[:n] for role, values in probs.items()},
                {"native_tokens": n, "forward_tokens": physical, "graph_calls": 0, "forward_calls": physical // 32})
    return validated


class Encoding:
    def __init__(self, text):
        self.ids = [ord(c) % 5000 for c in text]
        self.offsets = [(i, i + 1) for i in range(len(text))]


class Fast:
    def encode(self, text, add_special_tokens=False):
        return Encoding(text)


def runA_rows(split, n_words):
    rows = []
    for w in range(n_words):
        for index, label in enumerate(("safe", "unsafe", "safe", "unsafe")):
            content = "安" * 10 + ("害" * 20 if label == "unsafe" else "安" * 20)
            rows.append({"sample_id": f"{split}{w}-{index}", "task_key": f"{split}{w}", "family": f"{split}f{w}", "split": split,
                         "language": ("zh", "en")[w % 2], "label": label, "prompt_label": "unsafe" if index < 2 else "safe",
                         "index": index, "response_style": ("risk_reasoning", "compliance", "helpful_answer", "drift")[index],
                         "response_format": "answer", "onset_char": 10 if label == "unsafe" else None,
                         "onset_end_char": 12 if label == "unsafe" else None,
                         "messages": [{"role": "user", "content": "问" * 10}, {"role": "assistant", "content": content}]})
    return rows


class InputTests(unittest.TestCase):
    def test_train_sample_plus_all_calibration_no_dev(self):
        from make_stage1_input import select
        rows = runA_rows("train", 10) + runA_rows("calibration", 3) + runA_rows("dev", 3)
        chosen = select(rows, 4)
        self.assertEqual({r["split"] for r in chosen}, {"train", "calibration"})
        self.assertEqual(len({r["task_key"] for r in chosen if r["split"] == "train"}), 4)
        self.assertEqual(len({r["task_key"] for r in chosen if r["split"] == "calibration"}), 3)


@unittest.skipIf(torch is None, "torch not installed")
class CacheTests(unittest.TestCase):
    def test_cache_sources_labels_and_shards(self):
        import cache_features_l20 as cache
        model = fake_model()
        capture = cache.Capture(torch, model, fake_engine(model))
        data = {"train": [
                    {"sample_id": "p1", "family": "f1", "language": "zh", "target_role": "assistant", "source_label": "safe",
                     "weight": 1.0, "ids": list(range(40)), "anchors": [{"token_end_exclusive": 20, "label": 0, "confidence": 0.1}],
                     "augmentation": {"ids": list(range(45)), "anchors": [{"token_end_exclusive": 45, "label": 0, "confidence": 1.0}]}},
                    {"sample_id": "p2", "family": "f2", "language": "en", "target_role": "assistant", "source_label": "unsafe",
                     "weight": 0.2, "ids": list(range(30)), "anchors": [],
                     "augmentation": {"ids": list(range(35)), "anchors": [{"token_end_exclusive": 35, "label": 1, "confidence": 1.0}]}},
                    {"sample_id": "p3", "family": "f3", "language": "en", "target_role": "user", "source_label": "safe",
                     "weight": 1.0, "ids": list(range(30)), "anchors": [], "augmentation": {"ids": [1], "anchors": []}}],
                "calibration": [
                    {"sample_id": "c1", "family": "g1", "language": "zh", "target_role": "assistant", "source_label": "safe",
                     "ids": list(range(50)), "target_token_positions": list(range(20, 50))},
                    {"sample_id": "c2", "family": "g2", "language": "zh", "target_role": "assistant", "source_label": "unsafe",
                     "ids": list(range(50)), "target_token_positions": list(range(20, 50))}]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cache, "SHARD_POSITIONS", 5):
            tmp = Path(tmp)
            stats = cache.cache_prefix_v2(torch, capture, data, tmp)
            self.assertEqual(stats["train"]["records"], 3)          # p1 original + p1 augmented + p2 augmented; user row skipped
            train = [np.load(p) for p in sorted(tmp.glob("cache_prefix_v2_train_[0-9]*.npz"))]
            labels = np.concatenate([t["label"] for t in train])
            weights = np.concatenate([t["weight"] for t in train])
            self.assertEqual(labels.tolist(), [0, 0, 1])
            self.assertTrue(np.allclose(weights, [0.1, 1.0, 0.2]))
            cal = np.concatenate([np.load(p)["label"] for p in tmp.glob("cache_prefix_v2_calibration_[0-9]*.npz")])
            self.assertEqual(int(cal.sum()), 1)                       # the unsafe endpoint
            self.assertGreater(len(cal), 2)                           # safe interior positions too
            rows = runA_rows("train", 2) + runA_rows("calibration", 1)
            stats = cache.cache_text_rows(torch, capture, Fast(), None, rows, tmp, "runA")
            self.assertEqual(stats["train"]["records"], 8)
            shards = [np.load(p) for p in sorted(tmp.glob("cache_runA_train_[0-9]*.npz"))]
            self.assertGreater(len(shards), 1)                        # small shard size forces several shards
            first = shards[0]
            self.assertEqual(first["hidden"].shape[1], 16)
            self.assertEqual(first["projection"].shape[1], 512)
            self.assertIn(1, np.concatenate([s["label"] for s in shards]).tolist())
        capture.close()

    def test_user_role_prompts(self):
        import cache_features_l20 as cache
        model = fake_model()
        capture = cache.Capture(torch, model, fake_engine(model), role="user")
        rows = runA_rows("train", 2) + runA_rows("calibration", 1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stats = cache.cache_prompt_rows(torch, capture, Fast(), None, rows, tmp, "runA_prompts")
            self.assertEqual((stats["train"]["records"], stats["calibration"]["records"]), (4, 2))   # 2 prompts per word
            records = json.loads((tmp / "cache_runA_prompts_train_records.json").read_text())
            self.assertEqual({r["sample_id"] for r in records},
                             {"train0:prompt:unsafe", "train0:prompt:safe", "train1:prompt:unsafe", "train1:prompt:safe"})
            data = np.load(tmp / "cache_runA_prompts_train_000.npz")
            self.assertEqual(sum(len(r["char_ends"]) for r in records), len(data["label"]))
            self.assertEqual(records[0]["char_ends"][-1], 10)                  # the prompt is 10 characters
            data_user = [{"sample_id": "u1", "family": "f", "language": "zh", "target_role": "user", "source_label": "unsafe",
                          "weight": 1.0, "ids": list(range(30)), "anchors": [],
                          "augmentation": {"ids": list(range(35)), "anchors": [{"token_end_exclusive": 35, "label": 1, "confidence": 1.0}]}},
                         {"sample_id": "a1", "family": "g", "language": "zh", "target_role": "assistant", "source_label": "safe",
                          "weight": 1.0, "ids": list(range(30)), "anchors": [],
                          "augmentation": {"ids": list(range(35)), "anchors": [{"token_end_exclusive": 35, "label": 0, "confidence": 1.0}]}}]
            stats = cache.cache_prefix_v2(torch, capture, {"train": data_user, "calibration": []}, tmp, role="user")
            self.assertEqual(stats["train"]["records"], 1)                    # only the user-role record
        self.assertLess(capture.max_prob_diff, 1e-5)                           # the user head reproduced its runtime
        capture.close()


def write_synthetic_cache(directory, init):
    rng = np.random.default_rng(0)
    torch.save(init, directory / "head_assistant_init.pt")
    for source in ("prefix_v2", "runA"):
        for split, n in (("train", 3000), ("calibration", 800)):
            label = (rng.random(n) < 0.3).astype(np.int8)
            hidden = rng.standard_normal((n, 16)).astype(np.float32)
            hidden[:, 1] += 2.5 * label                                  # the signal the current head does not read
            with torch.no_grad():
                projection = init_projection(init)(torch.tensor(hidden)).numpy()
            np.savez(directory / f"cache_{source}_{split}_000.npz", hidden=hidden.astype(np.float16),
                     projection=projection.astype(np.float16), label=label, weight=np.ones(n, np.float32),
                     record=np.arange(n, dtype=np.int32))


def init_projection(init):
    projection = torch.nn.Sequential(torch.nn.Linear(16, 512), torch.nn.LayerNorm(512), torch.nn.SiLU())
    projection.load_state_dict({k.split(".", 1)[1]: v for k, v in init.items() if k.startswith("projection.")})
    return projection


@unittest.skipIf(torch is None, "torch not installed")
class TrainTests(unittest.TestCase):
    def test_both_variants_improve_and_keep_keys(self):
        init = {k: v.clone() for k, v in fake_model().heads["assistant"].state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_synthetic_cache(tmp, init)
            subprocess.run([sys.executable, str(HERE / "train_head.py"), str(tmp), "--output", str(tmp / "out"),
                            "--epochs", "3", "--batch", "512", "--lr", "3e-3"], check=True, capture_output=True)
            report = json.loads((tmp / "out/train_report.json").read_text())
            self.assertFalse(report["dev_read"])
            for variant in ("risk", "full"):
                v = report["variants"][variant]
                self.assertGreater(v["chosen_calibration_mean_auc"], v["history"][0]["mean"])
                weights = torch.load(tmp / f"out/head_{variant}.pt")
                self.assertEqual(set(weights), set(init))
                self.assertTrue(torch.equal(weights["category.weight"], init["category.weight"]))
                if variant == "risk":
                    self.assertTrue(torch.equal(weights["projection.0.weight"], init["projection.0.weight"]))
                fake_model().heads["assistant"].load_state_dict(weights, strict=True)

    def test_labels3_three_class_targets(self):
        init = {k: v.clone() for k, v in fake_model().heads["assistant"].state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_synthetic_cache(tmp, init)
            labels3 = tmp / "labels3"
            labels3.mkdir()
            rng = np.random.default_rng(1)
            for split in ("train", "calibration"):
                old = np.load(tmp / f"cache_runA_{split}_000.npz")["label"]
                new = np.where(old == 1, np.where(rng.random(len(old)) < 0.5, 1, 2), 0).astype(np.int8)
                weight = np.ones(len(old), np.float32)
                weight[::10] = 0                                     # positions without a target are dropped
                np.savez(labels3 / f"labels3_runA_{split}.npz", label=new, weight=weight)
            subprocess.run([sys.executable, str(HERE / "train_head.py"), str(tmp), "--output", str(tmp / "out"),
                            "--labels3", str(labels3), "--variants", "risk,wide", "--epochs", "2", "--batch", "512",
                            "--lr", "3e-3"], check=True, capture_output=True)
            report = json.loads((tmp / "out/train_report.json").read_text())
            self.assertEqual((report["version"], report["calibration_score"]), ("round6-stage1-head-v2-redline", "cut"))
            self.assertEqual(report["sources_left_out"], ["prefix_v2", "s2", "s5"])   # no labels3 file, no cache
            self.assertEqual(report["train_positions"]["runA"], 2700)
            by_class = report["train_positions_by_class"]["runA"]
            self.assertEqual(sum(by_class.values()), 2700)
            self.assertGreater(by_class["2"], 0)
            self.assertIsNotNone(report["variants"]["risk"]["chosen_calibration_mean_auc"])
            wide = report["variants"]["wide"]                      # T028 diagnostic: its own shape, never deployed
            self.assertEqual((wide["deployable"], wide["changed_keys"]), (False, []))
            self.assertIsNotNone(wide["chosen_calibration_mean_auc"])
            state = torch.load(tmp / "out/head_wide.pt")
            self.assertEqual(tuple(state["projection.0.weight"].shape), (2048, init["projection.0.weight"].shape[1]))


@unittest.skipIf(torch is None, "torch not installed")
class EvalTests(unittest.TestCase):
    def test_eval_loops_use_loaded_weights(self):
        import eval_head_l20 as ev
        model = fake_model()
        new = {k: v + 0.01 for k, v in model.heads["assistant"].state_dict().items()}
        model.heads["assistant"].load_state_dict(new, strict=True)
        capture = ev.ProbsCapture(torch, model, fake_engine(model))

        class Official:
            SPLITS = ("thinking",)
            EXPECTED_UNIQUE = 1

            @staticmethod
            def sequence_key(row):
                return tuple(row["ids"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            counts = ev.eval_runA(capture, lambda: capture.last, Fast(), None, runA_rows("calibration", 1) + runA_rows("dev", 1), tmp)
            self.assertEqual(counts, {"runA_calibration": 4, "runA_dev": 4})
            with gzip.open(tmp / "runA_calibration.jsonl.gz", "rt") as handle:
                first = json.loads(handle.readline())
            self.assertEqual(first["char_ends"], list(range(1, 31)))     # one per position, offsets in the response
            data = {s: [{"sample_id": f"{s}1", "family": "f", "language": "zh", "target_role": "assistant", "source_label": "safe",
                         "ids": list(range(40)), "target_token_positions": list(range(20, 40))}] for s in ("calibration", "dev")}
            self.assertEqual(ev.eval_prefix_v2(capture, lambda: capture.last, data, tmp), {"prefix_v2_calibration": 1, "prefix_v2_dev": 1})
            official = {"thinking": [{"ids": list(range(30)), "eval_start_index": 5, "sample_id": "o", "row_index": 0,
                                      "unique_id": 0, "label": "Safe"}]}
            self.assertEqual(ev.eval_official(capture, lambda: capture.last, Official, official, tmp), {"official_unique_sequences": 1})
            with gzip.open(tmp / "official_sequences.jsonl.gz", "rt") as handle:
                self.assertEqual(len(json.loads(handle.readline())["logprobs"]), 25)
        self.assertLess(capture.max_prob_diff, 1e-5)
        capture.close()

    def test_user_role_prompts_eval(self):
        import eval_head_l20 as ev
        model = fake_model()
        capture = ev.ProbsCapture(torch, model, fake_engine(model), role="user")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            counts = ev.eval_runA_prompts(capture, lambda: capture.last, Fast(), None,
                                          runA_rows("calibration", 1) + runA_rows("dev", 2), tmp)
            self.assertEqual(counts, {"runA_prompts_calibration": 2, "runA_prompts_dev": 4})   # 2 prompts per word
            with gzip.open(tmp / "runA_prompts_dev.jsonl.gz", "rt") as handle:
                first = json.loads(handle.readline())
            self.assertTrue(first["sample_id"].startswith("dev0:prompt:"))
            self.assertEqual(first["char_ends"], list(range(1, 11)))
            self.assertEqual(len(first["logprobs"]), 10)
        self.assertLess(capture.max_prob_diff, 1e-5)
        capture.close()


class AnalysisTests(unittest.TestCase):
    def test_compare_current_and_variant(self):
        rng = np.random.default_rng(2)

        def write_dir(directory, separation):
            directory.mkdir(parents=True)

            def lp(unsafe_position, restating):
                base = 2.5 if unsafe_position else (2.5 - separation if restating else -3.0)   # restating == harm when separation is 0
                return [0.0, base + rng.normal(0, .5), -4.0]

            for split in ("calibration", "dev"):
                with gzip.open(directory / f"runA_{split}.jsonl.gz", "wt") as out:
                    for i in range(80):
                        slot = i % 4
                        label = "unsafe" if slot in (1, 3) else "safe"
                        cls = "S" * 20 if label == "safe" else "PPPPPPOOUUUUUUUUUUUU"
                        out.write(json.dumps({"sample_id": f"{split}{i}", "family": f"f{i // 4}", "label": label,
                                              "language": "zh" if i % 8 < 4 else "en", "slot": slot,
                                              "response_style": "risk_reasoning" if slot == 0 else "x", "classes": cls,
                                              "logprobs": [lp(c == "U", slot == 0) for c in cls]}) + "\n")
                with gzip.open(directory / f"prefix_v2_{split}.jsonl.gz", "wt") as out:
                    for i in range(40):
                        label = ("safe", "unsafe")[i % 2]
                        out.write(json.dumps({"sample_id": f"p{split}{i}", "family": f"pf{i}", "language": ("zh", "en")[i % 4 // 2],
                                              "target_role": "assistant", "source_label": label,
                                              "logprobs": [lp(label == "unsafe" and k > 5, False) for k in range(12)]}) + "\n")
            with gzip.open(directory / "official_sequences.jsonl.gz", "wt") as seq, gzip.open(directory / "official_rows.jsonl.gz", "wt") as rows:
                for i in range(30):
                    label = ("safe", "unsafe")[i % 2]
                    seq.write(json.dumps({"sequence_id": f"q{i}", "logprobs": [lp(label == "unsafe" and k > 10, label == "safe")
                                                                               for k in range(30)]}) + "\n")
                    rows.write(json.dumps({"sample_id": f"o{i}", "split": "thinking", "label": label.capitalize(),
                                           "sequence_id": f"q{i}"}) + "\n")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_dir(tmp / "current", separation=0.0)
            write_dir(tmp / "eval" / "eval_risk", separation=3.0)
            out = tmp / "compare.json"
            subprocess.run([sys.executable, str(HERE / "analyze_stage1.py"), "--current", str(tmp / "current"), "--eval-dir",
                            str(tmp / "eval"), "--variants", "risk", "--bootstrap", "20", "--output", str(out)],
                           check=True, capture_output=True)
            result = json.loads(out.read_text())
            self.assertEqual(set(result["scorers"]), {"current", "risk"})
            self.assertGreater(result["scorers"]["risk"]["stream_auc_max_score"]["official_thinking"],
                               result["scorers"]["current"]["stream_auc_max_score"]["official_thinking"])
            self.assertFalse(result["dev_used_for_fitting"])


if __name__ == "__main__":
    unittest.main()
