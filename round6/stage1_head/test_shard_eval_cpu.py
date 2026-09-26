"""shard_eval.py on synthetic data (CPU; needs torch): N sharded eval_head_l20 loops merged back equal the
single-process files line for line, and the checks catch a record scored twice or a failed shard."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe"))
import shard_eval as se  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from test_stage1_cpu import Fast, fake_model, runA_rows  # noqa: E402


def steady_engine(model):
    """validated_prefixes whose hidden states depend only on the ids, so any process gets the same probabilities."""
    def validated(ids):
        n, physical = len(ids), 32 * -(-len(ids) // 32)
        gen = torch.Generator().manual_seed(zlib.crc32(json.dumps(ids).encode()))
        h = torch.randn(1, physical, 16, generator=gen)
        probs = {}
        with torch.no_grad():
            for role in ("assistant", "user"):
                head = model.heads[role]
                probs[role] = torch.softmax(head["risk"](head["projection"](h)), -1)[0].tolist()[:n]
        return probs, {"native_tokens": n, "forward_tokens": physical, "graph_calls": 0, "forward_calls": physical // 32}
    return validated


class Official:
    SPLITS = ("thinking", "answer")
    EXPECTED_UNIQUE = 4

    @staticmethod
    def sequence_key(row):
        return tuple(row["ids"])


def official_rows():
    seqs = [list(range(30 + 7 * i)) for i in range(4)]
    rows = {"thinking": [], "answer": []}
    for j, s in enumerate([0, 1, 1, 2, 3, 0, 2]):          # repeats: rows share a sequence
        split = "thinking" if j < 4 else "answer"
        rows[split].append({"ids": seqs[s], "eval_start_index": 3, "sample_id": f"o{j}", "row_index": j,
                            "unique_id": j, "label": "Safe"})
    return rows


def prefix_data():
    return {s: [{"sample_id": f"{s}{k}", "family": "f", "language": "zh", "target_role": ("assistant", "user")[k % 2],
                 "source_label": "safe", "ids": list(range(40 + k)), "target_token_positions": list(range(20, 40))}
                for k in range(7)] for s in ("calibration", "dev")}


@unittest.skipIf(torch is None, "torch not installed")
class ShardEvalTests(unittest.TestCase):
    def run_eval(self, out, shard, role="assistant"):
        import eval_head_l20 as ev
        model = fake_model()
        capture = ev.ProbsCapture(torch, model, steady_engine(model), role=role)
        out.mkdir(parents=True)
        order = {} if shard else None
        rows = runA_rows("calibration", 2) + runA_rows("dev", 3)
        if role == "assistant":
            counts = ev.eval_runA(capture, lambda: capture.last, Fast(), None, rows, out, shard=shard, order=order)
            counts.update(ev.eval_prefix_v2(capture, lambda: capture.last, prefix_data(), out, shard=shard, order=order))
            counts.update(ev.eval_official(capture, lambda: capture.last, Official, official_rows(), out,
                                           shard=shard, order=order))
        else:
            counts = ev.eval_runA_prompts(capture, lambda: capture.last, Fast(), None, rows, out, shard=shard,
                                          order=order)
            counts.update(ev.eval_prefix_v2(capture, lambda: capture.last, prefix_data(), out, role="user",
                                            shard=shard, order=order))
        if shard:
            (out / "order.json").write_text(json.dumps(order))
        capture.close()
        return counts

    def test_merged_shards_equal_single_process(self):
        for role in ("assistant", "user"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                single = self.run_eval(tmp / "single", None, role)
                parts = [self.run_eval(tmp / f"s{i}", (i, 3), role) for i in range(3)]
                se.merge_files([tmp / f"s{i}" for i in range(3)], tmp / "merged")
                compared = se.compare_dirs(tmp / "merged", tmp / "single")
                self.assertTrue(compared)
                self.assertTrue(all(c["identical"] for c in compared.values()), compared)
                total = {}
                for c in parts:
                    for k, v in c.items():
                        total[k] = total.get(k, 0) + v
                if role == "assistant":
                    self.assertEqual(total.pop("official_unique_sequences"), 3 * 4)
                    self.assertEqual(single.pop("official_unique_sequences"), 4)
                self.assertEqual(total, dict(single))

    def test_record_scored_twice_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.run_eval(tmp / "s0", (0, 2))
            self.run_eval(tmp / "s1", (0, 2))                       # same shard twice
            with self.assertRaises(ValueError):
                se.merge_files([tmp / "s0", tmp / "s1"], tmp / "merged")


class ReportTests(unittest.TestCase):
    def report(self, ok=True, weights="w"):
        return {"status": "completed", "integrity_pass": ok, "role": "assistant", "elapsed_seconds": 10.0,
                "max_prob_diff": 1e-7, "totals": {"sequences": 3, "forward_tokens": 96}, "shard": [0, 2],
                "variants": {"checkpoint": {"weights_sha256": weights, "files": {},
                                            "counts": {"runA_dev": 2, "official_unique_sequences": 5}}}}

    def test_merge_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "eval_checkpoint").mkdir()
            merged = se.merge_reports([self.report(), self.report()], Path(tmp))
        self.assertEqual(merged["totals"], {"sequences": 6, "forward_tokens": 192})
        self.assertEqual(merged["variants"]["checkpoint"]["counts"], {"runA_dev": 4, "official_unique_sequences": 5})
        self.assertNotIn("shard", merged)
        with self.assertRaises(ValueError):
            se.merge_reports([self.report(), self.report(ok=False)], Path("."))
        with self.assertRaises(ValueError):
            se.merge_reports([self.report(), self.report(weights="x")], Path("."))

    def test_compare_reports_logprob_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir()
            b.mkdir()
            for d, value in ((a, -0.5), (b, -0.5002)):
                with gzip.open(d / "x.jsonl.gz", "wb") as handle:
                    handle.write((json.dumps({"id": 1, "logprobs": [[value, -1.0, -2.0]]}) + "\n").encode())
                    handle.write((json.dumps({"id": 2, "logprobs": [[-0.1, -1.0, -2.0]]}) + "\n").encode())
            c = se.compare_dirs(a, b)["x.jsonl.gz"]
        self.assertEqual((c["identical"], c["identical_lines"], c["other_field_differences"]), (False, 1, 0))
        self.assertAlmostEqual(c["max_abs_logprob_diff"], 0.0002, places=6)



class AutotunePinTests(unittest.TestCase):
    def fake_module(self):
        import types

        class Config:
            def __init__(self, kwargs, num_warps=4, num_stages=2):
                self.kwargs, self.num_warps, self.num_stages, self.num_ctas, self.maxnreg = kwargs, num_warps, num_stages, 1, None

        def kernel():
            pass

        class Autotuner:
            benches = 0

            def __init__(self, configs, pick):
                self.fn = self.base_fn = kernel
                self.configs, self.cache, self.pick = configs, {}, pick

            def run(self, key):
                if key not in self.cache:
                    Autotuner.benches += 1
                    self.cache[key] = self.configs[self.pick]
                return self.cache[key]
        return types.SimpleNamespace(Autotuner=Autotuner, Config=Config)

    def test_record_then_pin(self):
        import importlib
        import autotune_pin as ap
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "autotune.json"
            mod = self.fake_module()
            ap = importlib.reload(ap)
            ap.install("record", path, mod, write_at_exit=False)
            configs = [mod.Config({"BT": 64}), mod.Config({"BT": 32}, num_warps=8)]
            quiet = mod.Autotuner(configs, pick=1)                        # what a lone process picks
            quiet.run((128, "torch.bfloat16"))
            path.write_text(json.dumps(ap.dump()))
            mod = self.fake_module()
            ap = importlib.reload(ap)
            ap.install("pin", path, mod)
            busy = mod.Autotuner([mod.Config({"BT": 64}), mod.Config({"BT": 32}, num_warps=8)], pick=0)  # contended pick
            self.assertEqual(busy.run((128, "torch.bfloat16")).kwargs, {"BT": 32})                     # pinned
            self.assertEqual(mod.Autotuner.benches, 0)
            busy.run((256, "torch.bfloat16"))                                                        # not in the file
            state = ap.state()
            self.assertEqual((state["mode"], state["pinned"], len(state["unpinned"])), ("pin", 1, 1))

if __name__ == "__main__":
    unittest.main()
