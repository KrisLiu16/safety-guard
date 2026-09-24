"""Small synthetic CPU fixtures; no neural models, tokenizers or GPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("canonical_cal", HERE / "calibrate_canonical_l20.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
audit = cal.load_local("evaluate_final_l20", HERE / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
metrics = audit.exact_module("stream_metrics", HERE / "stream_metrics.py", audit.METRICS_SHA)

CONTRACT = {"name": "canonical32-v1", "block_tokens": 32, "window": 512,
            "backbone_layers": 24, "partial_cache": "not_committed",
            "readout": "all_32_positions_both_roles", "max_input_tokens": 8192,
            "risk_labels": ["safe", "unsafe", "controversial"], "pad_token_id": 0}


class FakeRuntime:
    engine_metadata = {"execution_contract": CONTRACT}

    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def classify_prefixes(self, ids):
        self.calls.append(list(ids))
        n = len(ids)
        physical = (n + 31) // 32 * 32
        values = {"user": [[.25, .75, 0.] for _ in ids],
                  "assistant": [[1 - value / 100, value / 100, 0.] for value in ids]}
        result = {"native_tokens": n, "forward_tokens": physical, "real_forward_tokens": n,
                  "padding_tokens": physical - n, "forward_calls": physical // 32,
                  "generated_tokens": 0, "execution_contract": copy.deepcopy(CONTRACT),
                  "probabilities_by_role_all_tokens": values,
                  "engine_execution": {"eager_calls": physical // 32, "graph_calls": 0,
                      "forward_tokens": physical, "real_forward_tokens": n,
                      "padding_tokens": physical - n, "initialization_included": False}}
        if self.transform:
            self.transform(result)
        return result


def source(name="row", role="assistant"):
    return {"sample_id": name, "family": "family-" + name, "language": "zh", "target_role": role,
            "source_label": "safe", "ids": [90, 10, 20, 30], "target_token_positions": [1, 2, 3],
            "text_cut_views": [{"cut_chars": 1, "ids": [90, 10]},
                               {"cut_chars": 2, "ids": [90, 77]},
                               {"cut_chars": 3, "ids": [90, 77]},
                               {"cut_chars": 4, "ids": [90, 10, 20, 30]}]}


def simple_data(prefix, *, safe=.2, unsafe=.8):
    result = []
    for stratum in audit.STRATA:
        language, role = stratum.split("/")
        for label, p in (("safe", safe), ("unsafe", unsafe)):
            result.append({"sample_id": prefix + stratum + label, "family": prefix + stratum + label,
                "language": language, "target_role": role, "source_label": label,
                "endpoint_p_unsafe": p, "native_target_max_p_unsafe": p,
                "text_cut_max_p_unsafe": p, "stream_max_p_unsafe": p})
    return result


class CanonicalCalibrationTests(unittest.TestCase):
    def test_import_cpu_safe(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_exact_prefix_reuse_bpe_retokenization_and_role_boundary(self):
        runtime = FakeRuntime()
        row = source()
        observed, counts = cal.observations(runtime, [row])
        audit.validate_observations(observed, [row], metrics)
        self.assertEqual(runtime.calls, [[90, 10, 20, 30], [90, 77]])
        self.assertEqual(observed[0]["native_target_max_p_unsafe"], .3)
        self.assertEqual(observed[0]["stream_max_p_unsafe"], .77)
        self.assertEqual(observed[0]["endpoint_p_unsafe"], .3)
        self.assertEqual(counts["forward_tokens"], 64)
        self.assertEqual(counts["native_tokens"], 6)
        self.assertEqual(counts["padding_tokens"], 58)
        self.assertEqual(counts["exact_prefix_cuts_reused"], 1)
        self.assertEqual(counts["complete_input_cuts_reused"], 1)
        self.assertEqual(counts["duplicate_cut_calls_reused"], 1)

    def test_identical_ids_different_roles_do_not_share_scores(self):
        runtime = FakeRuntime()
        observed, _ = cal.observations(runtime, [source("a", "assistant"), source("u", "user")])
        self.assertEqual(len(runtime.calls), 4)
        self.assertEqual([r["endpoint_p_unsafe"] for r in observed], [.3, .75])

    def test_nan_even_in_non_target_head_or_nonfinal_position_rejected(self):
        def bad(result):
            result["probabilities_by_role_all_tokens"]["user"][0][1] = float("nan")
        with self.assertRaises(ValueError):
            cal.observations(FakeRuntime(bad), [source()])

    def test_accounting_contract_or_padding_rows_cannot_false_pass(self):
        for name in ("native_tokens", "forward_tokens", "padding_tokens", "forward_calls", "generated_tokens"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cal.observations(FakeRuntime(lambda r: r.__setitem__(name, r[name] + 1)), [source()])
        with self.assertRaises(ValueError):
            cal.observations(FakeRuntime(lambda r: r["probabilities_by_role_all_tokens"]["assistant"].append([1., 0., 0.])), [source()])
        with self.assertRaises(ValueError):
            cal.observations(FakeRuntime(lambda r: r["execution_contract"].__setitem__("block_tokens", 8)), [source()])

    def test_input_limits_before_runtime_and_target_endpoint(self):
        runtime = FakeRuntime()
        for ids in ([], [True], [-1], [0] * 8193):
            with self.subTest(n=len(ids)), self.assertRaises(ValueError):
                cal.validated_prefixes(runtime, ids, CONTRACT)
        self.assertEqual(runtime.calls, [])
        row = source()
        row["target_token_positions"] = [1, 2]
        with self.assertRaises(ValueError):
            cal.observations(runtime, [row])

    def test_calibration_uses_cal_only_strict_ties_and_initial_retention(self):
        calrows, dev = simple_data("c"), simple_data("d", safe=.2, unsafe=.2)
        result = metrics.evaluate(calrows, dev, baseline_whole_macro_recall=1.)
        self.assertEqual(result["whole"]["strata"]["zh/user"]["threshold"], .2)
        self.assertEqual(result["stream"]["macro_fpr"], 0.)
        self.assertEqual(result["stream"]["macro_recall"], 0.)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["selection_gates"]["whole_macro_recall_drop_at_most_2pp"])
        for row in calrows:
            for field in metrics.SCORE_FIELDS.values():
                row[field] = 1.
            row["native_target_max_p_unsafe"] = row["text_cut_max_p_unsafe"] = 1.
        result = metrics.evaluate(calrows, simple_data("e"))
        self.assertEqual(result["stream"]["strata"]["en/assistant"]["threshold"], 1.)

    def test_gate_failure_still_returns_completed_artifact_without_selecting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("manifest.json", "calibration.jsonl", "dev.jsonl",
                         "calibration_predictions.jsonl", "dev_predictions.jsonl"):
                (root / name).write_text("{}\n")
            args = SimpleNamespace(data_root=root)
            with mock.patch.object(metrics, "is_better", side_effect=AssertionError("Must not select")):
                artifact = cal.calibration_artifact(audit, metrics, simple_data("c"),
                    simple_data("d", safe=.9, unsafe=.8), digest="a" * 64, directory=root, args=args,
                    baseline=1., contract=CONTRACT, receipts={}, metadata={}, sources={})
            self.assertEqual(artifact["status"], "completed")
            self.assertFalse(artifact["canonical_quality_gate_pass"])
            self.assertFalse(artifact["selection_changed"])
            self.assertEqual(artifact["thresholds"]["stream"]["zh/user"], .2)
            self.assertEqual(artifact["threshold_comparison"], ">")
            json.dumps(artifact, allow_nan=False)

    def test_family_overlap_is_not_independent_calibration(self):
        calibration, development = simple_data("c"), simple_data("d")
        development[0]["family"] = calibration[0]["family"]
        with self.assertRaises(ValueError):
            metrics.evaluate(calibration, development)

    def test_failed_observation_keeps_completed_rows_not_success_artifact(self):
        runtime = FakeRuntime()
        row = source("broken")
        row["ids"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            with self.assertRaises(ValueError):
                cal.observations(runtime, [source(), row], prediction_path=path)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertFalse((path.parent / "calibration.json").exists())

    def test_lazy_library_imports_allowed_but_source_mutation_rejected(self):
        before = {"runtime_source_sha256": {"a": "sha"}, "model_loader_source_sha256": {},
                  "source_paths": {"a": "path"}, "third_party_source_sha256": {"lib": "sha1"}}
        after = copy.deepcopy(before)
        after["third_party_source_sha256"]["late"] = "new"
        self.assertTrue(cal.sources_unchanged(before, after))
        after["third_party_source_sha256"]["lib"] = "wrong"
        self.assertFalse(cal.sources_unchanged(before, after))

    def test_completed_training_gate_precedes_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            audit.write_json(training / "summary.json", {"status": "running"})
            with mock.patch.object(cal, "prepare_modules", side_effect=AssertionError("No GPU load")):
                result = cal.main(["--training-output", str(training), "--output", str(root / "out")])
            self.assertEqual(result, 1)
            report = audit.read_json(root / "out/audit.json")
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["integrity_pass"])
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
