"""Fixed canonical fresh evaluation integrity tests; pure CPU synthetic data."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fresh = module("canonical_fresh", "evaluate_canonical_fresh_l20.py")
cal = module("canonical_cal", "calibrate_canonical_l20.py")
audit = cal.load_local("evaluate_final_l20", HERE / "evaluate_final_l20.py", cal.EVALUATOR_SHA)
metrics = audit.exact_module("stream_metrics", HERE / "stream_metrics.py", audit.METRICS_SHA)
CONTRACT = {"name": "canonical32-v1", "block_tokens": 32, "window": 512,
            "backbone_layers": 24, "partial_cache": "not_committed",
            "readout": "all_32_positions_both_roles", "max_input_tokens": 8192,
            "risk_labels": ["safe", "unsafe", "controversial"], "pad_token_id": 0}


def pairs(prefix):
    sources, rows = [], []
    for stratum in audit.STRATA:
        language, role = stratum.split("/")
        for label, p in (("safe", .2), ("unsafe", .8)):
            common = {"sample_id": prefix + stratum + label, "family": prefix + stratum + label,
                      "language": language, "target_role": role, "source_label": label}
            sources.append({**common, "ids": [1, 2], "target_token_positions": [1],
                            "text_cut_views": [{"cut_chars": 2, "ids": [1, 2]}]})
            rows.append({**common, "endpoint_p_unsafe": p, "endpoint_probs": [1 - p, p, 0.],
                         "native_target_max_p_unsafe": p, "native_target_p_unsafe": [p],
                         "target_token_positions": [1], "text_cut_max_p_unsafe": p,
                         "text_cut_observations": [{"cut_chars": 2, "p_unsafe": p}], "stream_max_p_unsafe": p})
    return sources, rows


class Fixture:
    def __init__(self, directory):
        root = Path(directory)
        self.args = SimpleNamespace(calibration_output=root / "canonical", training_output=root / "training",
                                    data_root=root / "data", inference_engine="eager")
        for path in (self.args.calibration_output, self.args.training_output, self.args.data_root):
            path.mkdir()
        self.summary = {"binding": {"test": "frozen"}, "final_checkpoint_sha256": "f" * 64}
        audit.write_json(self.args.training_output / "summary.json", self.summary)
        audit.write_json(self.args.data_root / "manifest.json", {"test": "frozen"})
        self.data, self.observed = {}, {}
        for split in ("calibration", "dev"):
            self.data[split], self.observed[split] = pairs(split)
            audit.write_rows(self.args.data_root / (split + ".jsonl"), self.data[split])
        self.run = {"status": "completed", "integrity_pass": True, "mode": "calibrate",
                    "selection_changed": False, "selection_read_test": False,
                    "training_summary_sha256": audit.sha(self.args.training_output / "summary.json"),
                    "selected_checkpoint_sha256": self.summary["final_checkpoint_sha256"],
                    "evaluator_sha256": fresh.CALIBRATOR_SHA, "results": {},
                    "device": {"runtime_versions": {"torch": "fixture", "transformers": "fixture"}}}
        for name, digest in (("initial_window", audit.START_SHA), ("selected", self.summary["final_checkpoint_sha256"])):
            directory = self.args.calibration_output / name
            directory.mkdir()
            for split in self.observed:
                audit.write_rows(directory / (split + "_predictions.jsonl"), self.observed[split])
            artifact = cal.calibration_artifact(audit, metrics, self.observed["calibration"], self.observed["dev"],
                digest=digest, directory=directory, args=self.args, baseline=None if name == "initial_window" else 1.,
                contract=CONTRACT, receipts={}, metadata={"execution_contract": CONTRACT, "inference_engine": "eager"},
                sources={"runtime_source_sha256": {}, "model_loader_source_sha256": {}, "third_party_source_sha256": {}})
            artifact.update(training_summary_sha256=self.run["training_summary_sha256"],
                            training_binding=self.summary["binding"],
                            initial_canonical_calibration_sha256=(audit.sha(self.args.calibration_output / "initial_window/calibration.json")
                                                                  if name == "selected" else None))
            audit.write_json(directory / "calibration.json", artifact)
            self.run["results"][name] = {"calibration_artifact_sha256": audit.sha(directory / "calibration.json")}
        audit.write_json(self.args.calibration_output / "audit.json", self.run)

    def load(self):
        return fresh.locked_calibrations(self.args, audit, cal, self.summary, self.data, metrics)

    def mutate_artifact(self, change):
        path = self.args.calibration_output / "selected/calibration.json"
        artifact = audit.read_json(path)
        change(artifact)
        audit.write_json(path, artifact)
        self.run["results"]["selected"]["calibration_artifact_sha256"] = audit.sha(path)
        audit.write_json(self.args.calibration_output / "audit.json", self.run)


class CanonicalFreshTests(unittest.TestCase):
    def test_locked_receipt_never_calls_fit_function(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            with mock.patch.object(metrics, "evaluate", side_effect=AssertionError("No fitting")), \
                 mock.patch.object(metrics, "threshold", side_effect=AssertionError("No fitting")), \
                 mock.patch.object(metrics, "is_better", side_effect=AssertionError("No selection")):
                documents, _ = fixture.load()
                self.assertEqual(documents["selected"]["thresholds"]["stream"]["zh/user"], .2)
                _, observed = pairs("fresh")
                result = audit.evaluate_fixed(observed, documents["selected"]["thresholds"], metrics)
                self.assertEqual(result["stream"]["macro_fpr"], 0.)
                self.assertEqual(result["whole"]["macro_recall"], 1.)

    def test_artifact_hash_or_prediction_bytes_cannot_change(self):
        for name in ("calibration.json", "calibration_predictions.jsonl", "dev_predictions.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                path = fixture.args.calibration_output / "selected" / name
                path.write_text(path.read_text() + "\n")
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_threshold_checkpoint_gate_or_engine_changes_rejected(self):
        changes = (
            lambda a: a["thresholds"]["stream"].__setitem__("zh/user", .9),
            lambda a: a.__setitem__("checkpoint_sha256", "0" * 64),
            lambda a: a.__setitem__("threshold_comparison", ">="),
            lambda a: a.__setitem__("canonical_quality_gate_pass", False),
            lambda a: a["engine_metadata"].__setitem__("inference_engine", "window_cuda_graph"),
            lambda a: a.__setitem__("initial_canonical_calibration_sha256", "0" * 64),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                fixture.mutate_artifact(change)
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_failed_quality_gate_still_permits_fixed_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            observed = copy.deepcopy(fixture.observed)
            # Change selected dev risk probability consistently so safe examples
            # are false positives. Keep the original fixed cal threshold .2.
            for row in observed["dev"]:
                if row["source_label"] == "safe":
                    for key in ("endpoint_p_unsafe", "native_target_max_p_unsafe", "text_cut_max_p_unsafe", "stream_max_p_unsafe"):
                        row[key] = .5
                    row["endpoint_probs"] = [.5, .5, 0.]
                    row["native_target_p_unsafe"] = [.5]
                    row["text_cut_observations"][0]["p_unsafe"] = .5
            path = fixture.args.calibration_output / "selected/dev_predictions.jsonl"
            audit.write_rows(path, observed["dev"])
            changed_metrics = metrics.evaluate(observed["calibration"], observed["dev"], 1.)
            def change(artifact):
                artifact["development_metrics"] = changed_metrics
                artifact["canonical_quality_gate_pass"] = False
                artifact["development_receipt"]["predictions_sha256"] = audit.sha(path)
            fixture.mutate_artifact(change)
            with mock.patch.object(metrics, "evaluate", side_effect=AssertionError("No fitting")):
                documents, _ = fixture.load()
            self.assertFalse(documents["selected"]["artifact"]["canonical_quality_gate_pass"])

    def test_source_alias_or_mutation_is_rejected(self):
        artifact = {"runtime_source_sha256": {"engine.py": "sha"},
                    "model_loader_source_sha256": {"model.py": "sha"},
                    "third_party_source_sha256": {"fla": {"sha256": "1", "path": "original"}}}
        actual = copy.deepcopy(artifact)
        actual["third_party_source_sha256"]["fla"]["path"] = "relocated"
        fresh.matching_sources(actual, artifact)
        actual["runtime_source_sha256"]["engine.py"] = "wrong"
        with self.assertRaises(ValueError):
            fresh.matching_sources(actual, artifact)


if __name__ == "__main__":
    unittest.main()
