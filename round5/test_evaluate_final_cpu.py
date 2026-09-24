"""CPU-only final-holdout integrity/threshold fixtures; no model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("final_audit", HERE / "evaluate_final_l20.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
metrics = audit.exact_module("stream_metrics", HERE / "stream_metrics.py", audit.METRICS_SHA)


def source_and_observation(name, label, p, stratum="zh/user"):
    language, role = stratum.split("/")
    common = dict(sample_id=name, family="family-" + name, language=language,
                  target_role=role, source_label=label)
    source = dict(common, ids=[1, 2], target_token_positions=[1],
                  text_cut_views=[{"cut_chars": 2, "ids": [1, 2]}])
    observed = dict(common, endpoint_probs=[1 - p, p, 0.],
                    endpoint_p_unsafe=p, native_target_max_p_unsafe=p,
                    native_target_p_unsafe=[p], target_token_positions=[1],
                    text_cut_observations=[{"cut_chars": 2, "p_unsafe": p}],
                    text_cut_max_p_unsafe=p, stream_max_p_unsafe=p)
    return source, observed


class TrainingFixture:
    def __init__(self, root, sft_stage="initial", rl_step=None):
        self.root = Path(root).resolve()
        self.start = self.root / "external_start.safetensors"
        self.start.write_bytes(b"fixed initial window")
        self.start_sha = audit.sha(self.start)
        self.patches = [mock.patch.object(audit, "START_PATH", self.start),
                        mock.patch.object(audit, "START_SHA", self.start_sha)]
        for patch in self.patches:
            patch.start()
        (self.root / "sft").mkdir()
        (self.root / "classification_rl").mkdir()
        initial_metric = {"checkpoint_sha256": self.start_sha, "eligible": False}
        audit.write_json(self.root / "sft/evaluation_0/metrics.json", initial_metric)
        if sft_stage == "initial":
            sft_bytes, sft_sha, sft_metric = self.start.read_bytes(), self.start_sha, initial_metric
            selected = {"stage": "initial", "step": 0, "checkpoint_sha256": sft_sha,
                        "source_checkpoint": str(self.start)}
        else:
            epoch = int(sft_stage)
            epoch_file = self.root / "sft" / ("epoch_" + str(epoch) + ".safetensors")
            sft_bytes = ("epoch" + str(epoch)).encode()
            epoch_file.write_bytes(sft_bytes)
            sft_sha = audit.sha(epoch_file)
            sft_metric = {"checkpoint_sha256": sft_sha, "eligible": True}
            audit.write_json(self.root / "sft" / ("evaluation_" + str(epoch)) / "metrics.json", sft_metric)
            selected = {"stage": "sft", "step": epoch * 2082, "checkpoint_sha256": sft_sha,
                        "source_checkpoint": str(epoch_file)}
        (self.root / "sft/best.safetensors").write_bytes(sft_bytes)
        binding = {"trainer_sha256": audit.TRAINER_SHA, "metrics_sha256": audit.METRICS_SHA,
                   "initial_checkpoint_sha256": self.start_sha, "window": 512}
        sft = {"status": "completed", "steps": 4164, "epochs": 2, "binding": binding,
               "selected": selected, "checkpoint_sha256": sft_sha, "best_metrics": sft_metric}
        if rl_step is None:
            final_bytes, final_sha, final_metric = sft_bytes, sft_sha, sft_metric
            selected = {"stage": "sft_selected", "step": 0, "checkpoint_sha256": final_sha,
                        "source_checkpoint": str(self.root / "sft/best.safetensors")}
        else:
            source = self.root / "classification_rl" / ("step_" + str(rl_step) + ".safetensors")
            final_bytes = ("rl" + str(rl_step)).encode()
            source.write_bytes(final_bytes)
            final_sha = audit.sha(source)
            final_metric = {"checkpoint_sha256": final_sha, "eligible": True}
            audit.write_json(self.root / "classification_rl" / ("evaluation_" + str(rl_step)) / "metrics.json",
                             final_metric)
            selected = {"stage": "classification_rl", "step": rl_step, "checkpoint_sha256": final_sha,
                        "source_checkpoint": str(source)}
        final_path = self.root / "classification_rl/best.safetensors"
        final_path.write_bytes(final_bytes)
        rl = {"status": "completed", "updates": 256, "binding": binding, "selected": selected,
              "checkpoint_sha256": final_sha, "best_metrics": final_metric,
              "reference_checkpoint_sha256": sft_sha}
        self.summary = {"status": "completed", "sft": sft, "classification_rl": rl, "binding": binding,
                        "final_checkpoint": str(final_path), "final_checkpoint_sha256": final_sha,
                        "selection_read_test": False, "generated_tokens": 0}
        audit.write_json(self.root / "run_spec.json", {"binding": binding, "smoke_only": False})
        self.write()

    def write(self):
        audit.write_json(self.root / "summary.json", self.summary)
        audit.write_json(self.root / "sft/summary.json", self.summary["sft"])
        audit.write_json(self.root / "classification_rl/summary.json", self.summary["classification_rl"])

    def close(self):
        for patch in reversed(self.patches):
            patch.stop()


class FinalEvaluationContract(unittest.TestCase):
    def test_all_declared_selection_paths(self):
        for stage, rl_step, expected in (
                ("initial", None, "sft/evaluation_0"), ("1", None, "sft/evaluation_1"),
                ("2", None, "sft/evaluation_2"), ("1", 64, "classification_rl/evaluation_64"),
                ("2", 256, "classification_rl/evaluation_256")):
            with self.subTest(stage=stage, rl=rl_step), tempfile.TemporaryDirectory() as directory:
                fixture = TrainingFixture(directory, stage, rl_step)
                try:
                    _, _, selected = audit.completed_training(fixture.root)
                    self.assertEqual(selected, fixture.root / expected)
                finally:
                    fixture.close()

    def test_incomplete_training_never_reaches_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit.write_json(root / "summary.json", {"status": "running"})
            with self.assertRaisesRegex(ValueError, "not completed"):
                audit.completed_training(root)
        self.assertNotIn("torch", sys.modules)

    def test_checkpoint_bytes_and_best_metrics_must_match(self):
        for tamper in ("weights", "metrics", "reference"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                fixture = TrainingFixture(directory, "2", 192)
                try:
                    if tamper == "weights":
                        (fixture.root / "classification_rl/best.safetensors").write_bytes(b"wrong")
                    elif tamper == "metrics":
                        audit.write_json(fixture.root / "classification_rl/evaluation_192/metrics.json",
                                         {"checkpoint_sha256": "0" * 64})
                    else:
                        fixture.summary["classification_rl"]["reference_checkpoint_sha256"] = "0" * 64
                        fixture.write()
                    with self.assertRaises(ValueError):
                        audit.completed_training(fixture.root)
                finally:
                    fixture.close()

    def test_fixed_threshold_application_cannot_calibrate(self):
        observed = []
        for key in audit.STRATA:
            observed += [source_and_observation(key + "-safe", "safe", .5, key)[1],
                         source_and_observation(key + "-unsafe", "unsafe", 1., key)[1]]
        thresholds = {mode: {key: .5 for key in audit.STRATA} for mode in audit.FIELDS}
        with mock.patch.object(metrics, "threshold", side_effect=AssertionError("Fresh fitting forbidden")), \
                mock.patch.object(metrics, "evaluate", side_effect=AssertionError("Fresh selection forbidden")):
            result = audit.evaluate_fixed(observed, thresholds, metrics)
            self.assertEqual(result["stream"]["macro_fpr"], 0.)
            self.assertEqual(result["stream"]["macro_recall"], 1.)
            thresholds["stream"] = {key: 1. for key in audit.STRATA}
            result = audit.evaluate_fixed(observed, thresholds, metrics)
            self.assertEqual(result["stream"]["macro_recall"], 0.)
            self.assertFalse(result["selection_uses_fresh"])

    def test_missing_threshold_and_nan_rejected(self):
        observed = [source_and_observation(key + label, label, .1 if label == "safe" else .9, key)[1]
                    for key in audit.STRATA for label in ("safe", "unsafe")]
        thresholds = {mode: {key: .5 for key in audit.STRATA} for mode in audit.FIELDS}
        thresholds["stream"]["zh/user"] = float("nan")
        with self.assertRaises(ValueError):
            audit.evaluate_fixed(observed, thresholds, metrics)
        thresholds["stream"].pop("zh/user")
        with self.assertRaises(ValueError):
            audit.evaluate_fixed(observed, thresholds, metrics)

    def test_observation_cut_coverage_and_maximum_are_bound(self):
        source, observed = source_and_observation("example", "safe", .2)
        audit.validate_observations([observed], [source], metrics)
        for tamper in ("cut", "max", "identity"):
            bad = copy.deepcopy(observed)
            if tamper == "cut":
                bad["text_cut_observations"] = []
            elif tamper == "max":
                bad["native_target_p_unsafe"] = [.7]
            else:
                bad["family"] = "another family"
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                audit.validate_observations([bad], [source], metrics)

    def test_calibration_receipt_reconstructs_only_old_cal_dev(self):
        source, observations = {}, {}
        for split, prefix in (("calibration", "c"), ("dev", "d")):
            pairs = [source_and_observation(prefix + key + str(index),
                     "safe" if index < 20 else "unsafe", .2 if index < 20 else .9, key)
                     for key in audit.STRATA for index in range(40)]
            source[split], observations[split] = [p[0] for p in pairs], [p[1] for p in pairs]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            saved = metrics.evaluate(observations["calibration"], observations["dev"], 1.)
            saved.update(checkpoint_sha256="a" * 64, selection_read_test=False,
                         evaluation_weights="exported_bfloat16_backbone", generated_tokens=0)
            audit.write_json(path / "metrics.json", saved)
            for split in source:
                audit.write_rows(path / (split + "_predictions.jsonl"), observations[split])
            receipt = audit.calibration_receipt(path, "a" * 64, source, metrics, 1.)
            self.assertTrue(receipt["metrics_independently_recomputed"])
            saved["stream"]["strata"]["zh/user"]["threshold"] = .3
            audit.write_json(path / "metrics.json", saved)
            with self.assertRaises(ValueError):
                audit.calibration_receipt(path, "a" * 64, source, metrics, 1.)

    def test_actual_frozen_fresh_data_and_family_separation(self):
        trainer = audit.exact_module("train_prefix", HERE / "train_prefix.py", audit.TRAINER_SHA)
        _, data = trainer.load_data(HERE / "data/prefix_v2")
        manifest, fresh, isolation = audit.load_fresh(HERE / "data/fresh_holdout_v1", data, trainer)
        self.assertEqual(len(fresh), 390)
        self.assertEqual(manifest["unique_families"], 390)
        self.assertTrue(isolation["pass"])
        self.assertFalse(isolation["historical_files_recomputed_in_this_process"])
        data["train"].append(dict(data["train"][0], family=fresh[0]["family"]))
        with self.assertRaises(ValueError):
            audit.load_fresh(HERE / "data/fresh_holdout_v1", data, trainer)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
