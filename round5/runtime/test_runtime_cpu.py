"""Round5 bundle behavior/identity checks: no neural libraries or real weights."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import types
from unittest import mock

import standalone_model as standalone
import prepare_manifest
import package_model
import run_guard
import download_release
import test_canonical_l20 as core_audit


def fake_completed_training(root):
    """Synthetic unit-test fixture only; its bytes cannot be loaded as a model."""
    checkpoint = root / 'classification_rl/best.safetensors'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'unit-test-no-model-weights')
    digest = standalone.sha256_file(checkpoint)
    strata = ('zh/user', 'zh/assistant', 'en/assistant')
    metrics = {'whole': {'strata': {key: {'threshold': .4, 'comparison': '>', 'fpr': .01} for key in strata}, 'macro_recall': .8},
               'stream': {'strata': {key: {'threshold': .6, 'comparison': '>', 'fpr': .01} for key in strata}, 'macro_recall': .75},
               'baseline_whole_macro_recall': .8, 'thresholds_fitted_on': 'calibration only',
               'eligible': True, 'selection_gates': {'each_stream_stratum_fpr_at_most_5pct': True,
                                                   'whole_macro_recall_drop_at_most_2pp': True},
               'selection_read_test': False, 'checkpoint_sha256': digest}
    binding = {'trainer_sha256': standalone.TRAINER_SHA256, 'metrics_sha256': standalone.METRICS_SHA256,
               'initial_checkpoint_sha256': standalone.INITIAL_WINDOW_SHA256, 'window': 512}
    sft = {'status': 'completed', 'binding': binding, 'checkpoint_sha256': digest,
           'baseline_whole_macro_recall': .8, 'steps': 4164, 'epochs': 2,
           'selected': {'checkpoint_sha256': digest}}
    rl = {'status': 'completed', 'binding': binding, 'checkpoint_sha256': digest, 'updates': 256,
          'reference_checkpoint_sha256': digest, 'best_metrics': metrics,
          'eligible_candidate_found': True, 'selected': {'stage': 'sft_selected', 'checkpoint_sha256': digest}}
    final = {'status': 'completed', 'binding': binding, 'sft': sft, 'classification_rl': rl,
             'final_checkpoint': str(checkpoint), 'final_checkpoint_sha256': digest, 'selection_read_test': False}
    (root / 'sft').mkdir()
    for path, value in ((root / 'sft/summary.json', sft), (root / 'classification_rl/summary.json', rl),
                        (root / 'summary.json', final)):
        path.write_text(json.dumps(value))
    contract = {'name': 'canonical32-v1', 'block_tokens': 32, 'pad_token_id': 0, 'window': 512,
                'backbone_layers': 24, 'partial_cache': 'not_committed',
                'readout': 'all_32_positions_both_roles', 'max_input_tokens': 8192,
                'risk_labels': ['safe', 'unsafe', 'controversial']}
    runtime = Path(__file__).parent
    canonical = {'version': 'round5-canonical32-calibration-v1', 'status': 'completed',
                 'checkpoint_sha256': digest, 'execution_contract': contract, 'threshold_comparison': '>',
                 'thresholds': {mode: {key: row['threshold'] for key, row in metrics[mode]['strata'].items()}
                                for mode in ('whole', 'stream')},
                 'calibrated_strata': {mode: sorted(strata) for mode in ('whole', 'stream')},
                 'thresholds_fitted_on': 'calibration only', 'selection_changed': False, 'selection_read_test': False,
                 'arbitrary_bpe_schedule_fpr_guarantee': False,
                 'calibration_receipt': {'predictions_sha256': 'a' * 64},
                 'development_receipt': {'predictions_sha256': 'b' * 64},
                 'development_metrics': metrics, 'canonical_quality_gate_pass': True,
                 'runtime_source_sha256': {name: standalone.sha256_file(runtime / name)
                                          for name in ('canonical_block_engine.py', 'canonical_text_runtime.py')}}
    (root / 'canonical_calibration.json').write_text(json.dumps(canonical))
    return root / 'summary.json', checkpoint


class RuntimeCPU(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.summary, self.checkpoint = fake_completed_training(self.root)
        self.canonical = self.root / 'canonical_calibration.json'
        self.selection = prepare_manifest.prepare(self.summary, self.canonical)

    def tearDown(self):
        self.temporary.cleanup()

    def test_imports_and_default_do_not_load_neural_libraries(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        args = run_guard.parse_args([])
        self.assertEqual(args.inference_engine, 'eager')
        self.assertEqual(run_guard.parse_args(['--inference-engine', 'window_cuda_graph']).inference_engine, 'window_cuda_graph')

    def test_whole_and_stream_use_different_thresholds(self):
        whole = run_guard.threshold_result(self.selection, 'zh', 'user', {'unsafe': .5}, None, False)
        stream = run_guard.threshold_result(self.selection, 'zh', 'user', {'unsafe': .5}, None, True)
        self.assertEqual(whole['target_binary_decision'], 'unsafe')
        self.assertEqual(stream['target_binary_decision'], 'safe')
        self.assertEqual(whole['threshold']['mode'], 'whole')
        self.assertEqual(stream['threshold']['mode'], 'stream')

    def test_threshold_equality_is_safe_in_both_modes(self):
        for streaming, value in ((False, .4), (True, .6)):
            result = run_guard.threshold_result(self.selection, 'zh', 'assistant', {'unsafe': value}, None, streaming)
            self.assertEqual(result['target_binary_decision'], 'safe')
            self.assertEqual(result['threshold']['comparison'], '>')
            self.assertEqual(run_guard.threshold_result(self.selection, 'zh', 'assistant', {'unsafe': value + 1e-9}, None, streaming)['target_binary_decision'], 'unsafe')

    def test_missing_stratum_never_falls_back(self):
        for streaming in (False, True):
            result = run_guard.threshold_result(self.selection, 'en', 'user', {'unsafe': .99}, None, streaming)
            self.assertIsNone(result['target_binary_decision'])
            self.assertEqual(result['threshold']['kind'], 'stratum_not_calibrated')
            result = run_guard.threshold_result(self.selection, None, 'assistant', {'unsafe': .99}, None, streaming)
            self.assertIsNone(result['threshold']['value'])

    def test_explicit_override_is_strict_and_uncalibrated(self):
        result = run_guard.threshold_result(self.selection, 'en', 'user', {'unsafe': .5}, .5, True)
        self.assertEqual(result['target_binary_decision'], 'safe')
        self.assertEqual(result['threshold']['kind'], 'explicit_uncalibrated')
        with self.assertRaises(ValueError):
            run_guard.threshold_result(self.selection, 'en', 'user', {'unsafe': .5}, float('nan'), True)

    def test_no_provisional_manifest_or_wrong_weight_sha(self):
        final = json.loads(self.summary.read_text()); final['status'] = 'running'
        self.summary.write_text(json.dumps(final))
        with self.assertRaises(ValueError):
            prepare_manifest.prepare(self.summary, self.canonical)
        final['status'] = 'completed'; self.summary.write_text(json.dumps(final))
        self.checkpoint.write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            prepare_manifest.prepare(self.summary, self.canonical)

    def test_fallback_cannot_claim_promotion(self):
        selection = copy.deepcopy(self.selection)
        selection.update(checkpoint_sha256=standalone.INITIAL_WINDOW_SHA256, candidate='initial_window_fallback',
                         status='fallback_not_promoted', eligible_candidate_found=False,
                         promoted_from_initial=False, training_promoted_from_initial=False)
        standalone.validate_selection(selection)
        selection['status'] = 'research_candidate'
        with self.assertRaises(ValueError):
            standalone.validate_selection(selection)

    def test_test_read_or_ineligible_replacement_rejected(self):
        selection = copy.deepcopy(self.selection); selection['selection_read_test'] = True
        with self.assertRaises(ValueError):
            standalone.validate_selection(selection)
        selection = copy.deepcopy(self.selection); selection['eligible_candidate_found'] = False
        with self.assertRaises(ValueError):
            standalone.validate_selection(selection)

    def test_evidence_rejects_threshold_substitution(self):
        documents = {name: json.loads(Path(item['path']).read_text()) for name, item in self.selection['selection_evidence'].items()}
        selection = copy.deepcopy(self.selection); selection['thresholds']['stream']['zh/user'] = .2
        with self.assertRaises(ValueError):
            standalone.validate_evidence(selection, documents)

    def test_failed_canonical_gate_preserves_unvalidated_research_artifact(self):
        canonical = json.loads(self.canonical.read_text())
        development = canonical['development_metrics']
        development['stream']['strata']['zh/user']['fpr'] = .2
        development['selection_gates']['each_stream_stratum_fpr_at_most_5pct'] = False
        development['eligible'] = False
        canonical['canonical_quality_gate_pass'] = False
        self.canonical.write_text(json.dumps(canonical))
        selected = prepare_manifest.prepare(self.summary, self.canonical)
        self.assertTrue(selected['training_promoted_from_initial'])
        self.assertFalse(selected['promoted_from_initial'])
        self.assertEqual(selected['status'], 'research_unvalidated_runtime')

    def test_complete_cpu_only_bundle_verification_and_tamper_rejection(self):
        manifest = self.root / 'MODEL_MANIFEST.json'; manifest.write_text(json.dumps(self.selection))
        assets, window = self.root / 'assets', self.root / 'window'
        assets.mkdir(); window.mkdir()
        (assets / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'text_config': {'num_hidden_layers': 24, 'hidden_size': 1024}}))
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (assets / name).write_text('{}')
        (window / 'window_attention.py').write_text('# CPU fixture, never executed\n')
        bundle = self.root / 'bundle'
        argv = ['--manifest', str(manifest), '--model-assets', str(assets), '--window-code-dir', str(window), '--output', str(bundle), '--copy-weights']
        with mock.patch.object(package_model, 'runtime_versions', return_value={'fixture': 'no-neural-runtime'}), contextlib.redirect_stdout(io.StringIO()):
            package_model.main(argv)
        _, record, selection = standalone.verify_bundle(bundle)
        self.assertEqual(record['format'], 'round5-prefix-classifier-bundle-v1')
        self.assertEqual(selection['thresholds'], self.selection['thresholds'])
        self.assertEqual(record['inference_engine'], 'eager')
        self.assertFalse(record['graph_validation_passed'])
        with contextlib.redirect_stdout(io.StringIO()):
            run_guard.main(['--bundle', str(bundle), '--verify-only'])
        download_release.inspect_metadata(bundle)
        (bundle / 'selection/rl_summary.json').write_text('{}')
        with self.assertRaises(ValueError):
            standalone.verify_bundle(bundle)

    def test_audit_exact_modules_override_sys_path_decoy_and_keep_window_class(self):
        local, decoy = self.root / 'runtime', self.root / 'old_round4'
        local.mkdir(); decoy.mkdir()
        for name in core_audit.RUNTIME_MODULES:
            (local / (name + '.py')).write_text('ORIGIN = "new-canonical"\n')
            (decoy / (name + '.py')).write_text('ORIGIN = "old-prototype"\n')
        window = types.ModuleType('window_attention')
        with mock.patch.dict(sys.modules), mock.patch.object(sys, 'path', [str(decoy), *sys.path]):
            for name in core_audit.RUNTIME_MODULES:
                sys.modules.pop(name, None)
            sys.modules['window_attention'] = window
            receipt = core_audit.load_runtime_modules(local)
            self.assertIs(sys.modules['window_attention'], window)
            self.assertEqual(sys.modules['graph_stream'].ORIGIN, 'new-canonical')
            self.assertEqual(Path(receipt['runtime_source_paths']['graph_stream']), (local / 'graph_stream.py').resolve())
            (local / 'graph_stream.py').write_text('ORIGIN = "changed-after-load"\n')
            with self.assertRaises(RuntimeError):
                core_audit.runtime_source_receipt(local)

    def test_audit_rejects_cached_module_from_wrong_directory(self):
        local, decoy = self.root / 'runtime', self.root / 'old_round4'
        local.mkdir(); decoy.mkdir()
        for name in core_audit.RUNTIME_MODULES:
            (local / (name + '.py')).write_text('ORIGIN = "new-canonical"\n')
        stale = types.ModuleType('graph_stream'); stale.__file__ = str(decoy / 'graph_stream.py')
        with mock.patch.dict(sys.modules), mock.patch.object(sys, 'path', list(sys.path)):
            for name in core_audit.RUNTIME_MODULES:
                sys.modules.pop(name, None)
            sys.modules['graph_stream'] = stale
            with self.assertRaises(RuntimeError):
                core_audit.load_runtime_modules(local)

    def test_audit_rejects_equal_infinite_cache_before_comparison(self):
        tensor = types.SimpleNamespace(shape=(1,), dtype='float32')
        graph = types.ModuleType('graph_stream')
        graph.state_tensors = lambda cache: {'nonfinite': tensor}
        graph.state_metadata = lambda cache: {}
        fake_torch = types.SimpleNamespace(float32='float32',
                                         isfinite=lambda value: types.SimpleNamespace(all=lambda: False))
        with mock.patch.dict(sys.modules, graph_stream=graph):
            with self.assertRaisesRegex(AssertionError, 'Nonfinite'):
                core_audit.assert_cache(fake_torch, object(), object())


if __name__ == '__main__':
    unittest.main()
