"""CPU-only regression of the frozen official rules, IDs and coverage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import evaluate_canonical_official_l20 as evaluation


class OfficialCanonicalCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.round4 = cls.root.parent / 'round4'
        cls.rules = evaluation.load_exact('cpu_official_rules', cls.round4 / 'evaluate_official_adapted.py', evaluation.OFFICIAL_EVALUATOR_SHA)
        cls.data, cls.receipt = evaluation.load_official_inputs(cls.round4 / 'data/official_adapted_v1',
                                                             cls.root.parent / 'round1/benchmark')

    def test_imports_are_cpu_safe(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)

    def test_full_frozen_coverage_and_exact_overlap(self):
        self.assertEqual(self.receipt['rows'], 2441)
        self.assertEqual(self.receipt['unique_forward_sequences'], 1872)
        self.assertEqual(self.receipt['exact_prediction_cache_hits'], 569)
        self.assertEqual(self.receipt['splits']['thinking_loc']['unique_forward_sequences'], 0)
        self.assertEqual(self.receipt['splits']['thinking_loc']['exact_prediction_cache_hits'], 569)

    def test_unsafe_pair_priority_over_earlier_controversial_pair(self):
        self.assertEqual(self.rules.decision([2, 2, 0, 1, 1]), (1, 4))
        self.assertEqual(self.rules.decision([2, 2, 0, 1]), (2, 1))
        self.assertEqual(self.rules.decision([0, 1, 0, 2]), (0, -1))

    def test_argmax_ties_native_start_and_endpoint_are_unchanged(self):
        row = {'ids': list(range(6)), 'eval_start_index': 2}
        values = [[.1, .8, .1], [.1, .8, .1], [.5, .5, 0], [.1, .1, .8], [.1, .1, .8], [.8, .1, .1]]
        result = evaluation.classify_record(row, {'assistant': values}, self.rules)
        self.assertEqual(result['decision'], 2)
        self.assertEqual(result['decision_index_in_evaluated_native_tokens'], 2)
        self.assertEqual(result['endpoint_decision'], 0)
        self.assertEqual(result['endpoint_probabilities'], values[-1])

    def test_strict_and_loose_definition_and_unsafe_only_fpr(self):
        rows = [{'label': 'Safe', 'decision': 2, 'endpoint_decision': 0},
                {'label': 'Unsafe', 'decision': 2, 'endpoint_decision': 2}]
        self.assertEqual(self.rules.binary_metrics(rows)['fp'], 0)
        self.assertEqual(self.rules.binary_metrics(rows, loose=True)['fp'], 1)
        self.assertEqual(self.rules.binary_metrics(rows, loose=True)['tp'], 1)
        self.assertIsNone(self.rules.binary_metrics(rows[1:])['fpr'])

    def test_cache_identity_includes_full_ids_and_start(self):
        row = {'ids': [1, 2, 3], 'eval_start_index': 1}
        self.assertNotEqual(evaluation.sequence_key(row), evaluation.sequence_key(dict(row, eval_start_index=2)))
        self.assertNotEqual(evaluation.sequence_key(row), evaluation.sequence_key(dict(row, ids=[1, 2, 4])))
        self.assertEqual(evaluation.sequence_key(row), evaluation.sequence_key(copy.deepcopy(row)))

    def test_incomplete_coverage_rejected(self):
        incomplete = dict(self.data, response_loc=self.data['response_loc'][:-1])
        with self.assertRaises(ValueError):
            evaluation.coverage_receipt(incomplete)

    def test_source_and_native_hash_tampering_rejected(self):
        rows = copy.deepcopy(self.data['response_loc'])
        sources = evaluation.read_rows(self.root.parent / 'round1/benchmark/response_loc.jsonl')
        rows[0]['ids'][0] += 1
        with self.assertRaises(ValueError):
            evaluation.validate_prepared_rows('response_loc', rows, sources)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'; path.write_text('{}')
            with self.assertRaises(ValueError):
                evaluation.load_official_inputs(Path(tmp), Path(tmp))

    def test_actual_tokenizer_id_mismatch_never_replaces_saved_ids(self):
        row = copy.deepcopy(self.data['response_loc'][0]); original = row['ids'].copy()
        tokenizer = mock.Mock(); tokenizer.backend_tokenizer.truncation = None
        tokenizer.encode.return_value = [0]
        serialize = lambda messages: '\n\n'.join(m['role'].upper() + ':\n' + m['content'] for m in messages)
        data = {split: [row] for split in evaluation.SPLITS}
        with self.assertRaises(ValueError):
            evaluation.verify_native_ids(tokenizer, data, serialize)
        self.assertEqual(row['ids'], original)

    def test_historical_a0_and_bulk_keep_original_identity(self):
        a0 = evaluation.reference_metrics(self.round4 / 'validation_results/output/round4_validation/official_a0/metrics.json',
                                          evaluation.A0_METRICS_SHA, 'a0', None)
        self.assertEqual(a0['original_kind'], 'a0')
        self.assertIsNone(a0['original_checkpoint_sha256'])
        bulk = evaluation.reference_metrics(self.round4 / 'fast_runtime/results/output/fast_runtime/official_window/metrics.json',
                                            evaluation.BULK_METRICS_SHA, 'student', evaluation.INITIAL_WINDOW_SHA)
        self.assertEqual(bulk['original_checkpoint_sha256'], evaluation.INITIAL_WINDOW_SHA)
        with self.assertRaises(ValueError):
            evaluation.reference_metrics(Path(a0['path']), evaluation.A0_METRICS_SHA, 'student', evaluation.INITIAL_WINDOW_SHA)


if __name__ == '__main__':
    unittest.main()
