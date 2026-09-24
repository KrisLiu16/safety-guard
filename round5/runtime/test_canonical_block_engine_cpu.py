"""CPU guards/routing tests only; no claim of CUDA or model causality proof."""
import copy
from collections import defaultdict
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch

import canonical_block_engine as engine


def rows():
    return [[[.6, .3, .1] for _ in range(32)] for _ in range(2)]


def fake_validate(cache, prefix, _torch):
    if prefix == 0 and cache is None:
        return
    if not isinstance(cache, dict) or cache.get('seen') != prefix:
        raise ValueError('Fake cache position mismatch')


def fake_engine(mode='eager'):
    result = object.__new__(engine.CanonicalBlockEngine)
    result._lock = threading.RLock()
    result._phase = 'ready'
    result._torch = object()
    result._engine = mode
    result._vocab_size = 100
    result._pad = 0
    result._counts = defaultdict(int)
    result._metadata = {'execution_contract': engine.execution_contract(0), 'engine_type': mode}
    result._model = object()
    result.calls = []

    def eager(ids, cache, prefix, *, export_cache):
        result.calls.append((list(ids), prefix, export_cache))
        work = copy.deepcopy(cache) if cache is not None else {'seen': 0, 'ids': []}
        work['ids'].extend(ids)
        work['seen'] += 32
        return work if export_cache else None, engine.probability_lists(rows())

    result._eager_block = eager
    result._runner = SimpleNamespace(step=lambda ids, cache, export_cache:
                                    eager(ids, cache, cache['seen'], export_cache=export_cache),
                                    close=lambda: None)
    return result


class CanonicalBlockEngineCPU(unittest.TestCase):
    def test_contract_identical_for_both_backends(self):
        a, b = fake_engine('eager'), fake_engine('window_cuda_graph')
        self.assertEqual(a.engine_metadata['execution_contract'], b.engine_metadata['execution_contract'])
        contract = a.engine_metadata['execution_contract']
        self.assertEqual(contract, {'name': 'canonical32-v1', 'block_tokens': 32, 'pad_token_id': 0,
            'window': 512, 'backbone_layers': 24, 'partial_cache': 'not_committed',
            'readout': 'all_32_positions_both_roles', 'max_input_tokens': 8192,
            'risk_labels': ['safe', 'unsafe', 'controversial']})
        contract['risk_labels'][0] = 'modified'
        self.assertEqual(a.engine_metadata['execution_contract']['risk_labels'][0], 'safe')

    def test_shapes_and_real_prefix_preserved(self):
        for prefix in (0, 32, 480, 512, 8160):
            for valid in range(1, 33):
                ids = list(range(1, valid + 1))
                padded, count = engine.prepare_block(ids, prefix, 99, 100)
                self.assertEqual(len(padded), 32)
                self.assertEqual(count, valid)
                self.assertEqual(padded[:valid], ids)
                self.assertEqual(padded[valid:], [99] * (32 - valid))
                self.assertEqual(ids, list(range(1, valid + 1)))

    def test_invalid_block_or_alignment_rejected(self):
        for prefix in (-32, 1, 31, 513, 8192, True, 32.0):
            with self.assertRaises(ValueError):
                engine.prepare_block([1], prefix, 0, 100)
        for ids in ([], [1] * 33, [True], [-1], [100], ['1'], None):
            with self.assertRaises(ValueError):
                engine.prepare_block(ids, 0, 0, 100)
        for pad in (-1, 100, True, None):
            with self.assertRaises(ValueError):
                engine.prepare_block([1], 0, pad, 100)

    def test_invalid_engine_precedes_hardware_access(self):
        with patch.object(engine, 'require_l20', side_effect=AssertionError('must not run')):
            with self.assertRaises(ValueError):
                engine.CanonicalBlockEngine(None, None, inference_engine='silent_fallback')

    def test_hardware_guard_cannot_fall_back(self):
        with patch.object(engine, 'require_l20', side_effect=RuntimeError('L20 unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'L20 unavailable'):
                engine.CanonicalBlockEngine(None, None)

    def test_all_32_positions_and_both_heads_read(self):
        hidden = object()
        seen = []

        class Risk:
            def float(self): return self
            def softmax(self, axis):
                self_test.assertEqual(axis, -1)
                return [[[(i + 1) / 64, 1 - (i + 1) / 64, 0.] for i in range(32)]]

        def readout(value, role):
            self.assertIs(value, hidden)
            seen.append(role)
            return Risk(), None

        self_test = self
        actual = engine.all_position_readout(SimpleNamespace(readout=readout), hidden,
                                             SimpleNamespace(stack=lambda values: values))
        converted = engine.probability_lists(actual)
        self.assertEqual(seen, ['user', 'assistant'])
        self.assertEqual(len(converted['user']), 32)
        self.assertNotEqual(converted['user'][0], converted['user'][-1])

    def test_probability_validation_and_copy(self):
        original = rows()
        actual = engine.probability_lists(original)
        actual['user'][0][0] = 0.
        self.assertEqual(original[0][0][0], .6)
        for bad in ([], rows()[:1], [rows()[0][:-1], rows()[1]]):
            with self.assertRaises(RuntimeError): engine.probability_lists(bad)
        for value in (float('nan'), float('inf'), -1., 2., True):
            bad = rows(); bad[1][20][1] = value
            with self.assertRaises(RuntimeError): engine.probability_lists(bad)

    @patch.object(engine, 'validate_aligned_cache', fake_validate)
    def test_partial_never_exports_cache_or_mutates_input(self):
        for mode, prefix in (('eager', 32), ('window_cuda_graph', 512)):
            runner = fake_engine(mode)
            cache = {'seen': prefix, 'ids': [5] * prefix}
            before = copy.deepcopy(cache)
            result = runner.evaluate_block([7] * 11, cache, prefix_tokens=prefix)
            self.assertEqual(cache, before)
            self.assertIsNone(result['cache'])
            self.assertFalse(runner.calls[-1][2])
            self.assertEqual(result['valid_tokens'], 11)
            self.assertEqual(result['execution']['forward_tokens'], 32)
            self.assertEqual(result['execution']['real_forward_tokens'], 11)
            self.assertEqual(result['execution']['padding_tokens'], 21)
            self.assertEqual(result['generated_tokens'], 0)

    @patch.object(engine, 'validate_aligned_cache', fake_validate)
    def test_full_cache_independent_and_early_graph_mode_uses_eager(self):
        runner = fake_engine('window_cuda_graph')
        cache = {'seen': 480, 'ids': [5] * 480}
        result = runner.evaluate_block([7] * 32, cache, prefix_tokens=480)
        self.assertEqual(result['execution']['eager_calls'], 1)
        self.assertEqual(result['execution']['graph_calls'], 0)
        self.assertEqual(result['cache']['seen'], 512)
        self.assertIsNot(result['cache'], cache)
        result['cache']['ids'][0] = 99
        self.assertEqual(cache['ids'][0], 5)

    @patch.object(engine, 'validate_aligned_cache', fake_validate)
    def test_success_counts_are_physical_and_padding_explicit(self):
        runner = fake_engine('window_cuda_graph')
        runner.evaluate_block([7] * 9, {'seen': 512, 'ids': [5] * 512}, prefix_tokens=512)
        values = runner.stats()
        self.assertEqual(values['forward_calls'], 1)
        self.assertEqual(values['forward_tokens'], 32)
        self.assertEqual(values['graph_forward_tokens'], 32)
        self.assertEqual(values['real_forward_tokens'] + values['padding_tokens'], 32)
        self.assertEqual(values['attempted_forward_tokens'], 32)

    @patch.object(engine, 'validate_aligned_cache', fake_validate)
    def test_missing_graph_is_failure_not_eager_fallback(self):
        runner = fake_engine('window_cuda_graph'); runner._runner = None
        with self.assertRaisesRegex(RuntimeError, 'fallback is disabled'):
            runner.evaluate_block([7], {'seen': 512, 'ids': [5] * 512}, prefix_tokens=512)
        self.assertEqual(runner.stats()['phase'], 'failed')
        self.assertEqual(runner.calls, [])
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            runner.evaluate_block([7], None, prefix_tokens=0)

    @patch.object(engine, 'validate_aligned_cache', fake_validate)
    def test_failed_private_forward_preserves_caller_and_fails_closed(self):
        runner = fake_engine()
        cache = {'seen': 32, 'ids': [5] * 32}; before = copy.deepcopy(cache)

        def fail(ids, source, prefix, *, export_cache):
            private = copy.deepcopy(source); private['ids'].extend(ids)
            raise RuntimeError('simulated post-forward error')

        runner._eager_block = fail
        with self.assertRaisesRegex(RuntimeError, 'post-forward'):
            runner.evaluate_block([7], cache, prefix_tokens=32)
        self.assertEqual(cache, before)
        self.assertEqual(runner.stats()['phase'], 'failed')
        self.assertEqual(runner.stats()['failed_calls'], 1)
        self.assertEqual(runner.stats()['attempted_forward_tokens'], 32)

    def test_close_and_metadata_remain_safe(self):
        runner = fake_engine()
        runner.close(); runner.close()
        self.assertEqual(runner.stats()['phase'], 'closed')
        with self.assertRaisesRegex(RuntimeError, 'not ready'):
            runner.evaluate_block([1], None, prefix_tokens=0)

    @classmethod
    def tearDownClass(cls):
        if 'torch' in sys.modules:
            raise AssertionError('CPU-only engine tests imported torch')


if __name__ == '__main__':
    unittest.main()
