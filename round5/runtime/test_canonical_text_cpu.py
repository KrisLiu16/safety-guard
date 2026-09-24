"""Canonical session CPU state-machine tests; no neural library import."""
import copy
import sys
import unittest

from canonical_text_runtime import CanonicalTextRuntime, encode, LABELS


class Tokenizer:
    truncation = None
    def encode(self, text, add_special_tokens=False, truncation=False):
        # A deliberate BPE contraction: the formerly two-token 'ab' plus 'c'
        # becomes one token, reproducing a rollback over aligned boundaries.
        values = []
        while text:
            if text.startswith('abc'):
                values.append(1000); text = text[3:]
            else:
                values.append(ord(text[0]) + 1); text = text[1:]
        return values


class FakeEngine:
    def __init__(self, pad=0):
        self.pad, self.calls, self.fail = pad, [], False
        self.engine_metadata = {'execution_contract': {'name': 'canonical32-v1'}, 'startup_forward_tokens': 0}
    def clone_cache(self, cache):
        return copy.deepcopy(cache)
    def stats(self):
        return {'eager_calls': len(self.calls), 'graph_calls': 0}
    def close(self):
        pass
    def evaluate_block(self, token_ids, cache, *, prefix_tokens):
        if self.fail:
            if cache is not None:
                cache['ids'].append(-1)
            raise RuntimeError('Injected failure after cache mutation')
        previous = [] if cache is None else list(cache['ids'])
        assert len(previous) == prefix_tokens and prefix_tokens % 32 == 0
        self.calls.append((prefix_tokens, list(token_ids)))
        padded = list(token_ids) + [self.pad] * (32 - len(token_ids))
        total = sum(previous)
        probs = {'user': [], 'assistant': []}
        for token in padded:
            total += token
            for role in probs:
                risk = .1 + (total % 37) / 100 + (.02 if role == 'assistant' else 0)
                probs[role].append([.95 - risk, risk, .05])
        return {'cache': {'ids': previous + padded} if len(token_ids) == 32 else None,
                'probabilities_by_role': probs, 'valid_tokens': len(token_ids),
                'execution': {'eager_calls': 1, 'graph_calls': 0, 'forward_tokens': 32,
                              'real_forward_tokens': len(token_ids), 'padding_tokens': 32 - len(token_ids)}}


class CanonicalCPU(unittest.TestCase):
    def runtime(self, pad=0):
        tokenizer, engine = Tokenizer(), FakeEngine(pad)
        return CanonicalTextRuntime(None, tokenizer, engine=engine), tokenizer, engine

    def assert_reference(self, runtime, tokenizer, session, result):
        ids = encode(tokenizer, session.messages)
        reference = runtime.classify_prefixes(ids)
        expected = dict(zip(LABELS, reference['probabilities_by_role_all_tokens'][result['target_role']][-1]))
        self.assertEqual(result['risk_probabilities'], expected)
        self.assertEqual(session.token_ids, ids)
        self.assertEqual(session.aligned_cache_tokens, len(ids) // 32 * 32)
        self.assertEqual(result['forward_tokens'], result['real_forward_tokens'] + result['padding_tokens'])
        self.assertEqual(result['forward_tokens'], result['forward_calls'] * 32)
        if session._cache is not None:
            self.assertEqual(session._cache['ids'], ids[:session.aligned_cache_tokens])
        self.assertLessEqual(len(session.snapshot_positions), 2)

    def test_empty_message_under32_and_exact_boundaries(self):
        runtime, tokenizer, engine = self.runtime()
        for text in ('', 'x' * 25, 'x' * 26, 'x' * 27, 'x' * 58, 'x' * 90):
            session = runtime.new_session()
            result = session.begin_message('user', text)
            self.assert_reference(runtime, tokenizer, session, result)
            before = len(engine.calls)
            empty = session.append_text('')
            self.assertEqual(len(engine.calls), before)
            self.assertEqual(empty['forward_tokens'], 0)

    def test_every_arrival_schedule_matches_whole(self):
        runtime, tokenizer, engine = self.runtime()
        text = 'x ' * 60 + 'abc' + 'public reference ' * 50
        for width in (1, 7, 31, 32, 99, 2048):
            session = runtime.new_session()
            result = session.begin_message('user')
            for start in range(0, len(text), width):
                result = session.append_text(text[start:start + width])
                self.assert_reference(runtime, tokenizer, session, result)
            self.assertEqual(result['logical_input_tokens'], len(session.token_ids))
        self.assertTrue(all(position % 32 == 0 and 1 <= len(ids) <= 32 for position, ids in engine.calls))

    def test_bpe_contraction_and_lost_older_snapshot_rebuild(self):
        runtime, tokenizer, _ = self.runtime()
        for length in (24, 56, 88, 120, 184):
            session = runtime.new_session()
            before = session.begin_message('user', 'x' * length + 'ab')
            after = session.append_text('c')
            self.assertEqual(after['net_new_tokens'], -1)
            self.assertTrue(after['rollback'])
            self.assert_reference(runtime, tokenizer, session, after)
            grown = session.append_text('!')
            self.assert_reference(runtime, tokenizer, session, grown)

    def test_partial_padding_is_not_committed_or_reported_as_input(self):
        runtime, tokenizer, _ = self.runtime()
        session = runtime.new_session()
        result = session.begin_message('user', 'x' * 33)
        self.assertEqual(result['native_tokens'], 39)
        self.assertEqual(result['aligned_cache_tokens'], 32)
        self.assertEqual(result['padding_tokens'], 25)
        next_result = session.append_text('y')
        self.assertEqual(next_result['restore_position'], 32)
        self.assertEqual(next_result['replay_tokens'], 7)
        self.assertEqual(next_result['real_forward_tokens'], 8)
        self.assertEqual(next_result['net_new_tokens'], 1)
        self.assert_reference(runtime, tokenizer, session, next_result)

    def test_future_pad_cannot_change_any_real_prefix(self):
        left, _, _ = self.runtime(0); right, _, _ = self.runtime(123)
        for length in (1, 17, 31, 32, 33, 63, 64, 513):
            ids = list(range(1, length + 1))
            a, b = left.classify_prefixes(ids), right.classify_prefixes(ids)
            self.assertEqual(a['probabilities_by_role_all_tokens'], b['probabilities_by_role_all_tokens'])
            self.assertEqual(len(a['probabilities_by_role_all_tokens']['user']), length)

    def test_sessions_roles_cache_ownership_and_failure_transaction(self):
        runtime, tokenizer, engine = self.runtime()
        a, b = runtime.new_session(), runtime.new_session()
        a.begin_message('user', 'x' * 130); b.begin_message('assistant', 'y' * 74)
        self.assert_reference(runtime, tokenizer, a, a.append_text('!'))
        self.assert_reference(runtime, tokenizer, b, b.append_text('?'))
        result = a.begin_message('assistant', 'public answer')
        self.assert_reference(runtime, tokenizer, a, result)
        before = copy.deepcopy((a.messages, a.token_ids, a._cache, a._snapshots, a._probabilities))
        engine.fail = True
        with self.assertRaises(RuntimeError):
            a.append_text('suffix')
        self.assertEqual((a.messages, a.token_ids, a._cache, a._snapshots, a._probabilities), before)
        engine.fail = False
        self.assert_reference(runtime, tokenizer, a, a.append_text('suffix'))
        caches = [a._cache, *[s.cache for s in a._snapshots]]
        self.assertEqual(len({id(cache['ids']) for cache in caches}), len(caches))

    def test_8192_and_overflow_rejection_are_transactional(self):
        runtime, tokenizer, engine = self.runtime()
        session = runtime.new_session()
        result = session.begin_message('user', 'x' * (8192 - 6))
        self.assertEqual(result['native_tokens'], 8192)
        self.assertEqual(result['padding_tokens'], 0)
        before = copy.deepcopy((session.messages, session.token_ids, session._cache))
        calls = len(engine.calls)
        with self.assertRaises(ValueError):
            session.append_text('x')
        self.assertEqual((session.messages, session.token_ids, session._cache), before)
        self.assertEqual(len(engine.calls), calls)
        tokenizer.truncation = {'max_length': 16}
        with self.assertRaises(ValueError):
            session.append_text('')

    def test_no_neural_library_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)

    def test_end_synchronization_failure_does_not_commit_session(self):
        runtime, tokenizer, _ = self.runtime()
        session = runtime.new_session()
        session.begin_message('user', 'x' * 130)
        before = copy.deepcopy((session.messages, session.token_ids, session._cache, session._snapshots,
                                session._forward_tokens, session._wall_seconds))
        def fail():
            raise RuntimeError('Injected asynchronous completion failure')
        session._synchronize = fail
        with self.assertRaises(RuntimeError):
            session.append_text(' new text')
        self.assertEqual((session.messages, session.token_ids, session._cache, session._snapshots,
                          session._forward_tokens, session._wall_seconds), before)
        session._synchronize = lambda: None
        self.assert_reference(runtime, tokenizer, session, session.append_text(' new text'))


if __name__ == '__main__':
    unittest.main()
