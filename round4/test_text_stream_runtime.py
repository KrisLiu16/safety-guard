"""CPU-only tests of append/rollback transactions; no torch or neural model."""
import copy
from types import SimpleNamespace
import unittest

from text_stream_runtime import (
    MAX_INPUT_TOKENS, TextStreamSession, longest_common_prefix, serialize,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class MergeTokenizer(CharacterTokenizer):
    """A deterministic stand-in for suffix BPE merges, including token shrink."""
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        # abc becomes one token only when its third character arrives.
        out = []
        index = 0
        while index < len(ids):
            if ids[index:index + 3] == [ord('a'), ord('b'), ord('c')]:
                out.append(100000)
                index += 3
            else:
                out.append(ids[index])
                index += 1
        return out


class IgnoreSpacesTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        return super().encode(text.replace(' ', ''), add_special_tokens)


class RewriteTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        if text.endswith('!'):
            ids[0] += 100000
        return ids


class OverflowTokenizer(CharacterTokenizer):
    def __init__(self):
        self.return_overflow = False

    def encode(self, text, add_special_tokens=False):
        ids = super().encode(text, add_special_tokens)
        return SimpleNamespace(ids=ids[:16], overflowing=[ids[16:]]) if self.return_overflow else SimpleNamespace(ids=ids, overflowing=[])


class ConfigurableTruncationTokenizer(CharacterTokenizer):
    def __init__(self):
        self.truncation = None
        self.calls = 0

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        ids = super().encode(text, add_special_tokens)
        return ids[:self.truncation['max_length']] if self.truncation else ids


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.fail_after_mutation = False

    @staticmethod
    def probabilities(ids):
        unsafe = (sum((index + 1) * token for index, token in enumerate(ids)) % 101) / 125
        return {
            'user': {'safe': .9 - unsafe, 'unsafe': unsafe, 'controversial': .1},
            'assistant': {'safe': unsafe, 'unsafe': .9 - unsafe, 'controversial': .1},
        }

    def __call__(self, ids, cache):
        if cache is None:
            cache = {'ids': [], 'recurrent': {'state': []}}
        cache['ids'].extend(ids)
        cache['recurrent']['state'].append(sum(ids))
        self.calls.append(ids.copy())
        if self.fail_after_mutation:
            raise RuntimeError('injected failure after mutating the candidate cache')
        return cache, self.probabilities(cache['ids'])


class StreamStateMachineTests(unittest.TestCase):
    def make_session(self, tokenizer=None, chunk_tokens=1000):
        runner = FakeRunner()
        session = TextStreamSession(tokenizer or CharacterTokenizer(), runner, chunk_tokens=chunk_tokens)
        return session, runner

    def assert_reference(self, session, result):
        encoded = session._tokenizer.encode(serialize(session.messages), **session._encode_kwargs)
        ids = encoded.ids if hasattr(encoded, 'ids') else encoded
        self.assertEqual(session.token_ids, ids)
        self.assertEqual(session._cache['ids'], ids)
        self.assertEqual(result['risk_probabilities'], FakeRunner.probabilities(ids)[result['target_role']])

    def test_lcp(self):
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(longest_common_prefix([], [1]), 0)
        self.assertEqual(longest_common_prefix([1, 2], [1]), 1)

    def test_append_and_boundary_snapshots(self):
        session, runner = self.make_session()
        result = session.begin_message('user', 'x' * 140)
        self.assertEqual([len(chunk) for chunk in runner.calls], [64, 64, 18])
        self.assertEqual(result['snapshot_positions'], [64, 128])
        old_cache = copy.deepcopy(session._cache)
        snapshot = copy.deepcopy(session._snapshots)
        result = session.append_text('y' * 120)
        self.assertFalse(result['rollback'])
        self.assertEqual(result['forward_tokens'], 120)
        self.assertEqual(result['snapshot_positions'], [192, 256])
        self.assertEqual(snapshot[0].cache['ids'], old_cache['ids'][:64])
        self.assert_reference(session, result)

    def test_bpe_shrink_uses_snapshot(self):
        session, _ = self.make_session(MergeTokenizer())
        session.begin_message('user', 'x' * 140 + 'ab')
        previous = len(session.token_ids)
        result = session.append_text('c')
        self.assertEqual(result['net_new_tokens'], -1)
        self.assertEqual(result['restore_position'], 128)
        self.assertTrue(result['rollback'])
        self.assertFalse(result['fallback_full_replay'])
        self.assertEqual(result['forward_tokens'], previous - 1 - 128)
        self.assert_reference(session, result)

    def test_rollback_at_exact_snapshot_boundary(self):
        session, _ = self.make_session(MergeTokenizer())
        session.begin_message('user', 'x' * 58 + 'ab')
        result = session.append_text('c')
        self.assertEqual(result['common_prefix_tokens'], 64)
        self.assertEqual(result['restore_position'], 64)
        self.assertEqual(result['forward_tokens'], 1)
        self.assert_reference(session, result)

    def test_bpe_shrink_crosses_snapshot_boundary_then_expands(self):
        # 192 -> 191 makes position 64 necessary again although [128, 192]
        # were the only retained snapshots. Restoring from 128 is too late.
        runner = FakeRunner()
        session = TextStreamSession(MergeTokenizer(), runner)
        initial = session.begin_message('user', 'x' * 184 + 'ab')
        self.assertEqual(initial['native_tokens'], 192)
        self.assertEqual(initial['snapshot_positions'], [128, 192])
        original_128 = session._snapshots[0]
        shrunk = session.append_text('c')
        self.assertEqual(shrunk['native_tokens'], 191)
        self.assertEqual(shrunk['common_prefix_tokens'], 190)
        self.assertEqual(shrunk['restore_position'], 0)
        self.assertTrue(shrunk['fallback_full_replay'])
        self.assertEqual(shrunk['snapshot_positions'], [64, 128])
        self.assertIs(session._snapshots[1], original_128)
        self.assert_reference(session, shrunk)
        expanded = session.append_text('d')
        self.assertEqual(expanded['native_tokens'], 192)
        self.assertEqual(expanded['snapshot_positions'], [128, 192])
        self.assertFalse(expanded['rollback'])
        self.assertEqual(expanded['forward_tokens'], 1)
        self.assert_reference(session, expanded)

    def test_multiple_snapshot_boundary_shrinks_and_expansions(self):
        for boundary in (64, 128, 192, 256, 512, 1024):
            with self.subTest(boundary=boundary):
                session = TextStreamSession(MergeTokenizer(), FakeRunner())
                session.begin_message('user', 'x' * (boundary - 8) + 'ab')
                result = session.append_text('c')
                self.assertEqual(result['native_tokens'], boundary - 1)
                expected = [position for position in (boundary - 128, boundary - 64) if position > 0]
                self.assertEqual(result['snapshot_positions'], expected)
                self.assert_reference(session, result)
                result = session.append_text('d')
                self.assertEqual(result['native_tokens'], boundary)
                expected = [position for position in (boundary - 64, boundary) if position > 0]
                self.assertEqual(result['snapshot_positions'], expected)
                self.assert_reference(session, result)

    def test_fallback_when_rollback_precedes_both_snapshots(self):
        session, _ = self.make_session(RewriteTokenizer())
        session.begin_message('user', 'x' * 220)
        self.assertEqual(session.snapshot_positions, [128, 192])
        result = session.append_text('!')
        self.assertEqual(result['common_prefix_tokens'], 0)
        self.assertTrue(result['fallback_full_replay'])
        self.assertEqual(result['forward_tokens'], result['native_tokens'])
        self.assert_reference(session, result)

    def test_zero_token_change_and_empty_append(self):
        session, runner = self.make_session(IgnoreSpacesTokenizer())
        initial = session.begin_message('user', 'hello')
        count = len(runner.calls)
        result = session.append_text('  ')
        self.assertEqual(result['forward_tokens'], 0)
        self.assertEqual(result['net_new_tokens'], 0)
        self.assertEqual(result['risk_probabilities'], initial['risk_probabilities'])
        result = session.append_text('')
        self.assertEqual(len(runner.calls), count)
        self.assertEqual(result['input_characters'], 7)
        self.assert_reference(session, result)

    def test_cross_role_and_session_isolation(self):
        runner = FakeRunner()
        first = TextStreamSession(MergeTokenizer(), runner)
        second = TextStreamSession(MergeTokenizer(), runner)
        first.begin_message('user', 'x' * 130)
        second.begin_message('user', 'unrelated')
        second_state = copy.deepcopy(second._cache)
        result = first.begin_message('assistant', '回答 ab')
        result = first.append_text('c')
        self.assertEqual(result['target_role'], 'assistant')
        self.assertEqual(second._cache, second_state)
        self.assert_reference(first, result)
        # Public views cannot mutate the session.
        first.messages[-1]['content'] = 'edited'
        first.token_ids.clear()
        self.assert_reference(first, result)

    def test_mutating_failure_is_transactional(self):
        for tokenizer, addition in ((CharacterTokenizer(), 'z'), (MergeTokenizer(), 'c')):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                session, runner = self.make_session(tokenizer)
                session.begin_message('user', 'x' * 140 + 'ab')
                before = copy.deepcopy((session._messages, session._ids, session._cache,
                                        session._snapshots, session._probabilities,
                                        session._forward_tokens, session._wall_seconds))
                runner.fail_after_mutation = True
                with self.assertRaises(RuntimeError):
                    session.append_text(addition)
                after = (session._messages, session._ids, session._cache, session._snapshots,
                         session._probabilities, session._forward_tokens, session._wall_seconds)
                self.assertEqual(before, after)
                runner.fail_after_mutation = False
                result = session.append_text(addition)
                self.assert_reference(session, result)

    def test_too_long_rejected_before_forward(self):
        session, runner = self.make_session()
        session.begin_message('user', 'x')
        before = (session.messages, session.token_ids, len(runner.calls))
        with self.assertRaises(ValueError):
            session.append_text('x' * MAX_INPUT_TOKENS)
        self.assertEqual(before, (session.messages, session.token_ids, len(runner.calls)))

    def test_encoding_overflow_is_rejected_without_session_mutation(self):
        tokenizer = OverflowTokenizer()
        session, runner = self.make_session(tokenizer)
        session.begin_message('user', 'x' * 140)
        before = copy.deepcopy((session.messages, session.token_ids, session._cache, session._snapshots))
        calls = len(runner.calls)
        tokenizer.return_overflow = True
        with self.assertRaisesRegex(ValueError, 'overflowing'):
            session.append_text(' appended content')
        self.assertEqual(before, (session.messages, session.token_ids, session._cache, session._snapshots))
        self.assertEqual(len(runner.calls), calls)
        tokenizer.return_overflow = False
        self.assert_reference(session, session.append_text(' recovered'))

    def test_configured_truncation_is_rejected_before_encoding(self):
        tokenizer = ConfigurableTruncationTokenizer()
        session, runner = self.make_session(tokenizer)
        session.begin_message('user', 'x' * 140)
        before = copy.deepcopy((session.messages, session.token_ids, session._cache, session._snapshots))
        calls, encodes = len(runner.calls), tokenizer.calls
        tokenizer.truncation = {'max_length': 16}
        with self.assertRaisesRegex(ValueError, 'truncation is enabled'):
            session.append_text(' appended content')
        self.assertEqual(before, (session.messages, session.token_ids, session._cache, session._snapshots))
        self.assertEqual((len(runner.calls), tokenizer.calls), (calls, encodes))

    def test_hf_style_encode_receives_explicit_truncation_false(self):
        class HFStyleTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False, truncation=None, **kwargs):
                if truncation is not False:
                    raise AssertionError('HF-style encode must explicitly disable truncation')
                return super().encode(text, add_special_tokens)
        session, _ = self.make_session(HFStyleTokenizer())
        result = session.begin_message('user', 'test')
        self.assertEqual(result['native_tokens'], len('USER:\ntest'))

    def test_failed_new_message_and_invalid_output_are_transactional(self):
        session, runner = self.make_session()
        session.begin_message('user', 'x' * 140)
        before = copy.deepcopy((session.messages, session.token_ids, session._cache, session._snapshots))
        runner.fail_after_mutation = True
        with self.assertRaises(RuntimeError):
            session.begin_message('assistant', 'reply')
        self.assertEqual(before, (session.messages, session.token_ids, session._cache, session._snapshots))
        runner.fail_after_mutation = False
        original_runner = session._runner
        def invalid_output(ids, cache):
            cache, probabilities = original_runner(ids, cache)
            probabilities['user']['unsafe'] = float('nan')
            return cache, probabilities
        session._runner = invalid_output
        with self.assertRaises(ValueError):
            session.append_text('z')
        self.assertEqual(before, (session.messages, session.token_ids, session._cache, session._snapshots))

    def test_snapshots_share_no_mutable_fake_state(self):
        session, _ = self.make_session()
        session.begin_message('user', 'x' * 180)
        caches = [session._cache] + [snapshot.cache for snapshot in session._snapshots]
        for index, cache in enumerate(caches):
            for other in caches[index + 1:]:
                self.assertIsNot(cache, other)
                self.assertIsNot(cache['ids'], other['ids'])
                self.assertIsNot(cache['recurrent']['state'], other['recurrent']['state'])

    def test_variable_text_boundaries_match_reference(self):
        for sizes in ([1], [7, 1, 19, 3, 64]):
            session, _ = self.make_session(MergeTokenizer(), chunk_tokens=17)
            text = '公开资料 abc🧪，ab  English\n' * 70
            result = session.begin_message('user')
            offset, iteration = 0, 0
            while offset < len(text):
                end = min(len(text), offset + sizes[iteration % len(sizes)])
                result = session.append_text(text[offset:end])
                self.assert_reference(session, result)
                self.assertLessEqual(len(result['snapshot_positions']), 2)
                offset, iteration = end, iteration + 1
            self.assertGreater(result['native_tokens'], 512)

    def test_long_prefill_uses_large_blocks_and_only_two_clones(self):
        runner = FakeRunner()
        clones = []
        def tracked_clone(cache):
            clones.append(len(cache['ids']))
            return copy.deepcopy(cache)
        session = TextStreamSession(CharacterTokenizer(), runner, clone_cache=tracked_clone)
        result = session.begin_message('user', 'x' * (8192 - len('USER:\n')))
        self.assertEqual(result['native_tokens'], 8192)
        self.assertLessEqual(result['forward_calls'], 6)
        self.assertEqual(clones, [8128, 8192])
        self.assertEqual(result['snapshot_positions'], [8128, 8192])
        self.assertEqual(sum(len(chunk) for chunk in runner.calls), 8192)
        self.assert_reference(session, result)

    def test_long_fallback_replay_keeps_only_final_aligned_snapshots(self):
        runner = FakeRunner()
        clones = []
        def tracked_clone(cache):
            clones.append(len(cache['ids']))
            return copy.deepcopy(cache)
        session = TextStreamSession(RewriteTokenizer(), runner, clone_cache=tracked_clone,
                                    prefill_chunk_tokens=1024)
        session.begin_message('user', 'x' * (8191 - len('USER:\n')))
        calls_before = len(runner.calls)
        clones.clear()
        result = session.append_text('!')
        self.assertTrue(result['fallback_full_replay'])
        self.assertEqual(result['restore_position'], 0)
        self.assertEqual(result['forward_tokens'], 8192)
        self.assertEqual(result['replay_tokens'], 8191)
        self.assertLessEqual(len(runner.calls) - calls_before, 10)
        self.assertEqual(clones, [8128, 8192])
        self.assertEqual(result['snapshot_positions'], [8128, 8192])
        self.assert_reference(session, result)

    def test_long_append_reuses_old_cache_without_obsolete_snapshot_clones(self):
        runner = FakeRunner()
        clones = []
        def tracked_clone(cache):
            clones.append(len(cache['ids']))
            return copy.deepcopy(cache)
        session = TextStreamSession(CharacterTokenizer(), runner, clone_cache=tracked_clone)
        before = session.begin_message('user', 'x' * 140)
        clones.clear()
        result = session.append_text('y' * 6000)
        self.assertFalse(result['rollback'])
        self.assertEqual(result['forward_tokens'], 6000)
        self.assertLessEqual(result['forward_calls'], 7)
        self.assertEqual(result['snapshot_positions'], [6080, 6144])
        self.assertEqual(clones, [before['native_tokens'], 6080, 6144])
        self.assert_reference(session, result)


if __name__ == '__main__':
    unittest.main()
