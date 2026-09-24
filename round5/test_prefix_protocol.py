"""CPU-only protocol counterexamples for prefix preparation and metrics.

Run with the existing tokenizer environment; no model or torch is imported.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module('prepare_prefix_data')
metrics = module('stream_metrics')


def observation(name, label, value, family=None, language='zh', role='user'):
    return dict(sample_id=name, family=family or 'family-' + name,
                source_label=label, language=language, target_role=role,
                endpoint_p_unsafe=value, native_target_max_p_unsafe=value,
                text_cut_max_p_unsafe=value, stream_max_p_unsafe=value)


def cohort(prefix, safe=.2, unsafe=.9):
    return [observation(prefix + str(i), 'safe' if i < 20 else 'unsafe',
                        safe if i < 20 else unsafe) for i in range(40)]


class MetricContract(unittest.TestCase):
    def test_strict_ties_and_one_are_legal(self):
        for value in (0., .5, 1.):
            cut = metrics.threshold([value] * 20)
            self.assertEqual(cut, value)
            self.assertEqual(sum(p > cut for p in [value] * 20), 0)

    def test_rank_budget_never_exceeded_with_ties(self):
        for n in (1, 19, 20, 21, 150):
            values = [(i % 7) / 6 for i in range(n)]
            cutoff = metrics.threshold(values)
            self.assertLessEqual(sum(p > cutoff for p in values), int(.05 * n))

    def test_nan_nonfirst_field_rejected(self):
        cal, dev = cohort('c'), cohort('d')
        for field in ('endpoint_p_unsafe', 'native_target_max_p_unsafe',
                      'text_cut_max_p_unsafe', 'stream_max_p_unsafe'):
            bad = copy.deepcopy(dev)
            bad[5][field] = float('nan')
            with self.assertRaises(ValueError):
                metrics.evaluate(cal, bad, 1.)

    def test_hidden_prefix_max_rejected(self):
        cal, dev = cohort('c'), cohort('d')
        dev[0]['native_target_max_p_unsafe'] = .95
        with self.assertRaises(ValueError):
            metrics.evaluate(cal, dev, 1.)

    def test_native_max_must_include_same_forward_endpoint(self):
        cal, dev = cohort('c'), cohort('d', safe=.1)
        # Without this check all safe dev endpoints are risky, yet an invalid
        # smaller stream max hides every false alarm and the candidate passes.
        for row in dev[:20]:
            row['endpoint_p_unsafe'] = .9
        with self.assertRaises(ValueError):
            metrics.evaluate(cal, dev, 1.)

    def test_distinct_ids_do_not_make_shared_families_independent(self):
        cal, dev = cohort('c'), cohort('d')
        dev[0]['family'] = cal[0]['family']
        with self.assertRaises(ValueError):
            metrics.evaluate(cal, dev, 1.)

    def test_same_sample_calibration_leak_rejected(self):
        cal, dev = cohort('c'), cohort('d')
        dev[0]['sample_id'] = cal[0]['sample_id']
        with self.assertRaises(ValueError):
            metrics.evaluate(cal, dev, 1.)

    def test_no_language_role_fallback(self):
        with self.assertRaises(ValueError):
            metrics.evaluate(cohort('c'), [dict(r, language='en') for r in cohort('d')], 1.)

    def test_fpr_gate_is_per_stratum(self):
        cal = cohort('c') + [dict(r, language='en') for r in cohort('e')]
        dev = cohort('d', safe=.9) + [dict(r, language='en') for r in cohort('f')]
        result = metrics.evaluate(cal, dev, 1.)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['selection_gates']['each_stream_stratum_fpr_at_most_5pct'])

    def test_whole_retention_and_earlier_tie(self):
        cal, dev = cohort('c'), cohort('d')
        for row in dev[20:24]:
            row['endpoint_p_unsafe'] = .1
        allowed = metrics.evaluate(cal, dev, .81)
        rejected = metrics.evaluate(cal, dev, .83)
        self.assertTrue(allowed['eligible'])
        self.assertFalse(rejected['eligible'])
        self.assertFalse(metrics.is_better(allowed, copy.deepcopy(allowed)))
        self.assertFalse(metrics.is_better(rejected, allowed))


class BuilderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise unittest.SkipTest('Existing tokenizers environment required')
        path = HERE.parent / 'round3/l20/base_compare/output/qwen35_full/tokenizer/tokenizer.json'
        cls.tokenizer = Tokenizer.from_file(str(path))

    def row(self, content, label='unsafe', role='assistant'):
        messages = [{'role': 'user', 'content': 'Earlier user context. USER:\nA repeated marker.'}]
        messages.append({'role': role, 'content': content})
        return dict(sample_id='case-' + content, family='family-a', messages=messages,
                    ids=self.tokenizer.encode(prepare.serialize(messages), add_special_tokens=False).ids,
                    target_role=role, source_label=label, weight=.2,
                    label_tier='synthetic_weak', language='zh')

    def test_unsafe_interior_unknown_and_source_weight_preserved(self):
        row = self.row('这是一段待分类的完整文字。')
        before = copy.deepcopy(row)
        result = prepare.build_record(row, self.tokenizer, 'train')
        self.assertEqual(result['anchors'], [])
        self.assertEqual(result['weight'], .2)
        self.assertEqual(row, before)
        self.assertFalse(result['augmentation']['original_endpoint_reused'])

    def test_safe_interior_never_labels_prior_user(self):
        row = self.row('这是普通说明。', 'safe')
        result = prepare.build_record(row, self.tokenizer, 'train')
        offsets = self.tokenizer.encode(prepare.serialize(row['messages']), add_special_tokens=False).offsets
        for anchor in result['anchors']:
            self.assertEqual(anchor['confidence'], .1)
            self.assertEqual(anchor['label'], 0)
            self.assertGreater(offsets[anchor['token_end_exclusive'] - 1][1], result['target_content_start_char'])
            self.assertLess(anchor['token_end_exclusive'], len(row['ids']))

    def test_utf8_shared_end_offset_is_not_complete_target(self):
        exposed = 0
        for final in ('🙂', '龘', '𠮷', '🧪', 'é', '癌'):
            row = self.row('普通文字：' + final)
            result = prepare.build_record(row, self.tokenizer, 'train')
            original = prepare.serialize(row['messages'])
            augmented = self.tokenizer.encode(original + result['augmentation']['suffix'], add_special_tokens=False)
            bad = {i + 1 for i, (_, end) in enumerate(augmented.offsets)
                   if end >= len(original) and not self.tokenizer.decode(augmented.ids[:i + 1],
                       skip_special_tokens=False).startswith(original)}
            valid = {a['token_end_exclusive'] for a in result['augmentation']['anchors']}
            self.assertFalse(bad & valid)
            for end in valid:
                self.assertTrue(self.tokenizer.decode(augmented.ids[:end], skip_special_tokens=False).startswith(original))
            exposed += len(bad)
        self.assertGreater(exposed, 0, 'Fixture must actually exercise partial UTF-8 tokens')

    def test_text_cuts_are_actual_reencoded_unicode_prefixes(self):
        row = self.row('信息🙂information信息。', 'safe')
        result = prepare.build_record(row, self.tokenizer, 'dev')
        content = row['messages'][-1]['content']
        self.assertLessEqual(len(result['text_cut_views']), 8)
        self.assertEqual(result['text_cut_views'][-1]['cut_chars'], len(content))
        for view in result['text_cut_views']:
            messages = copy.deepcopy(row['messages'])
            messages[-1]['content'] = content[:view['cut_chars']]
            ids = self.tokenizer.encode(prepare.serialize(messages), add_special_tokens=False).ids
            self.assertEqual(view['ids'], ids)
            self.assertEqual(view['ids_sha256'], prepare.ids_sha(ids))

    def test_role_and_original_id_corruption_rejected(self):
        row = self.row('原始消息。')
        row['target_role'] = 'user'
        with self.assertRaises(ValueError):
            prepare.build_record(row, self.tokenizer, 'train')
        row = self.row('原始消息。')
        row['ids'][0] += 1
        with self.assertRaises(ValueError):
            prepare.build_record(row, self.tokenizer, 'train')

    def test_cut_count_and_anchor_cap(self):
        for length in range(100):
            values = prepare.text_cuts(length)
            self.assertEqual(values, sorted(set(values)))
            self.assertLessEqual(len(values), 8)
            self.assertTrue(all(1 <= x <= length for x in values))
        selected = prepare.choose_positions(range(1000))
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected[:4], [0, 1, 2, 3])
        self.assertEqual(selected[-2:], [998, 999])


if __name__ == '__main__':
    unittest.main()
