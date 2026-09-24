"""CPU state/data checks only; no torch import and no neural execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import train_prefix as training


class PrefixTrainingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.manifest, cls.data = training.load_data(cls.root / 'data/prefix_v2')

    def test_import_and_frozen_data_without_neural_libraries(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        self.assertEqual({s: len(rows) for s, rows in self.data.items()}, training.COUNTS)

    def test_each_original_is_seen_exactly_once_per_epoch(self):
        wanted = {row['sample_id'] for row in self.data['train']}
        for epoch in range(2):
            groups = training.epoch_groups(self.data['train'], epoch)
            seen = [row['sample_id'] for group in groups for row in group]
            self.assertEqual(len(groups), 2082)
            self.assertEqual(len(seen), len(wanted))
            self.assertEqual(set(seen), wanted)
            self.assertTrue(all(0 < len(group) <= 16 for group in groups))

    def test_augmentation_once_and_denominator_is_anchor_count(self):
        for row in self.data['train']:
            plans = [training.prefix_plan(row, epoch) for epoch in range(2)]
            self.assertEqual(sum(bool(aug) for _, aug, _ in plans), 1)
            for weak, augmented, count in plans:
                self.assertEqual(count, len(weak) + len(augmented))
                if weak:
                    # Dividing by sum(confidence) would erase the intended .1 down-weight.
                    self.assertLess(sum(a['confidence'] for a in weak + augmented), count)
                if row['source_label'] == 'unsafe':
                    self.assertEqual(weak, [])

    def test_rl_record_sampling_excludes_unknown_context_tier(self):
        pool = training.rl_pool(self.data['train'])
        self.assertEqual(len(pool), 32288)
        self.assertEqual(len({row['sample_id'] for row in pool}), len(pool))
        self.assertTrue(all(row['label_tier'] in training.RL_TIERS for row in pool))
        self.assertEqual([r['sample_id'] for r in pool[:16]],
                         [r['sample_id'] for r in training.rl_pool(self.data['train'])[:16]])

    def test_complete_view_coverage(self):
        digest, counts = training.view_digest(self.data)
        self.assertEqual(sum(counts.values()), 85399)
        self.assertEqual(digest, '661c3133168750289bafb8771d6e5d03303b717ea6c7386617d844c25fb5dede')
        row = self.data['train'][0]
        views = list(training.view_inputs(row, 'train'))
        self.assertEqual(views[1][1], views[0][1] + row['augmentation']['suffix'])
        row = self.data['dev'][0]
        views = list(training.view_inputs(row, 'dev'))
        for view, (_, text, _) in zip(row['text_cut_views'], views[1:]):
            expected = copy.deepcopy(row['messages'])
            expected[-1]['content'] = expected[-1]['content'][:view['cut_chars']]
            self.assertEqual(text, training.serialize(expected))

    def test_native_mismatch_fails_without_replacing_ids(self):
        tokenizer = mock.Mock()
        saved = [10, 20]
        tokenizer.encode.return_value = [10]
        with self.assertRaises(ValueError):
            training.check_encoding(tokenizer, 'literal', saved)
        self.assertEqual(saved, [10, 20])
        tokenizer.encode.assert_called_once_with('literal', add_special_tokens=False, truncation=False)

    def test_cut_schedule_reuses_endpoint_and_deduplicates_equivalent_views(self):
        row = copy.deepcopy(self.data['dev'][0])
        duplicate = copy.deepcopy(row)
        duplicate['sample_id'] += ':duplicate-diagnostic'
        table = {text: ids for _, text, ids in training.view_inputs(row, 'dev')}
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda text, **kwargs: table[text]
        jobs, plans = training.prepare_cut_jobs([row, duplicate], tokenizer)
        nonendpoint = {tuple(view['ids']) for view in row['text_cut_views'] if view['ids'] != row['ids']}
        self.assertEqual(len(jobs), len(nonendpoint))
        self.assertEqual(plans[row['sample_id']], plans[duplicate['sample_id']])
        self.assertTrue(any(item['job_index'] is None for item in plans[row['sample_id']]))
        self.assertEqual(tokenizer.encode.call_count, 2 * len(row['text_cut_views']))

    def test_resume_recovers_previous_selected_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, best = root / 'epoch.safetensors', root / 'best.safetensors'
            source.write_bytes(b'committed')
            best.write_bytes(b'uncommitted-newer-checkpoint')
            selected = {'checkpoint_sha256': training.sha(source), 'source_checkpoint': str(source)}
            training.reconcile_selected(selected, best)
            self.assertEqual(best.read_bytes(), b'committed')
            source.write_bytes(b'corrupt')
            best.write_bytes(b'wrong')
            with self.assertRaises(ValueError):
                training.reconcile_selected(selected, best)

    def test_resume_removes_only_uncommitted_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'losses.jsonl'
            path.write_text(''.join(json.dumps({'step': n}) + '\n' for n in range(1, 6)))
            training.trim_uncommitted_log(path, 3)
            self.assertEqual([json.loads(l)['step'] for l in path.read_text().splitlines()], [1, 2, 3])

    def test_selection_never_promotes_ineligible_or_equal_score(self):
        incumbent = {'eligible': False, 'selection_score': .8}
        self.assertFalse(training.stream_metrics.is_better({'eligible': False, 'selection_score': 1}, incumbent))
        self.assertTrue(training.stream_metrics.is_better({'eligible': True, 'selection_score': .7}, incumbent))
        self.assertFalse(training.stream_metrics.is_better({'eligible': True, 'selection_score': .7},
                                                          {'eligible': True, 'selection_score': .7}))


if __name__ == '__main__':
    unittest.main()
