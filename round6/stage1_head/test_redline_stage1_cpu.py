"""Stage 1 under the red-line policy (CPU; numpy only): cache relabelling and the 3-level comparison analysis."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe"))
from relabel_cache import WHOLE, load_positions, prefix_ends, relabel, runA_ends  # noqa: E402
from probe_common import serialize  # noqa: E402


def encode(text):
    return [ord(c) % 5000 for c in text], [(i, i + 1) for i in range(len(text))]


def runA_row(sample_id, label):
    content = "安" * 10 + ("害" * 20 if label == "unsafe" else "安" * 20)
    return {"sample_id": sample_id, "label": label, "onset_char": 10 if label == "unsafe" else None,
            "onset_end_char": 12 if label == "unsafe" else None,
            "messages": [{"role": "user", "content": "问" * 10}, {"role": "assistant", "content": content}]}


def probes(*pairs):
    return [{"cut": cut, "level": level} for cut, level in pairs]


class RelabelTests(unittest.TestCase):
    def test_runA_positions_get_levels_from_probes(self):
        rows = [runA_row("a", "unsafe"), runA_row("b", "safe"), runA_row("c", "safe"), runA_row("d", "unsafe")]
        ends = {r["sample_id"]: runA_ends(r, encode) for r in rows}
        self.assertTrue(all(1 <= e <= 30 for v in ends.values() for e in v))
        self.assertNotIn(11, ends["a"])                          # the old onset span (10, 12) was never cached
        records = [{"sample_id": r["sample_id"]} for r in rows]
        record = np.concatenate([np.full(len(ends[r["sample_id"]]), i, np.int32) for i, r in enumerate(rows)])
        old = np.concatenate([(np.asarray(ends[r["sample_id"]]) > 10).astype(np.int8) * (r["label"] == "unsafe") for r in rows])
        positions = {"label": old, "weight": np.ones(len(old), np.float32), "record": record}
        labels = {"a": {"label": "unsafe", "level": "unsafe",
                        "probes": probes((10, "safe"), (12, "controversial"), (18, "controversial"), (20, "unsafe"), (30, "unsafe"))},
                  "b": {"label": "controversial", "level": "controversial", "probes": probes((6, "safe"), (8, "controversial"), (30, "controversial"))},
                  "c": {"label": "unusable", "level": None, "probes": []}}
        label, weight, stats = relabel(positions, records, labels, lambda meta: ends[meta["sample_id"]])
        got = {}
        for j, rec in enumerate(record):
            got.setdefault(rows[rec]["sample_id"], []).append((ends[rows[rec]["sample_id"]][list(np.flatnonzero(record == rec)).index(j)],
                                                               int(label[j]) if weight[j] > 0 else None))
        for end, cls in got["a"]:
            expected = 0 if end <= 10 else 2 if 12 <= end <= 18 else 1 if end >= 20 else None
            self.assertEqual(cls, expected, end)                 # 2 = controversial, 1 = unsafe (head order)
        for end, cls in got["b"]:
            self.assertEqual(cls, 0 if end <= 6 else 2 if end >= 8 else None, end)
        self.assertTrue(all(cls is None for _, cls in got["c"] + got["d"]))
        self.assertEqual(stats["dropped_unusable"], len(ends["c"]))
        self.assertEqual(stats["dropped_unlabelled"], len(ends["d"]))
        self.assertEqual(stats["kept"] + sum(v for k, v in stats.items() if k.startswith("dropped_")), len(old))
        self.assertIn("1->2", stats["old_to_new_class"])          # an old unsafe position that is now controversial

    def test_count_mismatch_and_cache_reading(self):
        positions = {"label": np.zeros(3, np.int8), "weight": np.ones(3, np.float32), "record": np.zeros(3, np.int32)}
        labels = {"a": {"label": "safe", "level": "safe", "probes": probes((5, "safe"))}}
        _, weight, stats = relabel(positions, [{"sample_id": "a"}], labels, lambda meta: [1, 2])
        self.assertEqual((weight.sum(), stats["dropped_offsets"]), (0, 3))
        with tempfile.TemporaryDirectory() as tmp:
            for shard, (lab, rec) in enumerate((([0, 1], [0, 0]), ([1], [1]))):
                np.savez(Path(tmp) / f"cache_runA_train_{shard:03d}.npz", label=np.asarray(lab, np.int8),
                         weight=np.ones(len(lab), np.float32), record=np.asarray(rec, np.int32), hidden=np.zeros((len(lab), 2)))
            loaded = load_positions(tmp, "runA_train")
            self.assertEqual((loaded["label"].tolist(), loaded["record"].tolist()), ([0, 1, 1], [0, 0, 1]))

    def test_prefix_v2_views(self):
        messages = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "一二三四五六七八九十"}]
        ids, _ = encode(serialize(messages))
        start = len(serialize(messages)) - 10
        row = {"messages": messages, "ids": ids, "source_label": "safe",
               "anchors": [{"token_end_exclusive": start + 3}, {"token_end_exclusive": start + 7}],
               "augmentation": {"anchors": [{"token_end_exclusive": 99}] * 4},
               "target_token_positions": list(range(start, start + 10))}
        self.assertEqual(prefix_ends(row, {"view": "original"}, "train", encode), [3, 7])
        self.assertEqual(prefix_ends(row, {"view": "augmented"}, "train", encode), [WHOLE] * 4)
        cal = prefix_ends(row, {"view": "original"}, "calibration", encode)
        self.assertEqual(cal[0], WHOLE)
        self.assertTrue(all(1 <= e < 10 for e in cal[1:]))
        self.assertIsNone(prefix_ends({**row, "ids": ids[:-1]}, {"view": "original"}, "train", encode))


class UserRoleTests(unittest.TestCase):
    def test_prompt_records_use_cached_char_ends(self):
        records = [{"sample_id": "k0:prompt:unsafe", "char_ends": [2, 4, 6, 8, 10]},
                   {"sample_id": "k0:prompt:safe", "char_ends": [3, 6, 9]}]
        positions = {"label": np.asarray([1] * 5 + [0] * 3, np.int8), "weight": np.ones(8, np.float32),
                     "record": np.asarray([0] * 5 + [1] * 3, np.int32)}
        labels = {"k0:prompt:unsafe": {"label": "unsafe", "level": "unsafe",
                                       "probes": probes((4, "safe"), (6, "unsafe"), (10, "unsafe"))},
                  "k0:prompt:safe": {"label": "safe", "level": "safe", "probes": probes((9, "safe"))}}
        label, weight, stats = relabel(positions, records, labels, lambda meta: meta["char_ends"])
        self.assertEqual([int(x) if w else None for x, w in zip(label, weight)], [0, 0, 1, 1, 1, 0, 0, 0])
        self.assertEqual(stats["kept"], 8)

    def test_analysis_without_official_set(self):
        rng = np.random.default_rng(5)
        runA, prefix = [], []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for variant, separation in (("init", 0.0), ("risk", 3.0)):
                directory = tmp / "eval" / f"eval_{variant}"
                directory.mkdir(parents=True)
                for split in ("calibration", "dev"):
                    with gzip.open(directory / f"runA_prompts_{split}.jsonl.gz", "wt") as out:
                        for i in range(60):
                            cut = i % 2 == 1
                            out.write(json.dumps({"sample_id": f"{split}{i}:prompt:x", "family": f"f{i}", "language": ("zh", "en")[i % 4 // 2],
                                                  "char_ends": list(range(1, 11)),
                                                  "logprobs": [[0.0, -4.0, (2.5 if cut and e > 4 else -3.0 + (0 if cut else -separation))
                                                                + rng.normal(0, .3)] for e in range(1, 11)]}) + "\n")
                    with gzip.open(directory / f"prefix_v2_{split}.jsonl.gz", "wt") as out:
                        for i in range(40):
                            out.write(json.dumps({"sample_id": f"p{split}{i}", "family": f"pf{i}", "language": ("zh", "en")[i % 4 // 2],
                                                  "target_role": "user",
                                                  "logprobs": [[0.0, -4.0, (2.5 if i % 2 and k > 3 else -3.0) + rng.normal(0, .3)]
                                                               for k in range(8)]}) + "\n")
                    if variant == "init":
                        runA += [{"sample_id": f"{split}{i}:prompt:x", "label": ("safe", "unsafe")[i % 2], "level": ("safe", "unsafe")[i % 2],
                                  "stratum": ("normal", "redline")[i % 2],
                                  "onsets": {"controversial": {"prev_cut": 4, "cut": 5}} if i % 2 else {}} for i in range(60)]
                        prefix += [{"sample_id": f"p{split}{i}", "label": ("safe", "unsafe")[i % 2], "level": ("safe", "unsafe")[i % 2],
                                    "stratum": ("normal", "redline")[i % 2], "onsets": {}} for i in range(40)]
            for name, rows in (("runA.jsonl", runA), ("prefix.jsonl", prefix)):
                (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = tmp / "compare.json"
            subprocess.run([sys.executable, str(HERE / "analyze_redline.py"), "--role", "user", "--eval-dir", str(tmp / "eval"),
                            "--variants", "init,risk", "--runA-labels", str(tmp / "runA.jsonl"), "--prefix-labels", str(tmp / "prefix.jsonl"),
                            "--bootstrap", "20", "--output", str(out)], check=True, capture_output=True)
            result = json.loads(out.read_text())
            self.assertEqual(result["role"], "user")
            rules = result["scorers"]["risk"]["rules"]["threshold"]
            self.assertNotIn("official_thinking", rules)
            self.assertIn("early_fire", rules["runA_dev"])


class RedlineAnalysisTests(unittest.TestCase):
    def test_compare_under_red_line_labels(self):
        rng = np.random.default_rng(3)

        def lp(cut, restating, separation):
            base = 2.5 if cut else (2.5 - separation if restating else -3.0)
            return [0.0, -4.0, base + rng.normal(0, .5)]       # the controversial logit carries the cut score

        runA_labels, prefix_labels = [], []
        for split in ("calibration", "dev"):
            for i in range(80):
                level = ("safe", "controversial", "safe", "unsafe")[i % 4]
                runA_labels.append({"sample_id": f"{split}{i}", "label": level, "level": level,
                                    "stratum": ("normal", "redline", "non_redline_harm", "redline")[i % 4],
                                    "onsets": {"controversial": {"prev_cut": 8, "cut": 10}} if level != "safe" else {}})
            runA_labels.append({"sample_id": f"{split}x", "label": "unusable", "level": None, "stratum": None, "onsets": {}})
            for i in range(40):
                level = ("safe", "unsafe")[i % 2]
                prefix_labels.append({"sample_id": f"p{split}{i}", "label": level, "level": level,
                                      "stratum": "normal" if level == "safe" else "redline", "onsets": {}})

        def write_dir(directory, separation):
            directory.mkdir(parents=True)
            for split in ("calibration", "dev"):
                with gzip.open(directory / f"runA_{split}.jsonl.gz", "wt") as out:
                    for i in list(range(80)) + ["x"]:
                        cut_level = i != "x" and i % 4 in (1, 3)
                        restating = i != "x" and i % 4 == 2           # harm outside the red lines, now safe
                        ends = list(range(1, 21))
                        out.write(json.dumps({"sample_id": f"{split}{i}", "family": f"f{i}", "language": "zh" if i == "x" or i % 8 < 4 else "en",
                                              "char_ends": ends,
                                              "logprobs": [lp(cut_level and e >= 10, restating, separation) for e in ends]}) + "\n")
                with gzip.open(directory / f"prefix_v2_{split}.jsonl.gz", "wt") as out:
                    for i in range(40):
                        out.write(json.dumps({"sample_id": f"p{split}{i}", "family": f"pf{i}", "language": ("zh", "en")[i % 4 // 2],
                                              "target_role": "assistant",
                                              "logprobs": [lp(i % 2 == 1 and k > 5, False, separation) for k in range(12)]}) + "\n")
            with gzip.open(directory / "official_sequences.jsonl.gz", "wt") as seq, gzip.open(directory / "official_rows.jsonl.gz", "wt") as rows:
                for i in range(30):
                    label = ("safe", "unsafe")[i % 2]
                    seq.write(json.dumps({"sequence_id": f"q{i}", "logprobs": [lp(label == "unsafe" and k > 10, label == "safe", separation)
                                                                               for k in range(30)]}) + "\n")
                    rows.write(json.dumps({"sample_id": f"o{i}", "split": "thinking", "label": label.capitalize(),
                                           "sequence_id": f"q{i}"}) + "\n")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_dir(tmp / "eval" / "eval_init", separation=0.0)
            write_dir(tmp / "eval" / "eval_risk", separation=3.0)
            for name, rows in (("runA.jsonl", runA_labels), ("prefix.jsonl", prefix_labels)):
                (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = tmp / "compare.json"
            subprocess.run([sys.executable, str(HERE / "analyze_redline.py"), "--eval-dir", str(tmp / "eval"), "--variants", "init,risk",
                            "--runA-labels", str(tmp / "runA.jsonl"), "--prefix-labels", str(tmp / "prefix.jsonl"),
                            "--bootstrap", "20", "--output", str(out)], check=True, capture_output=True)
            result = json.loads(out.read_text())
            init, risk = result["scorers"]["init"], result["scorers"]["risk"]
            self.assertEqual(init["skipped_streams"], {"runA_calibration:unusable": 1, "runA_dev:unusable": 1})
            self.assertGreater(risk["stream_auc_max_score"]["runA_dev"], init["stream_auc_max_score"]["runA_dev"])
            rule = risk["rules"]["threshold"]["runA_dev"]
            self.assertEqual(set(rule["recall_by_level"]), {"controversial", "unsafe"})
            self.assertEqual(set(rule["fpr_by_stratum"]), {"normal", "non_redline_harm"})
            self.assertIn("early_fire", rule)
            # at the same calibration FPR, a readout that lets non-red-line harm pass keeps far more recall
            self.assertGreater(rule["recall"], init["rules"]["threshold"]["runA_dev"]["recall"] + 0.5)
            # T027: the official set scored by our labels (official safe restating a red line -> controversial)
            official = [{"sample_id": f"o{i}", "label": level, "level": level, "stratum": "redline" if level != "safe" else "normal",
                         "onsets": {}} for i in range(30)
                        for level in [("controversial" if i % 4 == 0 else "safe") if i % 2 == 0 else ("unsafe" if i % 4 == 1 else "safe")]]
            official.append({"sample_id": "o0x", "label": "unusable", "level": None, "stratum": None, "onsets": {}})
            (tmp / "official.jsonl").write_text("".join(json.dumps(r) + "\n" for r in official))
            done = subprocess.run([sys.executable, str(HERE / "analyze_redline.py"), "--eval-dir", str(tmp / "eval"), "--variants", "risk",
                                   "--runA-labels", str(tmp / "runA.jsonl"), "--prefix-labels", str(tmp / "prefix.jsonl"),
                                   "--official-labels", str(tmp / "official.jsonl"), "--bootstrap", "20", "--output", str(out)],
                                  check=True, capture_output=True, text=True)
            result = json.loads(out.read_text())
            self.assertEqual(result["official_labels"], "red-line policy (T027)")
            entry = result["scorers"]["risk"]["rules"]["threshold"]["official_thinking"]
            self.assertEqual(set(entry["crosstab"]), {"official_safe:controversial", "official_safe:safe",
                                                      "official_unsafe:unsafe", "official_unsafe:safe"})
            self.assertEqual(entry["crosstab"]["official_safe:controversial"]["n"], 8)
            self.assertIn("recall_by_level", entry)
            self.assertIn("official by our labels", done.stdout)


if __name__ == "__main__":
    unittest.main()
