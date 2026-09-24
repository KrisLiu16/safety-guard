"""Onset relocation (CPU only): clause cuts, bisection, judge-copy integrity, blind Tasks, comparison with v14."""
import filecmp
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
sys.path.insert(0, str(HERE))
from onset import bisect, clause_cuts, monotonic  # noqa: E402
from extract_onsets import relation, summarize  # noqa: E402


class CutTests(unittest.TestCase):
    def test_clause_cuts(self):
        self.assertEqual(clause_cuts("你好，世界。再见"), [3, 6, 8])
        self.assertEqual(clause_cuts("no boundary"), [11])
        self.assertEqual(clause_cuts("a. b! c"), [2, 5, 7])      # the cut sits right after the mark


class BisectTests(unittest.TestCase):
    def test_finds_first_unsafe_clause_in_log_calls(self):
        cuts = list(range(10, 330, 10))                      # 32 clauses
        for threshold in (10, 95, 200, 320):
            calls = []

            def judge(cut):
                calls.append(cut)
                return "unsafe" if cut >= threshold else "safe"

            result = bisect(cuts, judge)
            first = min(c for c in cuts if c >= threshold)
            self.assertEqual((result["status"], result["cut"]), ("located", first))
            self.assertEqual(result["prev_cut"], cuts[cuts.index(first) - 1] if first != cuts[0] else 0)
            self.assertLessEqual(len(calls), math.ceil(math.log2(len(cuts))) + 2)   # + whole + confirmation
            self.assertEqual(len(calls), len(set(calls)))        # no prefix judged twice

    def test_other_outcomes(self):
        cuts = [5, 10, 15, 20]
        self.assertEqual(bisect(cuts, lambda c: "safe")["status"], "whole_not_unsafe")
        self.assertEqual(bisect(cuts, lambda c: None if c == 10 else "unsafe" if c >= 15 else "safe")["status"], "judge_error")
        self.assertEqual(bisect(cuts, lambda c: "controversial" if c < 20 else "unsafe")["cut"], 20)   # controversial != unsafe
        noisy = {5: "safe", 10: "unsafe", 15: "safe", 20: "unsafe"}
        self.assertEqual(bisect(cuts, noisy.get)["status"], "nonmonotonic")    # caught by the confirmation probe
        self.assertEqual(bisect([5, 10], lambda c: "unsafe" if c >= 10 else "safe")["status"], "located")
        self.assertTrue(monotonic([{"cut": 5, "verdict": "safe"}, {"cut": 9, "verdict": "unsafe"}]))


class CopyTests(unittest.TestCase):
    def test_judge_files_are_byte_identical_copies(self):
        for name in ("pipeline.py", "taxonomy.py"):
            self.assertTrue(filecmp.cmp(HERE / "flow" / name, HERE.parent / "judge_v3/flow" / name, shallow=False),
                            f"{name} drifted from judge_v3; copy it again")


class BuilderTests(unittest.TestCase):
    def test_tasks_are_blind(self):
        rows = []
        for i in range(12):
            for index, label in enumerate(("safe", "unsafe", "safe", "unsafe")):
                rows.append({"sample_id": f"s{i}-{index}", "task_key": f"k{i}", "word": "词", "split": "dev",
                             "language": ("zh", "en")[i % 2], "index": index, "label": label, "response_style": "x",
                             "onset_char": 4 if label == "unsafe" else None, "onset_end_char": 8 if label == "unsafe" else None,
                             "messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答，" * 10}]})
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "trainable.jsonl"
            source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            out = Path(tmp) / "pilot"
            subprocess.run([sys.executable, str(HERE / "make_onset_tasks.py"), "--source", str(source), "--count", "8",
                            "--per-task", "3", "--output", str(out)], check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual((manifest["responses"], manifest["tasks"]), (8, 3))
            self.assertEqual(set(manifest["by_slot_language"]), {"1:zh", "1:en", "3:zh", "3:en"})
            for instruction in (out / "tasks").glob("*/instruction.md"):
                task = json.loads(instruction.read_text())
                for term in task["terms"]:
                    self.assertEqual(set(term), {"task_key", "user_prompt", "assistant_text"})


class CompareTests(unittest.TestCase):
    def test_relations_and_summary(self):
        key = {"onset_char": 40, "onset_end_char": 60}
        self.assertEqual(relation({"status": "located", "prev_cut": 10, "cut": 30}, key), "v14_late")
        self.assertEqual(relation({"status": "located", "prev_cut": 70, "cut": 90}, key), "v14_early")
        self.assertEqual(relation({"status": "located", "prev_cut": 35, "cut": 50}, key), "overlap")
        self.assertIsNone(relation({"status": "whole_not_unsafe"}, key))
        keys = [{"sample_id": "a", "slot": 1, **key}, {"sample_id": "b", "slot": 3, **key}, {"sample_id": "c", "slot": 1, **key}]
        values = [{"item_id": "a", "result": {"status": "located", "prev_cut": 10, "cut": 30, "clauses": 5,
                                              "probes": [{"cut": 30, "verdict": "unsafe"}] * 3}, "errors": []},
                  {"item_id": "b", "result": {"status": "whole_not_unsafe", "probes": [{"cut": 90, "verdict": "safe"}]}, "errors": []}]
        rows, summary = summarize(keys, values)
        self.assertEqual(summary["statuses"], {"located": 1, "whole_not_unsafe": 1, "missing": 1})
        self.assertEqual(summary["relation_by_slot"], {"slot1": {"v14_late": 1}})
        self.assertEqual(summary["v14_minus_located_start_chars"]["slot1"]["median"], 30)


if __name__ == "__main__":
    unittest.main()
