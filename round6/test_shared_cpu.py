"""Shared batch helpers (CPU only): multi-Run archive collection, screen exclusions, Task packing within limits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import aster_io  # noqa: E402
import task_pack  # noqa: E402


class CollectArchivesTests(unittest.TestCase):
    def test_two_runs_with_frozen_listings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            listings = []
            for n in range(2):
                path = tmp / f"attempts{n}.json"
                path.write_text(json.dumps({"run_id": f"id{n}", "truncated": False, "items": [
                    {"attempt_id": f"a{n}", "state": "completed", "archive": True},
                    {"attempt_id": f"b{n}", "state": "running", "archive": False}]}))
                listings.append(path)
            calls = []

            def call(args):
                calls.append(args[:2])
                if args[:2] == ["runs", "get"]:
                    return {"id": "id" + args[2][-1], "run_no": args[2], "status": "completed"}
                if args[:2] == ["runs", "archive"]:
                    Path(args[-1]).write_bytes(b"payload-" + args[3].encode())
                    return {"sha256": hashlib.sha256(b"payload-" + args[3].encode()).hexdigest()}
                raise AssertionError(args)

            paths, infos = aster_io.collect_archives(["run0", "run1"], listings, tmp / "archives", call)
            self.assertEqual([p.name for p in paths], ["a0.tar.gz", "a1.tar.gz"])
            self.assertEqual([i["run_no"] for i in infos], ["run0", "run1"])
            self.assertNotIn(["runs", "attempts"], calls)          # frozen listings replace the capped call
            with self.assertRaises(RuntimeError):                    # listing of another run
                aster_io.collect_archives(["run0"], [listings[1]], tmp / "archives", call)
            with self.assertRaises(ValueError):
                aster_io.collect_archives(["run0", "run1"], [listings[0]], tmp / "archives", call)


class PackTests(unittest.TestCase):
    def test_screen_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.jsonl"
            path.write_text("\n".join(json.dumps({"word": w, "verdict": v}, ensure_ascii=False)
                                      for w, v in (("甲", "no"), ("乙", "insult"), ("丙", "evasion"), ("丁", "missing"))))
            excluded, screened = task_pack.screen_exclusions(path)
            self.assertEqual(excluded, {"乙", "丙", "丁"})
            self.assertEqual(screened, {"甲", "乙", "丙", "丁"})

    def test_packing_limits_and_parts(self):
        seeds = [{"task_key": f"k{i}", "word": f"w{i}", "source_group": "g", "blob": "x" * 50} for i in range(25)]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(task_pack, "PART_TASKS", 4):
            out = task_pack.write_generation_tasks(seeds, Path(tmp) / "a", "t", {"prompt_version": "v"}, words_per_task=3)
            self.assertEqual(out["tasks"], 9)                          # ceil(25 / 3)
            self.assertEqual(out["archives"], {"pilot_part0.tar.gz": 4, "pilot_part1.tar.gz": 4, "pilot_part2.tar.gz": 1})
            self.assertEqual(out["task_of"]["k24"], "t-008")
            with tarfile.open(Path(tmp) / "a" / "pilot_part0.tar.gz") as archive:
                self.assertEqual(len({m.name.split("/")[0] for m in archive}), 4)
            small = task_pack.write_generation_tasks(seeds, Path(tmp) / "b", "t", {"prompt_version": "v"},
                                                     words_per_task=10, max_task_bytes=400)
            self.assertGreater(small["tasks"], 3)                     # the byte cap splits before 10 words
            with self.assertRaises(ValueError):
                task_pack.write_generation_tasks(seeds, Path(tmp) / "c", "t", {}, max_task_bytes=50)

    def test_judge_packer(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("judge_make_pilot", ROOT / "judge_v3/make_pilot.py")
        judge = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "judge_v3" / "flow"))
        spec.loader.exec_module(judge)
        by_word = {f"k{i}": [{"item_id": f"i{i}-{j}", "user_prompt": "问", "assistant_text": "答" * 20} for j in range(3)]
                   for i in range(7)}
        tasks = judge.pack(by_word, words_per_task=3, max_task_bytes=10_000)
        self.assertEqual([len(t) for t in tasks], [9, 9, 3])
        self.assertTrue(all(len({it["item_id"].split("-")[0] for it in t}) <= 3 for t in tasks))
        self.assertGreater(len(judge.pack(by_word, words_per_task=7, max_task_bytes=900)), 1)
        with self.assertRaises(ValueError):
            judge.pack(by_word, words_per_task=1, max_task_bytes=100)


class VerifiedFilterTests(unittest.TestCase):
    def test_all_items_must_be_safe(self):
        from verified_filter import verify
        rows = [{"sample_id": s, "split": "train", "response_style": "x"} for s in ("a", "b", "c", "d")]
        items = [{"sample_id": s, "item_id": f"{s}-{k}"} for s in ("a", "b", "c") for k in ("whole", "mid")]
        judgments = [{"item_id": "a-whole", "judge_label": "safe"}, {"item_id": "a-mid", "judge_label": "safe"},
                     {"item_id": "b-whole", "judge_label": "safe"}, {"item_id": "b-mid", "judge_label": "unsafe"},
                     {"item_id": "c-whole", "judge_label": "safe"}, {"item_id": "c-mid", "judge_label": None}]
        kept, reasons = verify(rows, items, judgments)
        self.assertEqual([r["sample_id"] for r in kept], ["a"])
        self.assertEqual(reasons, {"judged_unsafe": 1, "missing_verdict": 1, "not_judged": 1})


class S2BatchTests(unittest.TestCase):
    def test_screen_and_packing_in_the_s2_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            seeds = tmp / "seeds.jsonl"
            seeds.write_text("\n".join(json.dumps({"task_key": f"k{i}", "word": f"词{i}", "source_group": f"g{i % 3}",
                                                   "family": f"f{i}"}, ensure_ascii=False) for i in range(40)) + "\n", encoding="utf-8")
            screen = tmp / "screen.jsonl"
            screen.write_text("\n".join(json.dumps({"word": f"词{i}", "verdict": "insult" if i < 5 else "no"}, ensure_ascii=False)
                                        for i in range(38)) + "\n", encoding="utf-8")      # 词38, 词39 unscreened
            out = tmp / "batch"
            subprocess.run([sys.executable, str(ROOT / "s2_v15/make_pilot.py"), "--seeds", str(seeds), "--words", "100",
                            "--screen", str(screen), "--words-per-task", "10", "--name-prefix", "s2-batch", "--output", str(out)],
                           check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["words"], 33)                   # 40 - 5 flagged - 2 unscreened
            self.assertEqual((manifest["excluded_by_screen"], manifest["unscreened_words"]), (7, 2))
            self.assertEqual(manifest["tasks"], 4)
            with (out / "seeds.jsonl").open(encoding="utf-8") as handle:
                chosen = [json.loads(line) for line in handle]
            self.assertFalse({c["word"] for c in chosen} & {f"词{i}" for i in (0, 1, 2, 3, 4, 38, 39)})
            self.assertTrue(all(c["group_key"].startswith("s2-batch-") for c in chosen))


if __name__ == "__main__":
    unittest.main()
