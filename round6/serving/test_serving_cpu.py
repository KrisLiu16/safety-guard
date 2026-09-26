"""CPU tests for the serving pieces that need no GPU (T034 step A): v7's input packing against v6's row loop, and the
vectorised simulator's arrival arithmetic against bench_slot_l20.simulate's loop."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_v7_l20 import arrival_times, due_tokens  # noqa: E402
from slot_engine_v7 import pack_inputs  # noqa: E402


def pack_reference(part, nb, lb, length, max_slots):
    """BucketedStreamEngine._run (v6): the per-row loop, flattened in the v7 static-buffer layout."""
    ids = np.zeros((nb, lb), dtype=np.int64)
    lens = np.zeros(nb, dtype=np.int64)
    slots = np.full(nb, max_slots, dtype=np.int64)
    starts = np.zeros(nb, dtype=np.int64)
    for row, (s, chunk) in enumerate(part):
        ids[row, :len(chunk)] = chunk
        lens[row], slots[row], starts[row] = len(chunk), s, length[s]
    return np.concatenate([ids.ravel(), lens, slots, starts])


class PackTests(unittest.TestCase):
    def test_matches_v6_loop(self):
        rng = random.Random(0)
        for _ in range(300):
            max_slots = 64
            length = np.array([rng.randint(0, 5000) for _ in range(max_slots)], dtype=np.int64)
            lb = rng.choice((1, 2, 4, 8, 16, 32, 64))
            nb = rng.choice((1, 2, 4, 8, 16, 32, 48, 64))
            slots = rng.sample(range(max_slots), rng.randint(1, nb))
            part = [(s, [rng.randint(0, 248319) for _ in range(rng.randint(1, lb))]) for s in slots]
            out = np.full(nb * lb + 3 * nb, -7, dtype=np.int64)          # stale contents must all be overwritten
            lens = pack_inputs([c for _, c in part], np.array(slots, dtype=np.int64), nb, lb, length, max_slots, out)
            np.testing.assert_array_equal(out, pack_reference(part, nb, lb, length, max_slots))
            np.testing.assert_array_equal(lens, [len(c) for _, c in part])

    def test_single_token_bucket_accepts_numpy_and_lists(self):
        length = np.arange(8, dtype=np.int64) * 10
        out = np.zeros(4 + 12, dtype=np.int64)
        pack_inputs([[5], np.array([6])], np.array([3, 1]), 4, 1, length, 8, out)
        np.testing.assert_array_equal(out, [5, 6, 0, 0, 1, 1, 0, 0, 3, 1, 8, 8, 30, 10, 0, 0])


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.n = 60
        self.rate = np.array([rng.uniform(30, 60) for _ in range(self.n)])
        self.offset = np.array([rng.uniform(0, 1.0) for _ in range(self.n)])
        self.rng = rng

    def test_due_tokens_matches_loop(self):
        for _ in range(200):
            now = self.rng.uniform(0, 3)
            delivered = np.array([max(0, int((now - o) * r) - self.rng.randint(0, 80)) if now > o else 0
                                  for o, r in zip(self.offset, self.rate)], dtype=np.int64)
            want = [min((int((now - self.offset[s]) * self.rate[s]) if now > self.offset[s] else 0) - delivered[s], 64)
                    for s in range(self.n)]
            np.testing.assert_array_equal(due_tokens(now, self.offset, self.rate, delivered), want)

    def test_arrival_times_match_loop(self):
        t0 = 123.25
        delivered = np.array([self.rng.randint(0, 500) for _ in range(self.n)], dtype=np.int64)
        n = np.array([self.rng.randint(1, 64) for _ in range(self.n)], dtype=np.int64)
        want = [t0 + self.offset[s] + (delivered[s] + j + 1) / self.rate[s] for s in range(self.n) for j in range(n[s])]
        got = arrival_times(t0, self.offset, self.rate, delivered, n)
        self.assertEqual(got.tolist(), want)                               # same float64 arithmetic, same order


if __name__ == "__main__":
    unittest.main()
