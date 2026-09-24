"""CPU check of the ring-attention reference (the Triton kernel's contract) against naive per-session attention
over several ticks, with appends longer than the window, padded rows and sessions idle in a tick.
The Triton kernel itself is compared with this reference on the L20 (bench_v4_l20.kernel_check)."""
import unittest

import torch

from ring_attention_kernel import ring_attention_reference


class RingReferenceTests(unittest.TestCase):
    def test_matches_naive_window_attention_over_ticks(self):
        torch.manual_seed(0)
        hkv, group, d, window, pool = 2, 3, 8, 6, 5
        scale = d ** -0.5
        ring_k = torch.zeros(pool, hkv, window + 1, d)
        ring_v = torch.zeros_like(ring_k)
        ring_pos = torch.full((pool, window + 1), -1, dtype=torch.long)
        slots = torch.tensor([3, 0, 4])
        history_k = [[] for _ in slots]
        history_v = [[] for _ in slots]
        length = [0, 0, 0]
        for lens in ([3, 1, 9], [1, 0, 2], [8, 5, 1], [2, 2, 0]):       # 9 and 8 exceed the window; 0 = idle
            lb = max(lens)
            lens_t = torch.tensor(lens)
            starts = torch.tensor(length)
            q = torch.randn(3, hkv, group * lb, d)
            new_k = torch.randn(3, hkv, lb, d)
            new_v = torch.randn(3, hkv, lb, d)
            out = ring_attention_reference(q, ring_k, ring_v, new_k, new_v, ring_pos, slots, starts, lens_t, scale, window)
            for i, n in enumerate(lens):
                keys = history_k[i] + [new_k[i, :, j] for j in range(n)]
                values = history_v[i] + [new_v[i, :, j] for j in range(n)]
                for row in range(group * lb):
                    l = row % lb
                    if l >= n:
                        self.assertTrue(torch.equal(out[i, :, row], torch.zeros(hkv, d)))     # padded query row
                        continue
                    p = length[i] + l
                    lo = max(0, p - window + 1)
                    kk, vv = torch.stack(keys[lo:p + 1]), torch.stack(values[lo:p + 1])      # [T, Hkv, D]
                    w = torch.softmax(torch.einsum("hd,thd->ht", q[i, :, row], kk) * scale, -1)
                    expect = torch.einsum("ht,thd->hd", w, vv)
                    self.assertTrue(torch.allclose(out[i, :, row], expect, atol=1e-5), (lens, i, row))
                # commit like the engine: every real key goes to column pos % W (later ones overwrite earlier)
                for j in range(n):
                    p = length[i] + j
                    ring_k[slots[i], :, p % window] = new_k[i, :, j]
                    ring_v[slots[i], :, p % window] = new_v[i, :, j]
                    ring_pos[slots[i], p % window] = p
                history_k[i] += [new_k[i, :, j] for j in range(n)]
                history_v[i] += [new_v[i, :, j] for j in range(n)]
                length[i] += n

    def test_rows_are_grouped_per_kv_head_as_in_v3(self):
        """Row r of KV head h is query head h * G + r // Lb at token r % Lb, v3's reshape of [N, Hq, Lb, D]."""
        torch.manual_seed(1)
        n, hq, hkv, lb, d, window = 2, 4, 2, 3, 8, 4
        query = torch.randn(n, hq, lb, d)
        q = query.reshape(n, hkv, (hq // hkv) * lb, d)
        self.assertTrue(torch.equal(q[1, 1, 1 * lb + 2], query[1, 1 * (hq // hkv) + 1, 2]))
        empty = torch.zeros(4, hkv, window + 1, d)
        pos = torch.full((4, window + 1), -1, dtype=torch.long)
        new_k, new_v = torch.randn(n, hkv, lb, d), torch.randn(n, hkv, lb, d)
        out = ring_attention_reference(q, empty, empty, new_k, new_v, pos, torch.tensor([0, 1]), torch.zeros(n, dtype=torch.long),
                                       torch.tensor([lb, lb]), d ** -0.5, window)
        self.assertTrue(torch.allclose(out[:, :, 0::lb], new_v[:, :, :1].expand(-1, -1, hq // hkv, -1), atol=1e-6))  # first token sees itself only


if __name__ == "__main__":
    unittest.main()
