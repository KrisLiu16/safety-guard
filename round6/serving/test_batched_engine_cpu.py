"""CPU checks of the varlen conv and window-attention helpers against naive per-session math."""
import math
import unittest

import torch
import torch.nn.functional as F

from batched_engine import varlen_causal_conv, varlen_window_attention


def naive_conv(history, new, weight):
    """history: [H, C] all previous inputs of the session (zeros before start), new: [L, C]."""
    k = weight.shape[1]
    full = torch.cat([torch.zeros(k - 1, new.shape[1]), history, new])
    out = [(full[i:i + k] * weight.t()).sum(0) for i in range(full.shape[0] - k + 1)]
    return F.silu(torch.stack(out[-new.shape[0]:]))


class VarlenTests(unittest.TestCase):
    def test_conv_matches_per_session_convolution_over_several_ticks(self):
        torch.manual_seed(0)
        c, k, sessions = 6, 4, 3
        weight = torch.randn(c, k)
        state = torch.zeros(sessions, k - 1, c)
        history = [torch.zeros(0, c) for _ in range(sessions)]
        for lens in ([5, 1, 2], [1, 7, 1], [2, 2, 9]):          # includes appends shorter than K-1
            new = [torch.randn(n, c) for n in lens]
            out, state = varlen_causal_conv(torch.cat(new), state, lens, weight)
            start = 0
            for i, n in enumerate(lens):
                expect = naive_conv(history[i], new[i], weight)
                self.assertTrue(torch.allclose(out[start:start + n], expect, atol=1e-5))
                history[i] = torch.cat([history[i], new[i]])
                start += n
            for i in range(sessions):
                self.assertTrue(torch.allclose(state[i], torch.cat([torch.zeros(k - 1, c), history[i]])[-(k - 1):]))

    def test_window_attention_matches_naive_with_ring_and_long_appends(self):
        torch.manual_seed(1)
        hq, hkv, d, window, sessions = 4, 2, 8, 6, 3
        scale = d ** -0.5
        ring_k = torch.zeros(sessions, window, hkv, d)
        ring_v = torch.zeros(sessions, window, hkv, d)
        ring_pos = torch.full((sessions, window), -1, dtype=torch.long)
        all_k = [[] for _ in range(sessions)]
        all_v = [[] for _ in range(sessions)]
        length = [0] * sessions
        for lens in ([3, 1, 9], [1, 5, 2], [8, 1, 1], [2, 2, 2]):  # 9 and 8 exceed the window
            q = torch.randn(sum(lens), hq, d)
            k = torch.randn(sum(lens), hkv, d)
            v = torch.randn(sum(lens), hkv, d)
            positions = torch.cat([torch.arange(length[i], length[i] + n) for i, n in enumerate(lens)])
            out = varlen_window_attention(q, k, v, positions, lens, ring_k, ring_v, ring_pos, window, scale)
            start = 0
            for i, n in enumerate(lens):
                all_k[i].extend(k[start:start + n]); all_v[i].extend(v[start:start + n])
                keys, values = torch.stack(all_k[i]), torch.stack(all_v[i])
                for j in range(n):
                    p = length[i] + j
                    lo = max(0, p - window + 1)
                    kk = keys[lo:p + 1].repeat_interleave(hq // hkv, 1)
                    vv = values[lo:p + 1].repeat_interleave(hq // hkv, 1)
                    w = torch.softmax(torch.einsum("hd,thd->ht", q[start + j], kk) * scale, -1)
                    expect = torch.einsum("ht,thd->hd", w, vv)
                    self.assertTrue(torch.allclose(out[start + j], expect, atol=1e-5), (lens, i, j))
                # commit like the engine: last W tokens of this append into the ring
                for j in range(max(0, n - window), n):
                    p = length[i] + j
                    ring_k[i, p % window] = k[start + j]; ring_v[i, p % window] = v[start + j]
                    ring_pos[i, p % window] = p
                start += n
                length[i] += n


if __name__ == "__main__":
    unittest.main()
