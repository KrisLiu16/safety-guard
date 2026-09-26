"""profile_train.py on synthetic data (CPU): kernel families, busy time, attention FLOPs, and the batched loss against
train_distill's per-row loss (value, parts and gradient; needs torch, skipped without it)."""
from pathlib import Path
import random
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profile_train as pt  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class PureTest(unittest.TestCase):
    def test_kernel_family(self):
        cases = {
            "sm89_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32": "matmul",
            "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn": "matmul",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "gated deltanet (fla)",
            "l2norm_fwd_kernel": "gated deltanet (fla)",
            "fmha_cutlassF_bf16_aligned_64x64_rf_sm80": "attention",
            "Memcpy HtoD (Pinned -> Device)": "copy",
            "void at::native::(anonymous namespace)::multi_tensor_apply_kernel<...>": "multi-tensor (optimizer, clip)",
            "void at::native::vectorized_elementwise_kernel<4, ...>": "elementwise / indexing",
            "void at::native::reduce_kernel<512, 1, ...>": "reduction / norm / softmax",
            "sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_bf16": "short convolution",
            "triton_poi_fused_add_0": "other triton",
            "something_else": "other",
        }
        for name, family in cases.items():
            self.assertEqual(pt.kernel_family(name), family, name)

    def test_busy_ns(self):
        self.assertEqual(pt.busy_ns([(0, 10), (5, 15), (20, 25), (21, 22)]), 20)
        self.assertEqual(pt.busy_ns([]), 0)

    def test_attention_flops(self):
        # length 3, window 2: keys 1 + 2 + 2 = 5, 4 FLOPs per key for one head of size 1
        self.assertEqual(pt.attention_flops([3], 2, 1, 1, 1), 20)
        self.assertEqual(pt.attention_flops([3, 3], 2, 2, 4, 3), 2 * 20 * 2 * 4 * 3)

    def test_merge_groups(self):
        self.assertEqual(pt.merge_groups([[1], [2], [3], [4], [5]], 2), [[1, 2], [3, 4]])
        self.assertEqual(pt.merge_groups([[1], [2]], 1), [[1], [2]])

    def test_loss_plan_coefficients(self):
        rows = [{"role": "user", "positions": [1, 2], "classes": [0, 3], "token_weights": [0.5, 0.5],
                 "t_positions": [2, 3], "t_risk": [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]], "t_cat": [-1, 4],
                 "distill_weight": 0.2}]
        p = pt.loss_plan(rows, 0.5, 0.25)["user"]
        self.assertAlmostEqual(sum(p["kl_coef"]), 0.5 * 0.2)
        self.assertAlmostEqual(sum(p["ce_coef"]), 0.25 * 0.2)
        self.assertEqual(p["cats"], [-1, 4])
        self.assertEqual(pt.loss_plan(rows, 0.5, 0.0)["user"]["cats"], [-1, -1])


@unittest.skipIf(torch is None, "torch not installed")
class BatchedLossTest(unittest.TestCase):
    def fake_model(self):
        nn = torch.nn

        class Fake(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = nn.ModuleDict({role: nn.ModuleDict({
                    "projection": nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4), nn.SiLU()),
                    "risk": nn.Linear(4, 3), "category": nn.Linear(4, cats),
                    "general_projection": nn.Sequential(nn.Linear(8, 4), nn.LayerNorm(4), nn.SiLU()),
                    "general": nn.Linear(4, 3), "general_category": nn.Linear(4, cats)})
                    for role, cats in [("user", 9), ("assistant", 8)]})

            def readout(self, hidden, role):
                h = self.heads[role]["projection"](hidden.float())
                return self.heads[role]["risk"](h), self.heads[role]["category"](h)
        return Fake()

    def rows(self, rng, length):
        out = []
        for i in range(6):
            role = "user" if i % 2 else "assistant"
            n = rng.randint(1, length)
            positions = sorted(rng.sample(range(length), n))
            row = {"role": role, "positions": positions, "classes": [rng.randint(0, 3) for _ in positions],
                   "token_weights": [rng.random() for _ in positions], "distill_weight": 0.0}
            if i != 4:
                m = rng.randint(1, length)
                t_pos = sorted(rng.sample(range(length), m))
                row.update(t_positions=t_pos, t_risk=[[rng.random() + 1e-7 for _ in range(3)] for _ in t_pos],
                           t_cat=[rng.choice([-1, rng.randint(0, 7)]) for _ in t_pos],
                           distill_weight=0.0 if i == 2 else rng.random())
            out.append(row)
        out[0]["t_cat"] = [-1] * len(out[0]["t_cat"])
        return out

    def test_batched_equals_loop(self):
        torch.manual_seed(0)
        rng = random.Random(1)
        model = self.fake_model().double().float()
        rows = self.rows(rng, 12)
        table = pt.alert_table(torch, 0.4, "cpu")
        for category in (0.25, 0.0):
            hidden_a = torch.randn(len(rows), 12, 8, requires_grad=True)
            hidden_b = hidden_a.detach().clone().requires_grad_(True)
            losses, parts = pt.loop_loss(torch, model, hidden_a, rows, alert_cut=0.4, distill=0.5,
                                         category=category, device="cpu")
            loop_total = torch.stack(losses).sum()
            loop_total.backward()
            tensors = pt.plan_tensors(torch, pt.loss_plan(rows, 0.5, category), "cpu", pin=False)
            red, dist = pt.batched_loss(torch, model, hidden_b, tensors, table)
            (red + dist).backward()
            self.assertAlmostEqual(float(red + dist), float(loop_total), places=4)
            self.assertAlmostEqual(float(red), parts["redline"], places=4)
            self.assertAlmostEqual(float(dist), parts["distill"], places=4)
            self.assertTrue(torch.allclose(hidden_a.grad, hidden_b.grad, rtol=1e-4, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
