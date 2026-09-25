"""Stage 2 on synthetic data (CPU): per-token targets, token weights, batching, calibration metrics; the training
step and calibration scoring need torch (skipped without it). Neutral placeholder text only."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_targets as bt  # noqa: E402
import train_stage2 as ts  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def encode(text):
    return [ord(c) % 5000 for c in text], [(i, i + 1) for i in range(len(text))]


def labelled(level, probes, label=None):
    return {"label": label or level, "level": level, "probes": [{"cut": c, "level": l} for c, l in probes]}


def runA_row(sample_id, split, prompt_label="unsafe"):
    return {"sample_id": sample_id, "task_key": sample_id.split("-")[0], "prompt_label": prompt_label, "split": split,
            "family": "f" + sample_id, "language": "zh",
            "messages": [{"role": "user", "content": "问" * 6}, {"role": "assistant", "content": "安" * 10 + "害" * 10}]}


def prefix_row(sample_id, role, ids_ok=True):
    messages = [{"role": "user", "content": "问" * 4}, {"role": "assistant", "content": "答" * 8}]
    if role == "user":
        messages = messages[:1]
    ids, offsets = encode(bt.serialize(messages))
    start = len(bt.serialize(messages)) - len(messages[-1]["content"])
    return {"sample_id": sample_id, "target_role": role, "family": "pf", "language": "zh", "weight": 0.8,
            "messages": messages, "ids": ids if ids_ok else ids[:-1],
            "target_token_positions": [i for i, (_, e) in enumerate(offsets) if e > start],
            "augmentation": {"ids": ids + [7, 7, 7], "anchors": [{"token_end_exclusive": len(ids) + 3}]}}


class TargetTests(unittest.TestCase):
    def test_runA_answer_levels_per_token(self):
        row = runA_row("w1-0", "train")
        # answer: 10 safe chars, then controversial from the clause ending at 14, unsafe from 20
        lab = labelled("unsafe", [(10, "safe"), (14, "controversial"), (20, "unsafe")])
        made, reason, undetermined = bt.text_record(row, row["messages"], "runA", "assistant", lab, encode)
        self.assertIsNone(reason)
        start = len(bt.serialize(row["messages"])) - 20
        by_end = {p - start + 1: c for p, c in zip(made["positions"], made["classes"])}
        self.assertEqual(by_end[10], 0)                         # safe prefix
        self.assertNotIn(12, by_end)                            # inside the onset clause: no target
        self.assertEqual(by_end[14], 2)                         # controversial (head index 2)
        self.assertEqual(by_end[20], 1)                         # unsafe (head index 1)
        self.assertEqual(undetermined, 3 + 5)                   # 11-13 and 15-19

    def test_alert_targets(self):
        row = runA_row("w1-0", "train")
        safe = labelled("safe", [(20, "safe")])
        made, _, _ = bt.text_record(row, row["messages"], "runA", "assistant", {**safe, "alert": {"from": 10}}, encode)
        self.assertEqual(made["classes"], [0] * 10 + [bt.ALERT] * 10)          # onset known: safe before, alert after
        row["messages"][-1]["content"] = "安" * 9 + "，" + "害" * 10
        made, _, _ = bt.text_record(row, row["messages"], "runA", "assistant", {**safe, "alert": {"from": None}}, encode)
        self.assertEqual(made["classes"], [bt.ALERT] * 10)                     # last clause only, the rest untrained
        made, _, _ = bt.text_record(row, row["messages"], "runA", "assistant", safe, encode)
        self.assertEqual(made["classes"], [0] * 20)
        rows = [{"source": "runA", "role": "assistant", "weight": 1.0, "classes": [0, bt.ALERT, bt.ALERT, 1]}]
        ts.token_weights(rows, {"runA/assistant": 1})
        self.assertAlmostEqual(sum(rows[0]["token_weights"][:3]), 0.5)          # alert shares the safe class weight
        metrics = ts.calibration_metrics([{"group": "g", "level": "safe", "classes": [0, bt.ALERT], "cut": [0.1, 0.4]},
                                          {"group": "g", "level": "unsafe", "classes": [0, 1], "cut": [0.1, 0.9]}])
        self.assertEqual(metrics["groups"]["g"]["alert_mean_cut"], 0.4)
        self.assertEqual(metrics["groups"]["g"]["position_auc"], 1.0)           # alert is a negative

    def test_build_all_sources(self):
        runA = [runA_row("w1-0", "train"), runA_row("w1-1", "train"), runA_row("w2-0", "calibration", "safe")]
        safe_all = labelled("safe", [(20, "safe")])
        labels = {"runA": {"w1-0": safe_all, "w1-1": labelled("x", [], "unusable"), "w2-0": safe_all},
                  "runA_prompts": {"w1:prompt:unsafe": labelled("unsafe", [(3, "safe"), (6, "unsafe")]),
                                   "w2:prompt:safe": labelled("safe", [(6, "safe")])},
                  "prefix_v2": {"pa": labelled("unsafe", [(8, "unsafe")]), "pbad": safe_all},
                  "prefix_v2_user": {"pu": labelled("safe", [(4, "safe")])}}
        prefix = {"train": [prefix_row("pa", "assistant"), prefix_row("pbad", "assistant", ids_ok=False),
                            prefix_row("pu", "user")],
                  "calibration": [prefix_row("pmissing", "assistant")]}
        out, stats = bt.build(runA, prefix, labels, encode)
        train = {r["sample_id"]: r for r in out["train"]}
        self.assertEqual(sorted(train), ["pa", "pa:augmented", "pu", "pu:augmented", "w1-0", "w1:prompt:unsafe"])
        self.assertEqual({r["sample_id"] for r in out["calibration"]}, {"w2-0", "w2:prompt:safe"})
        self.assertEqual(stats["runA:assistant:train:skipped_label"], 1)
        self.assertEqual(stats["prefix_v2:assistant:train:skipped_ids_mismatch"], 1)
        self.assertEqual(stats["prefix_v2:assistant:calibration:skipped_label"], 1)
        prompt = train["w1:prompt:unsafe"]
        self.assertEqual((prompt["role"], prompt["classes"][:3], prompt["classes"][-1]), ("user", [0, 0, 0], 1))
        self.assertEqual(len(prompt["ids"]), len(bt.serialize(runA[0]["messages"][:1])))
        aug = train["pa:augmented"]
        self.assertEqual((aug["classes"], aug["weight"], aug["positions"]), ([1], 0.4, [len(aug["ids"]) - 1]))
        self.assertEqual(train["pa"]["classes"], [1])           # only the whole-text probe: earlier prefixes undetermined
        self.assertEqual(stats["prefix_v2:assistant:train:undetermined_tokens"], 7)


class TrainerPureTests(unittest.TestCase):
    def records(self):
        return [{"sample_id": "a", "source": "runA", "role": "assistant", "weight": 1.0, "classes": [0, 0, 1], "ids": [1] * 5},
                {"sample_id": "b", "source": "runA", "role": "assistant", "weight": 1.0, "classes": [0, 2], "ids": [1] * 9},
                {"sample_id": "c", "source": "prefix_v2", "role": "user", "weight": 0.5, "classes": [0, 0], "ids": [1] * 3}]

    def test_token_weights(self):
        rows = self.records()
        info = ts.token_weights(rows, ts.parse_mix("runA/assistant=1,prefix_v2/user=1"))
        total = sum(w for r in rows for w in r["token_weights"])
        self.assertAlmostEqual(total, 1.0)
        a, b, c = (r["token_weights"] for r in rows)
        self.assertAlmostEqual(sum(a[:2]) + b[0], 0.5 / 3)          # safe share of runA/assistant
        self.assertAlmostEqual(a[2], 0.5 / 3)                        # the one unsafe token carries its class share
        self.assertAlmostEqual(sum(c), 0.5)
        self.assertEqual(info["runA/assistant"]["share"], 0.5)
        with self.assertRaises(ValueError):
            ts.token_weights(self.records(), {"runA/assistant": 1.0})

    def test_groups_and_microbatches(self):
        rows = [{"sample_id": str(i), "ids": [1] * (i + 1)} for i in range(40)]
        groups = ts.epoch_groups(rows, 0)
        self.assertEqual(sorted(r["sample_id"] for g in groups for r in g), sorted(r["sample_id"] for r in rows))
        self.assertTrue(all(len(g) <= ts.EFFECTIVE_BATCH for g in groups))
        self.assertNotEqual(groups, ts.epoch_groups(rows, 1))
        packed = ts.microbatches(rows[:10] + [{"sample_id": "long", "ids": [1] * 100}], micro_tokens=24)
        self.assertTrue(all(len(m) <= ts.MICROBATCH for m in packed))
        self.assertTrue(all(len(m) == 1 or max(len(r["ids"]) for r in m) * len(m) <= 24 for m in packed))
        self.assertEqual([r["sample_id"] for r in packed[-1]], ["long"])
        self.assertAlmostEqual(ts.lr_scale(32, 1000), 0.5 * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * 0.032))))
        self.assertAlmostEqual(ts.lr_scale(1000, 1000), 0.1)

    def test_calibration_metrics(self):
        scored = [{"group": "runA/assistant", "level": "unsafe", "classes": [0, 1], "cut": [0.1, 0.9]},
                  {"group": "runA/assistant", "level": "safe", "classes": [0, 0], "cut": [0.2, 0.3]},
                  {"group": "prefix_v2/user", "level": "safe", "classes": [0], "cut": [0.5]}]
        out = ts.calibration_metrics(scored)
        g = out["groups"]["runA/assistant"]
        self.assertEqual((g["position_auc"], g["stream_auc"], g["stream_positive"]), (1.0, 1.0, 1))
        self.assertIsNone(out["groups"]["prefix_v2/user"]["stream_auc"])
        self.assertEqual(out["mean_stream_auc"], 1.0)


@unittest.skipIf(torch is None, "torch not installed")
class TrainerTorchTests(unittest.TestCase):
    def fake(self):
        torch.manual_seed(0)

        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.heads = torch.nn.ModuleDict({role: torch.nn.ModuleDict({"risk": torch.nn.Linear(8, 3)})
                                                  for role in ("user", "assistant")})

            def forward(self, ids, mask=None, use_cache=False):
                return Output(torch.cumsum(self.embed(ids), dim=1))

            def readout(self, hidden, role):
                return self.heads[role]["risk"](hidden), None
        return Model()

    def test_soft_alert_target(self):
        targets = ts.soft_targets(torch, [0, 3], 0.4, "cpu")
        self.assertTrue(torch.allclose(targets, torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.0, 0.4]])))

    def test_steps_reduce_the_loss_and_scores_cover_every_position(self):
        model = self.fake()
        rows = []
        for i in range(16):
            unsafe = i % 2 == 1
            ids = [3] * 4 + ([9] * 4 if unsafe else [5] * 4)
            rows.append({"sample_id": str(i), "source": "runA", "role": ("assistant", "user")[i % 4 // 2], "weight": 1.0,
                         "level": "unsafe" if unsafe else "safe", "ids": ids, "positions": list(range(8)),
                         "classes": [0] * 4 + ([1] * 4 if unsafe else [0] * 4)})
        ts.token_weights(rows, {"runA/assistant": 1.0, "runA/user": 1.0})
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        losses = [ts.train_step(torch, model, optimizer, rows, pad=0, device="cpu", micro_tokens=16, scale=1.0)["loss"]
                  for _ in range(30)]
        self.assertLess(losses[-1], losses[0] * 0.5)
        scored = ts.calibration_scores(torch, model, rows, pad=0, device="cpu", micro_tokens=16)
        self.assertEqual(sorted(len(s["cut"]) for s in scored), [8] * 16)
        self.assertGreater(ts.calibration_metrics(scored)["mean_stream_auc"], 0.9)


if __name__ == "__main__":
    unittest.main()
