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
import train_distill as td  # noqa: E402

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

    def test_extra_sources(self):
        user = {"sample_id": "e1:user", "split": "train", "family": "e1", "language": "en",
                "messages": [{"role": "user", "content": "q" * 6}]}
        answer = {"sample_id": "e1:assistant", "split": "dev", "family": "e1", "language": "en",
                  "messages": [{"role": "user", "content": "q" * 6}, {"role": "assistant", "content": "a" * 8}]}
        extra = [("en_redline_prompts", "user", [user], {"e1:user": labelled("unsafe", [(6, "unsafe")])}),
                 ("en_redline", "assistant", [answer], {})]
        out, stats = bt.build([], {}, {"runA": {}, "runA_prompts": {}}, encode, (), extra)
        record = out["train"][0]
        self.assertEqual((record["source"], record["role"], record["classes"]), ("en_redline_prompts", "user", [1]))
        self.assertEqual(stats["en_redline:assistant:dev:skipped_label"], 1)
        with self.assertRaises(ValueError):
            bt.build([], {}, {"runA": {}, "runA_prompts": {}}, encode, (), [("x", "assistant", [user], {})])

    def test_teacher_attached_at_every_content_position(self):
        row = runA_row("w1-0", "train")
        lab = labelled("safe", [(20, "safe")])
        # teacher tokens end at content chars 5, 10 and 20; the student has one token per char (encode)
        teacher = {"ends": [5, 10, 20], "risk": [[0.9, 0.05, 0.05], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]], "cat": [1, 2, 2]}
        made, _, _ = bt.text_record(row, row["messages"], "runA", "assistant", lab, encode, teacher=teacher)
        self.assertEqual(len(made["t_positions"]), 20 - 4)          # content ends 5..20 have a teacher token
        self.assertEqual(made["t_risk"][0], [0.9, 0.05, 0.05])       # end 5 -> teacher token ending at 5
        self.assertEqual(made["t_risk"][5], [0.3, 0.6, 0.1])         # end 10
        self.assertEqual(made["t_cat"][0], -1)                       # teacher safe -> no category target
        self.assertEqual(made["t_cat"][-1], 2)
        plain, _, _ = bt.text_record(row, row["messages"], "runA", "assistant", lab, encode)
        self.assertNotIn("t_positions", plain)

    def test_teacher_end_only_for_user_texts(self):
        runA = [runA_row("w1-0", "train")]
        labels = {"runA": {"w1-0": labelled("safe", [(20, "safe")])},
                  "runA_prompts": {"w1:prompt:unsafe": labelled("unsafe", [(6, "unsafe")])}}
        per_token = {"ends": [3, 6, 20], "risk": [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], "cat": [0, 0, 2]}
        teacher = {("w1-0", "assistant"): per_token, ("w1:prompt:unsafe", "user"): per_token}
        end = {"w1:prompt:unsafe": {"end_risk": [0.1, 0.8, 0.1], "end_cat": 4}}
        out, _ = bt.build(runA, {}, labels, encode, teacher=teacher, teacher_end=end)
        rows = {r["sample_id"]: r for r in out["train"]}
        prompt, answer = rows["w1:prompt:unsafe"], rows["w1-0"]
        self.assertEqual(prompt["t_positions"], [len(prompt["ids"]) - 1])     # the last prompt token only
        self.assertEqual((prompt["t_risk"], prompt["t_cat"]), ([[0.1, 0.8, 0.1]], [4]))
        self.assertEqual(len(answer["t_positions"]), 20 - 2)                  # answers keep per-token targets
        out, _ = bt.build(runA, {}, labels, encode, teacher=teacher, teacher_end={})
        self.assertNotIn("t_positions", {r["sample_id"]: r for r in out["train"]}["w1:prompt:unsafe"])
        safe_end = bt.end_targets([4, 5], {"end_risk": [0.7, 0.2, 0.1], "end_cat": 3})
        self.assertEqual(safe_end["t_cat"], [-1])                             # teacher safe: no category target

    def test_distill_only_rows_carry_teacher_targets_and_no_redline_positions(self):
        answer = {"sample_id": "bench:X:1", "split": "train", "language": "en",
                  "messages": [{"role": "user", "content": "问" * 6}, {"role": "assistant", "content": "答" * 12}]}
        prompt = {"sample_id": "bench:X:2", "split": "train", "language": "en",
                  "messages": [{"role": "user", "content": "问" * 8}]}
        per_token = {"ends": [3, 6, 12], "risk": [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], "cat": [0, 0, 2]}
        teacher = {("bench:X:1", "assistant"): per_token}
        end = {"bench:X:2": {"end_risk": [0.1, 0.8, 0.1], "end_cat": 4}}
        out, stats = bt.build([], {}, {}, encode, teacher=teacher, teacher_end=end,
                              distill_only=[("bench_fit", "assistant", [answer]), ("bench_fit", "user", [prompt])])
        rows = {r["sample_id"]: r for r in out["train"]}
        a, p = rows["bench:X:1"], rows["bench:X:2"]
        self.assertEqual((a["positions"], a["classes"], p["positions"], p["classes"]), ([], [], [], []))
        self.assertEqual(len(a["t_positions"]), 12 - 2)                      # per token from the teacher's first end
        self.assertEqual((p["t_positions"], p["t_risk"]), ([len(p["ids"]) - 1], [[0.1, 0.8, 0.1]]))
        self.assertEqual((stats["bench_fit:assistant:train:records"], stats["bench_fit:user:train:records"]), (1, 1))
        info = ts.token_weights(list(rows.values()), ts.parse_mix("runA/assistant=1"))
        self.assertEqual((a["token_weights"], info), ([], {}))                # no red-line weight, no mix needed
        _, stats = bt.build([], {}, {}, encode, teacher={}, teacher_end={},
                            distill_only=[("bench_fit", "assistant", [answer])])
        self.assertEqual(stats["bench_fit:assistant:train:skipped_no_teacher"], 1)
        with self.assertRaises(ValueError):
            bt.build([], {}, {}, encode, teacher=teacher, distill_only=[("bench_fit", "assistant", [{**answer, "split": "dev"}])])

    def test_leader_prompt_label_filter(self):
        import make_leader_prompt_labels as mk
        row = {"sample_id": "s1", "split": "train", "language": "zh", "response_style": "place_normal",
               "messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]}
        self.assertEqual(mk.label(row)["label"], "safe")
        self.assertIsNone(mk.label(row, {}))
        self.assertIsNone(mk.label(row, {"s1": {"level": "controversial", "excluded": False}}))
        self.assertEqual(mk.label(row, {"s1": {"level": "safe", "excluded": False}})["sample_id"], "s1:prompt")
        self.assertIsNone(mk.label({**row, "response_style": "satire_restated"}, {"s1": {"level": "safe"}}))


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


@unittest.skipIf(torch is None, "torch not installed")
class BatchedStepTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(1)

        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden

        def head(cats):
            return torch.nn.ModuleDict({
                "projection": torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.LayerNorm(6), torch.nn.SiLU()),
                "risk": torch.nn.Linear(6, 3), "category": torch.nn.Linear(6, cats),
                "general_projection": torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.LayerNorm(6), torch.nn.SiLU()),
                "general": torch.nn.Linear(6, 3), "general_category": torch.nn.Linear(6, cats)})

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(64, 8)
                self.heads = torch.nn.ModuleDict({"user": head(9), "assistant": head(8)})

            def forward(self, ids, mask=None, use_cache=False):
                return Output(torch.cumsum(self.embed(ids), dim=1))

            def readout(self, hidden, role):
                h = self.heads[role]["projection"](hidden.float())
                return self.heads[role]["risk"](h), self.heads[role]["category"](h)
        return Model()

    def rows(self):
        out = []
        for i in range(10):
            n = 5 + i % 4
            row = {"sample_id": str(i), "source": "runA", "role": ("assistant", "user")[i % 2], "weight": 1.0,
                   "ids": [(7 * i + k) % 60 + 1 for k in range(n)], "positions": list(range(2, n)),
                   "classes": [(i + k) % 4 for k in range(n - 2)], "split": "train"}
            if i % 3:
                row.update(t_positions=list(range(1, n)), t_risk=[[0.6, 0.3, 0.1]] * (n - 1),
                           t_cat=[-1 if k % 2 else i % 7 for k in range(n - 1)])
            out.append(row)
        out.append({"sample_id": "t", "source": "bench_fit", "role": "user", "weight": 1.0, "ids": [4, 5, 6],
                    "positions": [], "classes": [], "split": "train", "t_positions": [2], "t_risk": [[0.2, 0.7, 0.1]],
                    "t_cat": [3]})                                    # teacher-only row
        mix = {"runA/assistant": 1.0, "runA/user": 1.0, "bench_fit/user": 0.5}
        ts.token_weights(out, mix)
        td.distill_weights(out, mix)
        return out

    def test_batched_step_equals_per_row_step(self):
        import batched_loss as bl
        grads, results = [], []
        for batched in (False, True):
            model = self.model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            common = {"pad": 0, "device": "cpu", "micro_tokens": 64, "scale": 2.0, "distill": 0.5, "category": 0.25}
            if batched:
                out = bl.train_step(torch, model, optimizer, self.rows(), rows=16,
                                    table=bl.alert_table(torch, 0.4, "cpu"), **common)
            else:
                out = td.train_step(torch, model, optimizer, self.rows(), alert_cut=0.4, rows_per_pass=16, **common)
            results.append(out)
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        a, b = results
        for key in ("loss", "redline", "distill", "gradient_norm"):
            self.assertAlmostEqual(a[key], b[key], places=4, msg=key)
        self.assertEqual(a["input_tokens"], b["input_tokens"])
        self.assertGreater(a["distill"], 0)
        self.assertTrue(torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6))

if __name__ == "__main__":
    unittest.main()
