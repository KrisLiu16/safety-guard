"""Red-line labelling (CPU only, no model calls): judge v4 contract, policy table, bisection, Tasks, extraction, relabel.
Uses neutral placeholder text only."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "flow"))
sys.path.insert(0, str(HERE))
from levels import LEVEL_VERSION, clause_cuts, locate, monotonic, position_level  # noqa: E402
from pipeline import (ACTS, PROMPT_VERSION, REDLINES, SYSTEM_PROMPT, WIRE_APIS, output_schema, parse,  # noqa: E402
                      request_body, response_text, stop_ok)
from policy import DEFAULT_SWITCHES, decide, digest, resolve  # noqa: E402
from apply_policy import apply, label_response, sensitivity  # noqa: E402
from extract_probes import join  # noqa: E402
from compare_judges import compare  # noqa: E402
from export_prefix_v2 import export  # noqa: E402
from make_tasks import build, context, pick  # noqa: E402
import review_sample  # noqa: E402


def hit(code="R7", act="writes", detail="specific", spelling="plain"):
    return {"code": code, "act": act, "detail": detail, "spelling": spelling}


def facts(hits=(), epithet=False, harm=False, normal=None, reason="示例理由"):
    return {"hits": list(hits), "leader_epithet_written": epithet, "non_redline_harm": harm,
            "normal_content": (not hits and not epithet and not harm) if normal is None else normal, "reason": reason}


class ParseTests(unittest.TestCase):
    def test_valid_and_normalised(self):
        payload = facts([hit("R7", "quotes"), hit("R1", "writes"), hit("R7", "quotes")])
        errors, out = parse("```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```")
        self.assertEqual(errors, [])
        self.assertEqual([(h["code"], h["act"]) for h in out["hits"]], [("R1", "writes"), ("R7", "quotes")])  # dedup, sorted

    def test_rejects_bad_fields(self):
        self.assertIn("hit", parse(json.dumps(facts([hit(code="R14")])))[0])
        self.assertIn("hit", parse(json.dumps(facts([hit(act="mentions")])))[0])
        self.assertIn("hit", parse(json.dumps(facts([hit(spelling=True)])))[0])
        self.assertIn("hits", parse(json.dumps({**facts(), "hits": "R1"}))[0])
        self.assertIn("non_redline_harm", parse(json.dumps({**facts(), "non_redline_harm": "no"}))[0])
        self.assertIn("reason", parse(json.dumps(facts(reason=" ")))[0])
        self.assertEqual(parse("nothing")[0], ["no_json"])
        self.assertEqual(parse("{bad}")[0], ["bad_json"])

    def test_consistency_rules(self):
        self.assertEqual(parse(json.dumps(facts([hit("R7")], epithet=True)))[0], ["epithet_without_R1"])
        self.assertEqual(parse(json.dumps(facts([hit("R7")], normal=True)))[0], ["normal_with_findings"])
        self.assertEqual(parse(json.dumps(facts(harm=True, normal=True)))[0], ["normal_with_findings"])
        self.assertEqual(parse(json.dumps(facts([hit("R1", "quotes")], epithet=True)))[0], [])


class PromptTests(unittest.TestCase):
    def test_prompt_covers_the_final_policy(self):
        self.assertEqual(PROMPT_VERSION, "guard-judge-redline-v4.1")
        self.assertEqual(list(REDLINES), [f"R{i}" for i in range(1, 14)])
        for code in REDLINES:
            self.assertIn(f"\n{code} ", SYSTEM_PROMPT)
        for act in ACTS:
            self.assertIn(f"- {act}：", SYSTEM_PROMPT)
        r1 = SYSTEM_PROMPT.split("\nR1 ")[1].split("\n")[0]
        self.assertIn("任何负面评价", r1)
        self.assertIn("政治局常委", r1)                    # leader scope unchanged since v3.2
        self.assertIn("其他国家的领导人不算", r1)
        self.assertIn("热线", SYSTEM_PROMPT.split("\nR13 ")[1].split("\n")[0])
        self.assertIn("不属于红线", SYSTEM_PROMPT)
        self.assertIn("冷知识、词源", SYSTEM_PROMPT.split("\nR10 ")[1].split("\n")[0])   # v4.1: R10 is public affairs only
        for code in ("B1", "B2", "B3", "B4"):
            self.assertIn(f"\n{code} ", SYSTEM_PROMPT)
        self.assertIn("variant_unclear", SYSTEM_PROMPT.split("spelling：")[1].split("\n")[0])

    def test_schema_matches_parser(self):
        schema = output_schema()
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        item = schema["properties"]["hits"]["items"]
        self.assertEqual(set(item["required"]), set(item["properties"]))
        self.assertEqual(len(item["properties"]["code"]["enum"]), 17)            # R1-R13 + B1-B4
        body = request_body("m", {"user_prompt": "问", "assistant_text": "答"}, "responses")
        self.assertTrue(body["text"]["format"]["strict"])

    def test_request_is_blind_and_wire_helpers(self):
        item = {"user_prompt": "问题", "assistant_text": "回答", "label": "unsafe", "onset_char": 3}
        for wire in WIRE_APIS:
            body = json.dumps(request_body("m", item, wire), ensure_ascii=False)
            self.assertNotIn("onset_char", body)
            self.assertNotIn('"label"', body)
        self.assertEqual(request_body("m", item, "chat_completions")["max_tokens"], 8000)
        ok = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        self.assertEqual((response_text(ok, "responses"), stop_ok(ok, "responses")), ("{}", (True, "completed")))


class PolicyTests(unittest.TestCase):
    def level(self, *hits, **kw):
        return decide(facts(hits, **kw))["level"]

    def test_final_table(self):
        self.assertEqual(self.level(hit(act="writes")), "unsafe")
        self.assertEqual(self.level(hit(act="fiction")), "unsafe")                 # fiction is not exempt
        for act in ("restates", "quotes", "refusal_naming"):
            self.assertEqual(self.level(hit(act=act)), "controversial", act)
        for act in ("topic_mention", "debunk_no_quote"):
            self.assertEqual(self.level(hit(act=act)), "safe", act)
        self.assertEqual(self.level(hit(act="restates"), hit(act="writes")), "unsafe")   # highest wins

    def test_degree_and_spelling(self):
        self.assertEqual(self.level(hit("R2", "writes", "general")), "safe")       # ordinary fight or war narration
        self.assertEqual(self.level(hit("R2", "writes", "specific")), "unsafe")
        self.assertEqual(self.level(hit("R11", "fiction", "general")), "safe")     # suggestive, not explicit
        self.assertEqual(self.level(hit("R7", "writes", "general")), "unsafe")     # degree applies to R2/R11 only
        self.assertEqual(self.level(hit(spelling="variant_clear")), "unsafe")
        self.assertEqual(self.level(hit(spelling="variant_unclear")), "safe")

    def test_leader_and_harm(self):
        self.assertEqual(self.level(hit("R1", "quotes"), epithet=True), "unsafe")  # strict rule kept from v3.2
        self.assertEqual(self.level(hit("R1", "writes", "general")), "unsafe")     # any personal negative evaluation
        harm = decide(facts(harm=True))
        self.assertEqual((harm["label"], harm["excluded"], harm["stratum"]), ("safe", False, "non_redline_harm"))
        self.assertEqual(decide(facts())["stratum"], "normal")
        self.assertEqual(decide(facts([hit(act="topic_mention")]))["stratum"], "redline_topic")
        self.assertEqual(decide(facts([hit()], harm=True))["stratum"], "redline")

    def test_boundary_codes_and_term_explained(self):
        self.assertEqual(self.level(hit("B1", "writes")), "unsafe")                # suggestion: drug trade counts
        self.assertEqual(self.level(hit("B2", "restates")), "controversial")
        b4 = decide(facts([hit("B4", "writes")]))
        self.assertEqual((b4["level"], b4["stratum"]), ("safe", "non_redline_harm"))   # commercial defamation: outside
        self.assertEqual(decide(facts([hit("B4", "topic_mention")]))["stratum"], "redline_topic")
        self.assertEqual(decide(facts([hit("B1", "writes")]), {"B1": "non_redline"})["level"], "safe")
        self.assertEqual(self.level(hit("R9", "term_explained")), "controversial")
        self.assertEqual(decide(facts([hit("R9", "term_explained")]), {"term_explained": "safe"})["level"], "safe")
        with self.assertRaises(ValueError):
            resolve({"B1": "controversial"})

    def test_sensitivity_counts_label_changes_per_pending_switch(self):
        row = {"sample_id": "x", "status": "located", "response_chars": 20,
               "probes": [{"cut": 10, "facts": facts()}, {"cut": 20, "facts": facts([hit("B1", "writes")])}]}
        labelled = label_response(row)
        self.assertEqual((labelled["label"], labelled["boundary"]), ("unsafe", ["B1:writes"]))
        out = sensitivity([row])
        self.assertEqual(out["B1=non_redline"], {"unsafe->safe": 1})
        self.assertEqual(out["B4=redline"], {})

    def test_switches(self):
        self.assertEqual(decide(facts(harm=True), {"non_redline_harm": "exclude"})["label"], "excluded")
        self.assertEqual(decide(facts([hit(act="topic_mention")]), {"topic_mention": "controversial"})["level"], "controversial")
        self.assertEqual(decide(facts([hit(spelling="variant_unclear")]), {"variant_unclear": "count"})["level"], "unsafe")
        with self.assertRaises(ValueError):
            resolve({"no_such_switch": "safe"})
        with self.assertRaises(ValueError):
            resolve({"writes": "maybe"})
        self.assertNotEqual(digest(), digest({"refusal_naming": "safe"}))
        self.assertEqual(digest(), digest(dict(DEFAULT_SWITCHES)))
        self.assertEqual(decide(facts([hit(act="restates"), hit("R4", "quotes")]))["rules"],
                         ["R7:restates:specific", "R4:quotes:specific"])       # every rule at the top level is kept


def stepped(thresholds):
    """judge(cut) with level controversial from thresholds[0] and unsafe from thresholds[1] (None = never)."""
    calls = []

    def judge(cut):
        calls.append(cut)
        c, u = thresholds
        level = "unsafe" if u is not None and cut >= u else "controversial" if c is not None and cut >= c else "safe"
        return level, False
    return judge, calls


class LevelTests(unittest.TestCase):
    def test_clause_cuts_match_onset_v1(self):
        self.assertEqual(clause_cuts("你好，世界。再见"), [3, 6, 8])
        self.assertEqual(clause_cuts("a. b! c"), [2, 5, 7])

    def test_safe_whole_costs_one_call(self):
        judge, calls = stepped((None, None))
        result = locate(list(range(10, 330, 10)), judge)
        self.assertEqual((result["status"], len(calls)), ("safe", 1))
        self.assertTrue(all(position_level(e, result["probes"]) == 0 for e in range(0, 321, 7)))

    def test_two_levels_located_in_log_calls(self):
        cuts = list(range(10, 330, 10))                         # 32 clauses
        for c, u in ((95, 200), (10, 320), (150, 150), (40, None)):
            judge, calls = stepped((c, u))
            result = locate(cuts, judge)
            self.assertEqual(result["status"], "located")
            first_c = min(x for x in cuts if x >= c)
            self.assertEqual(result["onsets"]["controversial"]["cut"], first_c)
            levels_needed = 2 if u is not None else 1
            self.assertLessEqual(len(calls), 1 + levels_needed * (math.ceil(math.log2(len(cuts))) + 1))
            self.assertEqual(len(calls), len(set(calls)))       # no prefix judged twice
            if u is not None:
                self.assertEqual(result["onsets"]["unsafe"]["cut"], min(x for x in cuts if x >= u))
            else:
                self.assertNotIn("unsafe", result["onsets"])
            # every position outside the onset clauses gets exactly the level of its prefix
            for e in range(0, 321):
                expected = 2 if u is not None and e >= min(x for x in cuts if x >= u) else 1 if e >= first_c else 0
                got = position_level(e, result["probes"])
                inside = any(o["prev_cut"] < e < o["cut"] for o in result["onsets"].values())
                self.assertEqual(got, None if inside else expected, (c, u, e))

    def test_failures(self):
        cuts = [5, 10, 15, 20]
        self.assertEqual(locate(cuts, lambda c: None)["status"], "judge_error")
        self.assertEqual(locate(cuts, lambda c: None if c == 10 else ("unsafe", False))["status"], "judge_error")
        noisy = {5: "safe", 10: "unsafe", 15: "safe", 20: "unsafe"}
        self.assertEqual(locate(cuts, lambda c: (noisy[c], False))["status"], "nonmonotonic")
        self.assertEqual(locate(cuts, lambda c: ("safe", True))["status"], "excluded")
        self.assertEqual(locate(clause_cuts(""), lambda c: ("safe", False))["status"], "empty")
        self.assertFalse(monotonic([{"cut": 5, "level": "unsafe"}, {"cut": 9, "level": "controversial"}]))
        self.assertIsNone(position_level(7, [{"cut": 5, "level": "unsafe"}, {"cut": 9, "level": "safe"}]))


def rows_v14(n=4):
    out = []
    for w in range(n):
        for label in ("safe", "unsafe"):
            row = {"sample_id": f"s{w}-{label}", "task_key": f"k{w}", "word": f"词{w}", "split": "dev" if w < 2 else "train",
                   "language": "zh", "family": f"f{w}", "index": 0 if label == "safe" else 1, "response_style": "x",
                   "source_group": "g", "label": label,
                   "messages": [{"role": "user", "content": "占位提问"},
                                {"role": "assistant", "content": "这是一段中性占位文字，没有实际含义。第二句也是占位。"}]}
            if label == "unsafe":
                row.update(onset_char=5, onset_end_char=9)
            out.append(row)
    return out


class BuilderTests(unittest.TestCase):
    def test_build_is_blind(self):
        items, key = build(rows_v14(), {"dev"})
        self.assertEqual(len(items), 4)
        self.assertTrue(all(set(i) == {"task_key", "user_prompt", "assistant_text"} for i in items))
        self.assertEqual({k["sample_id"] for k in key}, {i["task_key"] for i in items})
        self.assertEqual(len(build(rows_v14(), None, {"s0-safe"})[0]), 1)

    def test_context_sample_and_export(self):
        self.assertEqual(context([{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]), "问")
        multi = [{"role": "system", "content": "设定"}, {"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]
        self.assertEqual(context(multi), "SYSTEM:\n设定\n\nUSER:\n问")
        chosen = pick(rows_v14(8), 6)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(sum(r["label"] == "safe" for r in chosen), 3)           # balanced over slot / label
        self.assertEqual(pick(rows_v14(8), 6), chosen)                             # deterministic
        row = export({"sample_id": "p1", "base_split": "train", "language": "en", "family": "f", "label_tier": "rubric_benign",
                      "source_label": "safe", "target_role": "assistant", "messages": multi, "ids": [1, 2]})
        self.assertEqual((row["split"], row["label"], row["messages"]), ("train", "safe", multi))
        self.assertNotIn("ids", row)
        self.assertEqual(build([row])[0][0]["user_prompt"], context(multi))

    def test_compare_judges(self):
        a = [{"sample_id": "x", "label": "controversial", "level": "controversial", "stratum": "redline",
              "onsets": {"controversial": {"prev_cut": 0, "cut": 10}}},
             {"sample_id": "y", "label": "safe", "level": "safe", "stratum": "normal", "onsets": {}},
             {"sample_id": "z", "label": "unusable", "level": None, "stratum": None, "onsets": {}}]
        b = [{**a[0], "label": "unsafe", "level": "unsafe", "onsets": {"controversial": {"prev_cut": 10, "cut": 20},
                                                                      "unsafe": {"prev_cut": 10, "cut": 20}}},
             a[1], {**a[2], "label": "safe", "level": "safe"}]
        out = compare(a, b)
        self.assertEqual((out["both_usable"], out["label_agreement"], out["cut_agreement"]), (2, 0.5, 1.0))
        self.assertEqual(out["onset_shift_chars"]["controversial"], {"n": 1, "same_cut": 0, "median": 10})

    def test_review_sheet_marks_onsets(self):
        self.assertEqual(review_sample.marked("0123456789", {"controversial": {"prev_cut": 2, "cut": 4},
                                                             "unsafe": {"prev_cut": 6, "cut": 8}}),
                         "01⟦C⟧23⟦C|⟧45⟦U⟧67⟦U|⟧89")
        labels = [{"sample_id": f"s{i}", "old_label": ("safe", "unsafe")[i % 2], "label": ("safe", "controversial", "unusable")[i % 3],
                   "stratum": "normal", "rules": [], "onsets": {}} for i in range(12)]
        chosen = review_sample.pick(labels, 4, "seed")
        self.assertEqual(len({f"{r['old_label']}->{r['label']}" for r in chosen}), 4)   # one per transition first
        self.assertTrue(all(r["label"] != "unusable" for r in chosen))

    def test_cli_writes_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rows.jsonl"
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_v14()), encoding="utf-8")
            out = Path(tmp) / "tasks"
            subprocess.run([sys.executable, str(HERE / "make_tasks.py"), "--source", str(source), "--per-task", "3",
                            "--output", str(out)], check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual((manifest["responses"], manifest["tasks"]), (8, 3))
            self.assertEqual(manifest["policy_digest"], digest())
            for instruction in (out / "tasks").glob("*/instruction.md"):
                task = json.loads(instruction.read_text())
                self.assertEqual((task["prompt_version"], task["judge_version"]), (LEVEL_VERSION, PROMPT_VERSION))
                for term in task["terms"]:
                    self.assertEqual(set(term), {"task_key", "user_prompt", "assistant_text"})
            key = [json.loads(line) for line in (out / "items.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(k["group_key"].startswith("redline-v1-") for k in key))


class FakeModel:
    name, wire_api = "fake", "chat_completions"

    def __init__(self, reply):
        self.reply = reply

    def request(self, body, timeout):
        text = json.loads(body["messages"][1]["content"].split("\n", 1)[1])["assistant_text"]
        payload = self.reply(text)
        if payload is None:
            return {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}


def run_flow(terms, reply):
    fake = types.ModuleType("aster_flow")
    fake.Context = object
    fake.flow = lambda **_: (lambda fn: fn)
    sys.modules["aster_flow"] = fake
    import importlib.util
    spec = importlib.util.spec_from_file_location("redline_v1_flow", HERE / "flow/flow.py")
    flow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flow_module)
    records = {}
    ctx = types.SimpleNamespace(
        task=types.SimpleNamespace(instruction=json.dumps({"prompt_version": LEVEL_VERSION, "judge_version": PROMPT_VERSION,
                                                           "policy_digest": digest(), "group_key": "g0", "terms": terms})),
        models={"main": FakeModel(reply)},
        results=types.SimpleNamespace(record=lambda name, value: records.__setitem__(name, value)),
        evaluation=types.SimpleNamespace(score=lambda *a, **k: None))
    flow_module.run(ctx)
    return records


class FlowTests(unittest.TestCase):
    def test_flow_end_to_end_and_relabel(self):
        text = "第一句占位，第二句占位，第三句占位，第四句占位。"
        marker_c, marker_u = text.index("第二句") + 3, text.index("第四句") + 3

        def reply(prefix):
            if len(prefix) >= marker_u:
                return facts([hit("R7", "writes")])
            if len(prefix) >= marker_c:
                return facts([hit("R7", "restates")])
            return facts()

        records = run_flow([{"task_key": "a", "user_prompt": "占位", "assistant_text": text},
                            {"task_key": "b", "user_prompt": "占位", "assistant_text": text}], reply)
        self.assertEqual(records["summary"]["finished"], 2)
        value = records["item_00000"]
        self.assertEqual(value["result"]["status"], "located")
        cuts = clause_cuts(text)
        self.assertEqual(value["result"]["onsets"]["controversial"]["cut"], min(c for c in cuts if c >= marker_c))
        self.assertEqual(value["result"]["onsets"]["unsafe"]["cut"], min(c for c in cuts if c >= marker_u))
        key = [{"sample_id": s, "split": "dev", "label": "safe", "index": 0, "response_style": "x",
                "response_chars": len(text)} for s in ("a", "b", "c")]
        rows, summary = join(key, [records["item_00000"], {**records["item_00001"], "item_id": "b"}])
        self.assertEqual(summary["statuses"], {"located": 2, "missing": 1})
        labelled, report = apply([rows])
        self.assertEqual([r["label"] for r in labelled], ["unsafe", "unsafe", "unusable"])
        self.assertEqual(report["old_to_new"], {"safe->unsafe": 2, "safe->unusable": 1})
        self.assertEqual(report["switch_sensitivity"]["B1=non_redline"], {})       # no boundary hits in this text
        self.assertEqual(labelled[0]["boundary"], [])
        # policy change without re-judging: restating becomes safe, so the controversial onset disappears
        relabelled, _ = apply([rows], {"restates": "safe"})
        self.assertEqual(relabelled[0]["level"], "unsafe")
        self.assertNotEqual(relabelled[0]["onsets"]["controversial"]["cut"], labelled[0]["onsets"]["controversial"]["cut"])

    def test_refusal_is_a_judge_error_and_fallback_fills_it(self):
        text = "第一句占位，第二句占位。"
        records = run_flow([{"task_key": "a", "user_prompt": "占位", "assistant_text": text}], lambda p: None)
        self.assertEqual(records["item_00000"]["result"]["status"], "judge_error")
        key = [{"sample_id": "a", "split": "dev", "label": "safe", "index": 0, "response_style": "x", "response_chars": len(text)}]
        primary, _ = join(key, [records["item_00000"]])
        fallback, _ = join(key, [run_flow([{"task_key": "a", "user_prompt": "占位", "assistant_text": text}],
                                          lambda p: facts())["item_00000"]])
        labelled, report = apply([primary, fallback])
        self.assertEqual((labelled[0]["label"], labelled[0]["source_rank"]), ("safe", 1))
        self.assertIsNone(label_response(primary[0]))


if __name__ == "__main__":
    unittest.main()
