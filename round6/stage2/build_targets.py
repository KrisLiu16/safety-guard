"""Stage 2, step 1 (CPU, in the job's pod where the Round5 data and the tokenizer live): per-token 3-class targets.

Stage 1 showed that a readout of the frozen backbone stops at a calibration AUC of about 0.87 (T028), so stage 2
trains the backbone. Every target token of every record gets the red-line level of its prefix from the stored
probes (levels.position_level), as the head's class index: 0 safe, 1 unsafe, 2 controversial. Tokens inside a
located onset clause, or inside a word-rule span (apply_policy.raise_from), are undetermined and get no target;
records whose label is unusable or excluded are skipped.
v2 (T030, POLICY section 9): a response labelled with an alert (harm outside the red lines, apply_policy.py) gets
class 3 ("alert", a soft target in train_stage2.py) at every safe position ending after the alert start: the
generation onset when known, else the start of the last clause, and then the positions before it get no target.
Records:
  runA          Run A answers, [user, assistant], assistant head; labels from the answer-side labelling
  runA_prompts  each distinct Run A prompt alone, [user], user head (sample id <task_key>:prompt:<label>, as T023)
  leader_v1     T031 leader-sensitivity conversations (--leader-rows, --leader-labels), assistant head; with
                --leader-prompt-labels also the conversation up to its last user message, user head
                (source leader_v1_prompts, sample id "<sample_id>:prompt")
  prefix_v2     Round5 prefix_v2 records of both roles (head = target_role) with their frozen ids; a record whose
                re-tokenized ids differ is skipped and counted. Train records also give their augmented view
                (the complete target plus a neutral suffix): its anchors take the whole-text level, at half weight.
Output: records_<split>.jsonl per split (sample_id, source, role, split, family, language, weight, level, ids,
positions, classes) and report.json (counts by source/role/split/class, skips, input hashes).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probe"))
sys.path.insert(0, str(HERE.parent / "redline_v1" / "flow"))
from levels import clause_cuts, position_level  # noqa: E402
from probe_common import serialize  # noqa: E402

CLASS_INDEX = {0: 0, 1: 2, 2: 1}          # level rank (safe, controversial, unsafe) -> head class index
ALERT = 3                                 # safe level, raised score (soft target in train_stage2.py)
LEVEL_RANK = {"safe": 0, "controversial": 1, "unsafe": 2}
MAX_TOKENS = 8192
AUGMENTED_WEIGHT = 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usable(labelled):
    return labelled is not None and labelled["label"] not in ("unusable", "excluded")


def content_targets(encode, messages):
    """(ids, target positions, char end of each relative to the last message's content)."""
    text = serialize(messages)
    content = messages[-1]["content"]
    start = len(text) - len(content)
    ids, offsets = encode(text)
    positions = [i for i, (_, end) in enumerate(offsets) if end > start]
    return list(ids), positions, [min(offsets[i][1] - start, len(content)) for i in positions]


def alert_start(labelled, content):
    """(char offset after which safe positions are alert, whether earlier positions keep their targets) or None."""
    alert = labelled.get("alert")
    if not alert:
        return None
    if alert.get("from") is not None:
        return alert["from"], True
    cuts = clause_cuts(content)
    return (cuts[-2] if len(cuts) > 1 else 0), False


def classify(positions, ends, labelled, alert=None):
    """Pure: keep the positions whose prefix level is determined; returns (positions, classes, undetermined).
    alert: alert_start() output; safe positions ending after it become ALERT, and when the start is only the last
    clause, the positions before it are left out."""
    kept, classes = [], []
    for position, end in zip(positions, ends):
        rank = position_level(end, labelled["probes"])
        if rank is None:
            continue
        cls = CLASS_INDEX[rank]
        if alert is not None and cls == 0:
            if end > alert[0]:
                cls = ALERT
            elif not alert[1]:
                continue
        kept.append(position)
        classes.append(cls)
    return kept, classes, len(positions) - len(kept)


def record(meta, ids, positions, classes, labelled, weight):
    return {**meta, "weight": weight, "level": labelled["level"], "ids": ids, "positions": positions, "classes": classes}


def teacher_targets(positions, ends, teacher, min_cat=0.5):
    """Pure (v3, T033): align a teacher's per-token output to student positions by character ends. Each student
    position takes the teacher token with the largest end <= its own end (the teacher has seen no more text than the
    student); positions before the first teacher token are skipped. Returns {"t_positions", "t_risk" (safe, unsafe,
    controversial), "t_cat" (teacher category where the teacher's non-safe probability > min_cat, else -1)}."""
    import bisect
    out = {"t_positions": [], "t_risk": [], "t_cat": []}
    for position, end in zip(positions, ends):
        j = bisect.bisect_right(teacher["ends"], end) - 1
        if j < 0:
            continue
        risk = teacher["risk"][j]
        out["t_positions"].append(position)
        out["t_risk"].append(risk)
        out["t_cat"].append(teacher["cat"][j] if 1 - risk[0] > min_cat else -1)
    return out


def end_targets(positions, teacher_end, min_cat=0.5):
    """Pure (v4, T035): Qwen3Guard-Stream trains and reads its query head only at the closing <|im_end|> of the user
    turn, so its outputs inside a prompt are untrained (v3 learned them and flagged 189 of 194 ordinary questions). A
    user text gets one teacher target, the end-of-turn output of teacher_label_l20.py --end-user, at the student's
    last content position."""
    if not positions:
        return {"t_positions": [], "t_risk": [], "t_cat": []}
    risk = teacher_end["end_risk"]
    return {"t_positions": [positions[-1]], "t_risk": [risk],
            "t_cat": [teacher_end["end_cat"] if 1 - risk[0] > min_cat else -1]}


def attach(positions, ends, teacher):
    return end_targets(positions, teacher) if "end_risk" in teacher else teacher_targets(positions, ends, teacher)


def text_record(row, messages, source, role, labelled, encode, sample_id=None, teacher=None):
    """Run A answer or prompt. Returns (record or None, skip reason or None, undetermined count). teacher (v3): the
    teacher's output for this text; its distributions are attached at every content position (teacher_targets)."""
    ids, all_positions, all_ends = content_targets(encode, messages)
    if not 1 <= len(ids) <= MAX_TOKENS:
        return None, "length", 0
    positions, classes, undetermined = classify(all_positions, all_ends, labelled,
                                                alert_start(labelled, messages[-1]["content"]))
    if not positions:
        return None, "no_target", undetermined
    meta = {"sample_id": sample_id or row["sample_id"], "source": source, "role": role, "split": row["split"],
            "family": row.get("family") or row.get("task_key"), "language": row.get("language")}
    made = record(meta, ids, positions, classes, labelled, 1.0)
    if teacher is not None:
        made.update(attach(all_positions, all_ends, teacher))
    return made, None, undetermined


def distill_record(row, source, role, encode, teacher):
    """Pure (v5, T037): a text with teacher targets only. No red-line positions, so it moves the general head (and the
    shared backbone through it) and never the red-line head. Returns (record or None, skip reason or None)."""
    if teacher is None:
        return None, "no_teacher"
    ids, all_positions, all_ends = content_targets(encode, row["messages"])
    if not 1 <= len(ids) <= MAX_TOKENS:
        return None, "length"
    made = {"sample_id": row["sample_id"], "source": source, "role": role, "split": row["split"],
            "family": row.get("family") or row.get("task_key"), "language": row.get("language"), "weight": 1.0,
            "level": "distill_only", "ids": ids, "positions": [], "classes": []}
    made.update(attach(all_positions, all_ends, teacher))
    if not made["t_positions"]:
        return None, "no_target"
    return made, None


def prefix_records(row, split, labelled, encode, teacher=None):
    """prefix_v2 record (and its augmented view in train). Returns (records, skip reason or None, undetermined).
    teacher (v3): attached to the main record at every content position; the augmented view gets none."""
    ids, positions, ends = content_targets(encode, row["messages"])
    if ids != list(row["ids"]):
        return [], "ids_mismatch", 0
    targets = set(row["target_token_positions"])
    pairs = [(p, e) for p, e in zip(positions, ends) if p in targets]
    kept, classes, undetermined = classify([p for p, _ in pairs], [e for _, e in pairs], labelled,
                                           alert_start(labelled, row["messages"][-1]["content"]))
    meta = {"sample_id": row["sample_id"], "source": "prefix_v2", "role": row["target_role"], "split": split,
            "family": row["family"], "language": row["language"]}
    out = [record(meta, ids, kept, classes, labelled, float(row["weight"]))] if kept else []
    if out and teacher is not None:
        out[0].update(attach(positions, ends, teacher))
    if split == "train" and row.get("augmentation", {}).get("anchors"):
        whole = CLASS_INDEX[LEVEL_RANK[labelled["level"]]]
        if whole == 0 and labelled.get("alert"):
            whole = ALERT
        anchors = [a["token_end_exclusive"] - 1 for a in row["augmentation"]["anchors"]]
        out.append(record({**meta, "sample_id": row["sample_id"] + ":augmented"}, list(row["augmentation"]["ids"]),
                          anchors, [whole] * len(anchors), labelled, float(row["weight"]) * AUGMENTED_WEIGHT))
    return out, (None if out else "no_target"), undetermined


def prompt_rows(rows):
    """Distinct Run A prompts: (row, sample id); same ids as make_tasks.prompt_rows and the T023 cache."""
    seen = set()
    for row in rows:
        sample_id = f"{row['task_key']}:prompt:{row['prompt_label']}"
        if sample_id not in seen:
            seen.add(sample_id)
            yield row, sample_id


def build(runA_rows, prefix_data, labels, encode, leader_rows=(), extra=(), teacher=None, teacher_end=None,
          distill_only=()):
    """Pure. runA_rows: stage-1 Run A input rows; prefix_data: split -> prefix_v2 rows; labels: name -> {sample_id:
    labels.jsonl row} for runA, runA_prompts, prefix_v2, prefix_v2_user. extra: (source, role, rows, {sample_id:
    label row}) for other v14-row sources whose last message is the role's (e.g. T032 English). teacher_end (v4):
    {sample_id: end-of-turn row}; when given, user texts take only that target (end_targets). distill_only (v5):
    (source, role, rows) of teacher-only texts, train split only (distill_record). Returns (split -> records, stats)."""
    out, stats = collections.defaultdict(list), collections.Counter()

    def t(sid, role):
        if role == "user" and teacher_end is not None:
            return teacher_end.get(sid)
        return teacher.get((sid, role)) if teacher else None

    def add(key, made, reason, undetermined):
        stats[f"{key}:undetermined_tokens"] += undetermined
        if reason:
            stats[f"{key}:skipped_{reason}"] += 1
        for r in made:
            out[r["split"]].append(r)
            stats[f"{key}:records"] += 1
            for c in r["classes"]:
                stats[f"{key}:class{c}"] += 1

    for row in runA_rows:
        key = f"runA:assistant:{row['split']}"
        labelled = labels["runA"].get(row["sample_id"])
        if not usable(labelled):
            stats[f"{key}:skipped_label"] += 1
            continue
        made, reason, undetermined = text_record(row, row["messages"], "runA", "assistant", labelled, encode,
                                                 teacher=t(row["sample_id"], "assistant"))
        add(key, [made] if made else [], reason, undetermined)
    for row, sample_id in prompt_rows(runA_rows):
        key = f"runA_prompts:user:{row['split']}"
        labelled = labels["runA_prompts"].get(sample_id)
        if not usable(labelled):
            stats[f"{key}:skipped_label"] += 1
            continue
        made, reason, undetermined = text_record(row, [row["messages"][0]], "runA_prompts", "user", labelled, encode,
                                                 sample_id, teacher=t(sample_id, "user"))
        add(key, [made] if made else [], reason, undetermined)
    for row in leader_rows:
        key = f"leader_v1:assistant:{row['split']}"
        labelled = labels.get("leader_v1", {}).get(row["sample_id"])
        if not usable(labelled):
            stats[f"{key}:skipped_label"] += 1
            continue
        made, reason, undetermined = text_record(row, row["messages"], "leader_v1", "assistant", labelled, encode,
                                                 teacher=t(row["sample_id"], "assistant"))
        add(key, [made] if made else [], reason, undetermined)
    for row in leader_rows:
        key = f"leader_v1_prompts:user:{row['split']}"
        sample_id = row["sample_id"] + ":prompt"
        labelled = labels.get("leader_v1_prompts", {}).get(sample_id)
        if labelled is None:
            continue                                   # prompt labels exist only for the pass kinds
        if not usable(labelled):
            stats[f"{key}:skipped_label"] += 1
            continue
        made, reason, undetermined = text_record(row, row["messages"][:-1], "leader_v1_prompts", "user", labelled,
                                                 encode, sample_id, teacher=t(sample_id, "user"))
        add(key, [made] if made else [], reason, undetermined)
    for source, role, rows, table in extra:
        for row in rows:
            key = f"{source}:{role}:{row['split']}"
            if row["messages"][-1]["role"] != role:
                raise ValueError(f"{source}: {row['sample_id']} does not end with a {role} message")
            labelled = table.get(row["sample_id"])
            if not usable(labelled):
                stats[f"{key}:skipped_label"] += 1
                continue
            made, reason, undetermined = text_record(row, row["messages"], source, role, labelled, encode,
                                                     teacher=t(row["sample_id"], role))
            add(key, [made] if made else [], reason, undetermined)
    for split, rows in prefix_data.items():
        for row in rows:
            key = f"prefix_v2:{row['target_role']}:{split}"
            table = labels["prefix_v2"] if row["target_role"] == "assistant" else labels["prefix_v2_user"]
            labelled = table.get(row["sample_id"])
            if not usable(labelled):
                stats[f"{key}:skipped_label"] += 1
                continue
            made, reason, undetermined = prefix_records(row, split, labelled, encode,
                                                        teacher=t(row["sample_id"], row["target_role"]))
            add(key, made, reason, undetermined)
    for source, role, rows in distill_only:
        for row in rows:
            key = f"{source}:{role}:{row['split']}"
            if row["split"] != "train":
                raise ValueError(f"{source}: teacher-only rows must be train rows ({row['sample_id']})")
            if row["messages"][-1]["role"] != role:
                raise ValueError(f"{source}: {row['sample_id']} does not end with a {role} message")
            made, reason = distill_record(row, source, role, encode, t(row["sample_id"], role))
            add(key, [made] if made else [], reason, 0)
    return dict(out), dict(sorted(stats.items()))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_labels(paths):
    return {r["sample_id"]: r for path in paths for r in read_jsonl(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runA", type=Path, default=Path("/work/round6/stage1_head/input_v1/stage1_runA_v1.jsonl"))
    parser.add_argument("--runA-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix-user-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix-data", type=Path, default=Path("/work/round5/data/prefix_v2"))
    parser.add_argument("--leader-rows", type=Path, help="T031 examples.jsonl (v14 rows), with --leader-labels")
    parser.add_argument("--leader-labels", type=Path, nargs="*", default=[])
    parser.add_argument("--leader-prompt-labels", type=Path, nargs="*", default=[],
                        help="user-side labels for the leader rows (make_leader_prompt_labels.py)")
    parser.add_argument("--extra", nargs=4, action="append", default=[], metavar=("SOURCE", "ROLE", "ROWS", "LABELS"),
                        help="another v14-row source (repeatable): rows ending with a ROLE message, apply_policy labels")
    parser.add_argument("--distill-only", nargs=3, action="append", default=[], metavar=("SOURCE", "ROLE", "ROWS"),
                        help="v5: teacher-only v14 rows (repeatable, train split): no red-line label, general head only")
    parser.add_argument("--teacher", type=Path, action="append", default=[],
                        help="teacher_label_l20.py output (jsonl.gz) to attach (repeatable)")
    parser.add_argument("--teacher-end", type=Path, action="append", default=[],
                        help="v4: teacher_label_l20.py --end-user output (jsonl.gz, repeatable); user texts take only "
                             "this target")
    parser.add_argument("--tokenizer", type=Path, default=Path("/work/models/qwen35/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from tokenizers import Tokenizer
    fast = Tokenizer.from_file(str(args.tokenizer))
    if fast.truncation is not None or fast.padding is not None:
        raise ValueError("tokenizer truncation/padding must be disabled")

    def encode(text):
        encoding = fast.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    label_paths = {"runA": args.runA_labels, "runA_prompts": args.prompt_labels, "prefix_v2": args.prefix_labels,
                   "prefix_v2_user": args.prefix_user_labels, "leader_v1": args.leader_labels,
                   "leader_v1_prompts": args.leader_prompt_labels}
    labels = {name: read_labels(paths) for name, paths in label_paths.items()}
    prefix_data = {split: read_jsonl(args.prefix_data / f"{split}.jsonl") for split in ("train", "calibration")}
    leader_rows = read_jsonl(args.leader_rows) if args.leader_rows else []
    extra = [(source, role, read_jsonl(rows), read_labels([labels_path])) for source, role, rows, labels_path in args.extra]
    if any(role not in ("assistant", "user") for _, role, _, _ in extra):
        raise ValueError("--extra ROLE must be assistant or user")
    import gzip
    teacher = None
    for path in args.teacher:
        teacher = teacher or {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (row["id"], row["role"]) in teacher:
                    raise ValueError(f"teacher case {row['id']} ({row['role']}) appears twice")
                teacher[(row["id"], row["role"])] = row
    teacher_end = None
    for path in args.teacher_end:
        teacher_end = teacher_end or {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                if row["id"] in teacher_end:
                    raise ValueError(f"end-of-turn teacher case {row['id']} appears twice")
                teacher_end[row["id"]] = row
    if args.distill_only and any(role not in ("assistant", "user") for _, role, _ in args.distill_only):
        raise ValueError("--distill-only ROLE must be assistant or user")
    distill_only = [(source, role, read_jsonl(rows)) for source, role, rows in args.distill_only]
    records, stats = build(read_jsonl(args.runA), prefix_data, labels, encode, leader_rows, extra, teacher, teacher_end,
                           distill_only)
    if teacher is not None or teacher_end is not None:
        stats["teacher_records"] = sum(1 for rows in records.values() for r in rows if "t_positions" in r)
        stats["teacher_records_user"] = sum(1 for rows in records.values() for r in rows
                                            if "t_positions" in r and r["role"] == "user")
    args.output.mkdir(parents=True)
    for split, rows in records.items():
        with (args.output / f"records_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"version": "stage2-targets-v5" if distill_only else "stage2-targets-v4" if teacher_end is not None
              else "stage2-targets-v3", "class_index": {"safe": 0, "unsafe": 1, "controversial": 2, "alert": ALERT},
              "inputs": {str(p): sha(p) for p in [args.runA, args.tokenizer, *[p for ps in label_paths.values() for p in ps],
                                                  *([args.leader_rows] if args.leader_rows else []),
                                                  *args.teacher, *args.teacher_end,
                                                  *[Path(p) for e in args.extra for p in e[2:]],
                                                  *[Path(e[2]) for e in args.distill_only],
                                                  *[args.prefix_data / f"{s}.jsonl" for s in prefix_data]]},
              "records": {split: len(rows) for split, rows in records.items()},
              "tokens": {split: sum(len(r["ids"]) for r in rows) for split, rows in records.items()},
              "stats": stats, "files": {p.name: sha(p) for p in sorted(args.output.glob("records_*.jsonl"))}}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("records", "tokens")}, indent=1))


if __name__ == "__main__":
    main()
