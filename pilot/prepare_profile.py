#!/usr/bin/env python3
"""Create a new frozen-flow candidate; never submits or calls a model."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-word", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.examples_per_word < 10 or args.examples_per_word % 10:
        parser.error("examples-per-word must be a positive multiple of 10")
    if not 1 <= args.concurrency <= 512:
        parser.error("concurrency must be between 1 and 512")
    if args.out.exists():
        parser.error("output already exists; use a new profile directory")
    requests = args.examples_per_word // 10
    wall = requests * 650 + 120
    if wall > 86400:
        parser.error("requested task exceeds the platform wall-time budget")
    shutil.copytree(ROOT / "flow", args.out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    settings = json.loads((args.out / "settings.json").read_text())
    settings["examples_per_word"] = args.examples_per_word
    settings["run_concurrency"] = args.concurrency
    settings["profile_status"] = "prepared_not_submitted"
    (args.out / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    flow = (args.out / "flow.py").read_text().replace('"wall_seconds": 900', f'"wall_seconds": {wall}')
    (args.out / "flow.py").write_text(flow)
    pipeline = (args.out / "pipeline.py").read_text()
    pipeline = pipeline.replace('POLICY_VERSION = "guard-pilot-v1"', 'POLICY_VERSION = "guard-generation-v2"')
    pipeline = pipeline.replace('所有标签均为合成初标，',
        '真实姓名不能加上“虚构人物/虚构角色”前缀来假装匿名。原词可在正常背景中出现，风险行为如需匿名用与词条不同的甲某/乙某。'
        '不要将明确创作小说或虚构情节自动当作现实伤害意图；本批风险样本应明确区分现实意图与合法创作。'
        '安全输入不得暗含规避审计/欺诈等请求，不安全输入要有清楚的行为意图。'
        '所有标签均为合成初标，')
    # Separate source-topic from risk policy; vary context without changing the case slots.
    pipeline = pipeline.replace('"slots": slots(start, count),',
        '"slots": slots(start, count),\n'
        '            "batch_style": ["日常短句", "新闻辨析", "学术讨论", "合规咨询", "社区交流", '
        '"引用与反驳", "公共服务问答", "企业内审", "公共信息查询", "证据与来源核验"][(start // 10) % 10],')
    (args.out / "pipeline.py").write_text(pipeline)
    print(json.dumps({"directory": str(args.out), "examples_per_word": args.examples_per_word,
                      "requests_per_word": requests, "planned_examples_for_50_words": 50*args.examples_per_word,
                      "concurrency": args.concurrency,
                      "planned_model_requests_for_50_words": 50*requests,
                      "wall_seconds_per_task": wall, "submitted": False}, ensure_ascii=False))

if __name__ == "__main__":
    main()
