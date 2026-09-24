"""Render collected Round4 research results without running a model (stdlib only).

Usage:
    python3 round4/render_results.py
    python3 round4/render_results.py --results-root /path/to/results --stdout

Missing artifacts produce an explicitly incomplete report. Test scores never
select a candidate. A candidate needs a recorded development choice and a
completed, checkpoint-matched long-stream audit before it can be recommended
for further validation; this script never approves production deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
NAMES = ("full", "window", "memory", "classification_rl")
TITLES = {
    "full": "full / SFT",
    "window": "window / SFT",
    "memory": "memory / SFT",
    "classification_rl": "分类 RL",
    "a0": "原版 A0（质量参考）",
}
REQUIRED_PREFIXES = {513, 1025, 4097, 8192}
MISSING = "待结果"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, places=4):
    return f"{value:.{places}f}" if number(value) else MISSING


def pct(value):
    return f"{100 * value:.2f}%" if number(value) else MISSING


def integer(value):
    return f"{int(value):,}" if number(value) and value == int(value) else MISSING


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def mapping(value):
    return value if isinstance(value, dict) else {}


def metric_rows(metric):
    return {key: row for key, row in mapping(metric.get("strata")).items() if isinstance(row, dict)}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values and all(number(v) for v in values) else None


def total(values):
    values = list(values)
    return sum(values) if values and all(number(v) for v in values) else None


def macro(metric, key):
    return mean(row.get(key) for row in metric_rows(metric).values())


class Artifacts:
    def __init__(self, results_root, artifact_dir=None):
        self.root = Path(results_root).resolve()
        self.artifact = Path(artifact_dir).resolve() if artifact_dir else self.root / "output" / "round4"
        self.missing = []
        self.errors = []
        self.read_paths = []

    def read(self, relative, outer=False):
        path = (self.root if outer else self.artifact) / relative
        if not path.exists():
            self.missing.append(path)
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
        except (OSError, ValueError) as exc:
            self.errors.append((path, str(exc)))
            return {}
        self.read_paths.append(path)
        return value


def audit_gate(name, audit, command, summary):
    """Fail closed on incomplete, stale, or purely short-stream evidence."""
    entry = mapping(mapping(audit.get("results")).get(name))
    reasons = []
    if command.get("completed") is not True or command.get("returncode") != 0:
        reasons.append("审计命令未确认正常结束")
    if audit.get("status") != "completed":
        reasons.append("长流审计未完成")
    if entry.get("pass") is not True:
        reasons.append("该检查点未通过长流审计")
    if entry.get("long_stream_parity_pass") is not True:
        reasons.append("长流概率一致性未通过")
    if entry.get("variant") != "full" and entry.get("bounded_state_pass") is not True:
        reasons.append("有界缓存未通过")
    expected = summary.get("checkpoint_sha256")
    actual = entry.get("checkpoint_sha256")
    if not expected or actual != expected:
        reasons.append("训练与审计检查点 SHA 未匹配")
    records = [r for r in entry.get("records", []) if isinstance(r, dict)]
    prefixes = {r.get("prefix_tokens") for r in records}
    if not REQUIRED_PREFIXES.issubset(prefixes):
        reasons.append("缺少完整长前缀检查记录")
    long_records = [r for r in records if r.get("prefix_tokens") == 8192]
    if len(long_records) < 4 or any(
        set(mapping(r.get("risk_probability_errors"))) != {"user", "assistant"}
        for r in long_records
    ):
        reasons.append("8K 检查未覆盖四组双角色结果")
    groups = {}
    for record in records:
        schedule = record.get("schedule")
        if isinstance(schedule, list) and schedule and all(number(v) and v > 0 for v in schedule):
            key = (record.get("input_order"), tuple(schedule))
            groups.setdefault(key, set()).add(record.get("prefix_tokens"))
    if (len(groups) < 4 or {key[0] for key in groups} != {"forward", "reverse"}
            or any(not REQUIRED_PREFIXES.issubset(prefixes) for prefixes in groups.values())):
        reasons.append("输入顺序/分块方案的长前缀覆盖未齐")
    tolerance = entry.get("probability_tolerance")
    errors = [value for row in records for value in mapping(row.get("risk_probability_errors")).values()]
    if not number(tolerance) or tolerance <= 0 or not errors or any(
        not number(value) or value >= tolerance for value in errors
    ):
        reasons.append("数值误差证据缺失或超出容差")
    return not reasons, reasons


def render_report(results_root, artifact_dir=None):
    files = Artifacts(results_root, artifact_dir)
    final = files.read("final_summary.json")
    spec = files.read("risk_run_spec.json")
    audit = files.read("long_stream_audit.json")
    audit_command = files.read("post_training_audit_status.json", outer=True)
    locations = files.read("checkpoint_locations.json", outer=True)
    summaries, dev, test, runtimes = {}, {}, {}, {}
    final_sft = {
        row.get("variant"): row for row in final.get("sft", []) if isinstance(row, dict)
    }
    final_tests = mapping(final.get("sealed_test"))
    for name in NAMES:
        filename = "summary.json" if name == "classification_rl" else "sft_summary.json"
        summaries[name] = files.read(f"{name}/{filename}") or mapping(final_sft.get(name))
        dev[name] = files.read(f"{name}/exported_dev_metrics.json")
        test[name] = files.read(f"{name}/sealed_test_metrics.json") or mapping(final_tests.get(name))
        runtimes[name] = files.read(f"{name}/runtime.json")
    test["a0"] = files.read("a0_reference_metrics.json")
    gates = {name: audit_gate(name, audit, audit_command, summaries[name]) for name in NAMES}
    chosen = final.get("rl_base_variant") or summaries["classification_rl"].get("variant")
    chosen = chosen if chosen in NAMES[:3] else None
    sections = [
        "# Round4 训练结果",
        "生成时间（UTC）：" + datetime.now(timezone.utc).isoformat(timespec="seconds")
        + f"。本报告只读取 `{files.artifact}` 及收集器状态，不运行模型。",
    ]
    complete = final.get("status") == "completed"
    if complete:
        sections.append("训练主流程记录为 **completed**；这仅表示训练和评估流程结束，不能据此判断长流审计通过或允许部署。")
    else:
        sections.append("**结果尚未完整收集，无法确认端到端训练完成。** 缺失文件记为“待结果”，不将其解释为零分或失败。")
    status_rows = []
    for name in NAMES:
        summary = summaries[name]
        status_rows.append([
            TITLES[name], summary.get("status", MISSING),
            "已记录" if dev[name] else MISSING,
            "已记录" if test[name] else MISSING,
            "已记录" if runtimes[name].get("records") else MISSING,
            "通过且证据完整" if gates[name][0] else "未通过或证据未齐",
        ])
    sections += ["## 完成状态", table(["模型", "训练状态", "导出权重开发评估", "保留测试", "速度", "长流门槛"], status_rows)]
    sections += ["## 开发集选模", "阈值从独立校准集的安全样本选取；开发评分为宏平均风险召回减去 `5 × 平均超额误报率`，误报预算为 5%。保留测试只报告结果，不用于重新挑选架构或检查点。"]
    rows = []
    for name in NAMES:
        summary = summaries[name]
        rows.append([
            TITLES[name], integer(summary.get("steps", summary.get("updates"))),
            integer(summary.get("selected_step")), fmt(dev[name].get("selection_score")),
            pct(macro(dev[name], "recall")), pct(macro(dev[name], "fpr")), fmt(macro(dev[name], "pr_auc")),
        ])
    sections.append(table(["模型", "完成更新数", "选中步骤", "导出权重开发分数", "宏召回", "宏 FPR", "宏 PR-AUC"], rows))
    if chosen:
        sections.append(f"已记录的 RL 起点架构：**{chosen}**。该选择来自训练流程，报告不会按测试集结果改选其他架构。")
        eligible = [name for name in (chosen, "classification_rl")
                    if summaries[name].get("status") == "completed" and gates[name][0]
                    and number(dev[name].get("selection_score"))]
        if complete and eligible:
            # Stable ordering prefers the SFT checkpoint on a tied development score.
            candidate = max(eligible, key=lambda name: dev[name]["selection_score"])
            sections.append(f"在已固定架构的 SFT 与 RL 检查点之间，仅按导出权重开发分数和长流门槛，"
                            f"**{TITLES[candidate]}** 可作为下一步验收候选。分数相同保留 SFT；这不构成生产部署批准。")
        else:
            sections.append("当前不能给出完成长流验收的候选：训练结果、开发分数或匹配检查点的长流证据尚不齐全。")
    else:
        sections.append("RL 起点架构尚未记录，暂不推荐候选。")

    sections += ["## 保留测试的风险能力", "所有分层的阈值只来自各自校准集。宏平均对各语言/角色分层等权；它不是按样本量加权的总体指标。原版 A0 仅作同一新测试集上的质量参考。"]
    rows = []
    for name in (*NAMES, "a0"):
        strata = metric_rows(test[name])
        rows.append([TITLES[name], integer(total(r.get("n") for r in strata.values())),
                     pct(macro(test[name], "recall")), pct(macro(test[name], "fpr")),
                     fmt(macro(test[name], "pr_auc")), fmt(macro(test[name], "roc_auc"))])
    sections.append(table(["模型", "样本数", "宏召回", "宏 FPR", "宏 PR-AUC", "宏 ROC-AUC"], rows))
    rows = []
    for name in (*NAMES, "a0"):
        strata = metric_rows(test[name])
        for stratum, row in sorted(strata.items()):
            rows.append([TITLES[name], stratum, integer(row.get("safe")), integer(row.get("unsafe")),
                         pct(row.get("recall")), pct(row.get("fpr")), fmt(row.get("pr_auc")),
                         fmt(row.get("threshold_from_calibration"), 6)])
        if not strata:
            rows.append([TITLES[name], MISSING, MISSING, MISSING, MISSING, MISSING, MISSING, MISSING])
    sections.append(table(["模型", "语言/目标角色", "安全数", "风险数", "风险召回", "FPR", "PR-AUC", "校准阈值"], rows))

    sections += ["## SFT 与分类 RL 对比", "以下只比较已固定起点架构及其 RL 版本。RL 采样分类动作，不生成文本 token；测试差值仅作观察。"]
    if chosen:
        rows = []
        for name in (chosen, "classification_rl"):
            rows.append([TITLES[name], fmt(dev[name].get("selection_score")),
                         pct(macro(test[name], "recall")), pct(macro(test[name], "fpr")),
                         fmt(macro(test[name], "pr_auc")), "通过" if gates[name][0] else "未确认通过"])
        sections.append(table(["版本", "开发分数", "测试宏召回", "测试宏 FPR", "测试宏 PR-AUC", "长流门槛"], rows))
        differences = []
        for label, key in (("召回", "recall"), ("FPR", "fpr"), ("PR-AUC", "pr_auc")):
            before, after = macro(test[chosen], key), macro(test["classification_rl"], key)
            if number(before) and number(after):
                value = after - before
                differences.append(f"{label} {value * 100:+.2f} 个百分点" if key != "pr_auc" else f"{label} {value:+.4f}")
        if differences:
            sections.append("RL − SFT 的保留测试差值：" + "；".join(differences) + "。召回/PR-AUC 越高越好，FPR 越低越好。")
    else:
        sections.append("尚无已记录的 RL 起点，无法建立配对比较。")

    sections += ["## L20 推理速度与状态内存", "按相同初始 context 和 chunk 分组比较。ITPS 仅统计新输入 token；延迟包含分类结果返回 CPU。数据是预热后的单会话 token-ID 模型调用，不含分词、HTTP 或批处理。此处不引用旧实验速度，也不为 A0 补造同场速度。"]
    speed_rows = []
    for name in NAMES:
        for record in runtimes[name].get("records", []):
            if not isinstance(record, dict):
                continue
            context, chunk = record.get("context_tokens"), record.get("chunk_tokens")
            if not number(context) or not number(chunk):
                continue
            state = record.get("state_bytes_at_prefill")
            speed_rows.append((context, chunk, NAMES.index(name), [
                integer(context), integer(chunk), TITLES[name], fmt(record.get("itps"), 1),
                fmt(record.get("classifications_per_second"), 1), fmt(record.get("p50_ms"), 2),
                fmt(record.get("p95_ms"), 2), fmt(state / 1048576 if number(state) else None, 3),
            ]))
    if speed_rows:
        sections.append(table(["初始 context", "chunk", "模型", "ITPS", "分类次数/秒", "P50 ms", "P95 ms", "prefill 状态 MiB"], [r[3] for r in sorted(speed_rows)]))
    else:
        sections.append("尚无已收集的同场速度记录。")
    sections.append("状态 MiB 是记录的缓存底层存储量，不是全部 GPU 显存；full 的状态随历史增长，window/memory 的有界性以独立长流审计为准。")

    sections += ["## 8K 长流审计", "审计比较导出检查点在相同输入前缀上的一次性分类与增量分类，检查 513、1025、4097、8192 tokens、两种输入顺序、两组分块方案及两个角色头。长输入由开发样本拼接，仅验证数值和缓存，不证明自然长对话的风险识别能力。"]
    sections.append(f"审计命令完成状态：`{audit_command.get('completed', '未记录')}`；"
                    f"命令返回码：`{audit_command.get('returncode', '未记录')}`；"
                    f"审计文件状态：`{audit.get('status', '未记录')}`；"
                    f"文件记录的全部候选通过：`{audit.get('all_candidates_pass', '未记录')}`。"
                    "命令返回 0 或训练 completed 都不能替代逐候选审计通过。")
    rows = []
    for name in NAMES:
        entry = mapping(mapping(audit.get("results")).get(name))
        records = [r for r in entry.get("records", []) if isinstance(r, dict)]
        expected_sha = summaries[name].get("checkpoint_sha256")
        matched = bool(expected_sha and expected_sha == entry.get("checkpoint_sha256"))
        bounded = "不适用" if entry.get("variant") == "full" else str(entry.get("bounded_state_pass", "未记录"))
        rows.append([TITLES[name], fmt(entry.get("max_probability_error"), 6),
                     fmt(entry.get("probability_tolerance"), 4),
                     integer(sum(r.get("prefix_tokens") == 8192 for r in records)) if entry else MISSING,
                     bounded, "匹配" if matched else "未确认", "通过" if gates[name][0] else "未通过或证据未齐"])
    sections.append(table(["模型", "最大概率误差", "容差", "8K 记录数", "缓存有界", "SHA", "完整证据门槛"], rows))
    failures = [(name, reasons) for name, (passed, reasons) in gates.items() if not passed]
    if failures:
        sections.append("\n".join(f"- {TITLES[name]}：{'；'.join(reasons)}。" for name, reasons in failures))
    for name in NAMES:
        entry = mapping(mapping(audit.get("results")).get(name))
        if entry.get("error"):
            sections.append(f"{TITLES[name]} 审计异常：`{cell(entry.get('error_type', 'Error'))}` — {cell(entry['error'])}")
    if audit_command.get("error"):
        sections.append("审计执行异常：" + cell(audit_command["error"]))

    sections += ["## 检查点与运行记录", f"收集器记录的 PVC：`{locations.get('pvc', '未记录')}`。权重保留在 PVC；本报告没有下载权重，也没有重新计算远端文件 SHA。"]
    recorded_paths = locations.get("checkpoints", [])
    rows = []
    for name in NAMES:
        path = f"/work/output/round4/{name}/best.safetensors"
        rows.append([TITLES[name], path, "收集器已记录" if path in recorded_paths else "约定路径，待收集器确认",
                     summaries[name].get("checkpoint_sha256", MISSING)])
    sections.append(table(["模型", "PVC 内路径", "路径证据", "训练记录的 SHA-256"], rows))
    manifest = mapping(spec.get("data_manifest"))
    if manifest:
        sections.append("冻结数据量：" + "；".join(f"{split} {integer(count)} 条" for split, count in mapping(manifest.get("counts")).items()) + "。")
        hashes = mapping(manifest.get("output_hashes"))
        if hashes:
            sections.append(table(["数据分割", "冻结 SHA-256"], sorted(hashes.items())))

    sections += ["## 解释边界", "\n".join([
        "- 本轮基于已有 Qwen3.5 文本主干和风险头做后训练，不是从零预训练。所有神经网络计算在 L20。",
        "- 公共标签遵循来源政策；合成标签未独立核验并降权，benign 模板只在其构造情境内提供规则标签。",
        "- 本轮硬标签为 safe/unsafe；类别头及第三档风险没有独立新增监督和验收，不能由二分类结果推断其能力。",
        "- 公开参考测试缺少英文用户分层；精确去重不等于语义去重，也不能保证基础模型预训练从未见过测试内容。",
        "- 阈值的 5% 预算只约束有限校准样本；开发/测试的真实 FPR 单独报告，不是生产 SLA。",
        "- token-ID 流式一致性不能替代任意文本重分词回滚、HTTP 服务、并发吞吐或真实长期风险上下文验收。",
        "- A0 与当前分类器使用各自输入格式，质量差异不能只归因于架构；本报告不使用保留测试重新选模。",
        "- 所有候选都没有本报告授予的生产批准；未完成或失败的长流审计阻止候选进入下一步验收。",
    ])]
    if files.missing or files.errors:
        sections.append("## 缺失或无法读取的结果")
        if files.missing:
            sections.append("下列文件当前不存在；如对应指标来自 final_summary 的实际记录，表格仍可展示该记录，其余保持“待结果”。")
            sections.append("\n".join(f"- `{path}`" for path in files.missing))
        if files.errors:
            sections.append("\n".join(f"- `{path}`：{cell(error)}" for path, error in files.errors))
    sections += ["## 本报告读取的证据文件", "\n".join(f"- `{path}`" for path in files.read_paths) or "尚无可读取的结果文件。"]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--artifact-dir", type=Path, help="Override results-root/output/round4")
    parser.add_argument("--output", type=Path, default=HERE / "ROUND4_RESULTS.md")
    parser.add_argument("--stdout", action="store_true", help="Print without writing a report file")
    args = parser.parse_args()
    report = render_report(args.results_root, args.artifact_dir)
    if args.stdout:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(json.dumps({"report": str(args.output.resolve()), "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
