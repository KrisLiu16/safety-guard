"""CPU-only rendering of collected Round5 evidence, including incomplete runs.

No evaluator/runtime imports, subprocesses, network, tokenizer or neural calls.
Final collection takes precedence per artifact; training collection is fallback.
This reports evaluator decisions and verifies basic bindings/accounting, but
does not rerun quality evaluation or silently turn process completion into pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

STRATA = ('zh/user', 'zh/assistant', 'en/assistant')
SPLITS = {'thinking': 1059, 'thinking_loc': 569, 'response_loc': 813}
INITIAL_SHA = 'bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def cell(value, *, percent=False):
    if value is None:
        return 'pending / 未提供'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if not math.isfinite(value):
            return 'invalid / 非有限值'
        return f'{value * 100:.2f}%' if percent else f'{value:.6g}'
    if percent and isinstance(value, int):
        return f'{value * 100:.2f}%'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace('|', '\\|').replace('\n', ' ')


def passed(document, marker):
    return get(document, 'status') == 'completed' and get(document, marker) is True


def state(document, marker=None):
    if not document:
        return 'pending：尚无可读产物'
    status = document.get('status', '未提供 status')
    if marker is None:
        return str(status) + '（仅进程/产物状态）'
    return f'{status}；{marker}={cell(document.get(marker))}'


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.locations = (self.root / 'results/output/round5', self.root / 'training_results/output/round5')
        self.sources, self.missing, self.issues = {}, [], []

    def read(self, relative, *, local=False):
        candidates = (self.root / relative,) if local else tuple(p / relative for p in self.locations)
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            self.missing.append(relative)
            return {}
        path = existing[0]
        try:
            raw = path.read_bytes()
            document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(document, dict):
                raise ValueError('expected a JSON object')
            sha = hashlib.sha256(raw).hexdigest()
            self.sources[relative] = {'path': path, 'sha256': sha}
            if relative == 'prefix_v2/summary.json' and any(digest(p) != sha for p in existing[1:]):
                self.issues.append('训练与最终 collection 的 summary SHA 不一致；不能混用两轮结果。')
            return document
        except (OSError, ValueError, TypeError) as error:
            self.issues.append(f'{path}: 无法解析，{type(error).__name__}: {error}')
            return {}

    def same(self, description, actual, expected):
        if actual is not None and expected is not None and actual != expected:
            self.issues.append(description + '：证据不一致')


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'])
    lines.extend('| ' + ' | '.join(map(cell, row)) + ' |' for row in rows)
    lines.append('')


def metric_rows(name, metrics, *, strata=False):
    rows = []
    for mode in ('whole', 'stream'):
        value = get(metrics, mode) or {}
        if strata:
            for key in STRATA:
                row = get(value, 'strata', key) or {}
                rows.append([name, mode, key, row.get('n'), row.get('safe'), row.get('unsafe'),
                             cell(row.get('recall'), percent=True), cell(row.get('fpr'), percent=True),
                             row.get('tp'), row.get('fp'), row.get('threshold')])
        else:
            rows.append([name, mode, cell(value.get('macro_recall'), percent=True),
                         cell(value.get('macro_fpr'), percent=True)])
    return rows


def timing_errors(row):
    """Recount saved 128-call timing; never count replay/padding as new input."""
    errors = []
    try:
        latencies, counts = row['latencies_seconds'], row['per_append_counts']
        if row['measured_calls'] != 128 or len(latencies) != 128 or len(counts) != 128:
            errors.append('requires 128 measured calls, latencies and accounting rows')
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in latencies):
            errors.append('invalid raw latency')
        fields = ('net_new_tokens', 'forward_tokens', 'real_forward_tokens', 'padding_tokens', 'replay_tokens', 'forward_calls')
        for key in fields:
            if any(type(c[key]) is not int for c in counts) or sum(c[key] for c in counts) != row[key]:
                errors.append('per-append total differs: ' + key)
        for c in counts:
            if (c['forward_tokens'] != 32 * c['forward_calls']
                    or c['forward_tokens'] != c['real_forward_tokens'] + c['padding_tokens']
                    or min(c['forward_calls'], c['real_forward_tokens'], c['padding_tokens'], c['replay_tokens']) < 0):
                errors.append('bad physical/real/padding accounting'); break
        if row['net_new_tokens'] != row['end_context_tokens'] - row['start_context_tokens']:
            errors.append('net input differs from final minus initial length')
        if not 0 < row['start_context_tokens'] <= row['end_context_tokens'] <= 8192:
            errors.append('context outside 8192')
        seconds = row['seconds']
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid wall time')
        expected = {'itps': row['net_new_tokens'] / seconds, 'classifications_per_second': 128 / seconds,
                    'p50_ms': sorted(latencies)[63] * 1000, 'p95_ms': sorted(latencies)[121] * 1000}
        for key, value in expected.items():
            if not math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-9):
                errors.append('raw timing cannot reproduce ' + key)
        if seconds + 1e-9 < sum(latencies):
            errors.append('outer wall is shorter than raw latencies')
        delta = row['engine_execution_delta']
        for key in ('forward_calls', 'forward_tokens', 'real_forward_tokens', 'padding_tokens'):
            if delta[key] != row[key]:
                errors.append('actual engine delta differs: ' + key)
        if row['inference_engine'] == 'window_cuda_graph' and delta['graph_calls'] <= 0:
            errors.append('graph runtime did not replay a graph')
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    return errors


def render(root):
    evidence = Evidence(root)
    read = evidence.read
    delivery, live = read('delivery_status.json', local=True), read('LIVE_STATUS.json', local=True)
    summary = read('prefix_v2/summary.json')
    sft = read('prefix_v2/sft/summary.json') or get(summary, 'sft') or {}
    rl = read('prefix_v2/classification_rl/summary.json') or get(summary, 'classification_rl') or {}
    initial_bulk = read('prefix_v2/sft/evaluation_0/metrics.json')
    workflow = read('validation/workflow_status.json')
    selection = read('validation/MODEL_MANIFEST.json')
    bundle = read('validation/bundle/BUNDLE_MANIFEST.json')
    core, portable = read('validation/canonical_core_audit.json'), read('validation/bundle_audit.json')
    calibration = read('canonical32_calibration/audit.json')
    cal = {name: read(f'canonical32_calibration/{name}/calibration.json') for name in ('initial_window', 'selected')}
    fresh = read('canonical32_fresh390/audit.json')
    fresh_results = {name: read(f'canonical32_fresh390/{name}/metrics.json') or get(fresh, 'results', name) or {}
                     for name in ('initial_window', 'selected')}
    official = read('canonical32_official/metrics.json')
    bulk_fresh = read('fresh390_final/audit.json')
    fixed_sha = summary.get('final_checkpoint_sha256')
    for label, actual in [('selection', selection.get('checkpoint_sha256')), ('bundle', bundle.get('checkpoint_sha256')),
                          ('core', core.get('checkpoint_sha256')), ('portable', portable.get('checkpoint_sha256')),
                          ('RL summary', rl.get('checkpoint_sha256')), ('canonical calibration', calibration.get('selected_checkpoint_sha256')),
                          ('fresh390', fresh.get('selected_checkpoint_sha256')), ('official', official.get('checkpoint_sha256'))]:
        evidence.same(label + ' checkpoint SHA 与最终训练选择', actual, fixed_sha)
    for name, expected in (('initial_window', INITIAL_SHA), ('selected', fixed_sha)):
        evidence.same(name + ' canonical calibration SHA', get(cal[name], 'checkpoint_sha256'), expected)
        evidence.same(name + ' fresh390 SHA', get(fresh_results[name], 'checkpoint_sha256'), expected)
        key = f'canonical32_calibration/{name}/calibration.json'
        receipt = evidence.sources.get(key, {})
        evidence.same(name + ' fresh 校准 artifact SHA', get(fresh_results[name], 'calibration_artifact_sha256'), receipt.get('sha256'))
        evidence.same(name + ' calibration audit artifact SHA', get(calibration, 'results', name, 'calibration_artifact_sha256'), receipt.get('sha256'))
        if fresh_results[name].get('status') == 'completed' and fresh_results[name].get('coverage') != {
                'requested': 390, 'evaluated': 390, 'excluded': {}, 'unique_families': 390}:
            evidence.issues.append(name + ' fresh completed 但不是完整 390/390 family 覆盖')
    evidence.same('canonical execution contract', get(core, 'execution_contract'), get(cal['selected'], 'execution_contract'))
    evidence.same('selection canonical quality gate', selection.get('canonical_quality_gate_pass'), get(cal['selected'], 'canonical_quality_gate_pass'))
    training_receipt = evidence.sources.get('prefix_v2/summary.json', {})
    for name, document in [('workflow', workflow), ('calibration', calibration), ('fresh', fresh), ('official', official)]:
        evidence.same(name + ' training summary SHA', document.get('training_summary_sha256'), training_receipt.get('sha256'))

    lines = ['# Round5 端到端结果', '', '生成时间：' + datetime.now(timezone.utc).isoformat(), '',
             '这是 CPU 文件报告；本脚本不加载 tokenizer/模型，不执行训练、评测或云操作。缺失字段标为 pending；'
             '`status=completed`、进程退出 0、`integrity_pass` 和权重下载均不等于安全质量通过。'
             'running/failed 产物中保留的数值只能视为局部观察，不能称为完成的基准结果。', '',
             f'当前交付状态：**{cell(delivery.get("status"))}**。训练总报告：{state(summary)}。'
             f'canonical32 质量门槛：**{cell(get(cal["selected"], "canonical_quality_gate_pass"))}**。']
    if not summary or summary.get('status') != 'completed':
        lines += ['', '**训练最终选择尚未完成/取回，后续结果 pending；不能宣称已训练完成或已可交付。**']
    if live:
        lines += ['', f'最近本地状态快照（不是完成证明）：{cell(live.get("observed_at"))}，Pod={cell(live.get("phase"))}；'
                  f'SFT step={cell(get(live, "sft", "step"))}/{cell(get(live, "sft", "steps"))}；'
                  f'RL update={cell(get(live, "classification_rl", "step"))}。']
    lines += ['', '## 训练、固定选择与 fallback', '',
              '起点是既有 Qwen3.5 window/W512 分类器，不是从零预训练。训练先 SFT，再分类动作 RL；'
              'whole 为完整输入末点，stream 为该记录预定 native target tokens 和至多八个真实文本切点的最大风险。', '',
              f'固定初始权重 SHA：`{INITIAL_SHA}`。最终训练选择 SHA：`{cell(fixed_sha)}`。', '',
              f'MODEL_MANIFEST：status={cell(selection.get("status"))}，candidate={cell(selection.get("candidate"))}；'
              f'training_promoted_from_initial={cell(selection.get("training_promoted_from_initial"))}，'
              f'promoted_from_initial={cell(selection.get("promoted_from_initial"))}。', '',
              f'是否实际保留初始权重：{cell(fixed_sha == INITIAL_SHA if fixed_sha else None)}。'
              f'选择说明：{cell(selection.get("promotion_reason"))}。初始 fallback 或 serving gate 失败不算训练提升。']
    table(lines, ['阶段', '状态', '完成步数', '选中 stage/step', 'eligible', '选中权重 SHA'], [
        ['SFT', sft.get('status'), sft.get('steps'), get(sft, 'selected'), get(sft, 'best_metrics', 'eligible'), sft.get('checkpoint_sha256')],
        ['classification RL', rl.get('status'), rl.get('updates'), get(rl, 'selected'), get(rl, 'best_metrics', 'eligible'), rl.get('checkpoint_sha256')]])
    lines += ['训练原 bulk dev（用于训练过程固定选择；不能替代下面的 canonical32 dev）：']
    table(lines, ['权重', '模式', 'macro recall', 'macro FPR'],
          sum((metric_rows(name, metrics) for name, metrics in [('initial', initial_bulk), ('SFT best', sft.get('best_metrics')), ('RL selected', rl.get('best_metrics'))]), []))
    lines += ['## canonical32 校准与独立 dev gate', '', state(calibration, 'integrity_pass'), '',
              '阈值只在 900 条 calibration 上拟合；1200 条 dev 检查：每个 stream stratum FPR≤5%，'
              'whole macro recall 比 canonical 初始基线最多下降 2 个百分点。whole/stream 各有独立阈值，严格 `>`；'
              'canonical gate 不重新选权重，失败仍须保留原记录。']
    for name in ('initial_window', 'selected'):
        value = cal[name]
        lines += ['', f'{name}：gate={cell(value.get("canonical_quality_gate_pass"))}；'
                  f'gates={cell(get(value, "development_metrics", "selection_gates"))}；'
                  f'cal/dev records={cell(get(value, "calibration_receipt", "records"))}/{cell(get(value, "development_receipt", "records"))}；'
                  f'calibrated_strata={cell(value.get("calibrated_strata"))}。']
    headers = ['权重', '模式', 'stratum', 'n', 'safe', 'unsafe', 'recall', 'FPR', 'TP', 'FP', '固定阈值']
    table(lines, headers, sum((metric_rows(name, get(cal[name], 'development_metrics'), strata=True)
                              for name in ('initial_window', 'selected')), []))

    lines += ['## Fresh 390：冻结的留出评估', '', state(fresh, 'integrity_pass'), '',
              '预定覆盖 390 records / 390 recorded families；三个语言/角色层，每层 safe/unsafe 各 65。'
              '以下为各权重自己的固定 calibration 阈值，不在 fresh 上重拟合、不用 fresh 选权重。']
    for name in ('initial_window', 'selected'):
        value = fresh_results[name]
        reuse = ('未复用（null）' if value.get('observations_reused_from') is None
                 else value['observations_reused_from']) if 'observations_reused_from' in value else None
        lines += ['', f'{name}：{state(value)}；coverage={cell(value.get("coverage"))}；'
                  f'observations_reused_from={cell(reuse)}。']
    table(lines, headers, sum((metric_rows(name, fresh_results[name], strata=True) for name in ('initial_window', 'selected')), []))
    lines += [f'selected − initial（描述性差值）：{cell(fresh.get("selected_minus_initial"))}。', '',
              f'旧 bulk fresh 对照单独保留：{state(bulk_fresh, "integrity_pass")}。'
              '不把旧 bulk 的 threshold/概率当成 canonical32 结果。']

    lines += ['', '## 官方 adapted benchmark：2441 来源行', '', state(official, 'integrity_pass'), '',
              '计划保留 thinking 1059、thinking_loc 569、response_loc 813 行；同一权重中只对完全相同的 native IDs '
              '与 eval_start_index 复用预测：1872 条 unique forward sequences + 569 cache hits = 2441 来源行。'
              '重复行仍计分，不删行伪装更高覆盖；thinking 与 thinking_loc 重叠，禁止合并为总分。', '',
              '决策为原 unsafe 优先的“连续两个 argmax 类别”规则；strict 只把 unsafe 算阳性，loose 也把 controversial 算阳性。'
              '它不使用 calibration 阈值。location splits 全是 Unsafe，FPR 未定义；原 Qwen3 token 定位下标不能直接用于 Qwen3.5，定位指标不报。', '',
              f'实际来源行={cell(official.get("evaluated_source_rows"))}；实际执行计数={cell(official.get("execution_totals"))}。']
    if passed(official, 'integrity_pass'):
        if (official.get('evaluated_source_rows') != 2441
                or get(official, 'execution_totals', 'unique_forward_sequences') != 1872
                or get(official, 'execution_totals', 'exact_prediction_cache_hits') != 569):
            evidence.issues.append('official completed/integrity_pass 与 2441/1872/569 覆盖计数不符')
    references = get(official, 'historical_references', 'a0_original_qwen3guard_stream') or {}
    a0 = references.get('metrics') or {}
    official_rows = []
    for split, count in SPLITS.items():
        coverage = get(official, 'splits', split, 'coverage') or {}
        if passed(official, 'integrity_pass') and coverage != {'requested': count, 'evaluated': count, 'excluded': {}}:
            evidence.issues.append('official ' + split + ' 覆盖缺失或排除样本')
        for rule in ('native_two_consecutive_strict', 'native_two_consecutive_loose', 'endpoint_argmax_unsafe'):
            current, baseline = get(official, 'splits', split, rule) or {}, get(a0, 'splits', split, rule) or {}
            undefined = lambda row: 'undefined（无 safe）' if split != 'thinking' and row.get('n') is not None and row.get('fpr') is None else cell(row.get('fpr'), percent=True)
            official_rows.append([split, rule, current.get('n'), cell(current.get('recall'), percent=True), undefined(current),
                                  baseline.get('n'), cell(baseline.get('recall'), percent=True), undefined(baseline)])
    table(lines, ['split', '规则', 'selected n', 'selected recall', 'selected FPR', 'A0 n', 'A0 recall', 'A0 FPR'], official_rows)
    lines += [f'A0 是历史原始 Qwen3Guard-Stream 结果，不冒充本轮重新推理：reference SHA={cell(references.get("sha256"))}，'
              f'本轮 A0 model calls={cell(official.get("a0_model_calls_this_run"))}。tokenizer/template/粒度不同，所以是 adapted 比较。']

    lines += ['', '## canonical core 与真实文本速度', '', state(core, 'pass'), '',
              f'全32位置双头/状态={cell(core.get("graph_same_shape_probability_and_state_pass"))}；'
              f'future pad 因果性={cell(core.get("future_pad_causality_pass"))}；'
              f'到达顺序不变={cell(core.get("session_arrival_invariance_pass"))}；'
              f'不提交 dummy cache={cell(core.get("no_dummy_cache_commit_pass"))}。', '',
              f'padding case 数={cell(len(core["future_padding_cases"]) if isinstance(core.get("future_padding_cases"), list) else None)}；'
              f'跨 aligned 边界 BPE={cell(core.get("crossed_aligned_boundary_bpe"))}；'
              f'608-token 实际 dispatch={cell(core.get("fixed_608_execution"))}。', '',
              'ITPS 分子是测量期净新增原生 tokens（末长度−初长度）；重放及未来 padding 不能冒充新增输入，'
              '但其计算时间保留在分母。128 次真实文本 append，先完整预热同一 schedule；'
              '耗时含重分词、缓存复制/回滚、分类输出到 CPU 和同步，排除启动捕获。']
    timing_rows = []
    for row in core.get('actual_text_timings', []):
        errors = timing_errors(row)
        if errors:
            evidence.issues.append('core timing ' + str(row.get('inference_engine')) + ': ' + '; '.join(errors))
        timing_rows.append([row.get('inference_engine'), row.get('itps'), row.get('p50_ms'), row.get('p95_ms'),
                            row.get('net_new_tokens'), row.get('forward_tokens'), row.get('real_forward_tokens'),
                            row.get('padding_tokens'), row.get('replay_tokens'), row.get('seconds'),
                            'CPU计账一致' if not errors else 'INVALID：计账不一致'])
    if not timing_rows:
        timing_rows = [[engine] + [None] * 10 for engine in ('eager', 'window_cuda_graph')]
    table(lines, ['engine', 'ITPS', 'P50 ms', 'P95 ms', '净新增', 'physical', 'real含重放', 'padding', 'replay', 'wall秒', '复核'], timing_rows)
    for name in ('eager', 'graph'):
        meta = get(core, 'initialization', name) or {}
        lines += [f'{name} 启动：seconds={cell(meta.get("startup_seconds"))}，capture={cell(meta.get("capture_seconds"))}，'
                  f'auxiliary tokens={cell(meta.get("auxiliary_input_tokens"))}，'
                  f'arena bytes={cell(meta.get("shared_graph_arena_storage_bytes"))}。']
    lines += ['', 'core 是固定 checkpoint 的计算一致性验收，不是安全质量评估；旧 bulk 仅诊断。'
              'graph core 通过也不会自动把 eager 便携包宣称为 graph 交付。原 Round4 fast audit 的失败不被本报告覆盖。']

    lines += ['', '## 便携包、工作流与本地交付', '',
              f'workflow：{state(workflow, "pipeline_integrity_pass")}；portable：{state(portable, "pass")}。', '',
              f'独立 child：{state(portable.get("portable_child"), "pass")}；'
              f'无需原基座权重={cell(portable.get("old_base_weights_unneeded_by_bundle"))}；'
              f'无需 H24={cell(portable.get("initial_h24_checkpoint_unneeded_by_bundle"))}；'
              f'canonical loader 最大概率差={cell(get(portable, "portable_child", "max_old_loader_probability_error"))}。', '',
              f'child 隔离={cell(get(portable, "portable_child", "isolated_python_process"))}；'
              f'HF offline={cell(get(portable, "portable_child", "hf_offline"))}；'
              f'参数结构精确匹配={cell(get(portable, "portable_child", "parameter_structure_exact_match"))}；'
              f'RoPE dtype/SHA 匹配={cell(get(portable, "portable_child", "rope_buffer_dtype_and_sha_exact_match"))}；'
              f'未知依赖拒绝记录={cell(get(portable, "portable_child", "dependency_audit", "unexpected_denials"))}；'
              f'en/user 阈值未伪造={cell(get(portable, "portable_child", "jsonl", "english_user_missing_threshold_not_fabricated"))}。', '',
              f'包 checkpoint SHA=`{cell(bundle.get("checkpoint_sha256"))}`；'
              f'权重 bytes={cell(get(bundle, "files", "classifier.safetensors", "bytes"))}；'
              f'包 inference_engine={cell(bundle.get("inference_engine"))}；'
              f'包 graph_validation_passed={cell(bundle.get("graph_validation_passed"))}。', '',
              f'本地交付记录路径：{cell(delivery.get("bundle"))}；download attempts={cell(delivery.get("download_attempts"))}；'
              f'artifact job completed={cell(delivery.get("artifact_job_completed"))}。']
    release = Path(delivery['bundle']) if isinstance(delivery.get('bundle'), str) else evidence.root / 'release'
    local_manifest = release / 'BUNDLE_MANIFEST.json'
    local_weight = release / 'classifier.safetensors'
    if local_manifest.is_file():
        local_sha = digest(local_manifest)
        collected_sha = get(evidence.sources, 'validation/bundle/BUNDLE_MANIFEST.json', 'sha256')
        evidence.same('本地/验收 BUNDLE_MANIFEST SHA', local_sha, collected_sha)
        lines += ['', f'本地清单实际存在：[{local_manifest}]({local_manifest})，SHA=`{local_sha}`。'
                  f'本地权重存在={cell(local_weight.is_file())}；实际 bytes={cell(local_weight.stat().st_size if local_weight.is_file() else None)}。'
                  '本报告不重新读取大权重做 SHA；完整权重 SHA 验证依据 downloader/portable 产物，不能仅凭目录存在证明可用。']
        if local_weight.is_file():
            evidence.same('本地权重大小', local_weight.stat().st_size, get(bundle, 'files', 'classifier.safetensors', 'bytes'))
    else:
        lines += ['', '本地 release 清单尚未出现：pending；不可声称模型已经下载。']
        if str(delivery.get('status', '')).startswith('bundle_delivered'):
            evidence.issues.append('delivery 声称 delivered，但本地 BUNDLE_MANIFEST 不存在')
    if workflow.get('stages'):
        table(lines, ['阶段', 'status', 'process completed', 'returncode', 'artifact quality gate'],
              [[s.get('name'), s.get('status'), s.get('completed'), s.get('returncode'),
                get(s, 'artifact_receipt', 'canonical_quality_gate_pass')] for s in workflow['stages']])

    lines += ['', '## 固定数据与结论限制', '',
              '- Fresh 390 是 390 个已记录 family；准备/审计保留 26 个短消息文本重合。长度≥20 字符的归一化精确文本排除规则，不等于所有长度零重复或语义去污染。',
              '- en/user 缺少独立 calibration，不借用 zh/user 或 en/assistant 阈值，不报告其已校准覆盖。',
              '- 第三风险类别 controversial、细分类别 category、自然风险最早出现位置及 safe-prefix 放行均未验收。',
              '- 预定 native/text-cut 轨迹上的 episode FPR，不构成任意 BPE 到达 schedule 的线上 FPR 保证。',
              '- 官方与 fresh 都是描述性评估，不用于改权重/选 checkpoint/调阈值；生产批准始终不由本报告推导。', '',
              '## 证据完整性与来源', '',
              f'发现的文件一致性/计账问题：{len(evidence.issues)}。以下检查只处理已有文件，不补写不存在的结果。']
    lines.extend('- **' + issue + '**' for issue in evidence.issues)
    if not evidence.issues:
        lines.append('已有字段未发现跨文件身份或速度计账冲突；缺失证据仍为 pending，不能据此通过总验收。')
    lines += ['', '缺失/尚未取回：' + ('、'.join('`' + p + '`' for p in evidence.missing) if evidence.missing else '无预定 JSON 缺失。'), '']
    table(lines, ['证据', '实际读取路径', 'SHA256'],
          [[name, str(row['path']), row['sha256']] for name, row in evidence.sources.items()])
    return '\n'.join(lines).rstrip() + '\n', evidence


def self_test():
    import ast
    import tempfile
    def save(root, path, value):
        p = root / path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    with tempfile.TemporaryDirectory(prefix='round5-report-cpu-') as folder:
        root = Path(folder)
        text, evidence = render(root)
        assert 'pending' in text and '不能宣称已训练完成' in text and evidence.missing
        save(root, 'delivery_status.json', {'status': 'waiting_for_training'})
        save(root, 'training_results/output/round5/prefix_v2/summary.json',
             {'status': 'completed', 'final_checkpoint_sha256': INITIAL_SHA})
        save(root, 'results/output/round5/canonical32_calibration/selected/calibration.json',
             {'status': 'completed', 'checkpoint_sha256': INITIAL_SHA, 'canonical_quality_gate_pass': False})
        text, evidence = render(root)
        assert '质量门槛：**false**' in text and '实际保留初始权重：true' in text and '本地 release 清单尚未出现' in text
        assert '/training_results/' in str(evidence.sources['prefix_v2/summary.json']['path'])
        split_metrics = {name: {'coverage': {'requested': n, 'evaluated': n, 'excluded': {}},
                               'native_two_consecutive_strict': {'n': n, 'recall': .75, 'fpr': .1 if name == 'thinking' else None}}
                         for name, n in SPLITS.items()}
        save(root, 'results/output/round5/canonical32_official/metrics.json',
             {'status': 'completed', 'integrity_pass': True, 'evaluated_source_rows': 2441,
              'execution_totals': {'unique_forward_sequences': 1872, 'exact_prediction_cache_hits': 569},
              'splits': split_metrics, 'historical_references': {'a0_original_qwen3guard_stream': {'metrics': {'splits': split_metrics}}}})
        text, evidence = render(root)
        assert '75.00%' in text and 'undefined（无 safe）' in text and not evidence.issues
        save(root, 'results/output/round5/prefix_v2/summary.json',
             {'status': 'completed', 'final_checkpoint_sha256': '1' * 64})
        _, evidence = render(root)
        assert any('summary SHA 不一致' in issue for issue in evidence.issues)
        save(root, 'results/output/round5/canonical32_calibration/audit.json', {'status': 'failed', 'integrity_pass': False})
        text, _ = render(root)
        assert 'failed；integrity_pass=false' in text
    counts = dict(net_new_tokens=1, forward_tokens=32, real_forward_tokens=2, padding_tokens=30, replay_tokens=1, forward_calls=1)
    row = {'inference_engine': 'window_cuda_graph', 'measured_calls': 128, 'latencies_seconds': [.01] * 128,
           'per_append_counts': [counts] * 128, **{k: v * 128 for k, v in counts.items()},
           'seconds': 1.3, 'itps': 128 / 1.3, 'classifications_per_second': 128 / 1.3,
           'p50_ms': 10., 'p95_ms': 10., 'start_context_tokens': 512, 'end_context_tokens': 640,
           'engine_execution_delta': {**{k: v * 128 for k, v in counts.items()}, 'graph_calls': 128}}
    assert not timing_errors(row)
    assert timing_errors({**row, 'itps': 4096 / 1.3})  # Padding inflation is rejected.
    assert timing_errors({**row, 'p95_ms': 11.})
    official_source = Path(__file__).resolve().with_name('evaluate_canonical_official_l20.py')
    source_tree = ast.parse(official_source.read_text())
    counts = next(ast.literal_eval(node.value) for node in source_tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COUNTS' for t in node.targets))
    assert SPLITS == counts and sum(SPLITS.values()) == 2441
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'tokenizers' not in sys.modules
    return {'cpu_checks': 'passed', 'checks': ['missing_pending', 'training_fallback', 'false_gate_not_promoted',
        'final_precedence_and_conflict', 'failed_artifact_retained', 'raw_timing_recount', 'padding_itps_inflation_rejected', 'bad_p95_rejected',
        'official_split_counts_match_frozen_source', 'completed_official_fixture_and_undefined_fpr'],
        'model_calls': 0, 'network_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true', help='Temporary-file CPU checks; no result files changed')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False)); return 0
    report, evidence = render(args.root)
    destination = args.output or args.root / 'END_TO_END_RESULTS.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(report, encoding='utf-8'); temporary.replace(destination)
    print(json.dumps({'report': str(destination.resolve()), 'sources': len(evidence.sources),
                      'missing_pending': len(evidence.missing), 'evidence_issues': evidence.issues,
                      'model_calls': 0, 'network_calls': 0}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
