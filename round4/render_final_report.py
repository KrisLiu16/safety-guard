"""Summarize actual training, streaming and portable-model artifacts (CPU only)."""
import json
from pathlib import Path
from render_results import table, fmt, pct, number

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / 'results/output/round4'
VALIDATION = ROOT / 'validation_results/output/round4_validation'
FAST = ROOT / 'fast_runtime/results/output/fast_runtime'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def paired_timing_sections(documents, selection):
    sections = ['## 与原版 A0 的同场效率对照',
                '单会话、batch=1，两种实现都将用户风险概率返回 CPU。每组用独立缓存完整预热相同轨迹，再测 128 次追加。分词、prefill、HTTP 和文本回滚不计入此处的增量 ITPS；原生 token 边界可能不同，不与前轮 24 次调用的数字计算提速。']
    indexed = {}
    rows = []
    for name in ('candidate', 'a0'):
        document = documents[name]
        indexed[name] = {}
        for record in document.get('records', []):
            key = (record.get('context_tokens'), record.get('chunk_tokens'))
            indexed[name][key] = record
            stored = record.get('state_bytes_at_prefill')
            rows.append([name, *key, record.get('end_context_tokens'), record.get('measured_calls'),
                         fmt(record.get('itps'), 1), fmt(record.get('p95_ms'), 2),
                         fmt(stored / 1048576, 2) if number(stored) else '待结果'])
    if rows:
        sections.append(table(['模型', '初始 context', 'chunk', '最终 context', '计时次数',
                               'ITPS', 'P95 ms', 'prefill 状态 MiB'], rows))
    else:
        sections.append('同场效率结果尚未收集。')
    candidate, reference = documents['candidate'], documents['a0']
    expected_keys = {(context, chunk) for context in (512, 4096) for chunk in (1, 8, 32)}
    uuid = candidate.get('device', {}).get('uuid')
    paired = (candidate.get('status') == reference.get('status') == 'completed'
              and candidate.get('protocol_version') == reference.get('protocol_version')
              == 'matched-classifier-token-id-v2'
              and candidate.get('protocol') == reference.get('protocol')
              and bool(candidate.get('trace', {}).get('common_text_sha256'))
              and candidate.get('trace', {}).get('common_text_sha256')
              == reference.get('trace', {}).get('common_text_sha256')
              and uuid not in (None, '', 'unavailable')
              and uuid == reference.get('device', {}).get('uuid')
              and bool(candidate.get('candidate_manifest_sha256'))
              and candidate.get('candidate_manifest_sha256') == reference.get('candidate_manifest_sha256')
              and bool(selection.get('checkpoint_sha256'))
              and candidate.get('model', {}).get('checkpoint_sha256') == selection.get('checkpoint_sha256')
              and all(set(indexed[name]) == expected_keys
                      and len(documents[name].get('records', [])) == len(expected_keys)
                      and all(record.get('measured_calls') == 128
                              and len(record.get('latency_seconds', [])) == 128
                              for record in indexed[name].values())
                      for name in ('candidate', 'a0')))
    if paired:
        ratios = []
        for key in sorted(expected_keys):
            a, b = indexed['candidate'][key], indexed['a0'][key]
            values = (a.get('itps'), b.get('itps'), a.get('p95_ms'), b.get('p95_ms'))
            if all(number(value) and value > 0 for value in values):
                ratios.append([*key, fmt(values[0] / values[1], 3), fmt(values[2] / values[3], 3)])
        sections.append(table(['初始 context', 'chunk', '候选/A0 ITPS 比', '候选/A0 P95 比'], ratios))
        sections.append('ITPS 比大于 1 表示候选吞吐更高；P95 比小于 1 表示候选延迟更低。两边模型、内核和分类头实现不同，比较的是各自当前可用的分类路径。')
    elif candidate or reference:
        sections.append('成对协议、设备、候选身份或完整覆盖尚未对齐，暂不计算提速比例。')
    sections.append('修正版 v2 从 512 或 4096 tokens 起步，最长到 8192，满足 A0 的长度上限。首版 A0 超长失败原样保留，不与修正版混配。状态存储不含模型权重或临时工作区；速度不证明风险质量。')
    return sections


def main():
    selection = read(ROOT / 'MODEL_MANIFEST.json')
    trained = read(TRAIN / 'final_summary.json')
    stages = read(VALIDATION / 'stage_status.json')
    stream = read(VALIDATION / 'text_stream_audit.json')
    bundle = read(VALIDATION / 'bundle_audit.json')
    graph = read(VALIDATION / 'graph_stream_audit.json')
    timings = {name: read(FAST / f'baseline_recheck/speed_{name}_v2.json') for name in ('candidate', 'a0')}
    keyword = {'candidate': read(FAST / 'baseline_recheck/keyword_primary_metrics.json'),
               'a0': read(VALIDATION / 'keyword_a0_metrics.json')}
    candidate = selection.get('candidate')
    chosen_stream = stream.get('results', {}).get(candidate, {})
    official = {name: read(VALIDATION / f'official_{name}/metrics.json') for name in ('student', 'a0')}
    sections = ['# 本轮端到端训练与推理结果',
                '本报告只汇总实测文件。训练曲线、保留测试质量、分类 RL 对照及 token-ID 速度见 [Round4 详细结果](ROUND4_RESULTS.md)。',
                table(['项目', '实际状态'], [
                    ['主训练流程', trained.get('status', '待结果')],
                    ['最终验证任务', stages.get('status', '待结果')],
                    ['固定开发集候选', candidate or '待选择'],
                    ['实际文本流检查', str(chosen_stream.get('pass', '待结果'))],
                    ['便携模型检查', str(bundle.get('pass', '待结果'))],
                    ['权重 SHA-256', selection.get('checkpoint_sha256', '待结果')],
                ])]
    sections += ['## 风险能力',
                 '当前仍存在流式质量阻碍：thinking 分片中，主候选按两连续 Unsafe 规则的 FPR 为 90.41%，A0 为 22.24%。端点分数的改善不能代表前缀安全可用；下一轮将单独训练并校准前缀判断。',
                 '下表使用用户指定 Qwen3GuardTest 的完整三个分片。学生使用训练时的角色格式和自己的 tokenizer；A0 使用官方模板和 tokenizer。两连续风险判决按各自 token 边界执行，所以这是适配后的能力观测，不声称复现原 Qwen3 token 定位或 128-token 延迟统计。']
    rows = []
    for name, document in official.items():
        for split in ('thinking', 'thinking_loc', 'response_loc'):
            measured = document.get('splits', {}).get(split, {})
            coverage = measured.get('coverage', {})
            strict = measured.get('native_two_consecutive_strict', {})
            loose = measured.get('native_two_consecutive_loose', {})
            endpoint = measured.get('endpoint_argmax_unsafe', {})
            rows.append([name, split, f"{coverage.get('evaluated', '?')}/{coverage.get('requested', '?')}",
                         fmt(strict.get('f1')), fmt(loose.get('f1')), pct(strict.get('recall')),
                         '无安全样本，无法估计' if strict.get('fpr') is None and measured else pct(strict.get('fpr')),
                         fmt(endpoint.get('f1'))])
    sections.append(table(['模型', '分片', '覆盖', 'strict F1', 'loose F1', 'strict 召回', 'strict FPR', '端点 F1'], rows))
    sections.append('thinking_loc 与 thinking 重叠，三片不合并为独立总分；两个 loc 分片均无安全样本。测试结果不参与候选选择。')
    sections += ['## 同词不同语境诊断',
                 '256 个四划分之外的词族构成 1024 条固定模板诊断，覆盖 29 个来源组。中文正常文本处理与明确诈骗请求保留同一词语；英文另观察隐私侵害请求及助手安全拒绝。它们是受控模板诊断，不是自然对话金标，不参与训练、选模或阈值调整。']
    rows = []
    for name, document in keyword.items():
        for template, values in document.get('by', {}).get('template', {}).items():
            rows.append([name, template, values.get('n'), values.get('calibrated_scored'),
                         pct(values.get('calibrated_fpr')) if values.get('calibrated_safe') else '不适用',
                         pct(values.get('calibrated_recall')) if values.get('calibrated_unsafe') else '不适用',
                         pct(values.get('uncalibrated_argmax_unsafe_recall')) if values.get('unsafe') else '不适用'])
    if rows:
        sections.append(table(['模型', '模板', '数量', '有现成阈值的数量', '校准 FPR', '校准召回', '未校准 argmax 召回'], rows))
    else:
        sections.append('诊断模型结果尚未收集。')
    sections.append('英文用户组缺少独立校准阈值，其校准结果留空，不借用助手阈值；仅报告明确标注为未校准的 argmax 诊断。只有安全或只有风险样本的单模板不计算不存在的另一类分母。')
    sections += paired_timing_sections(timings, selection)
    sections += ['## 实际文本流',
                 '会话接收文字片段，重分词后比较最长相同 token 前缀；BPE 尾部变化时恢复完整 GDN/注意力/历史记忆快照并重放。最近两个快照与当前状态彼此独立，异常追加在提交之前失败。每个成功追加返回分类概率，生成 token 数为零。']
    rows = []
    for name in ('full', 'window', 'memory', 'classification_rl'):
        record = stream.get('results', {}).get(name, {})
        speed = record.get('text_stream_performance', {})
        state_bytes = record.get('session_state_bytes_after')
        rows.append([name, str(record.get('pass', '待结果')), fmt(record.get('max_probability_error'), 6),
                     fmt(speed.get('itps_net_native_input'), 1), fmt(speed.get('p95_ms'), 2),
                     str(speed.get('net_new_native_tokens', '待结果')),
                     str(speed.get('actual_forward_tokens', '待结果')),
                     fmt(state_bytes / 1048576, 2) if isinstance(state_bytes, int) else '待结果'])
    sections.append(table(['模型', '文本流检查', '最大概率差', '文本 ITPS', '追加 P95 ms', '新增净 token', '实际前向 token', '当前+快照 MiB'], rows))
    sections.append('文本 ITPS 的分母包含分词、快照复制、回放和 CPU 可见分类结果；分子只计净新增输入，回放 token 不冒充新输入。每次追加净 token 数可因 BPE 合并而减少；用完整测量区间的净增长比较。此项不含 HTTP 或等待文字到达，不与纯 token-ID 内核数字混用。')
    sections += ['## 固定窗口 CUDA Graph 实验',
                 '这是 window/W512、窗口已满、batch=1、每次 8 个 token 的独立推理实验；不改变模型权重，当前不作为便携包的默认文本入口。比较包含外部会话状态复制与 CPU 可见双角色分类结果，不含分词、回滚源选择和 HTTP。']
    if graph.get('pass') is True:
        performance = graph.get('performance', {})
        sections.append(table(['路径', 'ITPS', 'P95 ms'], [
            [name, fmt(performance.get(name, {}).get('itps'), 1), fmt(performance.get(name, {}).get('p95_ms'), 2)]
            for name in ('eager', 'graph')]))
        sections.append('Graph / eager 的 ITPS 比率：' + fmt(performance.get('itps_ratio_graph_over_eager'), 3)
                        + '；P95 比率：' + fmt(performance.get('p95_ratio_graph_over_eager'), 3)
                        + '。正确性通过不等于更快，收益按这些实际数值判断。')
    elif graph:
        sections.append('该有限原型未通过 L20 验收。失败记录：' + str(graph.get('error', graph.get('status')))
                        + '。不据此宣称推理加速，也不影响已保存的训练权重。')
    else:
        sections.append('尚无 L20 实验结果，不宣称捕获可用或已有加速。')
    sections += ['## 模型交付',
                 '便携包包含完整分类器权重、配置、分词器、代码和运行依赖锁。加载器直接构造文本模型并加载完整 state_dict，不要求原基座权重或旧训练目录。整段和文本流入口见 [使用说明](BUNDLE_USAGE.md)。',
                 f"便携检查产物：`{VALIDATION / 'bundle_audit.json'}`；文件缺失或 pass 不为 true 时，不能声称独立加载已验收。",
                 '本轮继承 Qwen3.5 的预训练语义主干，进行了全参数风险后训练和分类 RL；不是从零重新预训练语言模型。保留全部 24 层，窗口与历史记忆版本改变了读历史信息的方法。',
                 '硬标签训练与本轮质量验收针对 safe/unsafe。第三档 controversial、风险类别头、英文用户独立公开测试以及按风险证据位置训练的放行/暂存/拦截策略仍未获得本轮完整验收；三档概率接口不应被误解为三档都有独立训练证据。']
    destination = ROOT / 'END_TO_END_RESULTS.md'
    destination.write_text('\n\n'.join(sections) + '\n')
    print(json.dumps({'report': str(destination), 'model_calls': 0}))


if __name__ == '__main__':
    main()
