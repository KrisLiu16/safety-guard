"""Paired held-out comparisons, family bootstrap, and a concise reproducible report."""
from collections import defaultdict
import json
from pathlib import Path
import random
from runtime import ROOT
from evaluate import read,write,metrics,summarize_chinese,official_metrics
from stream_evaluate import summarize_stream

def paired_ci(before,after,bt,at):
    b={r['sample_id']:r for r in before};a={r['sample_id']:r for r in after};assert b.keys()==a.keys()
    groups=defaultdict(list)
    for sid,r in b.items():groups[r['family']].append(sid)
    rng=random.Random(20260922);names=list(groups);deltas=[]
    for _ in range(1000):
        ids=[sid for name in rng.choices(names,k=len(names)) for sid in groups[name]]
        mb=metrics([b[sid] for sid in ids],bt);ma=metrics([a[sid] for sid in ids],at)
        if mb['tp']+mb['fn']:deltas.append(ma['recall']-mb['recall'])
    deltas.sort()
    return {'family_count':len(names),'resamples':len(deltas),'recall_delta_95_percentile':
        [deltas[int(.025*len(deltas))],deltas[min(len(deltas)-1,int(.975*len(deltas)))]]}

if __name__=='__main__':
    # Summarize the immutable saved predictions with the current transparent action accounting.
    for name in ['baseline','post','merged','precision_control']:
        directory=ROOT/name
        write(directory/'chinese_metrics.json',summarize_chinese(read(directory/'chinese_predictions.jsonl')))
        rows=read(directory/'official_all_predictions.jsonl')
        write(directory/'official_all_metrics.json',{s:official_metrics([r for r in rows if r['split']==s]) for s in ['thinking','thinking_loc','response_loc']})
        if (directory/'stream_predictions.jsonl').exists():write(directory/'stream_metrics.json',summarize_stream(read(directory/'stream_predictions.jsonl')))
    baseline=json.loads((ROOT/'baseline/chinese_metrics.json').read_text())
    post=json.loads((ROOT/'post/chinese_metrics.json').read_text())
    bp=read(ROOT/'baseline/chinese_predictions.jsonl');ap=read(ROOT/'post/chinese_predictions.jsonl')
    comparison={'threshold_rule':'Each checkpoint calibrated independently to empirical calibration FPR <=5%; actual test FPR reported separately.',
                'label_limit':'Chinese test labels are Luna synthetic; semantic spot review covers training examples only.', 'roles':{}}
    for role in ['user','assistant']:
        br=baseline['roles'][role];ar=post['roles'][role]
        comparison['roles'][role]={'before':br,'after':ar,'paired_test_ci':paired_ci(
            [r for r in bp if r['split']=='test' and r['target_role']==role],
            [r for r in ap if r['split']=='test' and r['target_role']==role],br['threshold'],ar['threshold'])}
    comparison['official_before']=json.loads((ROOT/'baseline/official_all_metrics.json').read_text())
    comparison['official_after']=json.loads((ROOT/'post/official_all_metrics.json').read_text())
    comparison['merged_chinese']=json.loads((ROOT/'merged/chinese_metrics.json').read_text())
    comparison['official_merged']=json.loads((ROOT/'merged/official_all_metrics.json').read_text())
    comparison['stream_before']=json.loads((ROOT/'baseline/stream_metrics.json').read_text())
    comparison['stream_after']=json.loads((ROOT/'post/stream_metrics.json').read_text())
    comparison['stream_merged']=json.loads((ROOT/'merged/stream_metrics.json').read_text())
    comparison['precision_control_chinese']=json.loads((ROOT/'precision_control/chinese_metrics.json').read_text())
    comparison['precision_control_official']=json.loads((ROOT/'precision_control/official_all_metrics.json').read_text())
    write(ROOT/'evaluation_before_after.json',comparison)
    perf={name:json.loads((ROOT/'performance'/f'{name}_performance.json').read_text()) for name in ['original','rl','rl_merged']}
    write(ROOT/'performance_before_after.json',perf)
    train=json.loads((ROOT/'training/summary.json').read_text());gen=json.loads((ROOT/'generated/generation_audit.json').read_text())
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    lines=['# M5 第一轮安审模型实验结果','',
        '这是一次从 Luna 合成数据到本地有限动作 RL 的完整实验。中文结果衡量对合成标签的一致性，不能替代独立人工金标。', '',
        '**主要结论：增量缓存带来约 10.8 倍的逐 token P95 改善；训练的能力收益有限，并有少量官方/流式指标退化。默认导出未合并训练路径，合并版可选。**', '',
        f"数据：目标 5,000 条，实际生成 {gen['raw_examples']:,} 条；49 个词任务成功、1 个中途失败，未自动补跑。结构合格 4,246 条；清洗与抽检后完整记录 {manifest['clean_after_spot_review']:,} 条，另有 74 条经 Codex 语义检查的训练前缀。", '',
        f"训练：Apple M5 / 32 GB，PyTorch MPS；BF16 主干，FP32 LoRA/风险头，共 {train['trainable_parameters']:,} 个可训练参数。实际 {train['steps']} 次更新、{train['examples']:,} 个训练实例、{train['tokens']:,} token，活跃训练 {train['active_seconds']/60:.1f} 分钟，峰值 MPS driver allocation {train['peak_mps_driver_bytes']/1024**3:.2f} GiB。开发集选中的导出 checkpoint 为 `{Path(train['checkpoint']).name}`，未使用测试集选模型。", '',
        '完整记录划分为 train 2,467 / dev 414 / calibration 410 / test 804；74 条前缀只加入训练。内存值是每次更新结束采样的 driver allocation 最大值，不是算子内部瞬时峰值或整机总内存；训练日志另含 RSS。', '',
        '## 中文冻结测试集','',
        '阈值分别在独立 calibration 划分上满足经验误拦率 ≤5%；下表同时报告真实测试误拦率。每个词族只出现在一个划分。', '',
        '| 目标 | 数量 | 原版 → RL F1（argmax） | 原版 → RL 召回（校准） | 原版 → RL 测试误拦 | 召回变化 95% 词族 bootstrap 区间 |',
        '|---|---:|---:|---:|---:|---|']
    for role,item in comparison['roles'].items():
        b=item['before']['splits']['test'];a=item['after']['splits']['test'];ci=item['paired_test_ci']['recall_delta_95_percentile']
        lines.append(f"| {role} | {b['argmax']['n']} | {b['argmax']['f1']:.3f} → {a['argmax']['f1']:.3f} | {b['calibrated']['recall']:.3f} → {a['calibrated']['recall']:.3f} | {b['calibrated']['false_positive_rate']:.3f} → {a['calibrated']['false_positive_rate']:.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines+=['','上面的误拦只统计 block，不能把 hold 当作无影响的放行。正常样本的暂挂及总干预另列如下：','',
        '| 目标 | 原版 → RL 正常样本 hold 数量 | 原版 → RL 正常样本总干预率（block + hold） |','|---|---|---|']
    for role,item in comparison['roles'].items():
        b=item['before']['splits']['test']['calibrated'];a=item['after']['splits']['test']['calibrated']
        lines.append(f"| {role} | {b['safe_review_count']} → {a['safe_review_count']} | {b['safe_total_intervention_rate']:.3f} → {a['safe_total_intervention_rate']:.3f} |")
    lines+=['','用户侧校准后的漏检由 2 条降至 1 条，但召回差的词族 bootstrap 区间包含 0，不能据此宣称稳定泛化提升。正常用户样本暂挂从 24/194 降到 12/194；这同样是合成标签下的观察。', '']
    lines+=['','为区分训练与数值精度的影响，额外跑了未训练、仅将原版风险头改为 FP32 的完整对照：','',
        '| 目标 | 原版 BF16 / 未训练 FP32 风险头 / RL 的 F1 | 对应校准召回 |', '|---|---|---|']
    for role in ['user','assistant']:
        variants=[baseline,comparison['precision_control_chinese'],post]
        tests=[d['roles'][role]['splits']['test'] for d in variants]
        lines.append('| '+role+' | '+' / '.join(f"{t['argmax']['f1']:.4f}" for t in tests)+' | '+' / '.join(f"{t['calibrated']['recall']:.4f}" for t in tests)+' |')
    lines+=['','## 官方 Qwen3GuardTest','',
        'strict 沿用官方脚本打印定义：仅 Unsafe 为阳性；loose 也计入 Controversial。定位两个分片全是风险样本，不能用于估计正常误拦；thinking_loc 与 thinking 存在包含关系，不合并计算总分。', '',
        '| 分片 | 评测覆盖 | strict F1 原版 → RL | loose F1 原版 → RL | 定位 exact hit 原版 → RL |', '|---|---|---|---|---|']
    for split,b in comparison['official_before'].items():
        a=comparison['official_after'][split];assert b['coverage']==a['coverage']
        lines.append(f"| {split} | {b['coverage']['evaluated']}/{b['coverage']['requested']} | {b['strict']['f1']:.3f} → {a['strict']['f1']:.3f} | {b['loose']['f1']:.3f} → {a['loose']['f1']:.3f} | {b['latency']['exact_hit_rate']:.3f} → {a['latency']['exact_hit_rate']:.3f} |")
    lines+=['','## 流式审查与合并后回归','',
        '流式控制器检查块内每一个新 token，两个连续 token 的风险概率达到阈值时拦截。阈值用 calibration 集的整个流式轨迹单独校准；不复用整段末尾分类阈值。以下均为冻结 test 的实测结果。', '',
        '大规模逐 token 评测采用完整因果前向，分块定位从这些读出计算；实际增量 KV 的全 token 一致性另外用 8 条开发例 × 4 种 chunk 检查。没有将全部测试例按真实网络到达时间回放。', '',
        '| 目标 | 原版 → RL → 合并版流式召回 | 原版 → RL → 合并版流式误拦 |', '|---|---|---|']
    for role in ['user','assistant']:
        values=[comparison[k]['roles'][role]['splits']['test'] for k in ['stream_before','stream_after','stream_merged']]
        lines.append('| '+role+' | '+' → '.join(f"{v['recall']:.3f}" for v in values)+' | '+' → '.join(f"{v['false_positive_rate']:.3f}" for v in values)+' |')
    lines+=['','**流式用户测试误拦约 9.3%，没有达到测试集 ≤5%。** 5% 只是 calibration 集的阈值约束。回答侧流式误拦由 4/242 增为 5/242，召回未改善；这不是“能力全面提升”的结果。', '',
        '合并模型使用相同全量官方分片和中文划分再次评测；逐项结果见 `evaluation_before_after.json` 中 merged 字段。合并后的正常用户暂挂为 14/194，高于未合并的 12/194，因此默认启动器保留未合并路径。chunk 1/8/16/32 保留块内所有风险读出，只延迟到块尾通知；其定位影响见各评测目录的 `official_chunk_metrics.json`。', '',
        '## 延迟与吞吐','',
        '所有前向计时都同步 MPS；每组先热身。下表为总长 1,024 token、历史 992 token、续流 32 token。块延迟不含等待文本到达；实际 buffer 等待另见 JSON 中以 100 token/s 建模的上界。', '',
        '| 权重 / 实现 | chunk | 每块 P95 ms | 续流 token/s | prefill P50 ms |', '|---|---:|---:|---:|---:|']
    for name,d in perf.items():
        for r in d['measurements']:
            if r.get('total_tokens')==1024 and (r['mode']=='published' and name=='original' or r['mode']=='incremental' and r['chunk'] in [1,16]):
                lines.append(f"| {name}/{r['mode']} | {r['chunk']} | {r['chunk_latency']['p95_ms']:.2f} | {r['continuation_tokens_per_second']:.1f} | {r['prefill_latency']['p50_ms']:.2f} |")
    lines+=['','完整测量覆盖 256/1,024/4,096/8,192 token，chunk 1/8/16/32，以及 batch 1/2/4。此性能网格是每块末 token 读出的前向微基准；真正流式演示计算块内全部新 token，实测时延另见 `demo_verification.json`。逐 token 模式两者读出范围相同。batch 指同时已就绪的请求；不据此宣称真实到达流量下的服务 SLO。', '',
        '| 合并版 batch / 就绪请求数 | 每请求长度 | 请求 P95 ms | requests/s |', '|---:|---:|---:|---:|']
    for r in perf['rl_merged']['measurements']:
        if r['mode']=='equal_length_batch' and r['tokens_per_request']==1024:
            lines.append(f"| {r['batch']} | 1,024 | {r['request_latency']['p95_ms']:.2f} | {r['requests_per_second']:.2f} |")
    production=json.loads((ROOT/'production_performance.json').read_text())
    startup=json.loads((ROOT/'startup_results.json').read_text())
    lines+=['',f"新进程启动到首条 JSON 结果的中位数：原版 {startup['original']['median_seconds']:.2f}s，未合并 RL {startup['rl']['median_seconds']:.2f}s，合并版 {startup['rl_merged']['median_seconds']:.2f}s。每种启动 3 次，包含 Python 导入、载入与首轮推理；没有清空 OS/Metal 缓存，也不是重启电脑后的冷机测量。", '']
    lines+=['','下表为合并版真实的全部新 token 读出，1K 总上下文、chunk=16，各 session 独立 KV，单设备轮询；客户可见延迟包含同一时刻已就绪请求之间的等待。', '',
        '| 活跃流会话数 | 服务 P95 ms | 含排队 P95 ms | 总续流 token/s |', '|---:|---:|---:|---:|']
    for r in production['results']:
        if r['chunk']==16:lines.append(f"| {r['active_sessions']} | {r['service_latency']['p95_ms']:.2f} | {r['ready_client_latency_including_queue']['p95_ms']:.2f} | {r['aggregate_new_tokens_per_second']:.1f} |")
    lines+=['','实测 batch=2/4 在 1K 输入上未增加吞吐，反而提高了每批等待；并发会话增加也主要增加排队。不会将这两项写成已实现的加速收益。分块降低调度成本，但 100 token/s 的输入下，chunk=16 还要计入最多 150 ms 的凑块等待。', '',
        '## 交付和限制','',
        '- 导出模型及 adapter：`export/`，含基座、adapter、风险头及可选合并权重；FP32 风险头由本地 runtime 明确恢复。',
        '- 可复现配置与依赖：`config.json`、`requirements.lock.txt`、模型/数据哈希清单。',
        '- 推荐入口：`export/run_guard.py --mode batch` 或 `--mode stream`，用 JSONL 标准输入；加 `--merged` 测试合并变体。',
        '- 分类统计、逐例预测、官方风险位置分布和完整性能数据均保留在本目录。',
        '- 单个训练 seed、合成中文标签及有限词族，不能证明通用安全能力或统计稳定提升；下一轮优先增加独立审核数据，针对实际退化类别改奖励/分布，再决定是否投入结构优化。',
        '- 本轮没有引入 DSA/MoE/MLA；主要加速来自同一模型的增量缓存与分块，adapter 合并只有小幅性能收益。', '']
    (ROOT/'ROUND1_REPORT.md').write_text('\n'.join(lines))
    base_lines=['# 原版 Qwen3Guard-Stream-0.6B 基线','',
        '固定 revision：419364a715de9840d47b1457982f64ff37f90ed4。Apple M5 / 32 GB，PyTorch MPS，BF16。中文数据为按词族隔离的 Luna 合成测试，不是人工金标。', '',
        '| 目标 | test 数量 | argmax F1 | 校准召回 | 测试 block 误拦 | 正常 hold 数量 |', '|---|---:|---:|---:|---:|---:|']
    for role,r in baseline['roles'].items():
        t=r['splits']['test'];m=t['calibrated']
        base_lines.append(f"| {role} | {m['n']} | {t['argmax']['f1']:.4f} | {m['recall']:.4f} | {m['false_positive_rate']:.4f} | {m['safe_review_count']} |")
    base_lines+=['','阈值仅在 calibration 上选择，使经验 block FPR ≤5%。hold 单独报告。', '',
        '| 官方分片 | 覆盖 | strict F1 | loose F1 | 定位 exact hit |', '|---|---|---:|---:|---:|']
    for split,r in comparison['official_before'].items():
        base_lines.append(f"| {split} | {r['coverage']['evaluated']}/{r['coverage']['requested']} | {r['strict']['f1']:.4f} | {r['loose']['f1']:.4f} | {r['latency']['exact_hit_rate']:.4f} |")
    base_lines+=['','strict/loose 沿用官方打印定义，两个定位分片均为风险例，thinking_loc 与 thinking 重叠，不相加为独立样本。', '',
        '| 总上下文 | 原包装器逐 token P95 ms | 正确 KV 逐 token P95 ms |', '|---:|---:|---:|']
    for n in [256,1024,4096,8192]:
        r={x['mode']:x for x in perf['original']['measurements'] if x.get('total_tokens')==n and x.get('chunk')==1}
        base_lines.append(f"| {n} | {r['published']['chunk_latency']['p95_ms']:.2f} | {r['incremental']['chunk_latency']['p95_ms']:.2f} |")
    base_lines+=['',f"原版新进程首条 JSON 中位数 {startup['original']['median_seconds']:.2f}s；OS/Metal 缓存未清空。热态逐 token 测试每格 96 个续流步，设备同步计时，不含输入到达等待。", '',
        '原始预测、校准阈值、置信区间及覆盖率在 `baseline/`；完整时延、吞吐和 driver allocation 在 `performance/original_performance.json`。独立缓存/梯度验证见 `smoke_results.json`，上游指标一致性验证见 `official_metrics_parity.json`。']
    (ROOT/'baseline_report.md').write_text('\n'.join(base_lines)+'\n')
    print('Wrote comparison JSON and ROUND1_REPORT.md')
