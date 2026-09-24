"""Compare local metric logic to the archived upstream evaluator on real predictions."""
import contextlib
import importlib.util
import io
import json
import random
import re
from pathlib import Path
from evaluate import ROOT,read,decision,official_metrics
spec=importlib.util.spec_from_file_location('upstream',ROOT/'source/eval_stream.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
official=module.SafetyEvaluator();rng=random.Random(20260922)
for _ in range(1000):
    ids=[rng.randrange(3) for _ in range(rng.randrange(1,100))]
    category,pos=decision(ids);reference=official.consecutive_unsafe([official.label_map[i] for i in ids])
    assert reference==(official.label_map[category],pos)
rows=read(ROOT/'baseline/official_all_predictions.jsonl');results={}
for split in ['thinking','thinking_loc','response_loc']:
    subset=[r for r in rows if r['split']==split and r['status']=='ok']
    path=ROOT/'baseline'/f'{split}_upstream_metric_input.jsonl'
    path.write_text(''.join(json.dumps(r|{'split_idx_eval':r['split_idx']})+'\n' for r in subset))
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        official.evaluate_f1(str(path),'response');official.evaluate_unsafe_latency(str(path))
    output=buf.getvalue();ours=official_metrics(subset)
    strict=float(re.search(r'F1 Score\(strict\): ([0-9]+\.[0-9]+)',output)[1])
    loose=float(re.search(r'F1 Score\(loose\): ([0-9]+\.[0-9]+)',output)[1])
    hit=float(re.search(r'Exact hit rate:\s*([0-9.]+)',output)[1])
    first=float(re.search(r'First 128 tokens stop rate:\s*([0-9.]+)',output)[1])
    assert abs(strict-ours['strict']['f1'])<=.000051
    assert abs(loose-ours['loose']['f1'])<=.000051
    assert abs(hit-ours['latency']['exact_hit_rate'])<1e-12
    assert abs(first-ours['latency']['published_first_128_stop_rate'])<1e-12
    results[split]={'status':'passed','upstream_output':output}
(ROOT/'official_metrics_parity.json').write_text(json.dumps(results,indent=2)+'\n')
print('Upstream decision/F1/location metrics match on all three full benchmark splits.')
