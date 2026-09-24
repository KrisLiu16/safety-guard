"""Shared data and scoring for an exploratory full-backbone comparison."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))


def serialize(messages):
    return '\n\n'.join(m['role'].upper()+':\n'+m['content'] for m in messages)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values=sorted(values)
    return {'p50_ms':statistics.median(values)*1000,
            'p95_ms':values[math.ceil(len(values)*.95)-1]*1000,
            'p99_ms':values[math.ceil(len(values)*.99)-1]*1000}


def summarize(dev, test):
    from sklearn.metrics import roc_auc_score
    keys=sorted({r['language']+'/'+r['target_role'] for r in test})
    result={}
    for key in keys:
        calibration=[r for r in dev if r['language']+'/'+r['target_role']==key]
        if not calibration:
            calibration=[r for r in dev if r['target_role']==key.split('/')[1]]
        safe=sorted([r['probs'][1] for r in calibration if r['source_label']=='safe'],reverse=True)
        threshold=math.nextafter(safe[min(math.floor(.05*len(safe)),len(safe)-1)],math.inf) if safe else .5
        rows=[r for r in test if r['language']+'/'+r['target_role']==key]
        y=[int(r['source_label']=='unsafe') for r in rows]
        p=[r['probs'][1] for r in rows]
        decisions={
            'argmax_unsafe':[max(range(3),key=lambda i:r['probs'][i])==1 for r in rows],
            'dev_fpr5_threshold':[v>=threshold for v in p],
        }
        metrics={'n':len(rows),'safe':y.count(0),'unsafe':y.count(1),
                 'unsafe_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
                 'threshold':threshold,'calibration_safe_n':len(safe)}
        for name,pred in decisions.items():
            tp=sum(a and b for a,b in zip(pred,y));fp=sum(a and not b for a,b in zip(pred,y))
            fn=sum(not a and b for a,b in zip(pred,y));tn=sum(not a and not b for a,b in zip(pred,y))
            metrics[name]={'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                           'recall':tp/(tp+fn) if tp+fn else None,
                           'fpr':fp/(fp+tn) if fp+tn else None,
                           'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
        result[key]=metrics
    return {'strata':result,'requested':len(test),'evaluated':len(test),
            'scope':'source labels; not project-policy calibrated; threshold selected only on development split'}
