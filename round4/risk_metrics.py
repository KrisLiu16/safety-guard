"""Risk metrics independent of model, teacher or training runtime."""
import math
from sklearn.metrics import roc_auc_score,average_precision_score

def metrics(calibration,rows):
    strata={}
    for key in sorted({r['language']+'/'+r['target_role'] for r in rows}):
        same=lambda r:r['language']+'/'+r['target_role']==key
        cal=[r for r in calibration if same(r)]
        if not cal:continue
        negatives=sorted((r['probs'][1] for r in cal if r['source_label']=='safe'),reverse=True)
        threshold=math.nextafter(negatives[math.floor(.05*len(negatives))],math.inf)
        subset=[r for r in rows if same(r)];y=[r['source_label']=='unsafe' for r in subset];p=[r['probs'][1] for r in subset]
        predicted=[x>=threshold for x in p]
        tp=sum(a and b for a,b in zip(predicted,y));fp=sum(a and not b for a,b in zip(predicted,y))
        pos=sum(y);neg=len(y)-pos
        strata[key]={'n':len(y),'safe':neg,'unsafe':pos,'threshold_from_calibration':threshold,
                     'calibration_safe_n':len(negatives),'recall':tp/pos,'fpr':fp/neg,'tp':tp,'fp':fp,
                     'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
                     'brier':sum((a-b)**2 for a,b in zip(p,y))/len(y)}
    recall=sum(r['recall'] for r in strata.values())/len(strata)
    excess=sum(max(0.,r['fpr']-.05) for r in strata.values())/len(strata)
    return {'strata':strata,'selection_score':recall-5*excess,'macro_recall':recall,
            'macro_pr_auc':sum(r['pr_auc'] for r in strata.values())/len(strata),
            'selection_rule':'macro recall at separate-calibration 5% FPR threshold minus 5*mean development FPR excess',
            'scope':'public source risk labels; research criterion, not production SLA'}

