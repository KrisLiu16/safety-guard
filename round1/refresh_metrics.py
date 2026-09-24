"""Recompute summaries from saved predictions, without any inference or fitting changes."""
from evaluate import ROOT,read,write,summarize_chinese,official_metrics
from stream_evaluate import summarize_stream

if __name__=='__main__':
    for name in ['baseline','post','merged','precision_control']:
        directory=ROOT/name
        write(directory/'chinese_metrics.json',summarize_chinese(read(directory/'chinese_predictions.jsonl')))
        rows=read(directory/'official_all_predictions.jsonl')
        write(directory/'official_all_metrics.json',{s:official_metrics([r for r in rows if r['split']==s]) for s in ['thinking','thinking_loc','response_loc']})
        if (directory/'stream_predictions.jsonl').exists():write(directory/'stream_metrics.json',summarize_stream(read(directory/'stream_predictions.jsonl')))
    print('Refreshed saved summaries; predictions and thresholds unchanged.')
