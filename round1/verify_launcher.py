"""Test the final recommended package, its optional merge, and failure/timeout handling."""
import json
from pathlib import Path
import subprocess
import sys
from runtime import ROOT
from evaluate import write

def invoke(mode,content,extra=()):
    result=subprocess.run([sys.executable,str(ROOT/'export/run_guard.py'),'--mode',mode,*extra],
        input=content,text=True,capture_output=True,cwd=ROOT/'export')
    if result.returncode:raise RuntimeError(result.stderr)
    return [json.loads(line) for line in result.stdout.splitlines() if line]

if __name__=='__main__':
    batch_input=(ROOT/'demo_batch.jsonl').read_text();stream_input=(ROOT/'demo_stream.jsonl').read_text()
    batch=invoke('batch',batch_input);stream=invoke('stream',stream_input);merged=invoke('batch',batch_input,['--merged'])
    for rows in [batch,merged]:assert [r['action'] for r in rows[0]]==['release','block'],rows
    assert stream[0]['action']=='block' and stream[-1]['action']=='release',stream
    for name,rows in [('recommended_batch',batch),('recommended_stream',stream),('optional_merged_batch',merged)]:
        (ROOT/f'{name}_output.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    events=['not-json',json.dumps({'messages':[]})]
    def event(sid):return {'session':sid,'messages':[{'role':'user','content':'你好，请用简单的话介绍一下图书馆。'}],'final':True}
    events += [json.dumps(event(str(i))) for i in range(5)]
    events += [json.dumps({'session':'0','close':True}),json.dumps(event('4'))]
    errors=invoke('stream','\n'.join(events)+'\n')
    assert errors[0]['action']==errors[1]['action']=='hold'
    assert all('error' not in r for r in errors[2:6])
    assert errors[6]['error']=='four_session_capacity_reached'
    assert errors[7]['action']=='closed' and 'error' not in errors[8]
    deadline=invoke('stream',json.dumps(event('deadline'))+'\n',['--deadline-ms','0.000000001'])[0]
    assert deadline['deadline_exceeded'] and deadline['action']=='hold' and not deadline['released_text']
    write(ROOT/'launcher_verification.json',{'status':'passed','recommended_variant':'unmerged',
        'batch_actions':[r['action'] for r in batch[0]],'stream_actions':[r['action'] for r in stream],
        'optional_merged_actions':[r['action'] for r in merged[0]],'invalid_json':'hold','empty_messages':'hold',
        'capacity_and_close':'passed','soft_deadline':'hold_without_releasing_text'})
    print('Final package: both variants, JSONL, four-session capacity, close, and soft deadline passed.')
