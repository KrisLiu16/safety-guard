"""Fresh-process launch to first JSON result; OS and Metal caches are not purged."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    payload=(ROOT/'demo_batch.jsonl').read_text();result={}
    for name,args in [
        ('original',[str(ROOT/'demo.py'),'--mode','batch','--calibration',str(ROOT/'baseline/chinese_metrics.json')]),
        ('rl',[str(ROOT/'export/run_guard.py'),'--mode','batch']),
        ('rl_merged',[str(ROOT/'export/run_guard.py'),'--mode','batch','--merged'])]:
        times=[]
        for repeat in range(3):
            with (ROOT/'startup_stderr.log').open('a') as error:
                begin=time.perf_counter()
                process=subprocess.Popen([sys.executable,*args],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True)
                process.stdin.write(payload);process.stdin.flush();process.stdin.close()
                first=process.stdout.readline();elapsed=time.perf_counter()-begin
                assert json.loads(first) and process.wait(timeout=30)==0
                times.append(elapsed)
        result[name]={'fresh_process_to_first_result_seconds':times,'median_seconds':statistics.median(times)}
    result['scope']='Includes Python imports, weight/adapter loading, MPS first inference and first JSON. One batch of two fixed short conversations. OS/Metal caches not purged; not a reboot-cold measurement.'
    (ROOT/'startup_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
