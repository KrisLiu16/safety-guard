"""Collect this experiment's reports, then release its GPU job; weights stay on PVC."""
import json
import argparse
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
POD='safety-guard-base-compare-20260923-r2-84k52'
JOB='safety-guard-base-compare-20260923-r2'
K=['kubectl','--kubeconfig',str(Path.home()/'.kube/cls-og1rjus2-private-latest')]


def run(args,timeout=30):
    return subprocess.run(K+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)


def main():
    global ROOT,POD,JOB
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',default=POD);parser.add_argument('--job',default=JOB)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--ready-marker',default='comparison_ready')
    parser.add_argument('--ack-marker',default='artifacts_collected')
    parser.add_argument('--artifact-relative',default='output')
    parser.add_argument('--checkpoint-path',action='append')
    parser.add_argument('--pre-collection-script',help='Optional L20 audit script to run after reports are ready, before releasing the job')
    parser.add_argument('--pre-collection-sha256',help='Required checksum of the optional remote audit script')
    args=parser.parse_args();ROOT=args.output_root.resolve();POD=args.pod;JOB=args.job
    if bool(args.pre_collection_script) != bool(args.pre_collection_sha256):
        parser.error('Audit script and checksum must be provided together')
    ROOT.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+21600
    audit_attempted=False
    while time.monotonic()<deadline:
        try:
            pod=run(['get','pod',POD,'-o','json'])
            if pod.returncode:
                time.sleep(30);continue
            phase=json.loads(pod.stdout)['status']['phase']
            if phase in ('Failed','Succeeded'):
                log=run(['logs',POD])
                (ROOT/'worker_final.log').write_bytes(log.stdout)
                print(json.dumps({'phase':phase,'weights_preserved_on_pvc':True}),flush=True)
                return
            ready=run(['exec',POD,'--','test','-f','/work/output/'+args.ready_marker])
            if ready.returncode:
                time.sleep(30);continue
            if args.pre_collection_script and not audit_attempted:
                # Additional validation cannot change the frozen training inputs.
                # Keep failure evidence and collect the original reports even if it fails.
                audit_attempted=True
                status={'script':args.pre_collection_script,'sha256':args.pre_collection_sha256}
                try:
                    digest=run(['exec',POD,'--','sha256sum',args.pre_collection_script])
                    if digest.returncode or digest.stdout.decode().split()[0]!=args.pre_collection_sha256:
                        raise RuntimeError('Post-training audit script checksum mismatch')
                    with (ROOT/'post_training_audit.log').open('wb') as dest:
                        audit=subprocess.run(K+['exec',POD,'--','/work/modern/bin/python',args.pre_collection_script],
                                             stdout=dest,stderr=subprocess.STDOUT,timeout=900)
                    status.update(returncode=audit.returncode,completed=audit.returncode==0)
                except (subprocess.TimeoutExpired,RuntimeError) as exc:
                    status.update(completed=False,error=str(exc))
                (ROOT/'post_training_audit_status.json').write_text(json.dumps(status,indent=2)+'\n')
            archive=ROOT/'comparison_reports.tar.gz'
            with archive.open('wb') as dest:
                result=subprocess.run(K+['exec',POD,'--','tar','czf','-','--exclude=*.safetensors','--exclude=*.pt','--exclude=*.tmp',
                                          '-C','/work',args.artifact_relative],stdout=dest,stderr=subprocess.PIPE,timeout=120)
            if result.returncode:raise RuntimeError(result.stderr.decode())
            subprocess.run(['tar','xzf',str(archive),'-C',str(ROOT)],check=True,timeout=60)
            (ROOT/'worker_final.log').write_bytes(run(['logs',POD]).stdout)
            # Reports are local; full checkpoints remain on the persistent PVC.
            (ROOT/'checkpoint_locations.json').write_text(json.dumps({
                'pvc':'safety-guard-base-compare-data',
                'checkpoints':args.checkpoint_path or ['/work/output/qwen3_full/classifier.safetensors',
                               '/work/output/qwen35_full/classifier.safetensors']},indent=2)+'\n')
            ack=run(['exec',POD,'--','touch','/work/output/'+args.ack_marker])
            if ack.returncode:raise RuntimeError(ack.stderr.decode())
            finished=run(['wait','--for=condition=complete','job/'+JOB,'--timeout=60s'],timeout=65)
            print(json.dumps({'reports_collected':True,'gpu_job_completed':finished.returncode==0,
                              'checkpoints_preserved_on_pvc':True}),flush=True)
            return
        except (subprocess.TimeoutExpired,RuntimeError) as exc:
            print(json.dumps({'collector_error':str(exc)}),flush=True)
            time.sleep(30)
    raise TimeoutError('Experiment collection deadline; PVC preserved')


if __name__=='__main__':main()
