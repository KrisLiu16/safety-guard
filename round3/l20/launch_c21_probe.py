"""Run the C21 diagnostic and exact-input-token speed probe on one free L20."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from launch_stage0_20k import free_l20, objects, run
from launch_classification_eval import digest, download_file

HERE = Path(__file__).resolve().parent
JOB = "safety-guard-c21-source-probe-20260923"
FILES = ("summary.json", "a0.json", "c14_stage0.json", "c21_init.json",
         "a0_predictions.jsonl", "c14_stage0_predictions.jsonl", "c21_init_predictions.jsonl")


def wait_pod(kubeconfig):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        for pod in objects(kubeconfig, "pods", "-l", f"job-name={JOB}")["items"]:
            if pod["status"]["phase"] == "Running":
                return pod["metadata"]["name"]
            if pod["status"]["phase"] == "Failed":
                raise RuntimeError("C21 Pod failed before upload")
        time.sleep(5)
    raise TimeoutError("C21 Pod not Running after 15 minutes")


def wait_ready(kubeconfig, pod):
    deadline = time.monotonic() + 10200
    while time.monotonic() < deadline:
        if run(kubeconfig, "exec", pod, "-c", "eval", "--", "test", "-f",
               "/work/output/ready", check=False).returncode == 0:
            return
        state = objects(kubeconfig, "pod", pod)["status"]["phase"]
        if state in ("Succeeded", "Failed"):
            logs = run(kubeconfig, "logs", pod, "-c", "eval", "--tail=80", check=False)
            raise RuntimeError(f"C21 Pod {state}: {logs.stdout[-4000:]}")
        time.sleep(10)
    raise TimeoutError("C21 probe exceeded deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", default="/Users/liuzhihao/.kube/cls-og1rjus2-private-latest")
    parser.add_argument("--out", type=Path, default=HERE / "c21_probe_output")
    args = parser.parse_args()
    manifest = json.loads((HERE / "bundle_c21_probe_manifest.json").read_text())
    if (manifest["bundle_sha256"] != digest(HERE / "bundle_c21_probe.tar")
            or manifest["source_valid_rows"] != 600):
        raise RuntimeError("C21 probe bundle mismatch")
    free = free_l20(args.kubeconfig)
    print(json.dumps({"free_l20_gpus_by_node": free}), flush=True)
    if max(free.values(), default=0) < 1:
        raise SystemExit("No free L20; no probe Job created")
    if run(args.kubeconfig, "get", "job", JOB, check=False).returncode == 0:
        raise RuntimeError(f"Job {JOB} already exists; inspect before re-running")
    run(args.kubeconfig, "create", "-f", str(HERE / "job_c21_probe.yaml"))
    run(args.kubeconfig, "patch", "job", JOB, "--type=merge", "-p",
        '{"spec":{"suspend":false}}')
    pod = wait_pod(args.kubeconfig)
    print(f"Uploading C21 probe to {pod}", flush=True)
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.tar.part", input_file=HERE / "bundle_c21_probe.tar")
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.sha256", input_file=HERE / "bundle_c21_probe.sha256")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "mv",
        "/work/bundle.tar.part", "/work/bundle.tar")
    print("Bundle uploaded; testing source-label quality and exact-token ITPS", flush=True)
    wait_ready(args.kubeconfig, pod)
    for name in FILES:
        download_file(args.kubeconfig, pod, name, args.out)
    summary = json.loads((args.out / "summary.json").read_text())
    if (not summary["not_policy_safety_aligned"]
            or set(summary["models"]) != {"a0", "c14_stage0", "c21_init"}):
        raise RuntimeError("C21 diagnostic output contract mismatch")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "touch", "/work/output/ack")
    complete = run(args.kubeconfig, "wait", f"job/{JOB}", "--for=condition=complete",
                   "--timeout=180s", check=False)
    if complete.returncode:
        raise RuntimeError("C21 Job did not complete after artifact acknowledgement")
    print(json.dumps({"status": "succeeded", "output": str(args.out.resolve()),
                      "user_auc": {name: x["quality"]["roles"]["user"]["unsafe_auc"]
                                   for name, x in summary["models"].items()}}), flush=True)


if __name__ == "__main__":
    main()
