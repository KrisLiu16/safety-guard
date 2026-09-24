"""Run direct-classifier quality and ITPS evaluation only on a free L20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from launch_stage0_20k import free_l20, objects, run

HERE = Path(__file__).resolve().parent
JOB = "safety-guard-classification-itps-20260923"
FILES = ("summary.json",
         "a0_original/classification_benchmark.json",
         "a0_original/official_subset_metrics.json",
         "a0_original/official_subset_predictions.jsonl",
         "c1_stage0_20k/classification_benchmark.json",
         "c1_stage0_20k/official_subset_metrics.json",
         "c1_stage0_20k/official_subset_predictions.jsonl")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wait_pod(kubeconfig):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        for pod in objects(kubeconfig, "pods", "-l", f"job-name={JOB}")["items"]:
            if pod["status"]["phase"] == "Running":
                return pod["metadata"]["name"]
            if pod["status"]["phase"] == "Failed":
                raise RuntimeError("Evaluation Pod failed before upload")
        time.sleep(5)
    raise TimeoutError("Evaluation Pod not Running after 15 minutes")


def wait_ready(kubeconfig, pod):
    deadline = time.monotonic() + 10200
    while time.monotonic() < deadline:
        if run(kubeconfig, "exec", pod, "-c", "eval", "--", "test", "-f",
               "/work/output/ready", check=False).returncode == 0:
            return
        item = objects(kubeconfig, "pod", pod)
        if item["status"]["phase"] in ("Succeeded", "Failed"):
            logs = run(kubeconfig, "logs", pod, "-c", "eval", "--tail=90", check=False)
            raise RuntimeError(f"Evaluation Pod {item['status']['phase']}: {logs.stdout[-4000:]}")
        time.sleep(10)
    raise TimeoutError("L20 classification evaluation exceeded deadline")


def download_file(kubeconfig, pod, name, out):
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    command = ["kubectl", f"--kubeconfig={kubeconfig}", "-n", "default",
               "exec", pod, "-c", "eval", "--", "cat", f"/work/output/{name}"]
    with part.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE)
    if result.returncode:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {name}: {result.stderr.decode(errors='replace')[:250]}")
    part.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", default="/Users/liuzhihao/.kube/cls-og1rjus2-private-latest")
    parser.add_argument("--out", type=Path, default=HERE / "classification_eval_output")
    args = parser.parse_args()
    manifest = json.loads((HERE / "bundle_classification_eval_manifest.json").read_text())
    if (manifest["bundle_sha256"] != digest(HERE / "bundle_classification_eval.tar")
            or not manifest["not_policy_safety_aligned"]):
        raise RuntimeError("Local benchmark bundle checksum/status mismatch")
    available = free_l20(args.kubeconfig)
    print(json.dumps({"free_l20_gpus_by_node": available}), flush=True)
    if max(available.values(), default=0) < 1:
        raise SystemExit("No free L20; no evaluation Job created")
    if run(args.kubeconfig, "get", "job", JOB, check=False).returncode == 0:
        raise RuntimeError(f"Job {JOB} already exists; inspect before re-running")
    run(args.kubeconfig, "create", "-f", str(HERE / "job_classification_eval.yaml"))
    run(args.kubeconfig, "patch", "job", JOB, "--type=merge", "-p",
        '{"spec":{"suspend":false}}')
    print(f"Created and resumed {JOB}", flush=True)
    pod = wait_pod(args.kubeconfig)
    print(f"Uploading benchmark bundle to {pod}", flush=True)
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.tar.part", input_file=HERE / "bundle_classification_eval.tar")
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.sha256", input_file=HERE / "bundle_classification_eval.sha256")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "mv",
        "/work/bundle.tar.part", "/work/bundle.tar")
    print("Bundle uploaded; measuring risk classification and input-token throughput", flush=True)
    wait_ready(args.kubeconfig, pod)
    for name in FILES:
        download_file(args.kubeconfig, pod, name, args.out)
    summary = json.loads((args.out / "summary.json").read_text())
    a0 = json.loads((args.out / "a0_original/classification_benchmark.json").read_text())
    c1 = json.loads((args.out / "c1_stage0_20k/classification_benchmark.json").read_text())
    if (not summary["not_policy_safety_aligned"] or not c1["http"]["first_append_has_classification"]
            or a0["has_lm_head"] or c1["has_lm_head"]
            or c1["http"]["input_tokens_per_second"] <= 0):
        raise RuntimeError("Downloaded benchmark contract invalid")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "touch", "/work/output/ack")
    completed = run(args.kubeconfig, "wait", f"job/{JOB}", "--for=condition=complete",
                    "--timeout=180s", check=False)
    if completed.returncode:
        raise RuntimeError("Evaluation Job failed to complete after artifact acknowledgement")
    print(json.dumps({"status": "succeeded", "output": str(args.out.resolve()),
                      "a0_parameters": a0["parameter_count"],
                      "c1_parameters": c1["parameter_count"],
                      "http_input_tokens_per_second": c1["http"]["input_tokens_per_second"]}), flush=True)


if __name__ == "__main__":
    main()
