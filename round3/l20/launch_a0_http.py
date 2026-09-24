"""Run current A0 HTTP classification workload on one available L20."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from launch_stage0_20k import free_l20, objects, run
from launch_classification_eval import digest, download_file

HERE = Path(__file__).resolve().parent
JOB = "safety-guard-a0-http-itps-20260923"


def wait_pod(kubeconfig):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        for pod in objects(kubeconfig, "pods", "-l", f"job-name={JOB}")["items"]:
            if pod["status"]["phase"] == "Running":
                return pod["metadata"]["name"]
            if pod["status"]["phase"] == "Failed":
                raise RuntimeError("A0 HTTP Pod failed before upload")
        time.sleep(5)
    raise TimeoutError("A0 HTTP Pod not Running after 15 minutes")


def wait_ready(kubeconfig, pod):
    deadline = time.monotonic() + 6600
    while time.monotonic() < deadline:
        if run(kubeconfig, "exec", pod, "-c", "eval", "--", "test", "-f",
               "/work/output/ready", check=False).returncode == 0:
            return
        state = objects(kubeconfig, "pod", pod)["status"]["phase"]
        if state in ("Succeeded", "Failed"):
            logs = run(kubeconfig, "logs", pod, "-c", "eval", "--tail=80", check=False)
            raise RuntimeError(f"A0 HTTP Pod {state}: {logs.stdout[-4000:]}")
        time.sleep(10)
    raise TimeoutError("A0 HTTP workload exceeded deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", default="/Users/liuzhihao/.kube/cls-og1rjus2-private-latest")
    parser.add_argument("--out", type=Path, default=HERE / "a0_http_output")
    args = parser.parse_args()
    manifest = json.loads((HERE / "bundle_a0_http_manifest.json").read_text())
    if manifest["bundle_sha256"] != digest(HERE / "bundle_a0_http.tar"):
        raise RuntimeError("A0 HTTP bundle checksum mismatch")
    free = free_l20(args.kubeconfig)
    print(json.dumps({"free_l20_gpus_by_node": free}), flush=True)
    if max(free.values(), default=0) < 1:
        raise SystemExit("No free L20; no A0 HTTP Job created")
    if run(args.kubeconfig, "get", "job", JOB, check=False).returncode == 0:
        raise RuntimeError(f"Job {JOB} already exists; inspect before re-running")
    run(args.kubeconfig, "create", "-f", str(HERE / "job_a0_http.yaml"))
    run(args.kubeconfig, "patch", "job", JOB, "--type=merge", "-p",
        '{"spec":{"suspend":false}}')
    pod = wait_pod(args.kubeconfig)
    print(f"Uploading A0 HTTP bundle to {pod}", flush=True)
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.tar.part", input_file=HERE / "bundle_a0_http.tar")
    run(args.kubeconfig, "exec", "-i", pod, "-c", "eval", "--", "sh", "-c",
        "cat > /work/bundle.sha256", input_file=HERE / "bundle_a0_http.sha256")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "mv",
        "/work/bundle.tar.part", "/work/bundle.tar")
    print("Bundle uploaded; measuring client-visible classification ITPS", flush=True)
    wait_ready(args.kubeconfig, pod)
    download_file(args.kubeconfig, pod, "a0_http_itps.json", args.out)
    result = json.loads((args.out / "a0_http_itps.json").read_text())
    if (result["has_lm_head"] or not result["classification_on_first_append"]
            or result["training_status"] != "pretrained_baseline_unaligned"):
        raise RuntimeError("A0 HTTP result contract mismatch")
    run(args.kubeconfig, "exec", pod, "-c", "eval", "--", "touch", "/work/output/ack")
    complete = run(args.kubeconfig, "wait", f"job/{JOB}", "--for=condition=complete",
                   "--timeout=180s", check=False)
    if complete.returncode:
        raise RuntimeError("A0 HTTP Job did not complete after artifact acknowledgement")
    print(json.dumps({"status": "succeeded", "output": str(args.out.resolve()),
                      "complete_8client_itps": result["complete"]["8"]["itps"],
                      "stream_8client_itps": result["streaming"]["8"]["itps"]}), flush=True)


if __name__ == "__main__":
    main()
