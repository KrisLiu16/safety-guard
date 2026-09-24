"""Read-only L20 scheduling and training-gate check for the third round."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
DEFAULT_KUBECONFIG = Path("/Users/liuzhihao/.kube/cls-og1rjus2-private-latest")


def kubectl_json(kubeconfig: Path, kind: str, all_namespaces: bool = False) -> dict:
    command = ["kubectl", f"--kubeconfig={kubeconfig}", "get", kind]
    if all_namespaces:
        command.append("-A")
    command += ["-o", "json"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"kubectl get {kind} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gpu_request(container: dict) -> int:
    return int(container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", "0"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "data/l20_resource_snapshot.json")
    args = parser.parse_args()
    nodes = kubectl_json(args.kubeconfig, "nodes")["items"]
    pods = kubectl_json(args.kubeconfig, "pods", all_namespaces=True)["items"]
    capacity = {}
    for node in nodes:
        if node["metadata"].get("labels", {}).get("naive-sandbox/gpu-product") == "L20":
            capacity[node["metadata"]["name"]] = int(
                node["status"]["allocatable"].get("nvidia.com/gpu", "0")
            )
    if not capacity:
        raise RuntimeError("No L20 nodes found in this cluster")
    occupied = []
    for pod in pods:
        node = pod["spec"].get("nodeName")
        if node not in capacity or pod["status"].get("phase") in ("Succeeded", "Failed"):
            continue
        containers = pod["spec"].get("containers", [])
        init_containers = pod["spec"].get("initContainers", [])
        requested = max(sum(map(gpu_request, containers)),
                        max(map(gpu_request, init_containers), default=0))
        if requested:
            occupied.append({
                "node": node,
                "namespace": pod["metadata"]["namespace"],
                "pod": pod["metadata"]["name"],
                "phase": pod["status"].get("phase"),
                "gpu_requested": requested,
            })
    free = dict(capacity)
    for item in occupied:
        free[item["node"]] -= item["gpu_requested"]
    candidates = json.loads((ROOT / "data/candidates/manifest.json").read_text())
    audit = json.loads((ROOT / "data/audit_report.json").read_text())
    final_gate_ready = bool(candidates["training_ready"] and audit["training_gate"]["ready"])
    stage0_manifest_path = ROOT / "data/stage0_distillation_20k/manifest.json"
    stage0_manifest = json.loads(stage0_manifest_path.read_text()) if stage0_manifest_path.exists() else {}
    stage0_data_ready = bool(
        stage0_manifest.get("train_rows", 0) > 0
        and stage0_manifest.get("dev_rows", 0) > 0
        and stage0_manifest.get("uses_source_safety_labels") is False
        and stage0_manifest.get("project_safety_alignment") is False
        and stage0_manifest.get("output_sha256", {}).get("train")
        and stage0_manifest.get("output_sha256", {}).get("dev")
    )
    gpu_free = any(value > 0 for value in free.values())
    snapshot = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster": "cls-og1rjus2",
        "gpu_type": "NVIDIA L20",
        "allocatable_gpus_by_node": capacity,
        "free_gpus_by_node": free,
        "occupants": occupied,
        "stage0_open_corpus_data_ready": stage0_data_ready,
        "final_alignment_data_ready": final_gate_ready,
        "can_submit_stage0_now": stage0_data_ready and gpu_free,
        "can_submit_final_alignment_now": final_gate_ready and gpu_free,
        "probe_read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "free_gpus_by_node": free,
        "occupant_pods": [item["pod"] for item in occupied],
        "stage0_open_corpus_data_ready": stage0_data_ready,
        "final_alignment_data_ready": final_gate_ready,
        "can_submit_stage0_now": snapshot["can_submit_stage0_now"],
        "can_submit_final_alignment_now": snapshot["can_submit_final_alignment_now"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
