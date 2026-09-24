"""Lock A0/C1 weights, Stage0 adapter, benchmark, and direct-classifier code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROUND3 = HERE.parent
ROOT = ROUND3.parent
A0 = ROOT / "round1/export/base_model"
C1 = ROUND3 / "checkpoints/c1_even14_init"
ADAPTER = HERE / "stage0_20k_output/stage0_adapter.safetensors"
RUNTIME = ROOT / "round1/export/runtime.py"
EVALUATE = ROOT / "round1/evaluate.py"
BENCHMARK = ROOT / "round1/benchmark"
SELECTION = ROOT / "round1/benchmark_selection.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclude_generated(info: tarfile.TarInfo):
    return None if "__pycache__" in Path(info.name).parts else info


def main():
    adapter_summary = json.loads((HERE / "stage0_20k_output/summary.json").read_text())
    c1_manifest = json.loads((C1 / "build_manifest.json").read_text())
    if (sha256(ADAPTER) != adapter_summary["adapter_sha256"]
            or sha256(C1 / "model.safetensors") != c1_manifest["model_sha256"]
            or sha256(A0 / "model.safetensors") != adapter_summary["teacher_model_sha256"]
            or not adapter_summary["not_policy_safety_aligned"]):
        raise RuntimeError("Input model/adapter checksum or status mismatch")
    bundle = HERE / "bundle_classification_eval.tar"
    with tarfile.open(bundle, "w") as archive:
        for source, dest in ((A0, "bundle/a0_model"), (C1, "bundle/c1_model"),
                             (ADAPTER, "bundle/stage0_adapter.safetensors"),
                             (RUNTIME, "bundle/runtime.py"),
                             (EVALUATE, "bundle/evaluate.py"),
                             (BENCHMARK, "bundle/benchmark"),
                             (SELECTION, "bundle/benchmark_selection.json"),
                             (ROUND3 / "classifier_runtime.py", "bundle/classifier_runtime.py"),
                             (ROUND3 / "serve_classifier.py", "bundle/serve_classifier.py"),
                             (ROUND3 / "benchmark_classification_cuda.py",
                              "bundle/benchmark_classification_cuda.py")):
            archive.add(source, arcname=dest, filter=exclude_generated)
    bundle_hash = sha256(bundle)
    (HERE / "bundle_classification_eval.sha256").write_text(
        f"{bundle_hash}  /work/bundle.tar\n")
    manifest = {
        "bundle_sha256": bundle_hash, "bundle_bytes": bundle.stat().st_size,
        "a0_model_sha256": sha256(A0 / "model.safetensors"),
        "c1_model_sha256": sha256(C1 / "model.safetensors"),
        "stage0_adapter_sha256": sha256(ADAPTER),
        "benchmark_selection_sha256": sha256(SELECTION),
        "classifier_runtime_sha256": sha256(ROUND3 / "classifier_runtime.py"),
        "http_adapter_sha256": sha256(ROUND3 / "serve_classifier.py"),
        "benchmark_script_sha256": sha256(ROUND3 / "benchmark_classification_cuda.py"),
        "metric": "input_tokens_per_second; no generated tokens",
        "not_policy_safety_aligned": True,
    }
    (HERE / "bundle_classification_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bundle_sha256": bundle_hash, "bundle_bytes": bundle.stat().st_size,
                      "metric": manifest["metric"]}), flush=True)


if __name__ == "__main__":
    main()
