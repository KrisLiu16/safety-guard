"""Freeze the original classifier and HTTP input-throughput smoke workload."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
R3 = HERE.parent
ROOT = R3.parent
A0 = ROOT / "round1/export/base_model"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_generated(info):
    return None if "__pycache__" in Path(info.name).parts else info


def main():
    bundle = HERE / "bundle_a0_http.tar"
    with tarfile.open(bundle, "w") as archive:
        archive.add(A0, arcname="bundle/a0_model", filter=no_generated)
        for source, name in (
            (ROOT / "round1/export/runtime.py", "runtime.py"),
            (R3 / "classifier_runtime.py", "classifier_runtime.py"),
            (R3 / "serve_classifier.py", "serve_classifier.py"),
            (R3 / "smoke_a0_http_cuda.py", "smoke_a0_http_cuda.py"),
        ):
            archive.add(source, arcname="bundle/" + name)
    digest = sha256(bundle)
    (HERE / "bundle_a0_http.sha256").write_text(f"{digest}  /work/bundle.tar\n")
    manifest = {
        "bundle_sha256": digest, "bundle_bytes": bundle.stat().st_size,
        "a0_model_sha256": sha256(A0 / "model.safetensors"),
        "runtime_sha256": sha256(R3 / "classifier_runtime.py"),
        "service_sha256": sha256(R3 / "serve_classifier.py"),
        "smoke_sha256": sha256(R3 / "smoke_a0_http_cuda.py"),
        "metric": "client-visible input tokens/s; no generation TPS",
        "not_policy_safety_aligned": True,
    }
    (HERE / "bundle_a0_http_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bundle_sha256": digest, "bytes": bundle.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
