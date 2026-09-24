"""Freeze the 21-layer architecture probe and independent source-valid labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
R3 = HERE.parent
ROOT = R3.parent
MODELS = {
    "a0_model": ROOT / "round1/export/base_model",
    "c14_model": R3 / "checkpoints/c1_even14_init",
    "c21_model": R3 / "checkpoints/c21_keep_final_init",
}
ADAPTER = HERE / "stage0_20k_output/stage0_adapter.safetensors"
DATA = R3 / "data/source_valid_probe.jsonl"
DATA_MANIFEST = R3 / "data/source_valid_probe_manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_generated(info):
    return None if "__pycache__" in Path(info.name).parts else info


def main():
    for name, path in MODELS.items():
        manifest = json.loads((path / "build_manifest.json").read_text()) if name != "a0_model" else None
        if manifest and (manifest["model_sha256"] != sha256(path / "model.safetensors")
                         or manifest["has_lm_head"]):
            raise RuntimeError(f"{name} checkpoint mismatch")
    adapter_summary = json.loads((HERE / "stage0_20k_output/summary.json").read_text())
    if sha256(ADAPTER) != adapter_summary["adapter_sha256"]:
        raise RuntimeError("Stage0 adapter mismatch")
    probe_manifest = json.loads(DATA_MANIFEST.read_text())
    if probe_manifest["training_use"] or sha256(DATA) != probe_manifest["output_sha256"]:
        raise RuntimeError("Source-valid diagnostic split mismatch")
    bundle = HERE / "bundle_c21_probe.tar"
    with tarfile.open(bundle, "w") as archive:
        for name, path in MODELS.items():
            archive.add(path, arcname="bundle/" + name, filter=no_generated)
        for source, name in (
            (ADAPTER, "stage0_adapter.safetensors"),
            (DATA, "source_valid_probe.jsonl"),
            (DATA_MANIFEST, "source_valid_probe_manifest.json"),
            (ROOT / "round1/export/runtime.py", "runtime.py"),
            (R3 / "classifier_runtime.py", "classifier_runtime.py"),
            (R3 / "probe_c21_cuda.py", "probe_c21_cuda.py"),
        ):
            archive.add(source, arcname="bundle/" + name)
    digest = sha256(bundle)
    (HERE / "bundle_c21_probe.sha256").write_text(f"{digest}  /work/bundle.tar\n")
    manifest = {
        "bundle_sha256": digest, "bundle_bytes": bundle.stat().st_size,
        "model_sha256": {name: sha256(path / "model.safetensors")
                         for name, path in MODELS.items()},
        "adapter_sha256": sha256(ADAPTER),
        "source_valid_manifest_sha256": sha256(DATA_MANIFEST),
        "source_valid_rows": probe_manifest["rows"],
        "metric": "direct classifier input tokens/s, not generation TPS",
        "not_policy_safety_aligned": True,
    }
    (HERE / "bundle_c21_probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bundle_sha256": digest, "bytes": bundle.stat().st_size,
                      "source_valid_rows": probe_manifest["rows"]}), flush=True)


if __name__ == "__main__":
    main()
