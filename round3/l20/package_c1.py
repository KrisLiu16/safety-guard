"""Package the credential-free C1 checkpoint and CUDA probe for L20."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
ROUND3 = HERE.parent
MODEL = ROUND3 / "checkpoints/c1_even14_init"
RUNTIME = ROUND3.parent / "round1/export/runtime.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    model_manifest = json.loads((MODEL / "build_manifest.json").read_text())
    if sha256(MODEL / "model.safetensors") != model_manifest["model_sha256"]:
        raise RuntimeError("C1 model checksum mismatch")
    bundle = HERE / "bundle_c1.tar"
    with tarfile.open(bundle, "w") as archive:
        archive.add(MODEL, arcname="bundle/c1_model")
        archive.add(RUNTIME, arcname="bundle/runtime.py")
        archive.add(HERE / "run_c1_cuda.py", arcname="bundle/run_c1_cuda.py")
    digest = sha256(bundle)
    (HERE / "bundle_c1.sha256").write_text(f"{digest}  /work/bundle.tar\n")
    manifest = {
        "bundle_sha256": digest,
        "bundle_bytes": bundle.stat().st_size,
        "model_sha256": model_manifest["model_sha256"],
        "architecture_version": model_manifest["architecture_version"],
        "contains_credentials": False,
    }
    (HERE / "bundle_c1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

