"""Package pinned open corpora and teacher/student weights for L20-only Stage 0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
ROUND3 = HERE.parent
ROOT = ROUND3.parent
TEACHER = ROOT / "round1/export/base_model"
STUDENT = ROUND3 / "checkpoints/c1_even14_init"
DATA = ROUND3 / "data/stage0_distillation"
RUNTIME = ROOT / "round1/export/runtime.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    student_manifest = json.loads((STUDENT / "build_manifest.json").read_text())
    data_manifest = json.loads((DATA / "manifest.json").read_text())
    if student_manifest["model_sha256"] != sha256(STUDENT / "model.safetensors"):
        raise ValueError("Student checksum mismatch")
    if data_manifest["project_safety_alignment"] or data_manifest["uses_source_safety_labels"]:
        raise ValueError("Stage0 data contract mismatch")
    for split in ("train", "dev"):
        if data_manifest["output_sha256"][split] != sha256(DATA / f"{split}.jsonl"):
            raise ValueError("Stage0 data file mismatch")
    bundle = HERE / "bundle_stage0.tar"
    with tarfile.open(bundle, "w") as archive:
        archive.add(TEACHER, arcname="bundle/teacher")
        archive.add(STUDENT, arcname="bundle/student")
        archive.add(DATA, arcname="bundle/data")
        archive.add(RUNTIME, arcname="bundle/runtime.py")
        archive.add(HERE / "train_stage0_cuda.py", arcname="bundle/train_stage0_cuda.py")
    digest = sha256(bundle)
    (HERE / "bundle_stage0.sha256").write_text(f"{digest}  /work/bundle.tar\n")
    result = {
        "bundle_sha256": digest,
        "bundle_bytes": bundle.stat().st_size,
        "teacher_sha256": sha256(TEACHER / "model.safetensors"),
        "student_sha256": student_manifest["model_sha256"],
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "train_examples": data_manifest["train_rows"],
        "dev_examples": data_manifest["dev_rows"],
        "uses_project_safety_labels": False,
        "contains_credentials": False,
    }
    (HERE / "bundle_stage0_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

