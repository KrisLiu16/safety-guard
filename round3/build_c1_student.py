"""Initialize the 14-layer causal classifier by selecting even A0 layers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "round1/export/base_model"
DESTINATION = ROOT / "checkpoints/c1_even14_init"
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
SUPPORT_FILES = (
    "LICENSE", "configuration_qwen3.py", "modeling_qwen3_guard.py",
    "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DESTINATION)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    original = json.loads((source / "config.json").read_text())
    n = original["num_hidden_layers"]
    if n != 28:
        raise ValueError(f"Expected the audited 28-layer checkpoint, got {n}")
    selected = list(range(0, n, 2))
    mapping = {source_idx: target_idx for target_idx, source_idx in enumerate(selected)}
    weights = {}
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as archive:
        keys = list(archive.keys())
        for key in keys:
            match = LAYER.match(key)
            if match:
                source_idx = int(match.group(1))
                if source_idx not in mapping:
                    continue
                output_key = f"model.layers.{mapping[source_idx]}.{match.group(2)}"
            else:
                output_key = key
            if output_key in weights:
                raise ValueError(f"Duplicate output key: {output_key}")
            weights[output_key] = archive.get_tensor(key).contiguous()
    actual_layers = sorted({int(m.group(1)) for k in weights if (m := LAYER.match(k))})
    if actual_layers != list(range(len(selected))):
        raise ValueError(f"Incomplete layer mapping: {actual_layers}")
    parameters = sum(t.numel() for t in weights.values())
    config = dict(original)
    config["num_hidden_layers"] = len(selected)
    config["max_window_layers"] = len(selected)
    config["use_cache"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c1-build-", dir=out.parent) as scratch:
        temp = Path(scratch)
        (temp / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        for name in SUPPORT_FILES:
            shutil.copy2(source / name, temp / name)
        save_file(weights, temp / "model.safetensors")
        manifest = {
            "architecture_version": "c1-even14-init-v1",
            "status": "untrained_pruned_initialization",
            "source_path": str(source),
            "source_model_sha256": sha256(source / "model.safetensors"),
            "source_config_sha256": sha256(source / "config.json"),
            "source_num_hidden_layers": n,
            "source_layers_retained": selected,
            "target_num_hidden_layers": len(selected),
            "parameter_count": parameters,
            "has_lm_head": any("lm_head" in key for key in weights),
            "model_sha256": sha256(temp / "model.safetensors"),
        }
        (temp / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temp / "README.md").write_text(
            "# C1 even-layer initialization\n\n"
            "This is a structural starting point, not a trained or evaluated safety classifier. "
            "Do not deploy its current classifications. See build_manifest.json for the frozen mapping.\n"
        )
        temp.rename(out)
    print(json.dumps({"output": str(out), "parameter_count": parameters,
                      "layers": len(selected), "has_lm_head": manifest["has_lm_head"]}))


if __name__ == "__main__":
    main()

