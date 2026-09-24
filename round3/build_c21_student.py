"""Initialize a 21-layer direct classifier, retaining the pretrained final layer."""
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
DEFAULT_OUT = ROOT / "checkpoints/c21_keep_final_init"
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
SUPPORT = ("LICENSE", "configuration_qwen3.py", "modeling_qwen3_guard.py",
           "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    config = json.loads((source / "config.json").read_text())
    if config["num_hidden_layers"] != 28:
        raise RuntimeError("Expected 28 pretrained layers")
    # Drop one layer in each group of four; preserve the final layer and head input.
    selected = [i for i in range(28) if i % 4 != 2]
    assert len(selected) == 21 and selected[-1] == 27
    mapping = {old: new for new, old in enumerate(selected)}
    weights = {}
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as archive:
        for key in archive.keys():
            match = LAYER.match(key)
            if match:
                old = int(match.group(1))
                if old not in mapping:
                    continue
                new_key = f"model.layers.{mapping[old]}.{match.group(2)}"
            else:
                new_key = key
            weights[new_key] = archive.get_tensor(key).contiguous()
    layer_indices = sorted({int(match.group(1)) for key in weights
                            if (match := LAYER.match(key))})
    if layer_indices != list(range(21)):
        raise RuntimeError("Incomplete C21 layer remap")
    config["num_hidden_layers"] = 21
    config["max_window_layers"] = 21
    config["use_cache"] = True
    params = sum(t.numel() for t in weights.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c21-build-", dir=out.parent) as scratch:
        temp = Path(scratch)
        (temp / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        for name in SUPPORT:
            shutil.copy2(source / name, temp / name)
        save_file(weights, temp / "model.safetensors")
        manifest = {
            "architecture_version": "c21-keep-final-init-v1",
            "status": "untrained_pruned_initialization",
            "source_model_sha256": sha256(source / "model.safetensors"),
            "source_layers_retained": selected,
            "target_num_hidden_layers": 21,
            "parameter_count": params,
            "has_lm_head": any("lm_head" in key for key in weights),
            "model_sha256": sha256(temp / "model.safetensors"),
        }
        (temp / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temp / "README.md").write_text(
            "# C21 pruned initialization\n\nUntrained architecture probe; not a safety service.\n")
        temp.rename(out)
    print(json.dumps({"output": str(out), "layers": 21,
                      "parameters": params, "has_lm_head": manifest["has_lm_head"]}))


if __name__ == "__main__":
    main()
