"""Package a fixed Round5 research candidate without reading base model weights.

Run in the verified L20 worker environment, after MODEL_MANIFEST.json is fixed.
Packaging itself only copies/checks files and reads package versions; it does
not import torch, instantiate a neural network, or alter trained checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from standalone_model import INFERENCE_ENGINE, GRAPH_ENGINE, BUNDLE_FORMAT, runtime_versions, sha256_file, validate_selection, validate_evidence

ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja", "LICENSE", "LICENSE.txt")
REQUIRED_ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json")


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "MODEL_MANIFEST.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-assets", type=Path, default=Path("/work/models/qwen35"))
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional relocated full checkpoint; its SHA must match the fixed manifest")
    parser.add_argument("--code-dir", type=Path, default=root)
    parser.add_argument("--window-code-dir", type=Path, default=Path("/work/window"))
    parser.add_argument("--copy-weights", action="store_true",
                        help="Copy even on the same filesystem; default attempts a hard link, then copies")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate input files/checkpoint SHA without writing or inspecting runtime packages")
    return parser.parse_args(argv)


def inspect_inputs(args):
    selection = validate_selection(json.loads(args.manifest.read_text(encoding="utf-8")))
    checkpoint = args.checkpoint or Path(selection["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("The complete checkpoint does not match MODEL_MANIFEST.json")
    sources = {"MODEL_MANIFEST.json": args.manifest, "classifier.safetensors": checkpoint}
    evidence = {}
    for name, item in selection['selection_evidence'].items():
        source = Path(item['path'])
        if sha256_file(source) != item['sha256']:
            raise ValueError('Completed selection evidence was modified: ' + name)
        sources['selection/' + name + '.json'] = source
        evidence[name] = json.loads(source.read_text())
    validate_evidence(selection, evidence)
    for name in REQUIRED_ASSETS:
        if not (args.model_assets / name).is_file():
            raise FileNotFoundError(f"Required local model asset is missing: {name}")
    config = json.loads((args.model_assets / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("Expected the trained Qwen3.5 config")
    text_config = config.get("text_config", config)
    if text_config.get("num_hidden_layers") != 24:
        raise ValueError("Expected the full retained 24-layer text backbone")
    if text_config.get("hidden_size") != 1024:
        raise ValueError("Expected the trained 0.8B backbone hidden width")
    for name in ASSETS:
        if (args.model_assets / name).is_file():
            sources[f"assets/{name}"] = args.model_assets / name
    templates = args.model_assets / "chat_templates"
    if templates.is_dir():
        for source in sorted(templates.glob("*.jinja")):
            sources[f"assets/chat_templates/{source.name}"] = source
    for name in ("standalone_model.py", "run_guard.py", "text_stream_runtime.py",
                 "graph_stream.py", "canonical_block_engine.py", "canonical_text_runtime.py"):
        sources[name] = args.code_dir / name
    if selection["variant"] in ("window", "memory"):
        sources["window_attention.py"] = args.window_code_dir / "window_attention.py"
    if selection["variant"] == "memory":
        sources["memory_attention.py"] = args.code_dir / "memory_attention.py"
    usage = args.code_dir / "BUNDLE_USAGE.md"
    if usage.is_file():
        sources["BUNDLE_USAGE.md"] = usage
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required bundle input is missing: {name}: {source}")
    return selection, sources


def main(argv=None):
    args = parse_args(argv)
    selection, sources = inspect_inputs(args)
    if args.check_only:
        print(json.dumps({"status": "inputs_verified", "candidate": selection["candidate"],
                          "checkpoint_sha256": selection["checkpoint_sha256"],
                          "files": sorted(sources), "model_calls": 0}))
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing bundle: {output}")
    versions = runtime_versions()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    weight_storage = "copied"
    try:
        for relative, source in sources.items():
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "classifier.safetensors" and not args.copy_weights:
                try:
                    os.link(source, destination)
                    weight_storage = "hard_linked"
                    continue
                except OSError:
                    pass
            shutil.copyfile(source, destination)
        requirements = "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n"
        (temporary / "requirements.txt").write_text(requirements, encoding="utf-8")
        records = {str(path.relative_to(temporary)): {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        } for path in sorted(temporary.rglob("*")) if path.is_file()}
        if records["classifier.safetensors"]["sha256"] != selection["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed while packaging")
        if records["MODEL_MANIFEST.json"]["sha256"] != sha256_file(args.manifest):
            raise RuntimeError("Selection manifest changed while packaging")
        if json.loads((temporary / "MODEL_MANIFEST.json").read_text(encoding="utf-8")) != selection:
            raise RuntimeError("Selection content changed after input validation")
        bundle_manifest = {
            "format": BUNDLE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate": selection["candidate"], "variant": selection["variant"],
            "inference_engine": INFERENCE_ENGINE,
            'selection_status': selection['status'], 'promoted_from_initial': selection['promoted_from_initial'],
            'thresholds': selection['thresholds'], 'threshold_comparison': '>',
            'execution_contract': selection['execution_contract'],
            'canonical_quality_gate_pass': selection['canonical_quality_gate_pass'],
            'graph_validation_passed': False,
            "text_engine_contract": {
                "default": INFERENCE_ENGINE, "experimental_option": GRAPH_ENGINE, "whole_input": "eager",
                'graph_validation_status': 'experimental; prior extended graph text whole-state audit failed; no full validation claim',
                "experimental_graph_capture_lengths": [32], "window": 512, "batch_size": 1,
                "graph_dispatch": "aligned cache has at least 512 tokens; always exactly32 physical tokens",
                "eager_dispatch": "aligned prefix below512 or explicit eager implementation; always exactly32 physical tokens",
                "capture_failure_fallback": False,
                "initialization_excluded_from_request_accounting": True,
            },
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "runtime_versions": versions,
            "python_version_at_packaging": platform.python_version(),
            "runtime_image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
            "runtime_requirements": {"physical_cuda_devices": 1, "gpu": "L20", "cuda": "12.8",
                                     "system_packages": ["gcc", "libc6-dev"]},
            "weight_storage_at_packaging": weight_storage,
            "standalone_model_weights": True,
            "reads_base_model_weights": False,
            "reads_initial_h24_checkpoint": False,
            "loading_dtype": {"backbone_parameters": "bfloat16", "heads": "float32",
                              "memory_feature_maps_and_gates": "float32",
                              "nonpersistent_buffers": "initialized by the ordinary text-model constructor"},
            "serialization": "uppercase role + ':\\n' + content; messages joined by '\\n\\n'; no special tokens",
            "max_input_tokens": 8192, "generated_tokens": 0, "production_approval": False,
            "third_risk_class_validated": False, "category_validated": False,
            "safe_prefix_release_validated": False,
            "bundle_neural_parity_validation": "required separately on L20; packaging is not validation",
            "files": records,
        }
        (temporary / "BUNDLE_MANIFEST.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Re-verify the portable inputs before the directory becomes visible.
        from standalone_model import verify_bundle
        verify_bundle(temporary)
        if output.exists():
            raise FileExistsError(f"Bundle destination appeared during packaging: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({"status": "packaged", "bundle": str(output),
                      "candidate": selection["candidate"], "variant": selection["variant"],
                      "inference_engine": INFERENCE_ENGINE,
                      "checkpoint_sha256": selection["checkpoint_sha256"],
                      "weight_storage": weight_storage, "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
