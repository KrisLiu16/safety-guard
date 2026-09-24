"""Load a complete Round4 classifier from a verified, self-contained bundle.

There is no from_pretrained model-weight call, network access, old H24 weight
dependency, vocabulary output head, or CPU/MPS neural-execution fallback.
Only config/tokenizer assets and the bundle's complete state_dict are read.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

RISK_LABELS = ("safe", "unsafe", "controversial")
ROLES = ("user", "assistant")
CORE_VERSIONS = {
    "torch": "2.7.1", "transformers": "5.17.0",
    "flash-linear-attention": "0.5.2", "fla-core": "0.5.2",
    "safetensors": "0.8.0",
}
RUNTIME_DISTRIBUTIONS = (*CORE_VERSIONS, "tokenizers", "triton", "huggingface-hub")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def valid_threshold(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def validate_selection(document):
    """Use the existing select_checkpoint.py manifest contract verbatim."""
    if not isinstance(document, dict) or document.get("status") != "research_candidate":
        raise ValueError("MODEL_MANIFEST.json must contain a fixed research_candidate")
    variant = document.get("variant")
    if variant not in ("full", "window", "memory"):
        raise ValueError("Unknown selected architecture")
    if document.get("candidate") not in (variant, "classification_rl"):
        raise ValueError("Selected checkpoint kind conflicts with its architecture")
    expected_window = None if variant == "full" else 512
    if document.get("window") != expected_window:
        raise ValueError("The selected architecture must use its fixed evaluation window")
    digest = document.get("checkpoint_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("A lowercase SHA-256 checkpoint digest is required")
    if not isinstance(document.get("checkpoint"), str) or not document["checkpoint"]:
        raise ValueError("Selected checkpoint path is missing")
    if document.get("risk_class_order") != list(RISK_LABELS):
        raise ValueError("Risk class order differs from the trained classifier")
    if document.get("hard_supervised_classes_this_round") != ["safe", "unsafe"]:
        raise ValueError("Unexpected supervision contract")
    if document.get("category_validated") is not False or document.get("generated_tokens") != 0:
        raise ValueError("Invalid category-validation or token-generation claim")
    if document.get("production_approval") is not False:
        raise ValueError("This bundle is a research candidate, not a production approval")
    thresholds = document.get("thresholds")
    if (not isinstance(thresholds, dict) or not thresholds
            or any(key not in ("zh/user", "zh/assistant", "en/user", "en/assistant")
                   or not valid_threshold(value) for key, value in thresholds.items())):
        raise ValueError("Selected calibration thresholds are invalid")
    if sorted(thresholds) != sorted(document.get("calibrated_strata", [])):
        raise ValueError("Calibrated strata do not match threshold keys")
    return document


def runtime_versions():
    versions = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Required runtime distribution is not installed: {name}") from error
    for name, expected in CORE_VERSIONS.items():
        actual = versions[name].split("+", 1)[0]
        if actual != expected:
            raise RuntimeError(f"{name} version {versions[name]} differs from the verified {expected}")
    return versions


def _inside(bundle, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Bundle paths must be nonempty relative paths")
    path = bundle / relative
    if ".." in Path(relative).parts or path.is_symlink() or not path.resolve().is_relative_to(bundle):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    return path


def verify_bundle(bundle):
    """Hash/layout verification is CPU-only; no torch import or model construction."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "round4-direct-classifier-bundle-v1":
        raise ValueError("Unsupported bundle format")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle has no file checksums")
    required = {"MODEL_MANIFEST.json", "classifier.safetensors", "assets/config.json",
                "assets/tokenizer.json", "assets/tokenizer_config.json", "standalone_model.py",
                "run_guard.py", "text_stream_runtime.py", "requirements.txt"}
    if not required.issubset(files):
        raise ValueError(f"Missing required bundled files: {sorted(required.difference(files))}")
    for relative, record in files.items():
        path = _inside(bundle, relative)
        if (not path.is_file() or not isinstance(record, dict)
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")):
            raise ValueError(f"Bundle file failed integrity verification: {relative}")
    selection = validate_selection(json.loads((bundle / "MODEL_MANIFEST.json").read_text(encoding="utf-8")))
    if files["classifier.safetensors"]["sha256"] != selection["checkpoint_sha256"]:
        raise ValueError("Bundled checkpoint differs from the fixed selection")
    if (manifest.get("variant") != selection["variant"]
            or manifest.get("candidate") != selection["candidate"]
            or manifest.get("checkpoint_sha256") != selection["checkpoint_sha256"]):
        raise ValueError("Bundle metadata conflicts with MODEL_MANIFEST.json")
    if selection["variant"] != "full" and "window_attention.py" not in files:
        raise ValueError("The window implementation is missing")
    if selection["variant"] == "memory" and "memory_attention.py" not in files:
        raise ValueError("The memory implementation is missing")
    return bundle, manifest, selection


def _load_module(name, path):
    # Load vendored modules explicitly. Existing experiment modules elsewhere in
    # sys.path must never substitute for the code whose checksum was verified.
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load bundled module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def enable_fla_short_recurrent():
    """Identical FLA wrappers and <=32-token recurrent dispatch to run_probe."""
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    def wrap(fn):
        def call(q, k, v, g, beta, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True, cu_seqlens=None, **kwargs):
            return fn(q, k, v, g=g, beta=beta, initial_state=initial_state,
                      output_final_state=output_final_state,
                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
        return call

    original = wrap(chunk_gated_delta_rule)

    def dispatch(q, k, v, g, beta, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True, cu_seqlens=None, **kwargs):
        fn = (fused_recurrent_gated_delta_rule
              if not torch.is_grad_enabled() and initial_state is not None and q.shape[1] <= 32
              else original)
        return fn(q, k, v, g=g, beta=beta, initial_state=initial_state,
                  output_final_state=output_final_state,
                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens, **kwargs)

    module.torch_chunk_gated_delta_rule = dispatch
    module.torch_recurrent_gated_delta_rule = wrap(fused_recurrent_gated_delta_rule)
    return {"attention": "torch sdpa", "gdn_chunk": "fla-0.5.2",
            "gdn_recurrent": "fla-0.5.2", "short_gdn_recurrent": True,
            "short_convolution": "Transformers torch implementation"}


def _build_classifier(backbone):
    import torch
    from torch import nn

    class Classifier(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            width = backbone.config.hidden_size
            self.heads = nn.ModuleDict({role: nn.ModuleDict({
                "projection": nn.Sequential(nn.Linear(width, 512, dtype=torch.float32),
                                            nn.LayerNorm(512, dtype=torch.float32), nn.SiLU()),
                "risk": nn.Linear(512, 3, dtype=torch.float32),
                "category": nn.Linear(512, categories, dtype=torch.float32),
            }) for role, categories in (("user", 9), ("assistant", 8))})

        def readout(self, hidden, role):
            projected = self.heads[role]["projection"](hidden.float())
            return self.heads[role]["risk"](projected), self.heads[role]["category"](projected)

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                     past_key_values=past_key_values, use_cache=use_cache)

        def step(self, ids, cache, cached):
            output = self(ids, past_key_values=cache, use_cache=cached)
            risk, category = self.readout(output.last_hidden_state[:, -1], "user")
            return risk, category, output.past_key_values if cached else None

    return Classifier(backbone)


def load_bundle(bundle):
    """Return (torch, model, tokenizer, metadata); execute only on one L20.

    A normal constructor initializes RoPE and every nonpersistent buffer. Only
    parameters are cast to BF16, so explicitly FP32 RoPE buffers retain their
    constructor precision. New memory parameters and classifier heads are then
    created in FP32. load_state_dict copies values into those destination
    dtypes; assign=True is intentionally never used.
    """
    bundle, manifest, selection = verify_bundle(bundle)
    if sha256_file(__file__) != manifest["files"]["standalone_model.py"]["sha256"]:
        raise RuntimeError("The active loader differs from the bundle; invoke its own run_guard.py")
    versions = runtime_versions()
    if versions != manifest.get("runtime_versions"):
        raise RuntimeError("Runtime versions differ from the packaged dependency lock")
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "L20" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Neural execution requires exactly one visible CUDA L20; CPU/MPS is disabled")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected the verified CUDA 12.8 build; found {torch.version.cuda}")
    torch.set_num_threads(4)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Construct the bundled model with the standard float32 default dtype")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config_json = json.loads((bundle / "assets/config.json").read_text(encoding="utf-8"))
    text_config = config_json.get("text_config", config_json)
    config = Qwen3_5TextConfig.from_dict(text_config)
    config._attn_implementation = "sdpa"
    kernels = enable_fla_short_recurrent()
    # Do not construct on meta and to_empty(): RoPE inv_freq is nonpersistent
    # and absent from the full state_dict; it must remain correctly initialized.
    with torch.device("cpu"):
        backbone = Qwen3_5TextModel(config)
    with torch.no_grad():
        for parameter in backbone.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    backbone = backbone.to("cuda")
    if selection["variant"] in ("window", "memory"):
        window_module = _load_module("window_attention", bundle / "window_attention.py")
        if selection["variant"] == "window":
            window_module.set_window(backbone, selection["window"])
        else:
            memory_module = _load_module("memory_attention", bundle / "memory_attention.py")
            memory_module.attach_memory(backbone, selection["window"])
    model = _build_classifier(backbone).to("cuda")
    state = load_file(str(bundle / "classifier.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        expected = (torch.float32 if name.startswith("heads.") or ".memory_" in name
                    else torch.bfloat16)
        if parameter.dtype != expected:
            raise RuntimeError(f"Loaded parameter has wrong dtype: {name}: {parameter.dtype}, expected {expected}")
    if any("lm_head" in name for name, _ in model.named_modules()):
        raise RuntimeError("A direct classifier must not contain a vocabulary output head")
    tokenizer = AutoTokenizer.from_pretrained(bundle / "assets", local_files_only=True,
                                             trust_remote_code=False)
    metadata = {"bundle": str(bundle), "candidate": selection["candidate"],
                "variant": selection["variant"], "checkpoint_sha256": selection["checkpoint_sha256"],
                "window_tokens": selection["window"], "runtime_versions": versions,
                "device": torch.cuda.get_device_name(0), "kernels": kernels,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "buffer_dtypes": {name: str(buffer.dtype) for name, buffer in model.named_buffers()},
                "risk_class_order": list(RISK_LABELS), "generated_tokens": 0,
                "production_approval": False,
                "third_risk_class_validated": False, "category_validated": False,
                "safe_prefix_release_validated": False,
                "selection": selection}
    return torch, model, tokenizer, metadata
