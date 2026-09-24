"""Load a complete Round5 classifier from a verified, self-contained bundle.

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
INFERENCE_ENGINE = "eager"
GRAPH_ENGINE = "window_cuda_graph"
BUNDLE_FORMAT = "round5-prefix-classifier-bundle-v1"
INITIAL_WINDOW_SHA256 = "bb16a3a6f87748ce302d6db125822b31add9a7d5210cd44804d9416be44f30d2"
TRAINER_SHA256 = '1aea85c2c1228aba6066c915922e790c7fb56cc26c4ef7d21329508937c13e7c'
METRICS_SHA256 = '20a803f17d846b729a64dc4f0d6ad28b22d714117197854755a47bd3f163faff'
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
    """Validate a completed Round5 selection, including an unpromoted fallback."""
    if (not isinstance(document, dict) or document.get("format") != "round5-prefix-selection-v1"
            or document.get("status") not in ("research_candidate", "fallback_not_promoted", 'research_unvalidated_runtime')):
        raise ValueError("MODEL_MANIFEST.json must contain a completed Round5 selection")
    if (document.get("inference_engine") != INFERENCE_ENGINE
            or document.get("variant") != "window" or document.get("backbone_layers") != 24
            or document.get("window") != 512):
        raise ValueError("Round5 requires the complete 24-layer W512 backbone and eager default")
    variant = document.get("variant")
    if variant not in ("full", "window", "memory"):
        raise ValueError("Unknown selected architecture")
    if document.get("candidate") not in ("initial_window_fallback", "sft", "classification_rl"):
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
    if (type(document.get('eligible_candidate_found')) is not bool
            or type(document.get('promoted_from_initial')) is not bool
            or type(document.get('training_promoted_from_initial')) is not bool
            or type(document.get('canonical_quality_gate_pass')) is not bool
            or document.get('selection_read_test') is not False
            or document.get('graph_validation_passed') is not False
            or document.get('threshold_comparison') != '>'):
        raise ValueError('Invalid selection, graph-validation or strict-threshold contract')
    training_promoted = document['eligible_candidate_found'] and digest != INITIAL_WINDOW_SHA256
    promoted = training_promoted and document['canonical_quality_gate_pass']
    expected_status = ('research_unvalidated_runtime' if not document['canonical_quality_gate_pass']
                       else 'research_candidate' if promoted else 'fallback_not_promoted')
    if (document['promoted_from_initial'] != promoted
            or document['training_promoted_from_initial'] != training_promoted
            or document['status'] != expected_status
            or (not document['eligible_candidate_found'] and digest != INITIAL_WINDOW_SHA256)
            or ((document['candidate'] == 'initial_window_fallback') != (digest == INITIAL_WINDOW_SHA256))):
        raise ValueError('A fallback cannot be promoted or replaced with an ineligible checkpoint')
    thresholds = document.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {'whole', 'stream'}:
        raise ValueError("Selected calibration thresholds are invalid")
    for mode, values in thresholds.items():
        if (not isinstance(values, dict) or not values
                or any(key not in ('zh/user', 'zh/assistant', 'en/user', 'en/assistant')
                       or not valid_threshold(value) for key, value in values.items())
                or sorted(values) != sorted(document.get('calibrated_strata', {}).get(mode, []))):
            raise ValueError('Invalid calibrated strata/thresholds for ' + mode)
    evidence = document.get('selection_evidence')
    if not isinstance(evidence, dict) or set(evidence) != {'summary', 'sft_summary', 'rl_summary', 'canonical_calibration'}:
        raise ValueError('Completed training selection evidence is required')
    for name, item in evidence.items():
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Invalid selection evidence: ' + name)
    validate_execution_contract(document.get('execution_contract'))
    return document


def validate_execution_contract(contract):
    expected = {'name': 'canonical32-v1', 'block_tokens': 32, 'window': 512, 'backbone_layers': 24,
                'partial_cache': 'not_committed', 'readout': 'all_32_positions_both_roles',
                'max_input_tokens': 8192, 'risk_labels': list(RISK_LABELS)}
    if (not isinstance(contract, dict) or set(contract) != set(expected) | {'pad_token_id'}
            or any(contract.get(key) != value for key, value in expected.items())
            or type(contract['pad_token_id']) is not int or contract['pad_token_id'] < 0):
        raise ValueError('Only the fixed canonical32 serving execution contract is supported')


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


def validate_evidence(selection, documents):
    """Derive runtime identity/thresholds from completed train+cal/dev evidence only."""
    final, sft, rl = (documents[name] for name in ('summary', 'sft_summary', 'rl_summary'))
    if (any(item.get('status') != 'completed' for item in (final, sft, rl))
            or sft.get('steps') != 4164 or sft.get('epochs') != 2 or rl.get('updates') != 256
            or final.get('sft') != sft or final.get('classification_rl') != rl
            or final.get('selection_read_test') is not False
            or final.get('binding') != sft.get('binding') or final.get('binding') != rl.get('binding')
            or final.get('final_checkpoint_sha256') != selection['checkpoint_sha256']
            or rl.get('checkpoint_sha256') != selection['checkpoint_sha256']
            or rl.get('reference_checkpoint_sha256') != sft.get('checkpoint_sha256')):
        raise ValueError('Completed training summary identities disagree')
    binding = final['binding']
    if (binding.get('trainer_sha256') != TRAINER_SHA256 or binding.get('metrics_sha256') != METRICS_SHA256
            or binding.get('initial_checkpoint_sha256') != INITIAL_WINDOW_SHA256 or binding.get('window') != 512
            or rl['selected'].get('checkpoint_sha256') != selection['checkpoint_sha256']
            or sft['selected'].get('checkpoint_sha256') != sft['checkpoint_sha256']):
        raise ValueError('Selection does not belong to the completed, frozen W512 training protocol')
    metrics = rl['best_metrics']
    if (metrics.get('checkpoint_sha256') != selection['checkpoint_sha256']
            or metrics.get('selection_read_test') is not False
            or metrics.get('thresholds_fitted_on') != 'calibration only'
            or metrics.get('baseline_whole_macro_recall') != sft.get('baseline_whole_macro_recall')):
        raise ValueError('Selected calibration/development evidence changed')
    thresholds = {}
    for mode in ('whole', 'stream'):
        strata = metrics[mode]['strata']
        if (set(strata) != {'zh/user', 'zh/assistant', 'en/assistant'}
                or any(row.get('comparison') != '>' for row in strata.values())):
            raise ValueError('Only strict > calibration thresholds are supported')
        thresholds[mode] = {name: row['threshold'] for name, row in strata.items()}
    gates = {'each_stream_stratum_fpr_at_most_5pct': all(row['fpr'] <= .05 for row in metrics['stream']['strata'].values()),
             'whole_macro_recall_drop_at_most_2pp': metrics['whole']['macro_recall'] >= sft['baseline_whole_macro_recall'] - .02 - 1e-12}
    eligible = all(gates.values())
    if (metrics.get('selection_gates') != gates or metrics.get('eligible') != eligible
            or rl.get('eligible_candidate_found') != eligible
            or selection['eligible_candidate_found'] != eligible):
        raise ValueError('Manifest original selection differs from fixed development gating')
    expected_kind = ('initial_window_fallback' if selection['checkpoint_sha256'] == INITIAL_WINDOW_SHA256
                     else 'classification_rl' if rl['selected']['stage'] == 'classification_rl' else 'sft')
    if selection['candidate'] != expected_kind:
        raise ValueError('Manifest attributes weights to the wrong selected stage')
    canonical = documents['canonical_calibration']
    expected = {'version': 'round5-canonical32-calibration-v1', 'status': 'completed',
                'checkpoint_sha256': selection['checkpoint_sha256'],
                'execution_contract': selection['execution_contract'], 'threshold_comparison': '>',
                'thresholds': selection['thresholds'], 'calibrated_strata': selection['calibrated_strata'],
                'thresholds_fitted_on': 'calibration only', 'selection_changed': False, 'selection_read_test': False,
                'arbitrary_bpe_schedule_fpr_guarantee': False}
    if any(canonical.get(name) != value for name, value in expected.items()):
        raise ValueError('Serving thresholds do not come from canonical32 calibration of the fixed selected weights')
    for name in ('calibration_receipt', 'development_receipt'):
        receipt = canonical.get(name)
        if not isinstance(receipt, dict) or len(receipt.get('predictions_sha256', '')) != 64:
            raise ValueError('Canonical calibration/development receipt missing')
    development = canonical['development_metrics']
    canonical_gates = {
        'each_stream_stratum_fpr_at_most_5pct': all(r['fpr'] <= .05 for r in development['stream']['strata'].values()),
        'whole_macro_recall_drop_at_most_2pp': development['whole']['macro_recall'] >= development['baseline_whole_macro_recall'] - .02 - 1e-12}
    canonical_eligible = all(canonical_gates.values())
    if (development.get('selection_gates') != canonical_gates or development.get('eligible') != canonical_eligible
            or canonical.get('canonical_quality_gate_pass') != canonical_eligible
            or selection['canonical_quality_gate_pass'] != canonical_eligible):
        raise ValueError('Canonical runtime acceptance status does not match its development gates')
    canonical_sources = canonical.get('runtime_source_sha256', {})
    if not {'canonical_block_engine.py', 'canonical_text_runtime.py'}.issubset(canonical_sources):
        raise ValueError('Canonical execution source checksums are required')
    return selection


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
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("Unsupported bundle format")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle has no file checksums")
    required = {"MODEL_MANIFEST.json", "classifier.safetensors", "assets/config.json",
                "assets/tokenizer.json", "assets/tokenizer_config.json", "standalone_model.py",
                "run_guard.py", "text_stream_runtime.py", "graph_stream.py",
                "canonical_block_engine.py", "canonical_text_runtime.py", "window_attention.py", "requirements.txt"}
    required.update({'selection/summary.json', 'selection/sft_summary.json', 'selection/rl_summary.json', 'selection/canonical_calibration.json'})
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
            or manifest.get("checkpoint_sha256") != selection["checkpoint_sha256"]
            or manifest.get("inference_engine") != INFERENCE_ENGINE
            or manifest.get('thresholds') != selection['thresholds']
            or manifest.get('threshold_comparison') != '>'
            or manifest.get('selection_status') != selection['status']
            or manifest.get('promoted_from_initial') != selection['promoted_from_initial']
            or manifest.get('graph_validation_passed') is not False):
        raise ValueError("Bundle metadata conflicts with MODEL_MANIFEST.json")
    for name, evidence in selection['selection_evidence'].items():
        if files['selection/' + name + '.json']['sha256'] != evidence['sha256']:
            raise ValueError('Bundled selection evidence differs from its fixed manifest')
    validate_evidence(selection, {name: json.loads((bundle / ('selection/' + name + '.json')).read_text())
                                  for name in selection['selection_evidence']})
    calibration = json.loads((bundle / 'selection/canonical_calibration.json').read_text())
    for name, digest in calibration['runtime_source_sha256'].items():
        if name not in files or files[name]['sha256'] != digest:
            raise ValueError('Canonical calibration was measured with different runtime code: ' + name)
    if (manifest.get('execution_contract') != selection['execution_contract']
            or manifest.get('canonical_quality_gate_pass') != selection['canonical_quality_gate_pass']):
        raise ValueError('Canonical runtime acceptance/definition differs from the selection manifest')
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
    if text_config.get('num_hidden_layers') != 24 or text_config.get('hidden_size') != 1024:
        raise ValueError('Round5 requires the retained 24-layer 1024-width backbone')
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
                "inference_engine": INFERENCE_ENGINE,
                "bundle_default_inference_engine": INFERENCE_ENGINE,
                "variant": selection["variant"], "checkpoint_sha256": selection["checkpoint_sha256"],
                "window_tokens": selection["window"], "runtime_versions": versions,
                "device": torch.cuda.get_device_name(0), "kernels": kernels,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "buffer_dtypes": {name: str(buffer.dtype) for name, buffer in model.named_buffers()},
                "risk_class_order": list(RISK_LABELS), "generated_tokens": 0,
                "production_approval": False,
                "selection_status": selection['status'], "promoted_from_initial": selection['promoted_from_initial'],
                "canonical_quality_gate_pass": selection['canonical_quality_gate_pass'],
                "threshold_comparison": '>', "thresholds": selection['thresholds'],
                "graph_validation_passed": False,
                "third_risk_class_validated": False, "category_validated": False,
                "safe_prefix_release_validated": False,
                "selection": selection}
    return torch, model, tokenizer, metadata
