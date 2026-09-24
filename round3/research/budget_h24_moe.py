"""Count a proposed classifier from public tensor metadata; no model execution."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "qwen35_08b_source_metadata.json").read_text())
header = source["header"]
prefix = "model.language_model."
base = sum(math.prod(item["shape"]) for name, item in header.items()
           if name.startswith(prefix))
cfg = source["config"]["text_config"]
d = cfg["hidden_size"]
layers = list(range(18, 24))
experts = 4
ffn = sum(math.prod(item["shape"]) for name, item in header.items()
          if name.startswith(prefix + "layers.18.mlp."))
assert ffn == 3 * d * cfg["intermediate_size"]
assert base == source["parameter_groups"]["model.language_model"]
# Budget assumption only: shared 1024->512 projection, then the existing
# user/assistant risk/category dimensions 3/9/3/8. No new policy labels.
head = d * 512 + 512 + 512 * (3 + 9 + 3 + 8) + (3 + 9 + 3 + 8)
router = len(layers) * d * experts  # bias-free routers
extra_experts = len(layers) * (experts - 1) * ffn
result = {
    "status": "architecture_budget_only_not_implemented",
    "source_repo": source["repo"], "source_revision": source["revision"],
    "text_backbone_parameters": base, "classification_head_budget": head,
    "h24_classifier_parameters": base + head,
    "moe_layer_indices_zero_based": layers, "experts_per_moe_layer": experts,
    "one_dense_ffn_parameters": ffn, "additional_expert_parameters": extra_experts,
    "router_parameters": router,
    "h24_m4_classifier_parameters": base + head + extra_experts + router,
    "top1_selected_path_parameter_convention": base + head + router,
    "top2_selected_path_parameter_convention": base + head + router + len(layers) * ffn,
    "selected_path_warning": "Includes the whole embedding table for parameter accounting; not FLOPs or actual bytes read per token. Batched routes can touch all experts.",
    "fixed_state_budget_bytes": {
        "gdn_fp32_18_layers": 18 * 16 * 128 * 128 * 4,
        "local_kv_bf16_6_layers_window512": 6 * 2 * 2 * 256 * 512 * 2,
        "conv_state_bf16_4_slots_18_layers": 18 * 6144 * 4 * 2,
    },
    "state_warning": "Proposed local-window variant only; excludes activations, padding, allocator, kernels, tokenizer and rollback state. Original full-attention variant is not fixed-state.",
}
assert result["h24_m4_classifier_parameters"] < 1_000_000_000
(HERE / "h24_moe_budget.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
