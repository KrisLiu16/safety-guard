"""Arithmetic budget for random-init pure GDN classifiers; no model run."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "qwen35_08b_source_metadata.json").read_text())
header = source["header"]
layer_prefix = "model.language_model.layers.0."
block = sum(math.prod(item["shape"]) for name, item in header.items()
            if name.startswith(layer_prefix))
d, layers, vocab = 1024, 24, 98304
head = d * 512 + 512 + 512 * 23 + 23
ffn = 3 * d * 3584
dense = vocab * d + layers * block + d + head
moe = dense + 6 * 3 * ffn + 6 * d * 4
result = {
    "status": "proposal_only_random_initialization_no_weights_loaded",
    "shape_reference": source["repo"] + "@" + source["revision"],
    "num_layers": layers, "hidden_size": d, "vocab_size_proposal": vocab,
    "gdn_and_ffn_block_parameters": block,
    "softmax_attention_layers": 0,
    "dense_classifier_parameters": dense,
    "last6_four_expert_classifier_parameters": moe,
    "gdn_fp32_recurrent_state_bytes": layers * 16 * 128 * 128 * 4,
    "conv_bf16_state_4slots_bytes": layers * 6144 * 4 * 2,
    "state_excludes": ["temporary activations", "tokenization and rollback snapshots",
                       "allocator", "optional additional role embeddings"],
    "pretraining_milestones_not_launched": [100_000_000, 1_000_000_000],
    "larger_budget_hypothesis_tokens": [10_000_000_000, 20_000_000_000],
    "budget_warning": "Milestones and token range are experimental choices, not a convergence guarantee or a scheduled job. Training throughput must be measured on L20.",
}
assert dense == 618529559 and moe == 816734999
assert moe < 1_000_000_000
(HERE / "r24_budget.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
