"""L20-only inference for exported Round4 direct risk classifiers.

The input is serialized once, then optionally split into exact token IDs.
This is not an incremental-text tokenizer or an HTTP serving runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

RISK_LABELS = ("safe", "unsafe", "controversial")
ROLES = ("user", "assistant")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", required=True, type=Path,
                        help="UTF-8 JSON file: messages array or {messages: [...]}; '-' reads stdin")
    parser.add_argument("--variant", required=True,
                        choices=("full", "window", "memory", "classification_rl"))
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Exported best.safetensors; no training state or initial checkpoint is needed")
    parser.add_argument("--rl-base-variant", choices=("full", "window", "memory"),
                        help="Architecture for classification_rl; otherwise read checkpoint-adjacent summary.json")
    parser.add_argument("--base-root", type=Path, default=Path("/work"),
                        help="Runtime root containing models/qwen35 (default: /work)")
    parser.add_argument("--base-code-dir", type=Path, default=Path("/work/input"))
    parser.add_argument("--window-code-dir", type=Path, default=Path("/work/window"))
    parser.add_argument("--memory-code-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--chunk-tokens", type=int, default=0,
                        help="0 = one complete forward; positive N = cached exact token-ID chunks")
    parser.add_argument("--max-input-tokens", type=int, default=8192,
                        help="Reject longer inputs without truncation (default: 8192)")
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument("--threshold", type=float,
                           help="Explicit unsafe probability threshold; not labelled as calibrated")
    threshold.add_argument("--calibration-metrics", type=Path,
                           help="Metrics JSON containing strata[language/role].threshold_from_calibration")
    parser.add_argument("--language", choices=("zh", "en"),
                        help="Required with --calibration-metrics; never inferred from text")
    args = parser.parse_args(argv)
    if args.chunk_tokens < 0 or args.max_input_tokens < 1:
        parser.error("--chunk-tokens must be >= 0 and --max-input-tokens must be >= 1")
    if args.threshold is not None and not valid_threshold(args.threshold):
        parser.error("--threshold must be a finite number in [0, 1]")
    if args.calibration_metrics is not None and args.language is None:
        parser.error("--language is required with --calibration-metrics")
    if args.rl_base_variant is not None and args.variant != "classification_rl":
        parser.error("--rl-base-variant is only valid with --variant classification_rl")
    return args


def valid_threshold(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def validate_messages(value):
    """Retain role/content bytes; do not strip or normalize labelled text."""
    messages = value.get("messages") if isinstance(value, dict) else value
    if not isinstance(messages, list) or not messages:
        raise ValueError("Input must contain a nonempty messages array")
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError(f"messages[{index}] requires a system/user/assistant role and string content")
        result.append({"role": role, "content": content})
    if result[-1]["role"] not in ROLES:
        raise ValueError("The final message must have role user or assistant")
    return result


def serialize(messages):
    """Identical to experiment_common and the frozen risk_v2 data builders."""
    return "\n\n".join(message["role"].upper() + ":\n" + message["content"] for message in messages)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def resolve_variant(args):
    if args.variant != "classification_rl":
        return args.variant
    summary_path = args.checkpoint.with_name("summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    recorded = summary.get("variant")
    if recorded is not None and recorded not in ("full", "window", "memory"):
        raise ValueError("RL summary.json contains an invalid architecture variant")
    if args.rl_base_variant is not None and recorded is not None and args.rl_base_variant != recorded:
        raise ValueError("--rl-base-variant conflicts with the checkpoint-adjacent RL summary")
    variant = args.rl_base_variant or recorded
    if variant is None:
        raise ValueError("RL architecture is unknown: supply --rl-base-variant or its summary.json")
    recorded_sha = summary.get("checkpoint_sha256")
    if recorded_sha is not None and sha256_file(args.checkpoint) != recorded_sha:
        raise ValueError("RL checkpoint does not match the SHA-256 in summary.json")
    return variant


def resolve_threshold(args, target_role):
    if args.threshold is not None:
        return args.threshold, {"kind": "explicit_uncalibrated", "value": args.threshold}
    if args.calibration_metrics is None:
        return None, {"kind": "none", "binary_decision_emitted": False}
    document = json.loads(args.calibration_metrics.read_text())
    key = args.language + "/" + target_role
    entry = document.get("strata", {}).get(key)
    if not isinstance(entry, dict) or "threshold_from_calibration" not in entry:
        raise ValueError(f"No calibration threshold for {key}; no language/role fallback is allowed")
    value = entry["threshold_from_calibration"]
    if not valid_threshold(value):
        raise ValueError(f"Invalid calibration threshold for {key}")
    return value, {"kind": "provided_calibration_report", "value": value, "stratum": key,
                   "path": str(args.calibration_metrics.resolve()),
                   "sha256": sha256_file(args.calibration_metrics),
                   "checkpoint_association_verified": False}


def load_classifier(args, variant):
    # Keep heavy imports behind the hardware check. There is no CPU/MPS fallback.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Model execution requires exactly one visible CUDA L20; CPU/MPS execution is disabled")
    device = torch.cuda.get_device_name(0)
    if "L20" not in device:
        raise RuntimeError(f"Model execution is restricted to L20; visible device is {device}")
    torch.set_num_threads(4)
    for directory in (args.base_code_dir, args.window_code_dir, args.memory_code_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing model-code directory: {directory}")
    if not (args.base_root / "models/qwen35").is_dir():
        raise FileNotFoundError(f"Missing local base model: {args.base_root / 'models/qwen35'}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path[:0] = [str(path.resolve()) for path in
                    (args.base_code_dir, args.window_code_dir, args.memory_code_dir)]
    import train_base
    from experiment_common import serialize as training_serialize
    from safetensors.torch import load_file
    from run_probe import short_recurrent

    # load() only reads local models/qwen35; it never needs training inputs,
    # the old H24 checkpoint, its summary, or its initial-checkpoint hash.
    train_base.ROOT = args.base_root.resolve()
    backbone, tokenizer, kernels = train_base.load("qwen35")
    model = train_base.Classifier(backbone).to("cuda")
    if variant == "window":
        from window_attention import set_window
        set_window(model.backbone, 512)
    elif variant == "memory":
        from memory_attention import attach_memory
        attach_memory(model.backbone, 512)
    state = load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    short_recurrent(True)
    if hasattr(model.backbone, "lm_head") or hasattr(model, "lm_head"):
        raise RuntimeError("A direct classifier must not have a vocabulary output head")
    return torch, model, tokenizer, training_serialize, {**kernels, "short_gdn_recurrent": True}, device


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main(argv=None):
    args = parse_args(argv)
    raw = sys.stdin.read() if str(args.messages) == "-" else args.messages.read_text(encoding="utf-8")
    messages = validate_messages(json.loads(raw))
    text = serialize(messages)
    target_role = messages[-1]["role"]
    threshold, threshold_metadata = resolve_threshold(args, target_role)
    variant = resolve_variant(args)
    torch, model, tokenizer, training_serialize, kernels, device = load_classifier(args, variant)
    if text != training_serialize(messages):
        raise RuntimeError("Input serialization does not match the training implementation")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids or len(token_ids) > args.max_input_tokens:
        raise ValueError(f"Input has {len(token_ids)} tokens; permitted range is 1..{args.max_input_tokens}; no truncation")
    model_limit = getattr(model.backbone.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and len(token_ids) > model_limit:
        raise ValueError(f"Input exceeds the base model's {model_limit}-token position limit")
    checkpoint_sha = sha256_file(args.checkpoint)
    emit({"event": "metadata", "variant": variant, "checkpoint_kind": args.variant,
          "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
          "base_root": str(args.base_root.resolve()), "device": device, "kernels": kernels,
          "window_tokens": 512 if variant != "full" else None,
          "target_role": target_role, "native_input_tokens": len(token_ids),
          "chunk_tokens": args.chunk_tokens, "risk_class_order": RISK_LABELS,
          "serialization": "uppercase role + ':\\n' + content; messages joined by '\\n\\n'; no special tokens",
          "serialized_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
          "token_ids_sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
          "threshold": threshold_metadata,
          "validation_limits": {
              "controversial": "preserved third class; no independent hard-label supervision or validation this round",
              "category": "not emitted; category heads were not supervised or validated this round",
              "other_role_head": "diagnostic only on this target; use target_role for the decision",
              "system_messages": "serializable, but not independently validated as a deployment input stratum",
              "prefix_semantics": "intermediate probabilities are not validated safe-prefix release decisions",
              "text_streaming": "whole input tokenized once; no text append re-tokenization, rollback, or session persistence",
              "production_approved": False},
          "generated_tokens": 0})
    ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    cached = args.chunk_tokens > 0
    size = args.chunk_tokens or len(token_ids)
    cache = None
    total_seconds = 0.0
    calls = 0
    with torch.inference_mode():
        for start in range(0, len(token_ids), size):
            end = min(len(token_ids), start + size)
            torch.cuda.synchronize()
            began = time.perf_counter()
            output = model(ids[:, start:end], past_key_values=cache, use_cache=cached)
            cache = output.past_key_values if cached else None
            hidden = output.last_hidden_state[:, -1]
            probabilities = {}
            for role in ROLES:
                logits, _ = model.readout(hidden, role)
                values = logits.float().softmax(-1)[0].cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError("Non-finite risk probabilities")
                probabilities[role] = dict(zip(RISK_LABELS, values))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            total_seconds += elapsed
            calls += 1
            unsafe = probabilities[target_role]["unsafe"]
            emit({"event": "classification", "chunk_index": calls - 1,
                  "input_token_start": start, "input_token_end": end,
                  "new_input_tokens": end - start, "prefix_tokens": end,
                  "final": end == len(token_ids), "target_role": target_role,
                  "risk_probabilities_by_head": probabilities,
                  "target_binary_decision": None if threshold is None else ("unsafe" if unsafe >= threshold else "safe"),
                  "target_unsafe_threshold": threshold,
                  "model_call_ms": elapsed * 1000,
                  "observed_native_itps": (end - start) / elapsed,
                  "generated_tokens": 0})
            del output, hidden, logits
    emit({"event": "complete", "native_input_tokens": len(token_ids), "classification_calls": calls,
          "model_call_seconds": total_seconds, "observed_native_itps": len(token_ids) / total_seconds,
          "timing_scope": "single input; model plus both role readouts and CPU-visible probabilities; includes cold first call; excludes loading, tokenization and JSON output; not a warmed benchmark",
          "generated_tokens": 0})


if __name__ == "__main__":
    main()
