"""Direct risk classification from the C1 Guard heads; no token generation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache

HERE = Path(__file__).resolve().parent
if not (HERE / "runtime.py").exists():
    sys.path.insert(0, str(HERE.parent / "round1"))
else:
    sys.path.insert(0, str(HERE))
from runtime import attach_lora, encode, load, merge_lora  # noqa: E402

SCHEMA_VERSION = "guard-classification-v1"
RISK_INDEX = {"safe": 0, "unsafe": 1, "controversial": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage0_classifier(model_path: Path, adapter_path: Path, device: str = "cuda",
                           merge_adapter: bool = True):
    """Restore exactly the trained C1 LoRA/head tensors and reject an LM head."""
    model_path, adapter_path = Path(model_path), Path(adapter_path)
    if device == "cuda" and (not torch.cuda.is_available()
                             or "L20" not in torch.cuda.get_device_name(0).upper()):
        raise RuntimeError("Formal inference requires an NVIDIA L20")
    model, tokenizer = load(dtype=torch.bfloat16, device=device, model_path=model_path)
    if hasattr(model, "lm_head"):
        raise RuntimeError("Expected a direct classifier without lm_head")
    trainables = attach_lora(model)
    state = load_file(str(adapter_path))
    expected = {name for name, _ in trainables}
    if set(state) != expected:
        raise RuntimeError("Adapter tensor names differ from C1 architecture")
    model.load_state_dict(state, strict=False)
    if merge_adapter:
        merge_lora(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    version = "c1-stage0-20k-" + sha256_file(adapter_path)[:12]
    return model, tokenizer, version


def load_baseline_classifier(model_path: Path, device: str = "cuda"):
    """Load the untouched, directly classifying A0 baseline."""
    model_path = Path(model_path)
    if device == "cuda" and (not torch.cuda.is_available()
                             or "L20" not in torch.cuda.get_device_name(0).upper()):
        raise RuntimeError("Formal inference requires an NVIDIA L20")
    model, tokenizer = load(dtype=torch.bfloat16, device=device, model_path=model_path)
    if hasattr(model, "lm_head"):
        raise RuntimeError("Expected a direct classifier without lm_head")
    version = "a0-original-" + sha256_file(model_path / "model.safetensors")[:12]
    return model, tokenizer, version


def _risk_category_logits(output, role: str):
    if role == "user":
        return output.query_risk_level_logits, output.query_category_logits
    if role == "assistant":
        return output.risk_level_logits, output.category_logits
    raise ValueError("Target role must be user or assistant")


def readout(output, role: str, model, model_version: str, total_tokens: int,
            forwarded_tokens: int, cache_reused_tokens: int,
            training_status: str = "stage0_teacher_distilled_unaligned") -> dict:
    risk_logits, category_logits = _risk_category_logits(output, role)
    if risk_logits.shape[-1] != 3:
        raise RuntimeError("Risk head must have three classes")
    risk = torch.softmax(risk_logits[0, -1].float(), dim=-1).cpu().tolist()
    category = torch.softmax(category_logits[0, -1].float(), dim=-1).cpu().tolist()
    category_map = (model.query_category_map if role == "user"
                    else model.response_category_map)
    if len(category) != len(category_map):
        raise RuntimeError("Category head and map differ")
    risk_probs = {name: float(risk[index]) for name, index in RISK_INDEX.items()}
    top_risk = max(risk_probs, key=risk_probs.get)
    categories = {category_map[index]: float(value)
                  for index, value in enumerate(category)}
    top_category = max(categories, key=categories.get)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "training_status": training_status,
        "target_role": role,
        "risk_level": top_risk,
        "risk_probabilities": risk_probs,
        "category": top_category,
        "category_distribution": categories,
        "category_semantics": "single_choice_source_head; topic is not an unsafe decision",
        "calibrated": False,
        "input_tokens_total": total_tokens,
        "input_tokens_forwarded": forwarded_tokens,
        "cache_reused_tokens": cache_reused_tokens,
    }


class PrefixClassifier:
    """One target message; every append returns a classification of the visible prefix."""

    def __init__(self, model, tokenizer, model_version: str, role: str,
                 training_status: str = "stage0_teacher_distilled_unaligned"):
        if role not in ("user", "assistant"):
            raise ValueError("Target role must be user or assistant")
        self.model, self.tokenizer = model, tokenizer
        self.model_version, self.role = model_version, role
        self.training_status = training_status
        self.ids: list[int] = []
        self.cache = DynamicCache()
        self.last_output = None
        self.total_forwarded_tokens = 0

    @torch.inference_mode()
    def append_token_ids(self, new_ids: list[int]) -> dict:
        """Forward only new IDs; attention still reads the growing historical KV."""
        if not new_ids or len(self.ids) + len(new_ids) > 8192:
            raise ValueError("Append must contain tokens and stay within 8192")
        reused = len(self.ids)
        output = self.model(input_ids=torch.tensor([new_ids], device=self.model.device),
                            past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1)
        self.ids.extend(new_ids)
        self.cache = output.past_key_values
        self.last_output = output
        self.total_forwarded_tokens += len(new_ids)
        return readout(output, self.role, self.model, self.model_version,
                       len(self.ids), len(new_ids), reused, self.training_status)

    @torch.inference_mode()
    def update_ids(self, ids: list[int]) -> dict:
        if not 0 < len(ids) <= 8192:
            raise ValueError("Expected 1–8192 input tokens")
        common = 0
        for old, new in zip(self.ids, ids):
            if old != new:
                break
            common += 1
        if common == len(ids) == len(self.ids) and self.last_output is not None:
            return readout(self.last_output, self.role, self.model,
                           self.model_version, len(ids), 0, common, self.training_status)
        common = min(common, len(ids) - 1)
        self.cache.crop(common)
        suffix = ids[common:]
        self.ids = list(self.ids[:common])
        return self.append_token_ids(suffix)

    def update_messages(self, messages: list[dict], partial: bool = True) -> dict:
        if not messages or messages[-1]["role"] != self.role:
            raise ValueError("Last message role differs from session target")
        ids = encode(self.tokenizer, messages, partial=partial)
        return self.update_ids(ids)


@torch.inference_mode()
def classify_full(model, tokenizer, model_version: str, messages: list[dict],
                  training_status: str = "stage0_teacher_distilled_unaligned") -> dict:
    ids = encode(tokenizer, messages, partial=False)
    output = model(input_ids=torch.tensor([ids], device=model.device),
                   use_cache=False, logits_to_keep=1)
    return readout(output, messages[-1]["role"], model, model_version,
                   len(ids), len(ids), 0, training_status)
