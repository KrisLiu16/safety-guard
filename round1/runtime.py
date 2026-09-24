"""Original Guard heads, MPS inference, incremental streams, and small LoRA adapters."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
LABELS = ['safe', 'unsafe', 'controversial']

def sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()

def load(dtype=torch.float32, device='mps', attention='sdpa', model_path=None):
    if device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise RuntimeError('Disable silent MPS CPU fallback before this experiment')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')
    model_path = Path(model_path) if model_path else ROOT/'model'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        local_files_only=True, torch_dtype=dtype, attn_implementation=attention).to(device).eval()
    assert model.query_risk_level_map == {0:'Safe',1:'Unsafe',2:'Controversial'}
    assert model.response_risk_level_map == model.query_risk_level_map
    if (model_path/'risk_heads_fp32.safetensors').exists():
        float_risk_heads(model)
        model.load_state_dict(load_file(str(model_path/'risk_heads_fp32.safetensors')),strict=False)
    return model, tokenizer

def encode(tokenizer, messages, partial=False):
    if not messages or messages[-1]['role'] not in ('user', 'assistant'):
        raise ValueError('Last message must have user or assistant role')
    # Use the published template, including its assistant thinking delimiters.
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    suffix = '<|im_end|>\n'
    if not text.endswith(suffix): raise ValueError('Unexpected chat template suffix')
    text = text[:-len(suffix)] if partial else text[:-1]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > 8192: raise ValueError(f'Context overflow: {len(ids)} > 8192')
    return ids

def logits(output, role):
    if role == 'user': return output.query_risk_level_logits
    if role == 'assistant': return output.risk_level_logits
    raise ValueError(role)

def forward_ids(model, ids, role, all_positions=False):
    x = torch.tensor([ids], device=model.device)
    output = model(input_ids=x, use_cache=False, logits_to_keep=0 if all_positions else 1)
    return logits(output, role)[0].float()

@torch.no_grad()
def batch_probs(model, tokenizer, conversations):
    """Right padding with explicit positions; gather each true final token."""
    encoded = [encode(tokenizer, m) for m in conversations]
    length = max(map(len, encoded))
    x = torch.full((len(encoded), length), tokenizer.pad_token_id, device=model.device, dtype=torch.long)
    mask = torch.zeros_like(x)
    for i, ids in enumerate(encoded):
        x[i, :len(ids)] = torch.tensor(ids, device=model.device)
        mask[i, :len(ids)] = 1
    output = model(input_ids=x, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp(min=0), use_cache=False)
    return [logits(output, messages[-1]['role'])[i, len(ids)-1].float().softmax(-1).cpu().tolist()
            for i, (messages, ids) in enumerate(zip(conversations, encoded))]

def bucketed_batch_probs(model, tokenizer, conversations, max_batch_size=4):
    """Bound GPU batch size, group similar lengths, then restore caller order."""
    if max_batch_size not in (1,2,4):raise ValueError('Supported local batch sizes: 1, 2, 4')
    order=sorted(range(len(conversations)),key=lambda i:len(encode(tokenizer,conversations[i])))
    results=[None]*len(conversations)
    for start in range(0,len(order),max_batch_size):
        indices=order[start:start+max_batch_size]
        values=batch_probs(model,tokenizer,[conversations[i] for i in indices])
        for i,value in zip(indices,values):results[i]=value
    return results

class Stream:
    """One conversation cache. Re-tokenization rolls back only the changed suffix."""
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        # Transformers 4.55's DynamicCache(config=...) forwards unsupported layer kwargs.
        # The no-config dynamic cache is equivalent for this model's full-attention layers.
        self.ids, self.cache, self.last = [], DynamicCache(), None
        self.processed_tokens = 0
        self.last_start = 0
        self.last_changed = False

    @torch.no_grad()
    def update_ids(self, ids, role, all_positions=False):
        if not 0 < len(ids) <= 8192: raise ValueError('Invalid context length')
        common = 0
        for a, b in zip(self.ids, ids):
            if a != b: break
            common += 1
        if common == len(ids) and common == len(self.ids) and self.last is not None:
            self.last_changed = False
            return logits(self.last, role)[0, -1].float().softmax(-1)
        # Recompute the last token if the new sequence ends at the common prefix.
        common = min(common, len(ids)-1)
        if self.cache is not None:
            self.cache.crop(common)
        new = ids[common:]
        output = self.model(input_ids=torch.tensor([new], device=self.model.device),
            past_key_values=self.cache, use_cache=True, logits_to_keep=0 if all_positions else 1)
        self.ids, self.cache, self.last = list(ids), output.past_key_values, output
        self.last_start, self.last_changed = common, True
        self.processed_tokens += len(new)
        return logits(output, role)[0, -1].float().softmax(-1)

    def update(self, messages, partial=True, all_positions=False):
        ids = encode(self.tokenizer, messages, partial=partial)
        return self.update_ids(ids, messages[-1]['role'], all_positions=all_positions)

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T @ self.B.T) * self.scale).to(x.dtype)

class FloatLinear(nn.Linear):
    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight, self.bias)

def attach_lora(model):
    for param in model.parameters(): param.requires_grad_(False)
    for layer in model.model.layers[-8:]:
        for name in ('q_proj', 'v_proj'):
            base = getattr(layer.self_attn, name)
            if not isinstance(base, LoRALinear): setattr(layer.self_attn, name, LoRALinear(base))
    float_risk_heads(model)
    for name in ('risk_level_head', 'query_risk_level_head'):
        for param in getattr(model, name).parameters(): param.requires_grad_(True)
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]

def float_risk_heads(model):
    # Shared category projection and category classifiers stay frozen.
    for name in ('risk_level_head', 'query_risk_level_head'):
        old = getattr(model, name)
        if not isinstance(old, FloatLinear):
            head = FloatLinear(old.in_features, old.out_features, bias=False, device=old.weight.device, dtype=torch.float32)
            head.weight.data.copy_(old.weight.float())
            setattr(model, name, head)

def save_adapter(model, path, metadata=None):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    adapters, heads = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        target = adapters if name.endswith(('.A', '.B')) else heads
        target[name] = param.detach().float().cpu().contiguous()
    save_file(adapters, str(path/'adapter.safetensors'))
    save_file(heads, str(path/'risk_heads.safetensors'))
    (path/'metadata.json').write_text(json.dumps(metadata or {}, ensure_ascii=False, indent=2)+'\n')

def restore_adapter(model, path):
    if getattr(model.config,'round1_adapter_merged',False):
        raise ValueError('Cannot apply an adapter again to an already merged model')
    attach_lora(model)
    state = {**load_file(str(Path(path)/'adapter.safetensors')),
             **load_file(str(Path(path)/'risk_heads.safetensors'))}
    expected = {n for n,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected: raise ValueError('Adapter parameters do not match architecture')
    model.load_state_dict(state, strict=False)
    return model.eval()

@torch.no_grad()
def merge_lora(model):
    for layer in model.model.layers[-8:]:
        for name in ('q_proj', 'v_proj'):
            module = getattr(layer.self_attn, name)
            if isinstance(module, LoRALinear):
                module.base.weight.add_((module.B.float() @ module.A.float() * module.scale).to(module.base.weight.dtype))
                setattr(layer.self_attn, name, module.base)
    return model
