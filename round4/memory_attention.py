"""Finite-window attention with learned associative memory of older keys.

Preserves all pretrained layers. New neural parameters: two 256->32 feature
maps and one gate per query head per attention layer. No output generation.
The compressed memory is independent of history length; exact local attention
still uses the inherited Q/K/V, gate and output projections.
"""
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from fla.ops.linear_attn import chunk_linear_attn,fused_recurrent_linear_attn
from window_attention import tiled_window_attention


class MemoryCache(DynamicCache):
    def __init__(self,config,window):
        super().__init__(config=config);self.window=window;self.seen={};self.memories={}
        self.first_attention=config.layer_types.index('full_attention')
    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        size=key_states.shape[-2]
        keys,values=super().update(key_states,value_states,layer_idx,*args,**kwargs)
        self.seen[layer_idx]=self.seen.get(layer_idx,0)+size
        layer=self.layers[layer_idx]
        # Keep W, not W-1: the oldest cached key enters the memory before the
        # next query and is excluded from that query's exact local window.
        layer.keys=keys[...,-self.window:,:].clone()
        layer.values=values[...,-self.window:,:].clone()
        return keys,values
    def get_seq_length(self,layer_idx=0):return self.seen.get(layer_idx,self.seen.get(self.first_attention,0))
    def crop(self,*args,**kwargs):raise NotImplementedError('Restore a saved full state and replay for text rollback')


def features(projection,x):
    z=projection(x).float()
    return torch.cat((z.softmax(-1),(-z).softmax(-1)),dim=-1).to(x.dtype)


def memory_read(q,k,v,window,phi_q,phi_k,initial_state=None,cached=False,key_mask=None):
    # q/k/v: B,H,T,D, with possibly W cached keys preceding the new queries.
    length=q.shape[-2];history=k.shape[-2]-length
    assert 0<=history<=window
    qf=features(phi_q,q).to(v.dtype).transpose(1,2).contiguous()
    # Eviction at query i is key(history+i-W). Earlier negative indices write zero.
    zeros=min(length,window-history)
    count=length-zeros
    selected_k=k[...,:count,:];selected_v=v[...,:count,:]
    kf=features(phi_k,selected_k).to(v.dtype) if count else qf.new_empty(k.shape[0],k.shape[1],0,qf.shape[-1])
    if key_mask is not None and count:kf=kf*key_mask[:,None,:count,None].to(kf.dtype)
    kf=F.pad(kf,(0,0,zeros,0));vf=F.pad(selected_v,(0,0,zeros,0))
    groups=q.shape[1]//k.shape[1]
    if groups!=1:kf=kf.repeat_interleave(groups,1);vf=vf.repeat_interleave(groups,1)
    kf=kf.transpose(1,2).contiguous();vf=vf.transpose(1,2).contiguous()
    fn=fused_recurrent_linear_attn if not torch.is_grad_enabled() and length<=32 else chunk_linear_attn
    kv_init,z_init=initial_state if initial_state is not None else (None,None)
    numerator,kv_state=fn(qf,kf,vf,scale=1.,initial_state=kv_init,
                          output_final_state=cached,normalize=False)
    # Accumulate the denominator in FP32. A BF16 running sum can stop changing
    # under single-token updates once history is long enough.
    z=kf.float().cumsum(dim=1)
    if z_init is not None:z=z+z_init
    denominator=(qf.float()*z).sum(-1,keepdim=True).clamp_min(1e-6)
    output=(numerator.float()/denominator).to(v.dtype)
    state=(kv_state,z[:,-1:].clone()) if cached else None
    return output.transpose(1,2),state


def attention_forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
    shape=hidden_states.shape[:-1];head_shape=(*shape,-1,self.head_dim)
    query,gate=torch.chunk(self.q_proj(hidden_states).view(*shape,-1,self.head_dim*2),2,dim=-1)
    gate=gate.reshape(*shape,-1)
    q=self.q_norm(query.reshape(head_shape)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1,2)
    v=self.v_proj(hidden_states).view(head_shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    initial=None
    if past_key_values is not None:
        if not isinstance(past_key_values,MemoryCache):raise TypeError('MemoryCache required')
        initial=past_key_values.memories.get(self.layer_idx)
        k,v=past_key_values.update(k,v,self.layer_idx)
    local=tiled_window_attention(q,k,v,self.guard_window,128,attention_mask,self.scaling,
                                 self.attention_dropout if self.training else 0.)
    older,state=memory_read(q,k,v,self.guard_window,self.memory_phi_q,self.memory_phi_k,
                           initial,past_key_values is not None,attention_mask)
    if past_key_values is not None:
        # FLA may return the final normalizer as a view into the whole chunk's
        # cumulative sums. Retain only the actual state, not that backing buffer.
        past_key_values.memories[self.layer_idx]=tuple(
            t.clone() if t.untyped_storage().nbytes()>t.numel()*t.element_size() else t for t in state)
    mix=self.memory_gate.tanh().to(local.dtype)[None,:,None,None]
    # No old keys exist for the first W queries. Do not attenuate the exact local path.
    if past_key_values is None:
        exists=(torch.arange(q.shape[-2],device=q.device)>=self.guard_window)[None,None,:,None]
    else:
        seen=past_key_values.seen[self.layer_idx]
        exists=(torch.arange(seen-q.shape[-2],seen,device=q.device)>=self.guard_window)[None,None,:,None]
    output=local+mix*exists*(older-local)
    output=output.transpose(1,2).reshape(*shape,-1).contiguous()
    return self.o_proj(output*torch.sigmoid(gate)),None


def backbone_forward(self,input_ids=None,attention_mask=None,past_key_values=None,use_cache=False,**kwargs):
    if isinstance(attention_mask,dict):raise ValueError('Only a 2D padding mask is supported')
    if use_cache or past_key_values is not None:
        if self.training:raise ValueError('Cached training needs explicit truncated BPTT')
        if attention_mask is not None and not bool(torch.all(attention_mask==1)):raise ValueError('Cached padding unsupported')
        if past_key_values is None:past_key_values=MemoryCache(self.config,self.guard_window)
        if not isinstance(past_key_values,MemoryCache):raise TypeError('MemoryCache required')
        masks={'full_attention':None,'linear_attention':None}
    else:masks={'full_attention':attention_mask,'linear_attention':attention_mask}
    return self.memory_original_forward(input_ids=input_ids,attention_mask=masks,
               past_key_values=past_key_values,use_cache=use_cache,**kwargs)


def attach_memory(backbone,window=512):
    if not hasattr(backbone,'memory_original_forward'):
        backbone.memory_original_forward=backbone.forward
        for layer in backbone.layers:
            if not hasattr(layer,'self_attn'):continue
            a=layer.self_attn
            a.memory_phi_q=nn.Linear(a.head_dim,32,bias=False,device=a.q_proj.weight.device,dtype=torch.float32)
            a.memory_phi_k=nn.Linear(a.head_dim,32,bias=False,device=a.q_proj.weight.device,dtype=torch.float32)
            a.memory_gate=nn.Parameter(torch.zeros(backbone.config.num_attention_heads,device=a.q_proj.weight.device))
            nn.init.orthogonal_(a.memory_phi_q.weight);nn.init.orthogonal_(a.memory_phi_k.weight)
            a.forward=MethodType(attention_forward,a)
        backbone.forward=MethodType(backbone_forward,backbone)
    backbone.guard_window=window
    for layer in backbone.layers:
        if hasattr(layer,'self_attn'):layer.self_attn.guard_window=window


def memory_state_bytes(cache):
    stores={}
    for state in cache.memories.values():
        for tensor in state:
            if tensor is not None:
                storage=tensor.untyped_storage();stores[storage.data_ptr()]=storage.nbytes()
    return sum(stores.values())
