"""Cross-session continuous-batching engine for the Qwen3.5 H24 window classifier.

One forward per scheduler tick: the new tokens of every active session are packed into a
single variable-length sequence. Per-session state lives in slot-indexed pools:
  - 18 GDN layers: recurrent state [slots, H, K, V] (fp32) and short-conv state [slots, 3, C];
  - 6 window-512 attention layers: rotated K/V ring buffers [slots, W, Hkv, D] plus one shared
    key-position table [slots, W] (all attention layers see the same token positions).
GDN uses FLA's chunk kernel with cu_seqlens and per-sequence initial/final states; the short
convolution and window attention are varlen-ized with index gathers and position masks.

Streaming-only contract (round 6 decision): there is no second whole-input path and no
bit-exactness guarantee across different chunkings. BPE stability is the caller's job: feed
only tokens that can no longer change (hold back the last k tokens until more text or flush).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def varlen_causal_conv(new_x, state, lens, weight):
    """Depthwise causal conv over packed sequences, each continuing from its own state.

    new_x: [T, C] packed new inputs; state: [N, K-1, C] last K-1 inputs per sequence;
    lens: python list of N lengths (sum T); weight: [C, K].
    Returns (silu(conv) [T, C] in new_x.dtype, new_state [N, K-1, C]).
    """
    n, keep, c = state.shape
    k = keep + 1
    device = new_x.device
    lens_t = torch.tensor(lens, device=device)
    seg = lens_t + keep                                   # extended segment length per sequence
    seg_start = torch.cumsum(seg, 0) - seg                # start of each extended segment
    ext = torch.empty(int(seg.sum()), c, dtype=new_x.dtype, device=device)
    state_rows = (seg_start[:, None] + torch.arange(keep, device=device)[None]).reshape(-1)
    ext[state_rows] = state.reshape(-1, c).to(new_x.dtype)
    seq_of_token = torch.repeat_interleave(torch.arange(n, device=device), lens_t)
    token_start = torch.cumsum(lens_t, 0) - lens_t
    offset_in_seq = torch.arange(new_x.shape[0], device=device) - token_start[seq_of_token]
    new_rows = seg_start[seq_of_token] + keep + offset_in_seq
    ext[new_rows] = new_x
    window = new_rows[:, None] - keep + torch.arange(k, device=device)[None]          # [T, K]
    out = (ext[window].float() * weight.t().float()[None]).sum(1)
    last = seg_start + seg - 1
    new_state = ext[last[:, None] - keep + 1 + torch.arange(keep, device=device)[None]]  # [N, K-1, C]
    return F.silu(out).to(new_x.dtype), new_state


def varlen_window_attention(q, k, v, positions, lens, ring_k, ring_v, ring_pos, window, scale):
    """Window attention of packed new tokens over each session's ring buffer plus its own new keys.

    q: [T, Hq, D]; k, v: [T, Hkv, D] (already rotated/normed); positions: [T] absolute positions;
    lens: list of N lengths; ring_k/ring_v: [N, W, Hkv, D]; ring_pos: [N, W] (-1 = empty).
    Returns [T, Hq, D].
    """
    n, device = len(lens), q.device
    lens_t = torch.tensor(lens, device=device)
    lmax = max(lens)
    seq_of_token = torch.repeat_interleave(torch.arange(n, device=device), lens_t)
    token_start = torch.cumsum(lens_t, 0) - lens_t
    col = torch.arange(q.shape[0], device=device) - token_start[seq_of_token]
    hq, hkv, d = q.shape[1], k.shape[1], k.shape[2]
    qp = torch.zeros(n, lmax, hq, d, dtype=q.dtype, device=device)
    kp = torch.zeros(n, lmax, hkv, d, dtype=k.dtype, device=device)
    vp = torch.zeros(n, lmax, hkv, d, dtype=v.dtype, device=device)
    qpos = torch.full((n, lmax), -1, dtype=torch.long, device=device)
    qp[seq_of_token, col] = q
    kp[seq_of_token, col] = k
    vp[seq_of_token, col] = v
    qpos[seq_of_token, col] = positions
    keys = torch.cat([ring_k.to(k.dtype), kp], 1).transpose(1, 2)          # [N, Hkv, W+L, D]
    values = torch.cat([ring_v.to(v.dtype), vp], 1).transpose(1, 2)
    key_pos = torch.cat([ring_pos, qpos], 1)                                 # [N, W+L]
    qq, kk = qpos[:, :, None], key_pos[:, None, :]
    allowed = (kk >= 0) & (kk <= qq) & (kk > qq - window)                    # [N, L, W+L]
    allowed |= (qq < 0)            # padded query rows: attend anywhere, output discarded (no NaN)
    out = F.scaled_dot_product_attention(qp.transpose(1, 2), keys, values, attn_mask=allowed[:, None],
                                         scale=scale, enable_gqa=hq != hkv)   # [N, Hq, L, D]
    return out.transpose(1, 2)[seq_of_token, col]


class BatchedStreamEngine:
    def __init__(self, model, max_slots, window=512, gdn_kernel="chunk"):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        self.model, self.backbone = model, model.backbone
        cfg = self.backbone.config
        self.types = list(cfg.layer_types[:cfg.num_hidden_layers])
        self.layers = list(self.backbone.layers[:cfg.num_hidden_layers])
        self.gdn_index = {i: n for n, i in enumerate(i for i, t in enumerate(self.types) if t == "linear_attention")}
        self.att_index = {i: n for n, i in enumerate(i for i, t in enumerate(self.types) if t == "full_attention")}
        self.window, self.max_slots = window, max_slots
        self.kernel = {"chunk": chunk_gated_delta_rule, "recurrent": fused_recurrent_gated_delta_rule}[gdn_kernel]
        gdn = self.layers[next(iter(self.gdn_index))].linear_attn
        att = self.layers[next(iter(self.att_index))].self_attn
        device, bf16 = "cuda", torch.bfloat16
        self.rec = torch.zeros(len(self.gdn_index), max_slots, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim,
                               dtype=torch.float32, device=device)
        self.conv = torch.zeros(len(self.gdn_index), max_slots, gdn.conv_kernel_size - 1, gdn.conv_dim,
                                dtype=bf16, device=device)
        self.ring_k = torch.zeros(len(self.att_index), max_slots, window, cfg.num_key_value_heads, att.head_dim,
                                  dtype=bf16, device=device)
        self.ring_v = torch.zeros_like(self.ring_k)
        self.ring_pos = torch.full((max_slots, window), -1, dtype=torch.long, device=device)
        self.length = [0] * max_slots

    def state_bytes_per_session(self):
        tensors = (self.rec, self.conv, self.ring_k, self.ring_v, self.ring_pos)
        return sum(t.element_size() * t.numel() for t in tensors) // self.max_slots

    def reset(self, slots=None):
        slots = range(self.max_slots) if slots is None else slots
        index = torch.tensor(list(slots), device="cuda")
        self.rec[:, index] = 0
        self.conv[:, index] = 0
        self.ring_k[:, index] = 0
        self.ring_v[:, index] = 0
        self.ring_pos[index] = -1
        for s in slots:
            self.length[s] = 0

    @torch.inference_mode()
    def step(self, requests, roles=("assistant",)):
        """requests: list of (slot, token_ids). Returns {role: [T, 3] probs} over the packed new tokens."""
        slots_list = [s for s, _ in requests]
        if len(set(slots_list)) != len(slots_list):
            raise ValueError("a session may appear once per tick")
        lens = [len(ids) for _, ids in requests]
        if min(lens) < 1:
            raise ValueError("empty append")
        device = "cuda"
        slots = torch.tensor(slots_list, device=device)
        ids = torch.tensor([t for _, row in requests for t in row], device=device)
        starts = torch.tensor([self.length[s] for s in slots_list], device=device)
        lens_t = torch.tensor(lens, device=device)
        seq_of_token = torch.repeat_interleave(torch.arange(len(lens), device=device), lens_t)
        token_start = torch.cumsum(lens_t, 0) - lens_t
        positions = starts[seq_of_token] + torch.arange(ids.shape[0], device=device) - token_start[seq_of_token]
        cu = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lens_t, 0)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.backbone.embed_tokens(ids)[None]
            cos, sin = self.backbone.rotary_emb(hidden, positions.view(1, 1, -1).expand(3, 1, -1))
            for i, layer in enumerate(self.layers):
                residual = hidden
                normed = layer.input_layernorm(hidden)
                if i in self.gdn_index:
                    mixed = self._gdn(layer.linear_attn, normed[0], self.gdn_index[i], slots, lens, cu)
                else:
                    mixed = self._attention(layer.self_attn, normed[0], self.att_index[i], slots, lens,
                                            positions, cos, sin)
                hidden = residual + mixed[None]
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = self.backbone.norm(hidden)[0]
            probs = {role: torch.softmax(self.model.readout(hidden, role)[0].float(), -1) for role in roles}
        # commit ring positions once (shared by all attention layers); keep the last W tokens per session
        keep = (positions - starts[seq_of_token]) >= (lens_t[seq_of_token] - self.window)
        self.ring_pos[slots[seq_of_token][keep], positions[keep] % self.window] = positions[keep]
        for s, n in zip(slots_list, lens):
            self.length[s] += n
        return probs

    def _gdn(self, mod, x, g_index, slots, lens, cu):
        t = x.shape[0]
        qkv = mod.in_proj_qkv(x)
        z = mod.in_proj_z(x).view(t, mod.num_v_heads, mod.head_v_dim)
        b, a = mod.in_proj_b(x), mod.in_proj_a(x)
        conv_out, new_conv = varlen_causal_conv(qkv, self.conv[g_index][slots], lens, mod.conv1d.weight[:, 0, :])
        self.conv[g_index, slots] = new_conv.to(self.conv.dtype)
        q, k, v = torch.split(conv_out, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(1, t, -1, mod.head_k_dim).contiguous()
        k = k.reshape(1, t, -1, mod.head_k_dim).contiguous()
        v = v.reshape(1, t, -1, mod.head_v_dim).contiguous()
        beta = b.sigmoid()[None]
        g = (-mod.A_log.float().exp() * F.softplus(a.float() + mod.dt_bias))[None]
        if mod.num_v_heads // mod.num_k_heads > 1:
            q = q.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
            k = k.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
        out, final = self.kernel(q, k, v, g=g, beta=beta, initial_state=self.rec[g_index][slots],
                                 output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        self.rec[g_index, slots] = final.float()
        out = mod.norm(out.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim)).reshape(t, -1)
        return mod.out_proj(out)

    def _attention(self, mod, x, a_index, slots, lens, positions, cos, sin):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        t = x.shape[0]
        query, gate = torch.chunk(mod.q_proj(x).view(t, -1, mod.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(t, -1)
        query = mod.q_norm(query)
        key = mod.k_norm(mod.k_proj(x).view(t, -1, mod.head_dim))
        value = mod.v_proj(x).view(t, -1, mod.head_dim)
        query, key = apply_rotary_pos_emb(query.transpose(0, 1)[None], key.transpose(0, 1)[None], cos, sin)
        query, key = query[0].transpose(0, 1), key[0].transpose(0, 1)                 # [T, H, D]
        out = varlen_window_attention(query, key, value, positions, lens, self.ring_k[a_index][slots],
                                      self.ring_v[a_index][slots], self.ring_pos[slots], self.window, mod.scaling)
        # write this layer's new K/V (last W tokens of each session) into its ring
        lens_t = torch.tensor(lens, device=x.device)
        seq_of_token = torch.repeat_interleave(torch.arange(len(lens), device=x.device), lens_t)
        token_start = torch.cumsum(lens_t, 0) - lens_t
        offset = torch.arange(t, device=x.device) - token_start[seq_of_token]
        keep = offset >= (lens_t[seq_of_token] - self.window)
        ring_slot, ring_col = slots[seq_of_token][keep], positions[keep] % self.window
        self.ring_k[a_index, ring_slot, ring_col] = key[keep].to(self.ring_k.dtype)
        self.ring_v[a_index, ring_slot, ring_col] = value[keep].to(self.ring_v.dtype)
        out = out.reshape(t, -1) * torch.sigmoid(gate)
        return mod.o_proj(out)
