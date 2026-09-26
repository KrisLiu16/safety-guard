"""L20 (dev pod): every fused_kernels op against the PyTorch code it replaces in the v6 engine, bit for bit, on the
Round5 weights and several (sessions, tokens) buckets. Prints one line per op and shape; exits 1 on any mismatch."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/bundle")
import standalone_model  # noqa: E402
from fused_kernels import (add_square, conv_silu_update, gated_finish, gdn_slot_v8, norm_rope, ring_attention_v8,  # noqa: E402
                           rms_finish, silu_mul, square_rows)
from gdn_slot_kernel import gdn_slot_recurrent  # noqa: E402
from ring_attention_kernel import ring_attention  # noqa: E402

SHAPES = ((1, 1), (7, 1), (77, 1), (96, 1), (5, 2), (33, 4), (12, 16), (3, 64))
FAILED = []


def check(name, shape, got, ref):
    same = got.shape == ref.shape and torch.equal(got, ref)
    diff = float((got.float() - ref.float()).abs().max()) if got.shape == ref.shape else float("nan")
    print(f"{'ok  ' if same else 'FAIL'} {name:28s} {str(shape):10s} max|d|={diff:.3g}", flush=True)
    if not same:
        FAILED.append((name, shape))


def main():
    torch.manual_seed(0)
    _, model, _, _ = standalone_model.load_bundle("/work/bundle")
    b = model.backbone
    gdn_layer = next(l for l in b.layers if hasattr(l, "linear_attn"))
    att_layer = next(l for l in b.layers if hasattr(l, "self_attn"))
    g, a = gdn_layer.linear_attn, att_layer.self_attn
    dev, bf16 = "cuda", torch.bfloat16
    ac = torch.autocast("cuda", dtype=bf16)
    with torch.inference_mode(), ac:
        for nb, lb in SHAPES:
            shape = (nb, lb)
            lens = torch.randint(1, lb + 1, (nb,), device=dev)
            lens[0] = lb
            if nb > 2:
                lens[-1] = 0                                              # a padded row
            slots = torch.randperm(600, device=dev)[:nb]
            slots[lens == 0] = 600                                          # padded rows use the scratch slot
            starts = torch.randint(0, 3000, (nb,), device=dev)

            # RMSNorm with residual add, and without
            res = torch.randn(nb, lb, 1024, device=dev).to(bf16) * 3
            mix = torch.randn(nb, lb, 1024, device=dev).to(bf16)
            norm = gdn_layer.post_attention_layernorm
            w1 = 1.0 + norm.weight.float()
            h, sq = add_square(res, mix)
            check("add", shape, h, res + mix)
            check("rmsnorm(add)", shape, rms_finish(h, sq.mean(-1, keepdim=True), w1, norm.eps), norm(res + mix))
            _, sq0 = add_square(res)
            check("rmsnorm", shape, rms_finish(res, sq0.mean(-1, keepdim=True), w1, norm.eps), norm(res))

            # MLP activation
            x = torch.randn(nb, lb, 1024, device=dev).to(bf16)
            gate_o, up_o = gdn_layer.mlp.gate_proj(x), gdn_layer.mlp.up_proj(x)
            check("silu_mul", shape, silu_mul(gate_o, up_o), gdn_layer.mlp.act_fn(gate_o) * up_o)

            # GDN: conv + silu + conv-state update
            qkv = g.in_proj_qkv(x)
            pool = (torch.randn(601, 3, g.conv_dim, device=dev) * 2).to(bf16)
            pool_ref = pool.clone()
            packed = conv_silu_update(qkv, pool, g.conv1d.weight.to(bf16).contiguous(), slots, lens)
            ext = torch.cat([pool_ref[slots].to(qkv.dtype), qkv], 1)
            conv = F.conv1d(ext.transpose(1, 2), g.conv1d.weight, groups=g.conv_dim)
            conv = F.silu(conv.float()).to(qkv.dtype).transpose(1, 2)
            index = (lens[:, None] + torch.arange(3, device=dev)[None])[:, :, None].expand(-1, -1, ext.shape[2])
            new_hist = torch.gather(ext, 1, index).to(pool_ref.dtype)
            keep = lens > 0
            pool_ref[slots[keep]] = new_hist[keep]
            check("conv_silu", shape, packed, conv.contiguous())
            check("conv_state", shape, pool, pool_ref)

            # GDN recurrence: v8 I/O against the v6 kernel on contiguous q/k/v
            q, k, v = torch.split(packed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
            q = q.reshape(nb, lb, -1, g.head_k_dim).contiguous()
            k = k.reshape(nb, lb, -1, g.head_k_dim).contiguous()
            v = v.reshape(nb, lb, -1, g.head_v_dim).contiguous()
            bb, aa = g.in_proj_b(x).contiguous(), g.in_proj_a(x).contiguous()
            state = (torch.randn(601, g.num_v_heads, g.head_k_dim, g.head_v_dim, device=dev) * 0.05).half()
            state_ref = state.clone()
            o, sq_o = gdn_slot_v8(packed, g.key_dim, g.num_k_heads, g.head_k_dim, g.num_v_heads, g.head_v_dim, aa, bb,
                                  g.A_log, g.dt_bias, state, slots, lens, bv=32, num_warps=2)
            o_ref = gdn_slot_recurrent(q, k, v, aa, bb, g.A_log, g.dt_bias, state_ref, slots, lens, bv=32, num_warps=2)
            check("gdn_out", shape, o, o_ref)
            check("gdn_state", shape, state, state_ref)
            check("gdn_squares", shape, sq_o, o_ref.float().pow(2))

            # gated RMSNorm
            z = g.in_proj_z(x).reshape(nb, lb, g.num_v_heads, g.head_v_dim)
            var = sq_o.view(-1, g.head_v_dim).mean(-1, keepdim=True)
            got = gated_finish(o.view(-1, g.head_v_dim), var, g.norm.weight, z.reshape(-1, g.head_v_dim).contiguous(),
                               g.norm.variance_epsilon)
            check("gated_norm", shape, got, g.norm(o_ref.reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)))

            # attention: q/k norm + rope + layouts, then ring attention with gate and ring write
            hq, hkv, d = 8, 2, a.head_dim
            pos = starts[:, None] + torch.arange(lb, device=dev)[None]
            cos, sin = b.rotary_emb(x, pos[None].expand(3, nb, lb))
            cos, sin = cos.contiguous(), sin.contiguous()
            qp, kp, vp = a.q_proj(x), a.k_proj(x), a.v_proj(x)
            query, gate = torch.chunk(qp.view(nb, lb, -1, d * 2), 2, dim=-1)
            gate = gate.reshape(nb, lb, -1)
            query = a.q_norm(query).transpose(1, 2)
            key = a.k_norm(kp.view(nb, lb, -1, d)).transpose(1, 2)
            value = vp.view(nb, lb, -1, d).transpose(1, 2)
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            q_ref = query.to(bf16).reshape(nb, hkv, (hq // hkv) * lb, d).contiguous()
            k_ref, v_ref = key.to(bf16).contiguous(), value.to(bf16).contiguous()
            sq_q = square_rows(qp, nb * lb, hq, hq * 2 * d, 2 * d, d)
            sq_k = square_rows(kp, nb * lb, hkv, hkv * d, d, d)
            var_q = sq_q.view(nb, lb, hq, d).mean(-1, keepdim=True)
            var_k = sq_k.view(nb, lb, hkv, d).mean(-1, keepdim=True)
            q8 = norm_rope(qp, var_q, 1.0 + a.q_norm.weight.float(), cos, sin, a.q_norm.eps, nb, lb, hq, hq // hkv,
                           hq * 2 * d, 2 * d, d)
            k8 = norm_rope(kp, var_k, 1.0 + a.k_norm.weight.float(), cos, sin, a.k_norm.eps, nb, lb, hkv, 1,
                           hkv * d, d, d)
            check("q_norm_rope", shape, q8, q_ref)
            check("k_norm_rope", shape, k8, k_ref)

            w = 512
            ring_k = (torch.randn(601, hkv, w + 1, d, device=dev)).to(bf16)
            ring_v = (torch.randn(601, hkv, w + 1, d, device=dev)).to(bf16)
            ring_pos = torch.full((601, w + 1), -1, dtype=torch.long, device=dev)
            for i in range(nb):                                             # a filled ring up to each session's start
                s0 = int(starts[i])
                p = torch.arange(max(0, s0 - w), s0, device=dev)
                ring_pos[slots[i], p % w] = p
            rk_ref, rv_ref = ring_k.clone(), ring_v.clone()
            out_ref = ring_attention(q_ref, rk_ref, rv_ref, k_ref, v_ref, ring_pos, slots, starts, lens, a.scaling, w)
            out_ref = out_ref.reshape(nb, hq, lb, d).transpose(1, 2).reshape(nb, lb, -1) * torch.sigmoid(gate)
            valid = torch.arange(lb, device=dev)[None] < lens[:, None]
            ring_col = torch.where(valid, pos % w, w)
            rk_ref[slots[:, None], :, ring_col] = k_ref.transpose(1, 2)
            rv_ref[slots[:, None], :, ring_col] = v_ref.transpose(1, 2)
            write = lb == 1
            out_same = ring_attention_v8(q_ref, ring_k.clone(), ring_v.clone(), k_ref, vp, qp, ring_pos, slots, starts, lens,
                                         a.scaling, w, write)
            check("ring_out_gated(same q/k)", shape, out_same, out_ref)
            out8 = ring_attention_v8(q8, ring_k, ring_v, k8, vp, qp, ring_pos, slots, starts, lens, a.scaling, w, write)
            if not write:
                ring_k[slots[:, None], :, ring_col] = k8.transpose(1, 2)
                ring_v[slots[:, None], :, ring_col] = vp.view(nb, lb, hkv, d)
            check("ring_out_gated", shape, out8, out_ref)
            real = slots[lens > 0]           # column W is the dump column: padded writes race there and it is never read
            check("ring_k_written", shape, ring_k[real][:, :, :w], rk_ref[real][:, :, :w])
            check("ring_v_written", shape, ring_v[real][:, :, :w], rv_ref[real][:, :, :w])
    print("FAILED" if FAILED else "ALL BIT-IDENTICAL", FAILED)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
