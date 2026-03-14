#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius
# Team: KernalForge
# Fix: Group configs by (K,V) to reduce JIT compilations from 10 to 4
# Remove static_shapes so B,T,H are runtime values — fits within 420s timeout
from task import input_t, output_t

import torch
import helion
import helion.language as hl


# 4 unique (K,V) pairs across all test+benchmark shapes:
#   (64,64)   — 7 shapes
#   (64,128)  — 1 shape
#   (100,100) — 1 shape
#   (128,128) — 2 shapes
# Each group compiles ONE Triton kernel. 4 compilations × ~60s = ~240s < 420s timeout.
KV_CONFIGS: dict[tuple[int, int], helion.Config] = {
    (64, 64): helion.Config(
        num_warps=4,
        num_stages=3,
        l2_groupings=[4],
        advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf",
    ),
    (64, 128): helion.Config(num_warps=8, num_stages=3),
    (100, 100): helion.Config(num_warps=8, num_stages=4, l2_groupings=[4]),
    (128, 128): helion.Config(num_warps=8, num_stages=4, l2_groupings=[8]),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(config=config)
    def kernel(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        beta_g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)
        BH = B * H

        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            a_chunk = A[b_idx, rt, h_idx, :].to(torch.bfloat16)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            beta_g_chunk = beta_g[b_idx, rt, h_idx].to(torch.float32)
            rhs_k = (k[b_idx, rt, h_idx, :].to(torch.float32) * beta_g_chunk[:, None]).to(torch.bfloat16)
            rhs_v = (v[b_idx, rt, h_idx, :].to(torch.float32) * beta_chunk[:, None]).to(torch.bfloat16)

            w_out[b_idx, rt, h_idx, :] = hl.dot(
                a_chunk,
                rhs_k,
                out_dtype=torch.float32,
            ).to(w_out.dtype)
            u_out[b_idx, rt, h_idx, :] = hl.dot(
                a_chunk,
                rhs_v,
                out_dtype=torch.float32,
            ).to(u_out.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {kv: _make_kernel(cfg) for kv, cfg in KV_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    K = k.shape[-1]
    V = v.shape[-1]
    beta_g = beta * torch.exp(g)
    kernel = _KERNELS[(K, V)]
    return kernel(k, v, beta, A, beta_g)
