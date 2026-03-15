#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius
# Team: KernalForge
# Fix: Group configs by (K,V) to reduce JIT compilations from 10 to 4
from task import input_t, output_t

import torch
import helion
import helion.language as hl


KV_CONFIGS: dict[tuple[int, int], helion.Config] = {
    (64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=2),
    (64, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=2),
    (100, 100): helion.Config(block_sizes=[16], num_warps=2, num_stages=3),
    (128, 128): helion.Config(block_sizes=[16], num_warps=2, num_stages=2),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(config=config)
    def kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        NT = T // C
        K = hl.specialize(K)
        V = hl.specialize(V)

        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)
        BH = B * H

        for flat_bh, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                g_chunk = g[b_idx, tc, h_idx].to(torch.float32)
                g_last = g[b_idx, tc.begin + C - 1, h_idx].to(torch.float32)

                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(h_out.dtype)

                proj = hl.dot(
                    w[b_idx, tc, h_idx, :].to(torch.float32),
                    state,
                    out_dtype=torch.float32,
                )
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj

                gated_diff = diff * torch.exp(g_last - g_chunk)[:, None]

                v_out[b_idx, tc, h_idx, tv] = diff.to(v_out.dtype)

                update = hl.dot(
                    k[b_idx, tc, h_idx, :].to(torch.float32).T,
                    gated_diff,
                    out_dtype=torch.float32,
                )
                state = state * torch.exp(g_last) + update

        return h_out, v_out

    return kernel


_KERNELS = {kv: _make_kernel(cfg) for kv, cfg in KV_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    K = k.shape[-1]
    V = u.shape[-1]
    kernel = _KERNELS[(K, V)]
    return kernel(k, w, u, g)
