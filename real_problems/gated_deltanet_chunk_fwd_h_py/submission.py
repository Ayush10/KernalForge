#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius
# Team: Kernal Forge
# Optimizations: tf32 dot precision, per-(K,V) tuned configs, removed redundant casts
from task import input_t, output_t

import torch
import helion
import helion.language as hl


KV_CONFIGS: dict[tuple[int, int], helion.Config] = {
    (64, 64): helion.Config(block_sizes=[16], num_warps=2, num_stages=3, l2_groupings=[8]),
    (64, 128): helion.Config(block_sizes=[16], num_warps=2, num_stages=2),
    (100, 100): helion.Config(block_sizes=[16], num_warps=2, num_stages=2, l2_groupings=[4]),
    (128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=1),
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
                g_chunk = g[b_idx, tc, h_idx]
                g_last_val = g[b_idx, tc.begin + C - 1, h_idx]

                h_out[b_idx, chunk_idx, h_idx, :, tv] = state

                proj = hl.dot(
                    w[b_idx, tc, h_idx, :],
                    state,
                    out_dtype=torch.float32,
                )
                diff = u[b_idx, tc, h_idx, tv] - proj

                v_out[b_idx, tc, h_idx, tv] = diff

                gated_diff = diff * torch.exp(g_last_val - g_chunk)[:, None]

                update = hl.dot(
                    k[b_idx, tc, h_idx, :].T,
                    gated_diff,
                    out_dtype=torch.float32,
                )
                state = state * torch.exp(g_last_val) + update

        return h_out, v_out

    return kernel


_KERNEL_CACHE: dict[tuple[int, int], callable] = {}


def _get_kernel(kv_shape: tuple[int, int]):
    kernel = _KERNEL_CACHE.get(kv_shape)
    if kernel is None:
        kernel = _make_kernel(KV_CONFIGS[kv_shape])
        _KERNEL_CACHE[kv_shape] = kernel
    return kernel


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    K = k.shape[-1]
    V = u.shape[-1]
    kernel = _get_kernel((K, V))
    return kernel(k, w, u, g)
