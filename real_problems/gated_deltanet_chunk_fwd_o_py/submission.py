#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius
# Team: Kernal Forge
# Optimizations: per-(K,V) tuned configs, lazy compilation
from task import input_t, output_t

import torch
import helion
import helion.language as hl


FULL_CHUNK_CONFIGS: dict[tuple[int, int], helion.Config] = {
    (64, 64): helion.Config(num_warps=4, num_stages=2),
}

BLOCKED_CONFIGS: dict[tuple[int, int], helion.Config] = {
    (64, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (100, 100): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (128, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
}


def _make_full_chunk_kernel(config: helion.Config):
    @helion.kernel(dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        C = 64
        NT = T // C
        K = hl.specialize(K)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, chunk_idx in hl.grid([BH, NT]):
            b_idx = flat_bh // H
            h_idx = flat_bh % H
            t_start = chunk_idx * C
            t_stop = t_start + C
            state_chunk = state[b_idx, chunk_idx, h_idx, :, :].to(torch.float32)

            for tc in hl.tile(t_start, t_stop, block_size=C):
                g_chunk = g[b_idx, tc, h_idx].to(torch.float32)
                q_chunk = q[b_idx, tc, h_idx, :].to(torch.float32)
                k_chunk = k[b_idx, tc, h_idx, :].to(torch.float32)
                v_chunk = v[b_idx, tc, h_idx, :].to(torch.float32)

                sim = hl.dot(q_chunk, k_chunk.T, out_dtype=torch.float32)
                sim = sim * torch.exp(g_chunk[:, None] - g_chunk[None, :])
                causal_mask = (tc.index[:, None] >= tc.index[None, :]).to(torch.float32)
                local_out = hl.dot(sim * causal_mask, v_chunk, out_dtype=torch.float32)
                global_out = hl.dot(q_chunk, state_chunk, out_dtype=torch.float32) * torch.exp(g_chunk)[:, None]
                out[b_idx, tc, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


def _make_blocked_kernel(config: helion.Config):
    @helion.kernel(dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        C = 64
        NT = T // C
        K = hl.specialize(K)
        QB = 16

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, chunk_idx in hl.grid([BH, NT]):
            b_idx = flat_bh // H
            h_idx = flat_bh % H
            t_start = chunk_idx * C
            t_stop = t_start + C

            for tc in hl.tile(t_start, t_stop, block_size=C):
                g_chunk = g[b_idx, tc, h_idx].to(torch.float32)
                k_chunk = k[b_idx, tc, h_idx, :].to(torch.float32)
                v_idx = tc.index

                for vv in hl.tile(v.shape[-1], block_size=None):
                    state_tile = state[b_idx, chunk_idx, h_idx, :, vv].to(torch.float32)
                    v_tile = v[b_idx, tc, h_idx, vv].to(torch.float32)

                    for tq in hl.tile(t_start, t_stop, block_size=QB):
                        q_gate = g[b_idx, tq, h_idx].to(torch.float32)
                        q_chunk = q[b_idx, tq, h_idx, :].to(torch.float32)
                        sim = hl.dot(q_chunk, k_chunk.T, out_dtype=torch.float32)
                        sim = sim * torch.exp(q_gate[:, None] - g_chunk[None, :])
                        causal_mask = (tq.index[:, None] >= v_idx[None, :]).to(torch.float32)
                        local_out = hl.dot(sim * causal_mask, v_tile, out_dtype=torch.float32)
                        global_out = hl.dot(q_chunk, state_tile, out_dtype=torch.float32) * torch.exp(q_gate)[:, None]
                        out[b_idx, tq, h_idx, vv] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_FULL_KERNEL_CACHE: dict[tuple[int, int], callable] = {}
_BLOCKED_KERNEL_CACHE: dict[tuple[int, int], callable] = {}


def _get_kernel(kv_shape: tuple[int, int]):
    if kv_shape in FULL_CHUNK_CONFIGS:
        kernel = _FULL_KERNEL_CACHE.get(kv_shape)
        if kernel is None:
            kernel = _make_full_chunk_kernel(FULL_CHUNK_CONFIGS[kv_shape])
            _FULL_KERNEL_CACHE[kv_shape] = kernel
        return kernel

    kernel = _BLOCKED_KERNEL_CACHE.get(kv_shape)
    if kernel is None:
        kernel = _make_blocked_kernel(BLOCKED_CONFIGS[kv_shape])
        _BLOCKED_KERNEL_CACHE[kv_shape] = kernel
    return kernel


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    K = q.shape[-1]
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _get_kernel((K, V))
    return kernel(q, k, v_new, h, g, scale)
