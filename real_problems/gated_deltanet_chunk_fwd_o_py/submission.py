#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius
from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple[int, int, int, int, int], helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=1),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2, l2_groupings=[4]),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=3, l2_groupings=[4]),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[8], num_warps=4, num_stages=1, l2_groupings=[4]),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=1, l2_groupings=[8]),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=1, l2_groupings=[8]),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[32], num_warps=8, num_stages=3, l2_groupings=[8]),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        NT = T // C
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, chunk_idx in hl.grid([BH, NT]):
            b_idx = flat_bh // H
            h_idx = flat_bh % H
            t_start = chunk_idx * C
            t_stop = t_start + C
            state_chunk = state[b_idx, chunk_idx, h_idx, :, :].to(torch.float32)

            for chunk_t in hl.tile(t_start, t_stop, block_size=C):
                g_chunk = g[b_idx, chunk_t, h_idx].to(torch.float32)
                k_chunk = k[b_idx, chunk_t, h_idx, :].to(torch.float32)
                v_chunk = v[b_idx, chunk_t, h_idx, :].to(torch.float32)

                for tq in hl.tile(t_start, t_stop, block_size=None):
                    q_gate = g[b_idx, tq, h_idx].to(torch.float32)
                    q_chunk = q[b_idx, tq, h_idx, :].to(torch.float32)
                    sim = hl.dot(q_chunk, k_chunk.T, out_dtype=torch.float32)
                    sim = sim * torch.exp(q_gate[:, None] - g_chunk[None, :])
                    causal_mask = (tq.index[:, None] >= chunk_t.index[None, :]).to(torch.float32)
                    sim = sim * causal_mask
                    local_out = hl.dot(
                        sim,
                        v_chunk,
                        out_dtype=torch.float32,
                    )
                    global_out = hl.dot(
                        q_chunk,
                        state_chunk,
                        out_dtype=torch.float32,
                    ) * torch.exp(q_gate)[:, None]
                    out[b_idx, tq, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(q, k, v_new, h, g, scale)
