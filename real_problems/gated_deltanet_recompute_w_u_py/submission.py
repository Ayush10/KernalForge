# Precompute beta_g = beta * exp(g) on host to eliminate exp() inside kernel
from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple[int, int, int, int, int], helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3, l2_groupings=[4]),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=4, l2_groupings=[4]),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=4, l2_groupings=[4]),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=4, l2_groupings=[8]),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=5, l2_groupings=[8]),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=4, l2_groupings=[8]),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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

            a_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            beta_g_chunk = beta_g[b_idx, rt, h_idx].to(torch.float32)

            for rk in hl.tile(K, block_size=None):
                rhs_k = k[b_idx, rt, h_idx, rk].to(torch.float32) * beta_g_chunk[:, None]
                w_out[b_idx, rt, h_idx, rk] = hl.dot(
                    a_chunk,
                    rhs_k,
                    out_dtype=torch.float32,
                ).to(w_out.dtype)

            for rv in hl.tile(V, block_size=None):
                rhs_v = v[b_idx, rt, h_idx, rv].to(torch.float32) * beta_chunk[:, None]
                u_out[b_idx, rt, h_idx, rv] = hl.dot(
                    a_chunk,
                    rhs_v,
                    out_dtype=torch.float32,
                ).to(u_out.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    # Precompute beta * exp(g) on device before kernel launch
    beta_g = beta * torch.exp(g)
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, beta_g)