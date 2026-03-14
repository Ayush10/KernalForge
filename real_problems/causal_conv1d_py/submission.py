# Retune two smallest benchmark shapes with smaller blocks for better SM occupancy
from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple[int, int, int, int], helion.Config] = {
    (1, 64, 64, 4): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=3),
    (1, 256, 256, 3): helion.Config(block_sizes=[128, 128], num_warps=4, num_stages=3),
    (1, 128, 64, 8): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=3),
    (4, 64, 128, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=2),
    (1, 768, 512, 4): helion.Config(block_sizes=[64, 64], num_warps=2, num_stages=3, l2_groupings=[1]),
    (1, 768, 2048, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=4, l2_groupings=[2]),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[128, 128], num_warps=8, num_stages=4, l2_groupings=[8]),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[256, 64], num_warps=8, num_stages=4, l2_groupings=[8]),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[256, 128], num_warps=8, num_stages=5, l2_groupings=[8]),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B, D, S = x.shape
        W = hl.specialize(w.size(1))
        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            rs_idx = rs.index
            in_bounds = rs_idx < S
            bias_tile = hl.zeros([rd, rs], dtype=torch.float32)
            bias_tile = bias_tile + b[rd][:, None]

            if W == 4:
                # Precompute all masks once to avoid redundant boolean ops per tap
                ib_base = in_bounds[None, :]
                ge1 = (rs_idx >= 1)
                ge2 = (rs_idx >= 2)
                ge3 = (rs_idx >= 3)
                mask0 = ib_base
                mask1 = (in_bounds & ge1)[None, :]
                mask2 = (in_bounds & ge2)[None, :]
                mask3 = (in_bounds & ge3)[None, :]

                x0 = hl.load(x, [bi, rd, rs_idx - 3], extra_mask=mask3)
                x1 = hl.load(x, [bi, rd, rs_idx - 2], extra_mask=mask2)
                x2 = hl.load(x, [bi, rd, rs_idx - 1], extra_mask=mask1)
                x3 = hl.load(x, [bi, rd, rs_idx], extra_mask=mask0)
                acc = bias_tile
                acc = acc + x0 * w[rd, 0][:, None]
                acc = acc + x1 * w[rd, 1][:, None]
                acc = acc + x2 * w[rd, 2][:, None]
                acc = acc + x3 * w[rd, 3][:, None]
            elif W == 3:
                # Precompute masks for W=3
                ib_base = in_bounds[None, :]
                ge1 = (rs_idx >= 1)
                ge2 = (rs_idx >= 2)
                mask0 = ib_base
                mask1 = (in_bounds & ge1)[None, :]
                mask2 = (in_bounds & ge2)[None, :]

                x0 = hl.load(x, [bi, rd, rs_idx - 2], extra_mask=mask2)
                x1 = hl.load(x, [bi, rd, rs_idx - 1], extra_mask=mask1)
                x2 = hl.load(x, [bi, rd, rs_idx], extra_mask=mask0)
                acc = bias_tile
                acc = acc + x0 * w[rd, 0][:, None]
                acc = acc + x1 * w[rd, 1][:, None]
                acc = acc + x2 * w[rd, 2][:, None]
            elif W == 8:
                acc = bias_tile
                for tap in range(8):
                    shift = 7 - tap
                    x_tap = hl.load(
                        x,
                        [bi, rd, rs_idx - shift],
                        extra_mask=(in_bounds & (rs_idx >= shift))[None, :],
                    )
                    acc = acc + x_tap * w[rd, tap][:, None]
            else:
                acc = bias_tile
                for tap in range(W):
                    shift = W - 1 - tap
                    x_tap = hl.load(
                        x,
                        [bi, rd, rs_idx - shift],
                        extra_mask=(in_bounds & (rs_idx >= shift))[None, :],
                    )
                    acc = acc + x_tap * w[rd, tap][:, None]

            y[rb, rd, rs] = acc[None, :, :]

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)