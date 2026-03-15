#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius
from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Autotuned per-shape configs via MultiFidelitySearch + PatternSearch on B200
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (N = num_tokens * hidden_dim // group_size, gsz = group_size)
    (1, 256, 64): helion.Config(block_sizes=[4], num_warps=4, num_stages=1),
    (4, 512, 128): helion.Config(block_sizes=[4], num_warps=4, num_stages=1),
    (16, 1024, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=1),
    (1, 4096, 128): helion.Config(block_sizes=[16], num_warps=8, num_stages=1, reduction_loops=[64]),
    (8, 4096, 128): helion.Config(block_sizes=[8], num_warps=1, num_stages=1),
    # Benchmark shapes - autotuned
    (16, 4096, 128): helion.Config(block_sizes=[16], load_eviction_policies=['', 'last', ''], num_warps=16, num_stages=2, reduction_loops=[64]),
    (256, 4096, 128): helion.Config(block_sizes=[32], num_warps=4, num_stages=1),
    (256, 8192, 128): helion.Config(block_sizes=[8], load_eviction_policies=['', 'last', ''], num_warps=4, num_stages=2),
    (4096, 7168, 128): helion.Config(block_sizes=[32], load_eviction_policies=['', 'last', ''], num_warps=8, num_stages=2),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,
        qout: torch.Tensor,
        scales_out: torch.Tensor,
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0
        EPS = 1e-10
        INV_MAX = 1.0 / 448.0

        for rr in hl.tile(nrows):
            row = data[rr, :].to(torch.float32)
            amax_pos = torch.amax(row, -1)
            amax_neg = -torch.amin(row, -1)
            amax = torch.maximum(amax_pos, amax_neg)
            amax = torch.clamp(amax, min=EPS)
            scale = amax * INV_MAX
            inv_scale = MAX_VAL / amax
            q = torch.clamp(row * inv_scale[:, None], min=-MAX_VAL, max=MAX_VAL)
            qout[rr, :] = q
            scales_out[rr] = scale
        return qout

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}

_cache_key = None
_cache_kernel = None


def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_kernel
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    key = (T, H, gsz)
    if key != _cache_key:
        _cache_key = key
        _cache_kernel = _KERNELS[key]

    _cache_kernel(x.reshape(N, gsz), x_q.reshape(N, gsz), x_s.reshape(N))

    return x_q, x_s
