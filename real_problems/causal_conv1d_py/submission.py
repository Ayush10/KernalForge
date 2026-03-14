#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius
from task import input_t, output_t

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_k(
    x_ptr, w_ptr, b_ptr, y_ptr,
    D, S, stride_b,
    W: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    GROUP_S: tl.constexpr,
):
    num_s_blocks = tl.cdiv(S, BLOCK_S)
    num_d_blocks = tl.cdiv(D, BLOCK_D)
    total_bd = tl.num_programs(1)

    pid_s = tl.program_id(0)
    pid_bd = tl.program_id(1)

    # Swizzle for L2 locality: group S-blocks together
    num_s_groups = tl.cdiv(num_s_blocks, GROUP_S)
    group_id = pid_s // GROUP_S
    first_s = group_id * GROUP_S
    group_size = tl.minimum(num_s_blocks - first_s, GROUP_S)
    pid_s_in_group = pid_s % GROUP_S
    # Interleave: within each S-group, cycle through all bd blocks
    # Actually use the swizzle2d approach
    pid_s, pid_bd = tl.swizzle2d(pid_s, pid_bd, num_s_blocks, total_bd, GROUP_S)

    pid_b = pid_bd // num_d_blocks
    pid_d = pid_bd % num_d_blocks

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    d_mask = d_off < D
    s_mask = s_off < S

    # Load bias [BLOCK_D]
    bias = tl.load(b_ptr + d_off, mask=d_mask, other=0.0)

    # Initialize accumulator with bias
    acc = bias[:, None] + tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

    # Base pointer for this batch
    x_base = x_ptr + pid_b * stride_b
    y_base = y_ptr + pid_b * stride_b

    # Fully unrolled convolution loop
    for k in tl.static_range(W):
        shift = W - 1 - k
        wk = tl.load(w_ptr + d_off * W + k, mask=d_mask, other=0.0)
        s_shifted = s_off - shift
        xk = tl.load(
            x_base + d_off[:, None] * S + s_shifted[None, :],
            mask=d_mask[:, None] & (s_shifted >= 0)[None, :] & s_mask[None, :],
            other=0.0,
        )
        acc += wk[:, None] * xk

    # Store result
    tl.store(
        y_base + d_off[:, None] * S + s_off[None, :],
        acc,
        mask=d_mask[:, None] & s_mask[None, :],
    )


# Per-shape configs: (BLOCK_D, BLOCK_S, num_warps, num_stages, GROUP_S)
# B200 has 148 SMs, 126MB L2
_CONFIGS = {
    # Test shapes
    (1, 64, 64, 4):   (64, 64, 4, 2, 1),
    (2, 128, 128, 4):  (64, 128, 4, 2, 1),
    (1, 256, 256, 3):  (64, 256, 4, 2, 1),
    (1, 128, 64, 8):   (64, 64, 4, 2, 1),
    (4, 64, 128, 4):   (64, 128, 4, 2, 1),
    # Benchmark shapes — tuned for B200
    (1, 768, 512, 4):   (32, 64, 4, 2, 8),    # grid=(8,24)=192
    (1, 768, 2048, 4):  (64, 128, 4, 3, 4),   # grid=(16,12)=192
    (1, 1536, 2048, 4): (64, 128, 8, 3, 8),   # grid=(16,24)=384
    (1, 2560, 2048, 4): (64, 128, 8, 3, 8),   # grid=(16,40)=640
    (1, 2560, 4096, 4): (64, 128, 8, 4, 8),   # grid=(32,40)=1280
}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    y = torch.empty_like(x)

    key = (B, D, S, W)
    cfg = _CONFIGS.get(key, (64, 128, 4, 2, 4))
    BLOCK_D, BLOCK_S, nw, ns, group_s = cfg

    grid = (triton.cdiv(S, BLOCK_S), B * triton.cdiv(D, BLOCK_D))

    _causal_conv1d_k[grid](
        x, weight, bias, y,
        D, S, D * S,
        W=W,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        GROUP_S=group_s,
        num_warps=nw,
        num_stages=ns,
    )
    return y
