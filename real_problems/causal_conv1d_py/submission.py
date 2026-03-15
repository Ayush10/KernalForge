#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius
from task import input_t, output_t

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d(
    x_ptr, w_ptr, b_ptr, y_ptr,
    D, S,
    W: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_bd = tl.program_id(1)  # batch*D index
    d = pid_bd % D

    b_val = tl.load(b_ptr + d)
    x_row = x_ptr + pid_bd * S
    y_row = y_ptr + pid_bd * S

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S

    acc = tl.where(s_mask, b_val, 0.0)

    for k in tl.static_range(W):
        shift = W - 1 - k
        wk = tl.load(w_ptr + d * W + k)
        s_shifted = s_off - shift
        xk = tl.load(
            x_row + s_shifted,
            mask=s_mask & (s_shifted >= 0),
            other=0.0,
        )
        acc += wk * xk

    tl.store(y_row + s_off, acc, mask=s_mask)


# Per-shape tuned configs: (BLOCK_S, num_warps, num_stages)
_CONFIGS = {
    (1, 768, 512, 4):   (256, 2, 2),
    (1, 768, 2048, 4):  (2048, 16, 4),
    (1, 1536, 2048, 4): (1024, 8, 2),
    (1, 2560, 2048, 4): (512, 1, 2),
    (1, 2560, 4096, 4): (4096, 8, 2),
}

_DEFAULT = (256, 4, 2)


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    y = torch.empty_like(x)

    bs, nw, ns = _CONFIGS.get((B, D, S, W), _DEFAULT)
    grid = (triton.cdiv(S, bs), B * D)

    _causal_conv1d[grid](
        x, weight, bias, y,
        D, S,
        W=W,
        BLOCK_S=bs,
        num_warps=nw,
        num_stages=ns,
    )
    return y
