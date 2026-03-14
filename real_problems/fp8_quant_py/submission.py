#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _fp8_quant_nopad(
    X, Q, S,
    N,
    GS: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    cols = tl.arange(0, GS)
    offs = rows[:, None] * GS + cols[None, :]
    x = tl.load(X + offs)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-10)
    inv_amax = 448.0 / amax
    q = x * inv_amax[:, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(Q + offs, q)
    tl.store(S + rows, amax * 2.232142857142857e-3)


@triton.jit
def _fp8_quant_masked(
    X, Q, S,
    N,
    GS: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    cols = tl.arange(0, GS)
    mask = rows < N
    offs = rows[:, None] * GS + cols[None, :]
    x = tl.load(X + offs, mask=mask[:, None], other=0.0)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-10)
    inv_amax = 448.0 / amax
    q = x * inv_amax[:, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(Q + offs, q, mask=mask[:, None])
    tl.store(S + rows, amax * 2.232142857142857e-3, mask=mask)


# Tuned on B200 with CUDA graph timing
_CFGS = {
    # Test shapes
    (1, 256, 64):      (4, 2, 1),
    (4, 512, 128):     (8, 4, 1),
    (16, 1024, 64):    (8, 1, 1),
    (1, 4096, 128):    (8, 4, 1),
    (8, 4096, 128):    (8, 4, 1),
    # Benchmark shapes
    (16, 4096, 128):   (8, 4, 1),
    (256, 4096, 128):  (8, 1, 1),
    (256, 8192, 128):  (4, 1, 1),
    (4096, 7168, 128): (16, 8, 1),
}

_prev_key = None
_prev_cfg = None


def custom_kernel(data: input_t) -> output_t:
    global _prev_key, _prev_cfg
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gs = H // G
    N = T * G

    key = (T, H, gs)
    if key != _prev_key:
        _prev_key = key
        _prev_cfg = _CFGS.get(key, (8, 4, 1))

    bn, nw, ns = _prev_cfg
    grid = ((N + bn - 1) // bn,)

    if N % bn == 0:
        _fp8_quant_nopad[grid](
            x, x_q, x_s, N,
            GS=gs, BN=bn,
            num_warps=nw, num_stages=ns,
        )
    else:
        _fp8_quant_masked[grid](
            x, x_q, x_s, N,
            GS=gs, BN=bn,
            num_warps=nw, num_stages=ns,
        )
    return x_q, x_s
