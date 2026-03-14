"""
kernel.py -- Helion kernel implementations.
THIS IS THE FILE THE AGENT MODIFIES.

Each kernel is a @helion.kernel() decorated function.
The bench.py harness looks up kernels by the attribute names:
    - matmul_kernel
    - softmax_kernel
    - layernorm_kernel
    - matmul_relu_kernel
    - vecadd_kernel

Start with these baselines and let the agent iterate.
"""

import torch
import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Matmul: C = A @ B
# ---------------------------------------------------------------------------

@helion.kernel()
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


# ---------------------------------------------------------------------------
# Softmax: row-wise softmax
# ---------------------------------------------------------------------------

@helion.kernel()
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        out[tile_m, :] = torch.softmax(row, dim=-1)
    return out


# ---------------------------------------------------------------------------
# LayerNorm: normalize over last dimension
# ---------------------------------------------------------------------------

@helion.kernel()
def layernorm_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        mean = torch.mean(row, dim=-1, keepdim=True)
        var = torch.mean(row * row, dim=-1, keepdim=True) - mean * mean
        normed = (row - mean) / torch.sqrt(var + 1e-5)
        out[tile_m, :] = normed.to(x.dtype)
    return out


# ---------------------------------------------------------------------------
# Fused Matmul + ReLU: relu(A @ B)
# ---------------------------------------------------------------------------

@helion.kernel()
def matmul_relu_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc)
    return out


# ---------------------------------------------------------------------------
# Vector Add: C = A + B
# ---------------------------------------------------------------------------

@helion.kernel()
def vecadd_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n] + y[tile_n]
    return out
