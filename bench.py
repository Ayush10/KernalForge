"""
bench.py -- Benchmarking harness for Helion kernel auto-research.
DO NOT MODIFY THIS FILE. The agent only modifies kernel.py.

Usage:
    python bench.py                  # Run all registered problems
    python bench.py --problem matmul # Run a specific problem
    python bench.py --full-autotune  # Run with full autotuning (slow but accurate)
    python bench.py --warmup 10 --iters 100  # Custom benchmark iterations
"""

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
from summary_tables import format_metric, format_ratio, render_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ITERS = 5
BENCH_ITERS = 50
ATOL = 1e-2
RTOL = 1e-2
DEVICE = "cuda"
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Problem Registry
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    name: str
    description: str
    input_shapes: list  # list of (shape, dtype) tuples
    reference_fn: Callable  # PyTorch eager reference
    kernel_attr: str  # attribute name in kernel.py
    metric: str  # "tflops" or "gbps"
    flop_count: Callable  # function(inputs) -> int, for TFLOPS calc
    byte_count: Callable  # function(inputs) -> int, for GB/s calc


def make_inputs(shapes_and_dtypes, device=DEVICE):
    """Generate random inputs from shape/dtype specs."""
    inputs = []
    for shape, dtype in shapes_and_dtypes:
        if dtype in (torch.float16, torch.float32, torch.bfloat16):
            inputs.append(torch.randn(shape, device=device, dtype=dtype))
        elif dtype in (torch.int32, torch.int64):
            inputs.append(torch.randint(0, 100, shape, device=device, dtype=dtype))
        else:
            inputs.append(torch.randn(shape, device=device, dtype=dtype))
    return inputs


# ---- Problem: Matrix Multiplication ----

def matmul_reference(x, y):
    return x @ y

def matmul_flops(inputs):
    x, y = inputs
    M, K = x.shape
    _, N = y.shape
    return 2 * M * N * K  # multiply-add = 2 FLOPs

def matmul_bytes(inputs):
    x, y = inputs
    M, K = x.shape
    _, N = y.shape
    elem = x.element_size()
    return (M * K + K * N + M * N) * elem


# ---- Problem: Softmax ----

def softmax_reference(x):
    return torch.softmax(x, dim=-1)

def softmax_flops(inputs):
    x = inputs[0]
    n = x.numel()
    return 5 * n  # exp, sum, div, max, sub (approximate)

def softmax_bytes(inputs):
    x = inputs[0]
    elem = x.element_size()
    return 2 * x.numel() * elem  # read + write


# ---- Problem: LayerNorm ----

def layernorm_reference(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:])

def layernorm_flops(inputs):
    x = inputs[0]
    n = x.numel()
    return 8 * n  # mean, var, sub, div, etc. (approximate)

def layernorm_bytes(inputs):
    x = inputs[0]
    elem = x.element_size()
    return 2 * x.numel() * elem


# ---- Problem: Fused Matmul + ReLU ----

def matmul_relu_reference(x, y):
    return torch.relu(x @ y)

def matmul_relu_flops(inputs):
    x, y = inputs
    M, K = x.shape
    _, N = y.shape
    return 2 * M * N * K + M * N  # matmul + relu

def matmul_relu_bytes(inputs):
    x, y = inputs
    M, K = x.shape
    _, N = y.shape
    elem = x.element_size()
    return (M * K + K * N + M * N) * elem


# ---- Problem: Vector Add (simple baseline) ----

def vecadd_reference(x, y):
    return x + y

def vecadd_flops(inputs):
    return inputs[0].numel()

def vecadd_bytes(inputs):
    x = inputs[0]
    elem = x.element_size()
    return 3 * x.numel() * elem  # read x, read y, write out


# ---- Register All Problems ----

PROBLEMS = {
    "matmul": Problem(
        name="matmul",
        description="Matrix multiplication: C = A @ B",
        input_shapes=[((4096, 4096), DTYPE), ((4096, 4096), DTYPE)],
        reference_fn=matmul_reference,
        kernel_attr="matmul_kernel",
        metric="tflops",
        flop_count=matmul_flops,
        byte_count=matmul_bytes,
    ),
    "softmax": Problem(
        name="softmax",
        description="Row-wise softmax",
        input_shapes=[((4096, 4096), DTYPE)],
        reference_fn=softmax_reference,
        kernel_attr="softmax_kernel",
        metric="gbps",
        flop_count=softmax_flops,
        byte_count=softmax_bytes,
    ),
    "layernorm": Problem(
        name="layernorm",
        description="Layer normalization over last dimension",
        input_shapes=[((4096, 4096), DTYPE)],
        reference_fn=layernorm_reference,
        kernel_attr="layernorm_kernel",
        metric="gbps",
        flop_count=layernorm_flops,
        byte_count=layernorm_bytes,
    ),
    "matmul_relu": Problem(
        name="matmul_relu",
        description="Fused matmul + ReLU: relu(A @ B)",
        input_shapes=[((4096, 4096), DTYPE), ((4096, 4096), DTYPE)],
        reference_fn=matmul_relu_reference,
        kernel_attr="matmul_relu_kernel",
        metric="tflops",
        flop_count=matmul_relu_flops,
        byte_count=matmul_relu_bytes,
    ),
    "vecadd": Problem(
        name="vecadd",
        description="Element-wise vector addition",
        input_shapes=[((16 * 1024 * 1024,), DTYPE), ((16 * 1024 * 1024,), DTYPE)],
        reference_fn=vecadd_reference,
        kernel_attr="vecadd_kernel",
        metric="gbps",
        flop_count=vecadd_flops,
        byte_count=vecadd_bytes,
    ),
}

# ---------------------------------------------------------------------------
# Benchmarking Utilities
# ---------------------------------------------------------------------------

def check_correctness(kernel_fn, reference_fn, inputs, atol=ATOL, rtol=RTOL):
    """Check kernel output matches reference."""
    with torch.no_grad():
        ref_out = reference_fn(*inputs)
        try:
            kernel_out = kernel_fn(*inputs)
        except Exception as e:
            return False, f"Kernel crashed: {e}"

    if kernel_out.shape != ref_out.shape:
        return False, f"Shape mismatch: kernel={kernel_out.shape}, ref={ref_out.shape}"
    if kernel_out.dtype != ref_out.dtype:
        # Allow dtype mismatch if values are close after casting
        kernel_out = kernel_out.to(ref_out.dtype)

    try:
        torch.testing.assert_close(kernel_out, ref_out, atol=atol, rtol=rtol)
        return True, "PASS"
    except AssertionError as e:
        max_diff = (kernel_out.float() - ref_out.float()).abs().max().item()
        return False, f"Max diff: {max_diff:.6f} | {e}"


def benchmark_kernel(kernel_fn, inputs, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark kernel latency using CUDA events."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            kernel_fn(*inputs)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    with torch.no_grad():
        for i in range(iters):
            start_events[i].record()
            kernel_fn(*inputs)
            end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Remove outliers (top/bottom 10%)
    times_ms.sort()
    trim = max(1, len(times_ms) // 10)
    trimmed = times_ms[trim:-trim] if len(times_ms) > 2 * trim else times_ms

    avg_ms = sum(trimmed) / len(trimmed)
    min_ms = min(times_ms)
    return avg_ms, min_ms


def compute_metric(problem, inputs, avg_ms):
    """Compute TFLOPS or GB/s from timing."""
    if problem.metric == "tflops":
        flops = problem.flop_count(inputs)
        tflops = flops / (avg_ms * 1e-3) / 1e12
        return tflops, "TFLOPS"
    else:
        nbytes = problem.byte_count(inputs)
        gbps = nbytes / (avg_ms * 1e-3) / 1e9
        return gbps, "GB/s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_problem(problem, full_autotune=False, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run a single problem: correctness check + benchmark."""
    print(f"\n{'='*60}")
    print(f"  Problem: {problem.name}")
    print(f"  {problem.description}")
    print(f"{'='*60}")

    # Set autotune effort via environment
    if not full_autotune:
        os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    else:
        os.environ.pop("HELION_AUTOTUNE_EFFORT", None)
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"

    # Reload kernel module to pick up changes
    if "kernel" in sys.modules:
        importlib.reload(sys.modules["kernel"])
    import kernel

    # Get the kernel function
    kernel_fn = getattr(kernel, problem.kernel_attr, None)
    if kernel_fn is None:
        print(f"  SKIP: kernel.{problem.kernel_attr} not found")
        return None

    # Generate inputs
    inputs = make_inputs(problem.input_shapes)

    # Correctness check
    correct, msg = check_correctness(kernel_fn, problem.reference_fn, inputs)
    status = "PASS" if correct else "FAIL"
    print(f"  Correctness: {status}")
    if not correct:
        print(f"  Detail: {msg}")
        return {"name": problem.name, "correct": False, "message": msg}

    # Benchmark
    print(f"  Benchmarking ({warmup} warmup, {iters} iters)...")
    avg_ms, min_ms = benchmark_kernel(kernel_fn, inputs, warmup=warmup, iters=iters)
    value, unit = compute_metric(problem, inputs, avg_ms)

    print(f"  Avg latency:  {avg_ms:.3f} ms")
    print(f"  Min latency:  {min_ms:.3f} ms")
    print(f"  Throughput:   {value:.2f} {unit}")

    # Also benchmark reference for comparison
    avg_ref, _ = benchmark_kernel(problem.reference_fn, inputs, warmup=warmup, iters=iters)
    ref_value, _ = compute_metric(problem, inputs, avg_ref)
    speedup = avg_ref / avg_ms
    print(f"  Reference:    {ref_value:.2f} {unit} ({avg_ref:.3f} ms)")
    print(f"  Speedup:      {speedup:.2f}x vs PyTorch eager")

    return {
        "name": problem.name,
        "correct": True,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "throughput": value,
        "unit": unit,
        "ref_throughput": ref_value,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Helion Kernel Benchmark")
    parser.add_argument("--problem", type=str, default=None,
                        help="Run a specific problem (default: all)")
    parser.add_argument("--full-autotune", action="store_true",
                        help="Run with full autotuning (slower but more accurate)")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help=f"Warmup iterations (default: {WARMUP_ITERS})")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS,
                        help=f"Benchmark iterations (default: {BENCH_ITERS})")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    print(f"Helion Kernel Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Autotune: {'full' if args.full_autotune else 'none (fast mode)'}")

    problems_to_run = PROBLEMS
    if args.problem:
        if args.problem not in PROBLEMS:
            print(f"Unknown problem: {args.problem}")
            print(f"Available: {', '.join(PROBLEMS.keys())}")
            sys.exit(1)
        problems_to_run = {args.problem: PROBLEMS[args.problem]}

    results = []
    for name, problem in problems_to_run.items():
        result = run_problem(problem, args.full_autotune, args.warmup, args.iters)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    summary_rows = []
    failures = []
    for r in results:
        summary_rows.append([
            r["name"],
            format_metric(r.get("throughput"), r.get("unit", "")),
            format_ratio(r.get("speedup")),
            "PASS" if r.get("correct") else "FAIL",
        ])
        if not r.get("correct"):
            failures.append(f"{r['name']}: {r.get('message', 'unknown error')}")

    print(render_table(["Kernel", "Throughput", "vs Eager", "Status"], summary_rows))
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")

    if args.json:
        print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
