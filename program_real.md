# Real Kernel Auto-Research Agent

You are optimizing the real GPU Mode Helion leaderboard kernels.
This file is read by `autoopt_real.py` on every iteration.

## Response Contract

- Return the full updated `submission.py` file as a single Python code block.
- Only modify the target problem's `submission.py`.
- Keep the file importable and CUDA-graph capturable.
- Preserve the public entry point: `custom_kernel(data)`.
- Keep `SHAPE_CONFIGS` valid for every listed test and benchmark shape.
- Put a short one-line comment near the top summarizing the experiment.
- Do not include prose outside the Python code block.

## Optimization Priorities

1. Pass all correctness tests first.
2. Keep the kernel capturable by CUDA graphs.
3. Reduce the benchmark geomean latency across the listed benchmark cases. The benchmark times `custom_kernel(data)` itself, so wrapper work counts.
4. Prefer targeted changes over full rewrites.

## What To Read

For each target problem, use:
- `submission.py`: the file you are editing
- `reference.py`: the correctness oracle and input generator
- `task.py`: typed input/output contract
- `task.yml`: description, tests, and benchmark shapes

## Ideas To Explore

### Shape-aware tuning
- Use per-shape `helion.Config` entries aggressively
- Tune block sizes, `num_warps`, and `num_stages` per shape
- Hoist shape-specialized constants with `hl.specialize` where appropriate

### Memory and reuse
- Avoid redundant loads and repeated casts
- Reuse tiles or partial reductions instead of recomputing them
- Reduce temporary allocations inside the kernel body

### Kernel structure
- Move cheap host-side setup outside tile loops
- Fuse pointwise epilogues when it avoids extra memory traffic
- Choose loop orders that improve locality and reduce wasted work
- Do not leave expensive tensor construction, padding, or reshaping in `custom_kernel(data)` if the kernel can absorb it

### Numerical discipline
- Use float32 accumulators when required for correctness
- Do not introduce numerical shortcuts that break the reference checks

## Constraints

- Do not edit `reference.py`, `task.py`, or `task.yml`.
- Do not change the semantics of `custom_kernel`.
- Do not remove benchmark/test shapes from `SHAPE_CONFIGS`.
- Do not define nested functions or helper functions inside `@helion.kernel` bodies.
- Do not use unsupported Python statements inside `@helion.kernel` such as nested defs, classes, lambdas, `with`, `try`, or `match`.
- Correctness is mandatory.
- Small, attributable changes are preferred.
