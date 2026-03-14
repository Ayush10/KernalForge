# helion-autoresearch

Autonomous Helion kernel optimization via AI agents. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: point an AI coding agent at `program.md`, let it iterate on `kernel.py`, and wake up to faster GPU kernels.
This repo supports both a manual agent workflow and an autonomous experiment loop. The autonomous loop defaults to the `claude` CLI and also supports `codex`. Each default run is treated as a 1-hour experiment session because `autoopt.py` defaults to a `60` minute budget.
The repo now has two tracks:
- `demo`: sample kernels for iteration on the optimizer itself
- `real`: the 5 vendored GPU Mode Helion leaderboard kernels

The demo optimizer now has two search modes:
- `structured`: the model returns a small JSON plan and the repo rewrites the target kernel deterministically
- `freeform`: the model returns a full `kernel.py` rewrite

The default `auto` mode uses structured search for `matmul` and `matmul_relu`, and falls back to freeform for the other problems.

## Quick Start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv and install deps
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# 3. Verify setup with a baseline run
python bench.py --problem matmul

# 4. Run all problems
python bench.py
```

PowerShell equivalent:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
python bench.py --problem matmul
```

If you use the bundled project venv instead of creating a new one:

```powershell
.\gpu-improve-new\Scripts\Activate.ps1
```

## Autonomous Loop

### Demo Track

If you want the system to propose, benchmark, and keep demo candidate kernels on its own:

```bash
# Log in once with your Claude subscription (default provider)
claude auth login

# Fast iteration mode (default: auto => structured for matmul)
python autoopt.py --problem matmul --budget 60 --iters 20

# Force the constrained JSON-plan path explicitly
python autoopt.py --problem matmul --search-mode structured --budget 60 --iters 20

# Slower but more accurate evaluation
python autoopt.py --problem matmul --full-autotune --budget 120 --iters 10

# Or use Codex instead of Claude
codex login
python autoopt.py --provider codex --problem matmul --budget 60 --iters 20
```

PowerShell equivalent:

```powershell
claude auth login
python autoopt.py --problem matmul --budget 60 --iters 20
python autoopt.py --problem matmul --search-mode structured --budget 60 --iters 20

codex login
python autoopt.py --provider codex --problem matmul --budget 60 --iters 20
```

In structured mode, the loop reads `program.md`, asks the selected CLI for a JSON optimization plan, rewrites the target kernel deterministically, evaluates the candidate with correctness checks, and only keeps it if it beats the current best by the configured threshold. In freeform mode, it asks for a full updated `kernel.py`. Generated responses, candidate files, accepted snapshots, JSON plans, and the JSONL experiment log are written under `experiments/`.
If you use Claude and `ANTHROPIC_API_KEY` is set in your environment, Claude Code may use API billing instead of your logged-in subscription.

Each run also gets its own experiment folder under `experiments/runs/` with:
- experiment number
- timestamps
- baseline and final kernel snapshots
- per-iteration event log
- per-iteration structured plans when `--search-mode structured` is active
- an auto-generated `report.html` with before/after and progress graphs

The HTML report now includes the experiment number, search mode, number of runs, batch results, and before/after graphs so each 1-hour session is easy to compare.

You can regenerate a report later with:

```powershell
python report_experiment.py
python report_experiment.py --experiment 1
```

### Real GPU Mode Track

The real kernels are vendored under `real_problems/`:
- `fp8_quant`
- `causal_conv1d`
- `gated_deltanet_chunk_fwd_h`
- `gated_deltanet_chunk_fwd_o`
- `gated_deltanet_recompute_w_u`

Run the real benchmark harness:

```powershell
python bench_real.py --problem fp8_quant --mode both
python bench_real.py
```

Run the autonomous optimizer for a real kernel:

```powershell
claude auth login
python autoopt_real.py --problem fp8_quant --budget 60 --iters 20

codex login
python autoopt_real.py --provider codex --problem causal_conv1d --budget 60 --iters 20
```

`autoopt_real.py` is now an advanced hybrid loop:
- it asks the model for a batch of structured JSON plans
- ranks those plans with a surrogate scorer built from heuristics plus prior experiment history
- materializes only the top-ranked plans into `submission.py` candidates
- runs correctness-only screening before full benchmarks
- benchmarks only the best screened candidates and promotes the winner

Useful knobs:

```powershell
python autoopt_real.py --problem fp8_quant --planner-batch-size 6 --materialize-top-k 3 --benchmark-top-k 2
```

By default, `autoopt_real.py` also prints a heartbeat every 60 seconds, for example:
`[heartbeat] Program responding | 1 min passed | stage=iter-003-benchmark-1`
You can change that cadence with `--heartbeat-seconds`.
At the end of each outer iteration it also prints an `Iteration Summary` table, and at the end of the run it prints an `Experiment Summary` table.

Real-kernel runs use the same `experiments/` reporting pipeline as the demo track, but the report now also shows screening stages and surrogate scores.

## Running the Agent

Open Claude Code (or Codex, Cursor, etc.) in this directory:

```
Hi, read program.md and let's start optimizing! Run bench.py first to get baselines, then begin the experiment loop.
```

## Project Structure

```
program.md      -- Agent instructions (human edits this)
kernel.py       -- Helion kernels (agent edits this)
bench.py        -- Benchmark harness (nobody touches this)
autoopt.py      -- Autonomous optimize/benchmark/promote loop
program_real.md -- Agent instructions for the real GPU Mode kernels
bench_real.py   -- Benchmark harness for the 5 real kernels
autoopt_real.py -- Autonomous loop for the 5 real kernels
real_problems/  -- Vendored GPU Mode Helion problem folders
report_experiment.py -- Generates HTML reports for experiment runs
experiments/    -- Auto-generated responses, candidates, accepted kernels, logs
```

## Adapting to Hackathon Problems

The repo already vendors the current 5 GPU Mode Helion problems under `real_problems/`.
If that leaderboard changes later, refresh those folders and keep the shared `eval.py` / `task.yml` contract intact.

## Tips for the Hackathon

- Use `--full-autotune` only on your best candidates (it takes minutes per kernel)
- Fast iteration mode (default) skips autotuning for ~10x faster feedback
- Focus on 2-3 problems max rather than spreading across everything
- The agent explores algorithmic structure; Helion's autotuner handles tile sizes
- Check `experiments/experiments.jsonl` to avoid repeating failed ideas
