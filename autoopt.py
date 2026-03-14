from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROGRAM_PATH = ROOT / "program.md"
KERNEL_PATH = ROOT / "kernel.py"
EXPERIMENTS_DIR = ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
RESPONSES_DIR = EXPERIMENTS_DIR / "responses"
CANDIDATES_DIR = EXPERIMENTS_DIR / "candidates"
ACCEPTED_DIR = EXPERIMENTS_DIR / "accepted"
LOG_PATH = EXPERIMENTS_DIR / "experiments.jsonl"
INDEX_PATH = EXPERIMENTS_DIR / "experiments_index.jsonl"
LATEST_REPORT_PATH = EXPERIMENTS_DIR / "latest_report.html"
SUPPORTED_PROVIDERS = ("claude", "codex")
SUPPORTED_SEARCH_MODES = ("auto", "structured", "freeform")
DEFAULT_PROVIDER = os.environ.get("AUTOOPT_PROVIDER", "claude").lower()
if DEFAULT_PROVIDER not in SUPPORTED_PROVIDERS:
    DEFAULT_PROVIDER = "claude"
DEFAULT_MODEL = os.environ.get("AUTOOPT_MODEL")
DEFAULT_SEARCH_MODE = os.environ.get("AUTOOPT_SEARCH_MODE", "auto").lower()
if DEFAULT_SEARCH_MODE not in SUPPORTED_SEARCH_MODES:
    DEFAULT_SEARCH_MODE = "auto"

sys.path.insert(0, str(ROOT))
import bench as bench_harness
import structured_search
from summary_tables import (
    format_improvement,
    format_metric,
    format_ratio,
    render_table,
    status_from_improvement,
)


SYSTEM_PROMPT = """You are an expert GPU kernel engineer optimizing Helion kernels.
You will receive:
- the project research instructions from program.md
- the full current kernel.py file
- the target problem and benchmark metric
- a short history of recent attempts

Return exactly one Python code block containing the full updated kernel.py file.

Rules:
- Keep the file importable.
- Keep all existing kernels defined unless a change is explicitly required.
- Focus changes on the target kernel attribute unless a small shared helper is necessary.
- Prefer one targeted optimization over a rewrite.
- Add a concise first-line comment summarizing the experiment you are trying.
- Do not include prose outside the Python code block.
"""


def ensure_dirs() -> None:
    for path in (
        EXPERIMENTS_DIR,
        RUNS_DIR,
        RESPONSES_DIR,
        CANDIDATES_DIR,
        ACCEPTED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def metric_label(problem: bench_harness.Problem) -> str:
    return "TFLOPS" if problem.metric == "tflops" else "GB/s"


def extract_code_block(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_summary(candidate_source: str) -> str:
    for line in candidate_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return "No summary comment provided."


def load_history(problem_name: str) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []

    history: list[dict[str, Any]] = []
    for raw_line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("problem") == problem_name:
            history.append(record)
    return history


def append_log(record: dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def next_experiment_number() -> int:
    max_number = 0
    for run_dir in RUNS_DIR.glob("experiment_*"):
        match = re.match(r"experiment_(\d+)_", run_dir.name)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def create_run_context(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now()
    experiment_number = next_experiment_number()
    run_name = (
        f"experiment_{experiment_number:03d}_{args.problem}_"
        f"{started_at.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "experiment_number": experiment_number,
        "run_name": run_name,
        "run_dir": run_dir,
        "events_path": run_dir / "events.jsonl",
        "summary_path": run_dir / "summary.json",
        "started_at": started_at,
        "started_at_iso": started_at.isoformat(timespec="seconds"),
    }


def record_event(
    run_context: dict[str, Any],
    run_events: list[dict[str, Any]],
    record: dict[str, Any],
) -> None:
    run_events.append(record)
    append_log(record)
    append_jsonl(run_context["events_path"], record)


def format_history(history: list[dict[str, Any]], limit: int) -> str:
    if not history:
        return "- No prior experiments recorded yet."

    lines: list[str] = []
    for record in history[-limit:]:
        throughput = record.get("throughput")
        unit = record.get("unit", "")
        throughput_text = "n/a"
        if isinstance(throughput, (int, float)):
            throughput_text = f"{throughput:.2f} {unit}".strip()
        lines.append(
            f"- iter {record['iter']:03d} | {record.get('result', 'unknown')} | "
            f"correct={record.get('correct')} | throughput={throughput_text} | "
            f"summary={record.get('summary', 'n/a')}"
        )
    return "\n".join(lines)


def build_freeform_user_prompt(
    *,
    problem_name: str,
    current_source: str,
    program_text: str,
    best_result: dict[str, Any],
    history: list[dict[str, Any]],
    history_limit: int,
) -> str:
    problem = bench_harness.PROBLEMS[problem_name]
    unit = metric_label(problem)
    return f"""## Target
Optimize the `{problem.kernel_attr}` implementation for problem `{problem.name}`.

- Problem description: {problem.description}
- Primary metric: {unit} (higher is better)
- Current best throughput: {best_result['throughput']:.2f} {best_result['unit']}
- Current best latency: {best_result['avg_ms']:.3f} ms
- Current speedup vs PyTorch eager: {best_result['speedup']:.2f}x

## Research Program
{program_text}

## Current kernel.py
```python
{current_source}
```

## Recent Experiment History
{format_history(history, history_limit)}

## Requirements
- Return the full updated `kernel.py` file as a single Python code block.
- Keep all existing imports and non-target kernels working.
- Preserve correctness.
- Try one concrete change that is likely to improve the metric.
"""


def load_python_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_autotune_env(full_autotune: bool) -> None:
    if full_autotune:
        os.environ.pop("HELION_AUTOTUNE_EFFORT", None)
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
    else:
        os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
        os.environ.pop("HELION_FORCE_AUTOTUNE", None)


def reset_random_seed() -> None:
    torch = bench_harness.torch
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def evaluate_kernel_file(
    *,
    kernel_path: Path,
    problem_name: str,
    warmup: int,
    iters: int,
    full_autotune: bool,
) -> dict[str, Any]:
    problem = bench_harness.PROBLEMS[problem_name]
    set_autotune_env(full_autotune)

    try:
        module = load_python_module(kernel_path, f"_candidate_{time.time_ns()}")
    except Exception as exc:
        return {
            "name": problem.name,
            "correct": False,
            "message": f"Import failed: {exc}",
        }

    kernel_fn = getattr(module, problem.kernel_attr, None)
    if kernel_fn is None:
        return {
            "name": problem.name,
            "correct": False,
            "message": f"Missing kernel attribute: {problem.kernel_attr}",
        }

    reset_random_seed()
    correctness_inputs = bench_harness.make_inputs(problem.input_shapes)
    correct, message = bench_harness.check_correctness(
        kernel_fn,
        problem.reference_fn,
        correctness_inputs,
    )
    if not correct:
        return {
            "name": problem.name,
            "correct": False,
            "message": message,
        }

    reset_random_seed()
    bench_inputs = bench_harness.make_inputs(problem.input_shapes)
    avg_ms, min_ms = bench_harness.benchmark_kernel(
        kernel_fn,
        bench_inputs,
        warmup=warmup,
        iters=iters,
    )
    throughput, unit = bench_harness.compute_metric(problem, bench_inputs, avg_ms)

    reset_random_seed()
    ref_inputs = bench_harness.make_inputs(problem.input_shapes)
    avg_ref_ms, _ = bench_harness.benchmark_kernel(
        problem.reference_fn,
        ref_inputs,
        warmup=warmup,
        iters=iters,
    )
    ref_throughput, _ = bench_harness.compute_metric(problem, ref_inputs, avg_ref_ms)

    return {
        "name": problem.name,
        "correct": True,
        "message": "PASS",
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "throughput": throughput,
        "unit": unit,
        "ref_throughput": ref_throughput,
        "speedup": avg_ref_ms / avg_ms,
    }


def require_provider_binary(provider: str) -> str:
    binary = shutil.which(provider)
    if binary is None:
        raise RuntimeError(f"Provider `{provider}` is not installed or not on PATH.")
    return binary


def format_process_error(
    provider: str,
    completed: subprocess.CompletedProcess[str],
) -> str:
    details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
    return f"{provider} exited with code {completed.returncode}: {details}"


def request_via_claude_cli(
    *,
    prompt: str,
    model: str | None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    binary = require_provider_binary("claude")
    command = [
        binary,
        "-p",
        "--output-format",
        "text",
        "--tools",
        "",
        "--system-prompt",
        system_prompt,
    ]
    if model:
        command.extend(["--model", model])
    command.append(prompt)

    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(format_process_error("claude", completed))

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError("claude returned an empty response.")
    return output


def request_via_codex_cli(
    *,
    prompt: str,
    model: str | None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    binary = require_provider_binary("codex")
    with tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        dir=RESPONSES_DIR,
        prefix="codex-last-message-",
        suffix=".txt",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)

    command = [
        binary,
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(ROOT),
        "-s",
        "read-only",
        "-o",
        str(output_path),
        "-",
    ]
    if model:
        command[2:2] = ["-m", model]

    combined_prompt = (
        f"{system_prompt}\n\n"
        "The following user prompt contains the current project state and exact task.\n\n"
        f"{prompt}"
    )

    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        input=combined_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(format_process_error("codex", completed))

    if output_path.exists():
        output = output_path.read_text(encoding="utf-8").strip()
        output_path.unlink(missing_ok=True)
    else:
        output = completed.stdout.strip()

    if not output:
        raise RuntimeError("codex returned an empty response.")
    return output


def request_candidate_source(
    *,
    provider: str,
    model: str | None,
    prompt: str,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    del max_tokens
    if provider == "claude":
        return request_via_claude_cli(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )
    if provider == "codex":
        return request_via_codex_cli(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def print_result(prefix: str, result: dict[str, Any]) -> None:
    if result.get("correct"):
        print(
            f"{prefix} {result['throughput']:.2f} {result['unit']} | "
            f"{result['avg_ms']:.3f} ms | {result['speedup']:.2f}x vs eager"
        )
    else:
        print(f"{prefix} FAILED | {result.get('message', 'unknown error')}")


def build_demo_performance_rows(
    *,
    problem_name: str,
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
) -> list[list[str]]:
    return [[
        problem_name,
        format_metric(current_result.get("throughput"), current_result.get("unit", "")),
        format_metric(baseline_result.get("throughput"), baseline_result.get("unit", "")),
        format_improvement(
            baseline_result.get("throughput"),
            current_result.get("throughput"),
            direction="higher",
        ),
        format_ratio(current_result.get("speedup")),
        status_from_improvement(
            baseline_result.get("throughput"),
            current_result.get("throughput"),
            direction="higher",
        ),
    ]]


def print_demo_performance_table(
    *,
    title: str,
    problem_name: str,
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
) -> None:
    print(title)
    print(
        render_table(
            ["Kernel", "Throughput", "Baseline", "vs Baseline", "vs Eager", "Status"],
            build_demo_performance_rows(
                problem_name=problem_name,
                baseline_result=baseline_result,
                current_result=current_result,
            ),
        )
    )


def run_loop(args: argparse.Namespace) -> int:
    ensure_dirs()

    if args.provider not in SUPPORTED_PROVIDERS:
        print(
            f"[!] Unknown provider `{args.provider}`. "
            f"Available: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        return 1

    if args.problem not in bench_harness.PROBLEMS:
        print(
            f"[!] Unknown problem `{args.problem}`. "
            f"Available: {', '.join(sorted(bench_harness.PROBLEMS))}"
        )
        return 1

    if not PROGRAM_PATH.exists():
        print(f"[!] Missing {PROGRAM_PATH}")
        return 1
    if not KERNEL_PATH.exists():
        print(f"[!] Missing {KERNEL_PATH}")
        return 1

    try:
        require_provider_binary(args.provider)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        return 1

    try:
        search_mode = structured_search.resolve_search_mode(
            args.problem, args.search_mode
        )
    except ValueError as exc:
        print(f"[!] {exc}")
        return 1

    if args.provider == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[!] ANTHROPIC_API_KEY is set. Claude Code may use API billing "
            "instead of your logged-in subscription. Unset it if you want "
            "subscription-backed usage."
        )

    program_text = PROGRAM_PATH.read_text(encoding="utf-8")
    current_source = KERNEL_PATH.read_text(encoding="utf-8")
    run_context = create_run_context(args)
    run_events: list[dict[str, Any]] = []
    history = load_history(args.problem)
    next_iter = max((record["iter"] for record in history), default=0) + 1

    baseline_snapshot_path = run_context["run_dir"] / "baseline_kernel.py"
    baseline_snapshot_path.write_text(current_source, encoding="utf-8")
    write_json(
        run_context["run_dir"] / "metadata.json",
        {
            "suite": "demo",
            "experiment_number": run_context["experiment_number"],
            "run_name": run_context["run_name"],
            "problem": args.problem,
            "provider": args.provider,
            "model": args.model,
            "search_mode": search_mode,
            "metric_label": "Throughput",
            "metric_direction": "higher",
            "planned_budget_minutes": args.budget,
            "started_at": run_context["started_at_iso"],
        },
    )

    print(f"\n{'=' * 72}")
    print("Autonomous Helion Optimizer")
    print(f"Experiment:     {run_context['experiment_number']:03d}")
    print(f"Problem:        {args.problem}")
    print(f"Provider:       {args.provider}")
    print(f"Search mode:    {search_mode}")
    print(f"Budget:         {args.budget:.1f} minutes")
    print(f"Max iterations: {args.iters}")
    print(f"Model:          {args.model or 'provider default'}")
    print(f"{'=' * 72}\n")

    print("[*] Benchmarking current kernel.py as the starting point...")
    best_result = evaluate_kernel_file(
        kernel_path=KERNEL_PATH,
        problem_name=args.problem,
        warmup=args.warmup,
        iters=args.bench_iters,
        full_autotune=args.full_autotune,
    )
    if not best_result.get("correct"):
        finished_at = datetime.now()
        summary = {
            "suite": "demo",
            "experiment_number": run_context["experiment_number"],
            "run_name": run_context["run_name"],
            "problem": args.problem,
            "provider": args.provider,
            "model": args.model,
            "search_mode": search_mode,
            "metric_label": "Throughput",
            "metric_unit": best_result.get("unit"),
            "metric_direction": "higher",
            "planned_budget_minutes": args.budget,
            "started_at": run_context["started_at_iso"],
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "actual_duration_minutes": round(
                (finished_at - run_context["started_at"]).total_seconds() / 60.0, 2
            ),
            "status": "failed_start",
            "iterations_requested": args.iters,
            "iterations_completed": 0,
            "candidate_runs": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "provider_error_count": 0,
            "parse_error_count": 0,
            "plan_error_count": 0,
            "syntax_error_count": 0,
            "baseline_result": best_result,
            "final_result": best_result,
            "report_path": None,
        }
        write_json(run_context["summary_path"], summary)
        append_jsonl(INDEX_PATH, summary)
        print_result("[!]", best_result)
        print("[!] Refusing to start from a broken baseline.")
        return 1

    write_json(run_context["run_dir"] / "baseline_result.json", best_result)
    print_result("[baseline]", best_result)
    print_demo_performance_table(
        title="Baseline Snapshot",
        problem_name=args.problem,
        baseline_result=best_result,
        current_result=best_result,
    )
    baseline_result = dict(best_result)
    deadline = time.time() + args.budget * 60
    accepted_count = 0
    iterations_completed = 0

    for offset in range(args.iters):
        if time.time() >= deadline:
            break

        iteration = next_iter + offset
        iter_tag = f"{args.problem}_iter_{iteration:03d}"
        print(f"\n[iter {iteration:03d}] Requesting a new candidate...")
        iterations_completed += 1

        history_text = format_history(history, args.history_limit)
        if search_mode == "structured":
            prompt = structured_search.build_structured_prompt(
                problem_name=args.problem,
                current_source=current_source,
                program_text=program_text,
                best_result=best_result,
                history_text=history_text,
            )
        else:
            prompt = build_freeform_user_prompt(
                problem_name=args.problem,
                current_source=current_source,
                program_text=program_text,
                best_result=best_result,
                history=history,
                history_limit=args.history_limit,
            )

        try:
            raw_response = request_candidate_source(
                provider=args.provider,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            record = {
                "suite": "demo",
                "experiment_number": run_context["experiment_number"],
                "run_name": run_context["run_name"],
                "iter": iteration,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "problem": args.problem,
                "provider": args.provider,
                "model": args.model,
                "search_mode": search_mode,
                "result": "provider_error",
                "correct": False,
                "summary": str(exc),
            }
            record_event(run_context, run_events, record)
            history.append(record)
            print(f"  [!] Provider error: {exc}")
            time.sleep(2)
            continue

        response_path = RESPONSES_DIR / f"{iter_tag}.txt"
        response_path.write_text(raw_response, encoding="utf-8")

        plan_path: Path | None = None
        if search_mode == "structured":
            try:
                raw_plan = structured_search.extract_json_object(raw_response)
                plan = structured_search.normalize_plan(args.problem, raw_plan)
                candidate_source = structured_search.render_candidate_source(
                    current_source,
                    plan,
                )
                plan_path = run_context["run_dir"] / f"{iter_tag}_plan.json"
                write_json(plan_path, plan)
                summary = plan["summary"]
            except Exception as exc:
                record = {
                    "suite": "demo",
                    "experiment_number": run_context["experiment_number"],
                    "run_name": run_context["run_name"],
                    "iter": iteration,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem": args.problem,
                    "provider": args.provider,
                    "model": args.model,
                    "search_mode": search_mode,
                    "result": "plan_error",
                    "correct": False,
                    "summary": str(exc),
                    "response_path": str(response_path.relative_to(ROOT)),
                }
                record_event(run_context, run_events, record)
                history.append(record)
                print(f"  [!] Structured plan error: {exc}")
                continue
        else:
            candidate_source = extract_code_block(raw_response)
            if not candidate_source:
                record = {
                    "suite": "demo",
                    "experiment_number": run_context["experiment_number"],
                    "run_name": run_context["run_name"],
                    "iter": iteration,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem": args.problem,
                    "provider": args.provider,
                    "model": args.model,
                    "search_mode": search_mode,
                    "result": "parse_error",
                    "correct": False,
                    "summary": "Model response did not include a Python code block.",
                    "response_path": str(response_path.relative_to(ROOT)),
                }
                record_event(run_context, run_events, record)
                history.append(record)
                print("  [!] Parse error: no Python code block found.")
                continue
            summary = extract_summary(candidate_source)

        try:
            compile(candidate_source, str(CANDIDATES_DIR / f"{iter_tag}.py"), "exec")
        except SyntaxError as exc:
            record = {
                "suite": "demo",
                "experiment_number": run_context["experiment_number"],
                "run_name": run_context["run_name"],
                "iter": iteration,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "problem": args.problem,
                "provider": args.provider,
                "model": args.model,
                "search_mode": search_mode,
                "result": "syntax_error",
                "correct": False,
                "summary": f"Syntax error: {exc.msg} on line {exc.lineno}",
                "response_path": str(response_path.relative_to(ROOT)),
            }
            record_event(run_context, run_events, record)
            history.append(record)
            print(f"  [!] Syntax error: {exc.msg} on line {exc.lineno}")
            continue

        candidate_path = CANDIDATES_DIR / f"{iter_tag}.py"
        candidate_path.write_text(candidate_source, encoding="utf-8")

        print(f"  [*] Benchmarking candidate: {summary}")
        try:
            result = evaluate_kernel_file(
                kernel_path=candidate_path,
                problem_name=args.problem,
                warmup=args.warmup,
                iters=args.bench_iters,
                full_autotune=args.full_autotune,
            )
        except Exception as exc:
            result = {
                "name": args.problem,
                "correct": False,
                "message": f"Benchmark crashed: {exc}",
            }

        improvement_ratio = None
        accepted = False
        if result.get("correct"):
            improvement_ratio = result["throughput"] / best_result["throughput"]
            required_ratio = 1.0 + (args.min_improvement / 100.0)
            accepted = improvement_ratio >= required_ratio

        if accepted:
            accepted_count += 1
            current_source = candidate_source
            best_result = result
            KERNEL_PATH.write_text(candidate_source, encoding="utf-8")
            accepted_path = ACCEPTED_DIR / f"{iter_tag}.py"
            accepted_path.write_text(candidate_source, encoding="utf-8")
            print_result("  [accepted]", result)
        else:
            print_result("  [rejected]", result)

        record = {
            "suite": "demo",
            "experiment_number": run_context["experiment_number"],
            "run_name": run_context["run_name"],
            "iter": iteration,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "problem": args.problem,
            "provider": args.provider,
            "model": args.model,
            "search_mode": search_mode,
            "result": "accepted" if accepted else "rejected",
            "correct": bool(result.get("correct")),
            "summary": summary,
            "metric_value": result.get("throughput"),
            "throughput": result.get("throughput"),
            "unit": result.get("unit"),
            "avg_ms": result.get("avg_ms"),
            "speedup": result.get("speedup"),
            "improvement_ratio": improvement_ratio,
            "min_improvement_pct": args.min_improvement,
            "message": result.get("message"),
            "candidate_path": str(candidate_path.relative_to(ROOT)),
            "response_path": str(response_path.relative_to(ROOT)),
            "plan_path": (
                str(plan_path.relative_to(ROOT))
                if plan_path is not None
                else None
            ),
        }
        record_event(run_context, run_events, record)
        history.append(record)

        print(f"[iter {iteration:03d}] Iteration complete")
        print_demo_performance_table(
            title="Performance Snapshot",
            problem_name=args.problem,
            baseline_result=baseline_result,
            current_result=best_result,
        )

    finished_at = datetime.now()
    final_kernel_path = run_context["run_dir"] / "final_kernel.py"
    final_kernel_path.write_text(current_source, encoding="utf-8")
    write_json(run_context["run_dir"] / "final_result.json", best_result)

    provider_error_count = sum(
        1 for event in run_events if event["result"] == "provider_error"
    )
    parse_error_count = sum(
        1 for event in run_events if event["result"] == "parse_error"
    )
    plan_error_count = sum(
        1 for event in run_events if event["result"] == "plan_error"
    )
    syntax_error_count = sum(
        1 for event in run_events if event["result"] == "syntax_error"
    )
    rejected_count = sum(1 for event in run_events if event["result"] == "rejected")
    benchmarked_candidates = sum(
        1 for event in run_events if event["result"] in {"accepted", "rejected"}
    )
    final_delta = best_result["throughput"] - baseline_result["throughput"]
    final_delta_pct = 0.0
    if baseline_result["throughput"]:
        final_delta_pct = (final_delta / baseline_result["throughput"]) * 100.0

    summary = {
        "suite": "demo",
        "experiment_number": run_context["experiment_number"],
        "run_name": run_context["run_name"],
        "problem": args.problem,
        "provider": args.provider,
        "model": args.model,
        "search_mode": search_mode,
        "metric_label": "Throughput",
        "metric_unit": best_result.get("unit"),
        "metric_direction": "higher",
        "planned_budget_minutes": args.budget,
        "started_at": run_context["started_at_iso"],
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "actual_duration_minutes": round(
            (finished_at - run_context["started_at"]).total_seconds() / 60.0, 2
        ),
        "status": "completed",
        "iterations_requested": args.iters,
        "iterations_completed": iterations_completed,
        "candidate_runs": benchmarked_candidates,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "provider_error_count": provider_error_count,
        "parse_error_count": parse_error_count,
        "plan_error_count": plan_error_count,
        "syntax_error_count": syntax_error_count,
        "baseline_result": baseline_result,
        "final_result": best_result,
        "final_delta": round(final_delta, 4),
        "final_delta_pct": round(final_delta_pct, 3),
        "run_dir": str(run_context["run_dir"].relative_to(ROOT)),
        "baseline_kernel_path": str(baseline_snapshot_path.relative_to(run_context["run_dir"])),
        "final_kernel_path": str(final_kernel_path.relative_to(run_context["run_dir"])),
        "events_path": str(run_context["events_path"].relative_to(run_context["run_dir"])),
        "report_path": None,
    }
    write_json(run_context["summary_path"], summary)

    from report_experiment import generate_report_for_run

    report_path = generate_report_for_run(run_context["run_dir"])
    shutil.copyfile(report_path, LATEST_REPORT_PATH)
    summary["report_path"] = str(report_path.relative_to(run_context["run_dir"]))
    summary["latest_report_path"] = str(LATEST_REPORT_PATH.relative_to(ROOT))
    write_json(run_context["summary_path"], summary)
    append_jsonl(INDEX_PATH, summary)

    print(f"\n{'=' * 72}")
    print("Run complete")
    print(f"Experiment number:    {run_context['experiment_number']:03d}")
    print(f"Accepted candidates: {accepted_count}")
    print(
        f"Best result:         {best_result['throughput']:.2f} {best_result['unit']} "
        f"| {best_result['avg_ms']:.3f} ms"
    )
    print_demo_performance_table(
        title="Experiment Performance Summary",
        problem_name=args.problem,
        baseline_result=baseline_result,
        current_result=best_result,
    )
    print(f"Experiment log:      {LOG_PATH}")
    print(f"Run report:          {report_path}")
    print(f"{'=' * 72}\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous experiment loop for Helion kernel optimization."
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=DEFAULT_PROVIDER,
        help="Model provider CLI to use (default: claude)",
    )
    parser.add_argument(
        "--search-mode",
        choices=SUPPORTED_SEARCH_MODES,
        default=DEFAULT_SEARCH_MODE,
        help="Candidate generation mode (default: auto)",
    )
    parser.add_argument(
        "--problem",
        default="matmul",
        help="Problem name from bench.py (default: matmul)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=60.0,
        help="Wall-clock budget in minutes (default: 60)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Maximum candidate iterations to try (default: 20)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Optional provider-specific model alias/name (default: provider default)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per benchmark (default: 5)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=25,
        help="Timed benchmark iterations per candidate (default: 25)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1.0,
        help="Minimum percentage improvement required to keep a candidate (default: 1.0)",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="How many recent experiments to include in the prompt (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Reserved for API-style providers; ignored for CLI providers",
    )
    parser.add_argument(
        "--full-autotune",
        action="store_true",
        help="Use full Helion autotuning during evaluation instead of fast mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
