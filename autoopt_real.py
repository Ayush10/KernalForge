from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import advanced_ranker
import advanced_real_search
import real_problem_suite as suite
from autoopt import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    append_jsonl,
    extract_code_block,
    request_candidate_source,
    require_provider_binary,
    write_json,
)
from report_experiment import generate_report_for_run
from summary_tables import (
    format_improvement,
    format_metric,
    render_table,
    status_from_improvement,
)

ROOT = Path(__file__).resolve().parent
PROGRAM_PATH = ROOT / "program_real.md"
EXPERIMENTS_DIR = ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
REAL_RESPONSES_DIR = EXPERIMENTS_DIR / "real_responses"
REAL_CANDIDATES_DIR = EXPERIMENTS_DIR / "real_candidates"
REAL_ACCEPTED_DIR = EXPERIMENTS_DIR / "real_accepted"
LOG_PATH = EXPERIMENTS_DIR / "experiments.jsonl"
INDEX_PATH = EXPERIMENTS_DIR / "experiments_index.jsonl"
LATEST_REPORT_PATH = EXPERIMENTS_DIR / "latest_report.html"

SEARCH_MODE = "advanced_hybrid"
COMPILER_OR_RESOURCE_FAILURES = {
    "shared_memory_oor",
    "jit_file_error",
    "compile_error",
    "unsupported_syntax",
    "control_flow_tensor_mismatch",
    "invalid_indexing",
    "shape_mismatch",
    "broadcast_mismatch",
}


def ensure_dirs() -> None:
    for path in (
        EXPERIMENTS_DIR,
        RUNS_DIR,
        REAL_RESPONSES_DIR,
        REAL_CANDIDATES_DIR,
        REAL_ACCEPTED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


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
        f"experiment_{experiment_number:03d}_real_{args.problem}_"
        f"{started_at.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    planner_dir = run_dir / "planner"
    materializer_dir = run_dir / "materializer"
    planner_dir.mkdir(parents=True, exist_ok=True)
    materializer_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiment_number": experiment_number,
        "run_name": run_name,
        "run_dir": run_dir,
        "planner_dir": planner_dir,
        "materializer_dir": materializer_dir,
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
    append_jsonl(LOG_PATH, record)
    append_jsonl(run_context["events_path"], record)


class RunHeartbeat:
    def __init__(self, interval_seconds: int) -> None:
        self.interval_seconds = max(1, interval_seconds)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stage = "starting"
        self._started_at = 0.0
        self._tick_count = 0

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._tick_count += 1
            elapsed_seconds = self._tick_count * self.interval_seconds
            with self._lock:
                stage = self._stage
            print(
                f"[heartbeat] Program responding | "
                f"{self._format_elapsed(elapsed_seconds)} passed | "
                f"stage={stage}"
            )

    def _format_elapsed(self, elapsed_seconds: int) -> str:
        if elapsed_seconds % 60 == 0:
            minutes = elapsed_seconds // 60
            unit = "min" if minutes == 1 else "mins"
            return f"{minutes} {unit}"
        return f"{elapsed_seconds} sec"


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
        if record.get("suite") == "real" and record.get("problem") == problem_name:
            history.append(record)
    return history


def format_history(history: list[dict[str, Any]], limit: int) -> str:
    interesting_results = {
        "accepted",
        "rejected",
        "test_fail",
        "resource_block",
        "syntax_error",
        "static_block",
        "parse_error",
        "plan_error",
        "provider_error",
        "materialize_error",
    }
    filtered = [record for record in history if record.get("result") in interesting_results]
    if not filtered:
        return "- No prior real-kernel experiments recorded yet."

    lines: list[str] = []
    for record in filtered[-limit:]:
        score_ms = record.get("metric_value")
        score_text = "n/a"
        if isinstance(score_ms, (int, float)):
            score_text = f"{score_ms:.4f} ms"
        focus = record.get("plan_focus_area", "n/a")
        lines.append(
            f"- iter {record['iter']:03d} | {record.get('result', 'unknown')} | "
            f"focus={focus} | correct={record.get('correct')} | "
            f"kind={record.get('failure_kind', 'n/a')} | "
            f"score={score_text} | summary={record.get('summary', 'n/a')}"
        )
    return "\n".join(lines)


def extract_summary(candidate_source: str, fallback: str) -> str:
    for line in candidate_source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return fallback


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _is_helion_kernel(func: ast.FunctionDef) -> bool:
    return any(
        _decorator_name(decorator) in {"helion.kernel", "kernel"}
        for decorator in func.decorator_list
    )


def _shape_mutating_call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    name = _decorator_name(node.func)
    if name.split(".")[-1] in {
        "reshape",
        "view",
        "unsqueeze",
        "squeeze",
        "permute",
        "transpose",
    }:
        return name
    return None


def _branch_shape_mutation_violations(func: ast.FunctionDef) -> list[str]:
    violations: list[str] = []
    for child in ast.walk(func):
        if not isinstance(child, ast.If) or not child.orelse:
            continue

        body_targets: set[str] = set()
        orelse_targets: set[str] = set()

        for statement in child.body:
            for branch_node in ast.walk(statement):
                if not isinstance(branch_node, ast.Assign):
                    continue
                if _shape_mutating_call_name(branch_node.value) is None:
                    continue
                for target in branch_node.targets:
                    if isinstance(target, ast.Name):
                        body_targets.add(target.id)

        for statement in child.orelse:
            for branch_node in ast.walk(statement):
                if not isinstance(branch_node, ast.Assign):
                    continue
                if _shape_mutating_call_name(branch_node.value) is None:
                    continue
                for target in branch_node.targets:
                    if isinstance(target, ast.Name):
                        orelse_targets.add(target.id)

        for target in sorted(body_targets & orelse_targets):
            violations.append(
                f"{func.name}: branch-dependent tensor rank mutation for `{target}` near line {child.lineno}"
            )
    return violations


def validate_helion_kernel_source(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    blocked_types = {
        ast.FunctionDef: "nested function definition",
        ast.AsyncFunctionDef: "nested async function definition",
        ast.ClassDef: "nested class definition",
        ast.Lambda: "lambda expression",
        ast.With: "with statement",
        ast.AsyncWith: "async with statement",
        ast.Try: "try statement",
        ast.Match: "match statement",
        ast.IfExp: "inline ternary expression",
    }

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not _is_helion_kernel(node):
            continue
        violations.extend(_branch_shape_mutation_violations(node))
        for child in ast.walk(node):
            if child is node:
                continue
            for blocked_type, label in blocked_types.items():
                if isinstance(child, blocked_type):
                    line = getattr(child, "lineno", "?")
                    violations.append(
                        f"{node.name}: unsupported {label} on line {line}"
                    )
                    break

    return violations


def classify_failure_kind(message: str | None) -> str:
    return advanced_ranker.classify_failure_message(message)


def choose_search_track(
    *,
    problem_name: str,
    iteration: int,
    run_events: list[dict[str, Any]],
) -> str:
    if problem_name not in advanced_ranker.DELTA_PROBLEMS:
        return "config_only"

    recent_failure_kinds = Counter(
        event.get("failure_kind")
        for event in run_events[-8:]
        if isinstance(event.get("failure_kind"), str)
    )
    if recent_failure_kinds["shared_memory_oor"] >= 1:
        return "config_only"
    if (
        recent_failure_kinds["compile_error"]
        + recent_failure_kinds["jit_file_error"]
        + recent_failure_kinds["invalid_indexing"]
    ) >= 2:
        return "config_only"
    return "config_only" if iteration % 2 == 1 else "structural_only"


def improvement_pct(
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
) -> float:
    baseline = baseline_result.get("score_ms")
    current = current_result.get("score_ms")
    if not isinstance(baseline, (int, float)) or not isinstance(current, (int, float)):
        return 0.0
    if baseline == 0:
        return 0.0
    return ((baseline - current) / baseline) * 100.0


def should_resource_block_candidate(
    *,
    problem_name: str,
    plan: dict[str, Any],
    code_features: dict[str, Any],
    history: list[dict[str, Any]],
) -> str | None:
    if problem_name not in advanced_ranker.DELTA_PROBLEMS:
        return None

    resource_risk = str(code_features.get("code_resource_risk", "low"))
    resource_pressure = float(code_features.get("code_resource_pressure", 0.0))
    failure_counts = advanced_ranker.history_failure_counts(history, problem_name)

    if resource_risk == "high" and failure_counts["shared_memory_oor"] >= 1:
        return (
            "Predicted shared-memory pressure is high and recent runs already hit shared-memory OOR."
        )
    if (
        problem_name == "gated_deltanet_chunk_fwd_o"
        and resource_pressure >= 90.0
    ):
        return "Predicted shared-memory pressure is too high for chunk_fwd_o benchmark shapes."
    if (
        plan.get("search_track") == "structural_only"
        and resource_pressure >= 80.0
        and failure_counts["shared_memory_oor"] >= 2
    ):
        return "Structural variant is likely to exceed the shared-memory budget after recent OOR failures."
    return None


def should_stop_for_plateau(
    *,
    args: argparse.Namespace,
    baseline_result: dict[str, Any],
    best_result: dict[str, Any],
    run_events: list[dict[str, Any]],
) -> str | None:
    candidate_events = [
        event
        for event in run_events
        if event.get("stage") in {"materialization", "test", "benchmark", "screening"}
    ]
    if len(candidate_events) < args.plateau_window:
        return None

    recent_events = candidate_events[-args.plateau_window :]
    compiler_resource_events = [
        event
        for event in recent_events
        if event.get("failure_kind") in COMPILER_OR_RESOURCE_FAILURES
    ]
    if not recent_events:
        return None

    improvement = improvement_pct(baseline_result, best_result)
    failure_ratio = len(compiler_resource_events) / len(recent_events)
    if improvement >= args.plateau_improvement_threshold:
        return None
    if failure_ratio < args.plateau_failure_ratio:
        return None
    return (
        f"Plateau detected: best improvement is {improvement:.2f}% while "
        f"{failure_ratio:.0%} of the last {len(recent_events)} candidate events "
        f"were compiler/resource failures."
    )


def print_result(prefix: str, result: dict[str, Any]) -> None:
    if result.get("correct"):
        print(
            f"{prefix} {result['score_ms']:.4f} ms geomean | "
            f"tests {result['tests_passed']}/{result['tests_total']} | "
            f"benchmarks {result['benchmarks_completed']}/{result['benchmarks_total']}"
        )
    else:
        print(f"{prefix} FAILED | {result.get('message', 'unknown error')}")


def print_summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    if not rows:
        return

    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    print(title)
    print(border)
    for label, value in rows:
        print(f"| {label.ljust(label_width)} | {value.ljust(value_width)} |")
    print(border)


def print_real_performance_table(
    *,
    title: str,
    problem_name: str,
    baseline_result: dict[str, Any],
    current_result: dict[str, Any],
) -> None:
    print(title)
    print(
        render_table(
            ["Kernel", "Latency", "Baseline", "vs Baseline", "Status"],
            [[
                problem_name,
                format_metric(current_result.get("score_ms"), "ms", digits=4),
                format_metric(baseline_result.get("score_ms"), "ms", digits=4),
                format_improvement(
                    baseline_result.get("score_ms"),
                    current_result.get("score_ms"),
                    direction="lower",
                ),
                status_from_improvement(
                    baseline_result.get("score_ms"),
                    current_result.get("score_ms"),
                    direction="lower",
                ),
            ]],
        )
    )


def build_iteration_summary_rows(
    *,
    iteration: int,
    elapsed_seconds: float,
    plans_generated: int,
    plans_materialized: int,
    best_result: dict[str, Any],
    run_events: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    iter_events = [event for event in run_events if event.get("iter") == iteration]
    benchmark_events = [
        event for event in iter_events if event.get("stage") == "benchmark"
    ]
    accepted_events = [event for event in benchmark_events if event.get("result") == "accepted"]
    rejected_events = [event for event in benchmark_events if event.get("result") == "rejected"]
    test_fail_events = [event for event in iter_events if event.get("result") == "test_fail"]
    screened_events = [event for event in iter_events if event.get("result") == "screened_out"]
    resource_blocks = [event for event in iter_events if event.get("result") == "resource_block"]
    provider_errors = [event for event in iter_events if event.get("result") == "provider_error"]
    parse_errors = [event for event in iter_events if event.get("result") == "parse_error"]
    plan_errors = [event for event in iter_events if event.get("result") == "plan_error"]
    syntax_errors = [event for event in iter_events if event.get("result") == "syntax_error"]
    static_blocks = [event for event in iter_events if event.get("result") == "static_block"]

    best_iter_score = None
    if benchmark_events:
        valid_scores = [
            event.get("score_ms")
            for event in benchmark_events
            if isinstance(event.get("score_ms"), (int, float))
        ]
        if valid_scores:
            best_iter_score = min(valid_scores)

    status = "accepted" if accepted_events else "no promotion"
    return [
        ("Iteration", f"{iteration:03d}"),
        ("Elapsed", f"{elapsed_seconds:.1f} sec"),
        ("Plans generated", str(plans_generated)),
        ("Plans materialized", str(plans_materialized)),
        ("Benchmarked", str(len(benchmark_events))),
        ("Accepted", str(len(accepted_events))),
        ("Rejected", str(len(rejected_events))),
        ("Test fails", str(len(test_fail_events))),
        ("Screened out", str(len(screened_events))),
        ("Resource blocks", str(len(resource_blocks))),
        ("Provider errors", str(len(provider_errors))),
        ("Parse errors", str(len(parse_errors))),
        ("Plan errors", str(len(plan_errors))),
        ("Syntax errors", str(len(syntax_errors))),
        ("Static blocks", str(len(static_blocks))),
        ("Best iter score", f"{best_iter_score:.4f} ms" if best_iter_score is not None else "n/a"),
        ("Current best", f"{best_result['score_ms']:.4f} ms"),
        ("Status", status),
    ]


def build_experiment_summary_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    final_result = summary["final_result"]
    baseline_result = summary["baseline_result"]
    return [
        ("Experiment", f"{summary['experiment_number']:03d}"),
        ("Problem", str(summary["problem"])),
        ("Search mode", str(summary["search_mode"])),
        ("Iterations run", str(summary["iterations_completed"])),
        ("Benchmarked", str(summary["candidate_runs"])),
        ("Accepted", str(summary["accepted_count"])),
        ("Rejected", str(summary["rejected_count"])),
        ("Test fails", str(summary.get("test_fail_count", 0))),
        ("Screened out", str(summary.get("screened_out_count", 0))),
        ("Resource blocks", str(summary.get("resource_block_count", 0))),
        ("Provider errors", str(summary.get("provider_error_count", 0))),
        ("Parse errors", str(summary.get("parse_error_count", 0))),
        ("Plan errors", str(summary.get("plan_error_count", 0))),
        ("Syntax errors", str(summary.get("syntax_error_count", 0))),
        ("Static blocks", str(summary.get("static_block_count", 0))),
        ("Baseline", f"{baseline_result['score_ms']:.4f} ms"),
        ("Final best", f"{final_result['score_ms']:.4f} ms"),
        (
            "Vs baseline",
            format_improvement(
                baseline_result.get("score_ms"),
                final_result.get("score_ms"),
                direction="lower",
            ),
        ),
        ("Duration", f"{summary['actual_duration_minutes']:.2f} min"),
        ("Stop reason", str(summary.get("plateau_reason", "n/a"))),
    ]


def combine_eval_results(
    test_result: dict[str, Any],
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    combined = dict(benchmark_result)
    combined["tests_ran"] = True
    combined["tests_total"] = test_result.get("tests_total")
    combined["tests_passed"] = test_result.get("tests_passed")
    combined["correct"] = bool(test_result.get("correct")) and bool(
        benchmark_result.get("correct")
    )
    if not combined["correct"] and combined.get("message") == "PASS":
        combined["message"] = test_result.get("message", "Evaluation failed.")
    return combined


def candidate_record_base(
    *,
    run_context: dict[str, Any],
    args: argparse.Namespace,
    iteration: int,
    plan: dict[str, Any],
    summary: str,
) -> dict[str, Any]:
    return {
        "suite": "real",
        "experiment_number": run_context["experiment_number"],
        "run_name": run_context["run_name"],
        "iter": iteration,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "problem": args.problem,
        "provider": args.provider,
        "model": args.model,
        "search_mode": SEARCH_MODE,
        "summary": summary,
        "plan_id": plan["plan_id"],
        "plan_signature": plan["plan_signature"],
        "plan_focus_area": plan["focus_area"],
        "plan_structural_changes": plan["structural_changes"],
        "plan_target_shapes": plan["target_shapes"],
        "plan_risk": plan["risk"],
        "plan_expected_gain": plan["expected_gain"],
        "plan_search_track": plan.get("search_track", "config_only"),
        "surrogate_score": plan["surrogate_score"],
        "surrogate_uncertainty": plan["surrogate_uncertainty"],
        "surrogate_rank": plan["surrogate_rank"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced hybrid optimizer for the real GPU Mode Helion kernels."
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=DEFAULT_PROVIDER,
        help="Model provider CLI to use (default: claude)",
    )
    parser.add_argument(
        "--problem",
        choices=suite.available_problem_names(),
        default="fp8_quant",
        help="Real problem name to optimize",
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
        default=10,
        help="Maximum outer iterations to try (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Optional provider-specific model alias/name (default: provider default)",
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
        default=12,
        help="How many recent experiments to include in prompts (default: 12)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Reserved for API-style providers; ignored for CLI providers",
    )
    parser.add_argument(
        "--planner-batch-size",
        type=int,
        default=6,
        help="How many structured plans to propose per iteration (default: 6)",
    )
    parser.add_argument(
        "--materialize-top-k",
        type=int,
        default=3,
        help="How many ranked plans to materialize into code (default: 3)",
    )
    parser.add_argument(
        "--benchmark-top-k",
        type=int,
        default=2,
        help="How many test-passing candidates to benchmark fully (default: 2)",
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for correctness-only screening (default: 900)",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for benchmark evaluation (default: 1800)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Print a program-responding heartbeat every N seconds (default: 60)",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=8,
        help="Candidate-event window for plateau detection (default: 8)",
    )
    parser.add_argument(
        "--plateau-improvement-threshold",
        type=float,
        default=3.0,
        help="Stop early if best improvement stays below this percent while failures dominate (default: 3.0)",
    )
    parser.add_argument(
        "--plateau-failure-ratio",
        type=float,
        default=0.6,
        help="Compiler/resource failure ratio required for plateau stop (default: 0.6)",
    )
    return parser.parse_args()


def run_loop(args: argparse.Namespace) -> int:
    ensure_dirs()
    heartbeat = RunHeartbeat(args.heartbeat_seconds)

    if not PROGRAM_PATH.exists():
        print(f"[!] Missing {PROGRAM_PATH}")
        return 1

    try:
        require_provider_binary(args.provider)
    except RuntimeError as exc:
        print(f"[!] {exc}")
        return 1

    problem = suite.load_problem(args.problem)
    program_text = PROGRAM_PATH.read_text(encoding="utf-8")
    current_source = problem.submission_path.read_text(encoding="utf-8")
    run_context = create_run_context(args)
    run_events: list[dict[str, Any]] = []
    history = load_history(args.problem)
    next_iter = max((record["iter"] for record in history), default=0) + 1

    baseline_snapshot_path = run_context["run_dir"] / "baseline_submission.py"
    baseline_snapshot_path.write_text(current_source, encoding="utf-8")
    write_json(
        run_context["run_dir"] / "metadata.json",
        {
            "suite": "real",
            "experiment_number": run_context["experiment_number"],
            "run_name": run_context["run_name"],
            "problem": args.problem,
            "provider": args.provider,
            "model": args.model,
            "search_mode": SEARCH_MODE,
            "metric_label": "Geomean Latency (ms)",
            "metric_unit": "ms",
            "metric_direction": "lower",
            "planned_budget_minutes": args.budget,
            "started_at": run_context["started_at_iso"],
            "problem_directory": str(problem.directory.relative_to(ROOT)),
            "planner_batch_size": args.planner_batch_size,
            "materialize_top_k": args.materialize_top_k,
            "benchmark_top_k": args.benchmark_top_k,
        },
    )

    print(f"\n{'=' * 72}")
    print("Advanced Real Helion Optimizer")
    print(f"Experiment:       {run_context['experiment_number']:03d}")
    print(f"Problem:          {args.problem}")
    print(f"Provider:         {args.provider}")
    print(f"Search mode:      {SEARCH_MODE}")
    print(f"Budget:           {args.budget:.1f} minutes")
    print(f"Outer iterations: {args.iters}")
    print(f"Plan batch size:  {args.planner_batch_size}")
    print(f"Materialize topK: {args.materialize_top_k}")
    print(f"Benchmark topK:   {args.benchmark_top_k}")
    print(f"Heartbeat:        every {args.heartbeat_seconds} sec")
    print(f"Plateau window:   {args.plateau_window} candidate events")
    print(f"Plateau thresh:   {args.plateau_improvement_threshold:.1f}%")
    print(f"Model:            {args.model or 'provider default'}")
    print(f"{'=' * 72}\n")

    heartbeat.start()
    try:
        heartbeat.set_stage("baseline-benchmark")
        print("[*] Benchmarking current submission.py as the starting point...")
        best_result = suite.run_problem_eval(args.problem, mode="both")
        if not best_result.get("correct") or best_result.get("score_ms") is None:
            finished_at = datetime.now()
            summary = {
                "suite": "real",
                "experiment_number": run_context["experiment_number"],
                "run_name": run_context["run_name"],
                "problem": args.problem,
                "provider": args.provider,
                "model": args.model,
                "search_mode": SEARCH_MODE,
                "metric_label": "Geomean Latency (ms)",
                "metric_unit": "ms",
                "metric_direction": "lower",
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
                "screened_out_count": 0,
                "resource_block_count": 0,
                "test_fail_count": 0,
                "provider_error_count": 0,
                "parse_error_count": 0,
                "plan_error_count": 0,
                "syntax_error_count": 0,
                "static_block_count": 0,
                "plateau_reason": None,
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
        print_real_performance_table(
            title="Baseline Snapshot",
            problem_name=args.problem,
            baseline_result=best_result,
            current_result=best_result,
        )
        baseline_result = dict(best_result)
        deadline = time.time() + args.budget * 60
        accepted_count = 0
        iterations_completed = 0
        plateau_reason: str | None = None

        for offset in range(args.iters):
            if time.time() >= deadline:
                break

            iteration = next_iter + offset
            iteration_started_at = time.monotonic()
            iter_tag = f"real_{args.problem}_iter_{iteration:03d}"
            history_text = format_history(history, args.history_limit)
            search_track = choose_search_track(
                problem_name=args.problem,
                iteration=iteration,
                run_events=run_events,
            )
            heartbeat.set_stage(f"iter-{iteration:03d}-planning")
            print(
                f"\n[iter {iteration:03d}] Planning a candidate batch... "
                f"(track={search_track})"
            )
            iterations_completed += 1

            planning_prompt = advanced_real_search.build_plan_prompt(
                problem=problem,
                current_source=current_source,
                program_text=program_text,
                best_result=best_result,
                history_text=history_text,
                batch_size=args.planner_batch_size,
                search_track=search_track,
            )

            try:
                planner_response = request_candidate_source(
                    provider=args.provider,
                    model=args.model,
                    prompt=planning_prompt,
                    max_tokens=args.max_tokens,
                    system_prompt=advanced_real_search.PLANNER_SYSTEM_PROMPT,
                )
            except Exception as exc:
                record = {
                    "suite": "real",
                    "experiment_number": run_context["experiment_number"],
                    "run_name": run_context["run_name"],
                    "iter": iteration,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem": args.problem,
                    "provider": args.provider,
                    "model": args.model,
                    "search_mode": SEARCH_MODE,
                    "stage": "planning",
                    "result": "provider_error",
                    "failure_kind": "provider_error",
                    "correct": False,
                    "summary": str(exc),
                }
                record_event(run_context, run_events, record)
                history.append(record)
                print(f"  [!] Planner provider error: {exc}")
                print(f"[iter {iteration:03d}] Iteration complete")
                print_summary_table(
                    "Iteration Summary",
                    build_iteration_summary_rows(
                        iteration=iteration,
                        elapsed_seconds=time.monotonic() - iteration_started_at,
                        plans_generated=0,
                        plans_materialized=0,
                        best_result=best_result,
                        run_events=run_events,
                    ),
                )
                print_real_performance_table(
                    title="Performance Snapshot",
                    problem_name=args.problem,
                    baseline_result=baseline_result,
                    current_result=best_result,
                )
                time.sleep(2)
                continue

            planner_response_path = run_context["planner_dir"] / f"{iter_tag}_raw.txt"
            planner_response_path.write_text(planner_response, encoding="utf-8")

            try:
                raw_plans = advanced_real_search.extract_json_payload(planner_response)
                plans = advanced_real_search.normalize_plans(
                    problem_name=args.problem,
                    raw_payload=raw_plans,
                    batch_size=args.planner_batch_size,
                    search_track=search_track,
                )
                plans = advanced_real_search.merge_seeded_plans(
                    problem_name=args.problem,
                    search_track=search_track,
                    model_plans=plans,
                    batch_size=args.planner_batch_size,
                )
                ranked_plans = advanced_ranker.rank_plans(plans, history)
            except Exception as exc:
                record = {
                    "suite": "real",
                    "experiment_number": run_context["experiment_number"],
                    "run_name": run_context["run_name"],
                    "iter": iteration,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "problem": args.problem,
                    "provider": args.provider,
                    "model": args.model,
                    "search_mode": SEARCH_MODE,
                    "stage": "planning",
                    "result": "plan_error",
                    "failure_kind": "parse_error",
                    "correct": False,
                    "summary": str(exc),
                    "response_path": str(planner_response_path.relative_to(ROOT)),
                }
                record_event(run_context, run_events, record)
                history.append(record)
                print(f"  [!] Planner parse error: {exc}")
                print(f"[iter {iteration:03d}] Iteration complete")
                print_summary_table(
                    "Iteration Summary",
                    build_iteration_summary_rows(
                        iteration=iteration,
                        elapsed_seconds=time.monotonic() - iteration_started_at,
                        plans_generated=0,
                        plans_materialized=0,
                        best_result=best_result,
                        run_events=run_events,
                    ),
                )
                print_real_performance_table(
                    title="Performance Snapshot",
                    problem_name=args.problem,
                    baseline_result=baseline_result,
                    current_result=best_result,
                )
                continue

            plan_batch_path = run_context["planner_dir"] / f"{iter_tag}_plans.json"
            write_json(plan_batch_path, {"plans": ranked_plans})
            print(
                f"  [*] Generated {len(ranked_plans)} plans; "
                f"top surrogate score {ranked_plans[0]['surrogate_score']:.3f} "
                f"(track={search_track})"
            )

            selected_plans = ranked_plans[: max(1, args.materialize_top_k)]
            passing_candidates: list[dict[str, Any]] = []

            for selected_idx, plan in enumerate(selected_plans, start=1):
                materialize_tag = f"{iter_tag}_{plan['plan_id']}"
                heartbeat.set_stage(
                    f"iter-{iteration:03d}-materialize-{selected_idx}"
                )
                print(
                    f"  [*] Materializing rank {selected_idx}: "
                    f"{plan['summary']} ({plan['focus_area']})"
                )
                materialization_prompt = advanced_real_search.build_materialization_prompt(
                    problem=problem,
                    current_source=current_source,
                    program_text=program_text,
                    plan=plan,
                    history_text=history_text,
                )

                try:
                    raw_candidate_response = request_candidate_source(
                        provider=args.provider,
                        model=args.model,
                        prompt=materialization_prompt,
                        max_tokens=args.max_tokens,
                        system_prompt=advanced_real_search.MATERIALIZER_SYSTEM_PROMPT,
                    )
                except Exception as exc:
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=plan["summary"],
                        ),
                        "stage": "materialization",
                        "result": "provider_error",
                        "failure_kind": "provider_error",
                        "correct": False,
                        "message": str(exc),
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print(f"    [!] Materializer provider error: {exc}")
                    continue

                response_path = REAL_RESPONSES_DIR / f"{materialize_tag}.txt"
                response_path.write_text(raw_candidate_response, encoding="utf-8")

                candidate_source = extract_code_block(raw_candidate_response)
                if not candidate_source:
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=plan["summary"],
                        ),
                        "stage": "materialization",
                        "result": "parse_error",
                        "failure_kind": "parse_error",
                        "correct": False,
                        "response_path": str(response_path.relative_to(ROOT)),
                        "message": "Model response did not include a Python code block.",
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print("    [!] Parse error: no Python code block found.")
                    continue

                summary = extract_summary(candidate_source, plan["summary"])
                candidate_path = REAL_CANDIDATES_DIR / f"{materialize_tag}.py"
                try:
                    compile(candidate_source, str(candidate_path), "exec")
                except SyntaxError as exc:
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=summary,
                        ),
                        "stage": "materialization",
                        "result": "syntax_error",
                        "failure_kind": "syntax_error",
                        "correct": False,
                        "response_path": str(response_path.relative_to(ROOT)),
                        "message": f"Syntax error: {exc.msg} on line {exc.lineno}",
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print(f"    [!] Syntax error: {exc.msg} on line {exc.lineno}")
                    continue

                static_violations = validate_helion_kernel_source(candidate_source)
                if static_violations:
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=summary,
                        ),
                        "stage": "materialization",
                        "result": "static_block",
                        "failure_kind": "unsupported_syntax",
                        "correct": False,
                        "response_path": str(response_path.relative_to(ROOT)),
                        "message": "; ".join(static_violations),
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print(f"    [!] Static block: {'; '.join(static_violations)}")
                    continue

                candidate_path.write_text(candidate_source, encoding="utf-8")
                code_features = advanced_ranker.score_materialized_candidate(
                    problem_name=args.problem,
                    source=candidate_source,
                    baseline_source=current_source,
                    plan=plan,
                    history=history,
                )

                heartbeat.set_stage(f"iter-{iteration:03d}-test-screen")
                print(
                    f"    [*] Test screening: surrogate={plan['surrogate_score']:.3f} "
                    f"static={code_features['code_static_score']:.3f} "
                    f"resource={code_features['code_resource_risk']}"
                )
                try:
                    test_result = suite.evaluate_candidate_source(
                        args.problem,
                        candidate_source,
                        mode="test",
                        timeout_seconds=args.test_timeout,
                    )
                except Exception as exc:
                    test_result = {
                        "correct": False,
                        "message": f"Test run crashed: {exc}",
                        "metric_value": None,
                    }

                if not test_result.get("correct"):
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=summary,
                        ),
                        **code_features,
                        "stage": "test",
                        "result": "test_fail",
                        "failure_kind": classify_failure_kind(test_result.get("message")),
                        "correct": False,
                        "response_path": str(response_path.relative_to(ROOT)),
                        "candidate_path": str(candidate_path.relative_to(ROOT)),
                        "message": test_result.get("message"),
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print(f"    [!] Test failed: {test_result.get('message')}")
                    continue

                resource_block_reason = should_resource_block_candidate(
                    problem_name=args.problem,
                    plan=plan,
                    code_features=code_features,
                    history=history,
                )
                if resource_block_reason:
                    record = {
                        **candidate_record_base(
                            run_context=run_context,
                            args=args,
                            iteration=iteration,
                            plan=plan,
                            summary=summary,
                        ),
                        **code_features,
                        "stage": "screening",
                        "result": "resource_block",
                        "failure_kind": "shared_memory_oor",
                        "correct": True,
                        "response_path": str(response_path.relative_to(ROOT)),
                        "candidate_path": str(candidate_path.relative_to(ROOT)),
                        "message": resource_block_reason,
                    }
                    record_event(run_context, run_events, record)
                    history.append(record)
                    print(f"    [!] Resource block: {resource_block_reason}")
                    continue

                benchmark_priority = round(
                    code_features["code_static_score"]
                    - (plan["surrogate_uncertainty"] * 0.35)
                    - (float(code_features.get("code_resource_pressure", 0.0)) * 0.004),
                    4,
                )
                passing_candidates.append(
                    {
                        "plan": plan,
                        "summary": summary,
                        "candidate_source": candidate_source,
                        "candidate_path": candidate_path,
                        "response_path": response_path,
                        "code_features": code_features,
                        "test_result": test_result,
                        "benchmark_priority": benchmark_priority,
                    }
                )

            if not passing_candidates:
                print(f"[iter {iteration:03d}] Iteration complete")
                print_summary_table(
                    "Iteration Summary",
                    build_iteration_summary_rows(
                        iteration=iteration,
                        elapsed_seconds=time.monotonic() - iteration_started_at,
                        plans_generated=len(ranked_plans),
                        plans_materialized=len(selected_plans),
                        best_result=best_result,
                        run_events=run_events,
                    ),
                )
                print_real_performance_table(
                    title="Performance Snapshot",
                    problem_name=args.problem,
                    baseline_result=baseline_result,
                    current_result=best_result,
                )
                plateau_reason = should_stop_for_plateau(
                    args=args,
                    baseline_result=baseline_result,
                    best_result=best_result,
                    run_events=run_events,
                )
                if plateau_reason:
                    print(f"[plateau] {plateau_reason}")
                    break
                continue

            passing_candidates.sort(
                key=lambda item: (
                    item["benchmark_priority"],
                    item["plan"]["surrogate_score"],
                    -item["plan"]["surrogate_uncertainty"],
                ),
                reverse=True,
            )
            benchmark_candidates = passing_candidates[: max(1, args.benchmark_top_k)]
            screened_out_candidates = passing_candidates[max(1, args.benchmark_top_k) :]

            for screened in screened_out_candidates:
                record = {
                    **candidate_record_base(
                        run_context=run_context,
                        args=args,
                        iteration=iteration,
                        plan=screened["plan"],
                        summary=screened["summary"],
                    ),
                    **screened["code_features"],
                    "stage": "screening",
                    "result": "screened_out",
                    "correct": True,
                    "response_path": str(screened["response_path"].relative_to(ROOT)),
                    "candidate_path": str(screened["candidate_path"].relative_to(ROOT)),
                    "benchmark_priority": screened["benchmark_priority"],
                    "message": "Passed tests but was not selected for full benchmarking.",
                }
                record_event(run_context, run_events, record)
                history.append(record)

            benchmark_outcomes: list[dict[str, Any]] = []
            for candidate_idx, candidate in enumerate(benchmark_candidates, start=1):
                heartbeat.set_stage(
                    f"iter-{iteration:03d}-benchmark-{candidate_idx}"
                )
                print(f"  [*] Full benchmark: {candidate['summary']}")
                try:
                    benchmark_result = suite.evaluate_candidate_source(
                        args.problem,
                        candidate["candidate_source"],
                        mode="benchmark",
                        timeout_seconds=args.benchmark_timeout,
                    )
                except Exception as exc:
                    benchmark_result = {
                        "correct": False,
                        "message": f"Benchmark run crashed: {exc}",
                        "metric_value": None,
                        "score_ms": None,
                    }

                combined_result = combine_eval_results(
                    candidate["test_result"],
                    benchmark_result,
                )
                improvement_ratio = None
                failure_kind = None
                if combined_result.get("correct") and combined_result.get("score_ms"):
                    improvement_ratio = (
                        best_result["score_ms"] / combined_result["score_ms"]
                    )
                else:
                    failure_kind = classify_failure_kind(combined_result.get("message"))

                benchmark_outcomes.append(
                    {
                        **candidate,
                        "combined_result": combined_result,
                        "improvement_ratio": improvement_ratio,
                        "failure_kind": failure_kind,
                    }
                )

            winner: dict[str, Any] | None = None
            required_ratio = 1.0 + (args.min_improvement / 100.0)
            for outcome in benchmark_outcomes:
                result = outcome["combined_result"]
                ratio = outcome["improvement_ratio"]
                if not result.get("correct") or ratio is None:
                    continue
                if ratio < required_ratio:
                    continue
                if (
                    winner is None
                    or result["score_ms"] < winner["combined_result"]["score_ms"]
                ):
                    winner = outcome

            if winner is not None:
                accepted_count += 1
                current_source = winner["candidate_source"]
                best_result = winner["combined_result"]
                problem.submission_path.write_text(current_source, encoding="utf-8")
                accepted_path = (
                    REAL_ACCEPTED_DIR / f"{iter_tag}_{winner['plan']['plan_id']}.py"
                )
                accepted_path.write_text(current_source, encoding="utf-8")

            for outcome in benchmark_outcomes:
                result = outcome["combined_result"]
                accepted = (
                    winner is not None
                    and outcome["plan"]["plan_id"] == winner["plan"]["plan_id"]
                )
                record = {
                    **candidate_record_base(
                        run_context=run_context,
                        args=args,
                        iteration=iteration,
                        plan=outcome["plan"],
                        summary=outcome["summary"],
                    ),
                    **outcome["code_features"],
                    "stage": "benchmark",
                    "result": "accepted" if accepted else "rejected",
                    "failure_kind": (
                        None if accepted else (outcome["failure_kind"] or "insufficient_gain")
                    ),
                    "correct": bool(result.get("correct")),
                    "metric_value": result.get("metric_value"),
                    "unit": result.get("unit"),
                    "avg_ms": result.get("avg_ms"),
                    "score_ms": result.get("score_ms"),
                    "improvement_ratio": outcome["improvement_ratio"],
                    "min_improvement_pct": args.min_improvement,
                    "response_path": str(outcome["response_path"].relative_to(ROOT)),
                    "candidate_path": str(outcome["candidate_path"].relative_to(ROOT)),
                    "benchmark_priority": outcome["benchmark_priority"],
                    "tests_total": result.get("tests_total"),
                    "tests_passed": result.get("tests_passed"),
                    "benchmarks_total": result.get("benchmarks_total"),
                    "benchmarks_completed": result.get("benchmarks_completed"),
                    "message": result.get("message"),
                }
                record_event(run_context, run_events, record)
                history.append(record)
                print_result(
                    "  [accepted]" if accepted else "  [rejected]",
                    result,
                )

            print(f"[iter {iteration:03d}] Iteration complete")
            print_summary_table(
                "Iteration Summary",
                build_iteration_summary_rows(
                    iteration=iteration,
                    elapsed_seconds=time.monotonic() - iteration_started_at,
                    plans_generated=len(ranked_plans),
                    plans_materialized=len(selected_plans),
                    best_result=best_result,
                    run_events=run_events,
                ),
            )
            print_real_performance_table(
                title="Performance Snapshot",
                problem_name=args.problem,
                baseline_result=baseline_result,
                current_result=best_result,
            )
            plateau_reason = should_stop_for_plateau(
                args=args,
                baseline_result=baseline_result,
                best_result=best_result,
                run_events=run_events,
            )
            if plateau_reason:
                print(f"[plateau] {plateau_reason}")
                break

        heartbeat.set_stage("final-report")
        finished_at = datetime.now()
        final_submission_path = run_context["run_dir"] / "final_submission.py"
        final_submission_path.write_text(current_source, encoding="utf-8")
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
        static_block_count = sum(
            1 for event in run_events if event["result"] == "static_block"
        )
        rejected_count = sum(
            1 for event in run_events if event["result"] == "rejected"
        )
        screened_out_count = sum(
            1 for event in run_events if event["result"] == "screened_out"
        )
        resource_block_count = sum(
            1 for event in run_events if event["result"] == "resource_block"
        )
        test_fail_count = sum(
            1 for event in run_events if event["result"] == "test_fail"
        )
        benchmarked_candidates = sum(
            1 for event in run_events if event["stage"] == "benchmark"
        )

        summary = {
            "suite": "real",
            "experiment_number": run_context["experiment_number"],
            "run_name": run_context["run_name"],
            "problem": args.problem,
            "provider": args.provider,
            "model": args.model,
            "search_mode": SEARCH_MODE,
            "metric_label": "Geomean Latency (ms)",
            "metric_unit": "ms",
            "metric_direction": "lower",
            "planned_budget_minutes": args.budget,
            "started_at": run_context["started_at_iso"],
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "actual_duration_minutes": round(
                (finished_at - run_context["started_at"]).total_seconds() / 60.0, 2
            ),
            "status": "plateau_stop" if plateau_reason else "completed",
            "iterations_requested": args.iters,
            "iterations_completed": iterations_completed,
            "candidate_runs": benchmarked_candidates,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "screened_out_count": screened_out_count,
            "resource_block_count": resource_block_count,
            "test_fail_count": test_fail_count,
            "provider_error_count": provider_error_count,
            "parse_error_count": parse_error_count,
            "plan_error_count": plan_error_count,
            "syntax_error_count": syntax_error_count,
            "static_block_count": static_block_count,
            "plateau_reason": plateau_reason,
            "planner_batch_size": args.planner_batch_size,
            "materialize_top_k": args.materialize_top_k,
            "benchmark_top_k": args.benchmark_top_k,
            "baseline_result": baseline_result,
            "final_result": best_result,
            "run_dir": str(run_context["run_dir"].relative_to(ROOT)),
            "baseline_kernel_path": str(
                baseline_snapshot_path.relative_to(run_context["run_dir"])
            ),
            "final_kernel_path": str(
                final_submission_path.relative_to(run_context["run_dir"])
            ),
            "events_path": str(
                run_context["events_path"].relative_to(run_context["run_dir"])
            ),
            "report_path": None,
        }
        write_json(run_context["summary_path"], summary)

        report_path = generate_report_for_run(run_context["run_dir"])
        shutil.copyfile(report_path, LATEST_REPORT_PATH)
        summary["report_path"] = str(report_path.relative_to(run_context["run_dir"]))
        summary["latest_report_path"] = str(LATEST_REPORT_PATH.relative_to(ROOT))
        write_json(run_context["summary_path"], summary)
        append_jsonl(INDEX_PATH, summary)

        print(
            f"[experiment {run_context['experiment_number']:03d}] "
            "Experiment complete"
        )
        print_summary_table(
            "Experiment Summary",
            build_experiment_summary_rows(summary),
        )
        print_real_performance_table(
            title="Experiment Performance Summary",
            problem_name=args.problem,
            baseline_result=baseline_result,
            current_result=best_result,
        )
        print(f"\n{'=' * 72}")
        print("Run complete")
        print(f"Experiment number:    {run_context['experiment_number']:03d}")
        print(f"Accepted candidates: {accepted_count}")
        print(f"Best result:         {best_result['score_ms']:.4f} ms geomean")
        print(f"Experiment log:      {LOG_PATH}")
        print(f"Run report:          {report_path}")
        print(f"{'=' * 72}\n")
        return 0
    finally:
        heartbeat.stop()


def main() -> int:
    args = parse_args()
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
