from __future__ import annotations

import math
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
REAL_PROBLEMS_DIR = ROOT / "real_problems"
EVAL_PATH = REAL_PROBLEMS_DIR / "eval.py"

PROBLEM_DIRS = {
    "fp8_quant": "fp8_quant_py",
    "causal_conv1d": "causal_conv1d_py",
    "gated_deltanet_chunk_fwd_h": "gated_deltanet_chunk_fwd_h_py",
    "gated_deltanet_chunk_fwd_o": "gated_deltanet_chunk_fwd_o_py",
    "gated_deltanet_recompute_w_u": "gated_deltanet_recompute_w_u_py",
}

TEST_RESULT_PATTERN = re.compile(r"Test\s+\d+:\s+(PASS|FAIL)")
BENCHMARK_PATTERN = re.compile(
    r"Benchmark\s+\d+:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
)


@dataclass(frozen=True)
class RealProblem:
    name: str
    directory_name: str
    directory: Path
    task: dict[str, Any]

    @property
    def description(self) -> str:
        return str(self.task.get("description", "")).strip()

    @property
    def tests(self) -> list[dict[str, Any]]:
        return list(self.task.get("tests", []))

    @property
    def benchmarks(self) -> list[dict[str, Any]]:
        return list(self.task.get("benchmarks", []))

    @property
    def ranking_by(self) -> str:
        return str(self.task.get("ranking_by", "geom")).strip().lower()

    @property
    def submission_path(self) -> Path:
        return self.directory / "submission.py"

    @property
    def reference_path(self) -> Path:
        return self.directory / "reference.py"

    @property
    def task_path(self) -> Path:
        return self.directory / "task.py"

    @property
    def task_yml_path(self) -> Path:
        return self.directory / "task.yml"


def available_problem_names() -> list[str]:
    return sorted(PROBLEM_DIRS)


def load_problem(problem_name: str) -> RealProblem:
    if problem_name not in PROBLEM_DIRS:
        raise ValueError(
            f"Unknown real problem `{problem_name}`. "
            f"Available: {', '.join(available_problem_names())}"
        )

    directory = REAL_PROBLEMS_DIR / PROBLEM_DIRS[problem_name]
    task_path = directory / "task.yml"
    if not directory.exists():
        raise FileNotFoundError(f"Missing problem directory: {directory}")
    if not task_path.exists():
        raise FileNotFoundError(f"Missing task.yml: {task_path}")

    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    return RealProblem(
        name=problem_name,
        directory_name=PROBLEM_DIRS[problem_name],
        directory=directory,
        task=task,
    )


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarise_latency(problem: RealProblem, benchmark_means_ms: list[float]) -> float | None:
    if not benchmark_means_ms:
        return None
    if problem.ranking_by == "geom":
        return geometric_mean(benchmark_means_ms)
    return sum(benchmark_means_ms) / len(benchmark_means_ms)


def parse_eval_output(
    problem: RealProblem,
    mode: str,
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Any]:
    test_statuses = TEST_RESULT_PATTERN.findall(stdout)
    benchmark_means_ms = [float(match) for match in BENCHMARK_PATTERN.findall(stdout)]
    score_ms = summarise_latency(problem, benchmark_means_ms)

    message = "PASS" if returncode == 0 else _extract_failure_message(stdout, stderr)

    return {
        "name": problem.name,
        "mode": mode,
        "correct": returncode == 0,
        "message": message,
        "tests_ran": mode in {"test", "both"},
        "tests_total": len(problem.tests),
        "tests_passed": sum(1 for status in test_statuses if status == "PASS"),
        "benchmarks_ran": mode in {"benchmark", "both"},
        "benchmarks_total": len(problem.benchmarks),
        "benchmarks_completed": len(benchmark_means_ms),
        "benchmark_means_ms": benchmark_means_ms,
        "score_ms": score_ms,
        "avg_ms": score_ms,
        "metric_value": score_ms,
        "unit": "ms",
        "metric_label": "Geomean Latency (ms)",
        "metric_direction": "lower",
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


def _extract_failure_message(stdout: str, stderr: str) -> str:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if "FAIL" in stripped or "Error" in stripped or "failed" in stripped.lower():
            return stripped
    stderr = stderr.strip()
    if stderr:
        return stderr
    return "Evaluation failed."


def run_problem_eval(
    problem_name: str,
    *,
    mode: str = "both",
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    problem = load_problem(problem_name)
    completed = subprocess.run(
        [sys.executable, str(EVAL_PATH), mode, str(problem.directory)],
        cwd=str(REAL_PROBLEMS_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    return parse_eval_output(
        problem,
        mode,
        completed.stdout,
        completed.stderr,
        completed.returncode,
    )


@contextmanager
def temporary_submission(problem_name: str, candidate_source: str):
    problem = load_problem(problem_name)
    original_source = problem.submission_path.read_text(encoding="utf-8")
    problem.submission_path.write_text(candidate_source, encoding="utf-8")
    try:
        yield problem
    finally:
        problem.submission_path.write_text(original_source, encoding="utf-8")


def evaluate_candidate_source(
    problem_name: str,
    candidate_source: str,
    *,
    mode: str = "both",
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    with temporary_submission(problem_name, candidate_source):
        return run_problem_eval(
            problem_name,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )
