from __future__ import annotations

import argparse

import real_problem_suite as suite
from summary_tables import render_table


def format_score(result: dict) -> str:
    score_ms = result.get("score_ms")
    if isinstance(score_ms, (int, float)):
        return f"{score_ms:.4f} ms geomean"
    return "n/a"


def print_result(problem_name: str, result: dict, *, show_output: bool) -> None:
    status = "PASS" if result.get("correct") else "FAIL"
    tests_text = (
        f"{result.get('tests_passed', 0)}/{result.get('tests_total', 0)}"
        if result.get("tests_ran")
        else "skipped"
    )
    benchmarks_text = (
        f"{result.get('benchmarks_completed', 0)}/{result.get('benchmarks_total', 0)}"
        if result.get("benchmarks_ran")
        else "skipped"
    )
    print(
        f"[{status}] {problem_name} | "
        f"tests={tests_text} | "
        f"benchmarks={benchmarks_text} | "
        f"score={format_score(result)}"
    )
    if result.get("message") and result.get("message") != "PASS":
        print(f"  {result['message']}")
    if show_output:
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the real GPU Mode Helion kernels."
    )
    parser.add_argument(
        "--problem",
        choices=suite.available_problem_names(),
        default=None,
        help="Run a single real problem (default: run all problems)",
    )
    parser.add_argument(
        "--mode",
        choices=("test", "benchmark", "both"),
        default="both",
        help="Evaluation mode to run (default: both)",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the underlying eval.py stdout/stderr for each problem",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_names = (
        [args.problem] if args.problem else suite.available_problem_names()
    )

    failed = False
    results: list[tuple[str, dict]] = []
    for problem_name in problem_names:
        result = suite.run_problem_eval(problem_name, mode=args.mode)
        print_result(problem_name, result, show_output=args.show_output)
        results.append((problem_name, result))
        failed = failed or not bool(result.get("correct"))

    if results:
        summary_rows = []
        failures = []
        for problem_name, result in results:
            tests_text = (
                f"{result.get('tests_passed', 0)}/{result.get('tests_total', 0)}"
                if result.get("tests_ran")
                else "skipped"
            )
            benchmarks_text = (
                f"{result.get('benchmarks_completed', 0)}/{result.get('benchmarks_total', 0)}"
                if result.get("benchmarks_ran")
                else "skipped"
            )
            summary_rows.append([
                problem_name,
                format_score(result),
                tests_text,
                benchmarks_text,
                "PASS" if result.get("correct") else "FAIL",
            ])
            if not result.get("correct"):
                failures.append(f"{problem_name}: {result.get('message', 'unknown error')}")

        print("\nSummary")
        print(
            render_table(
                ["Kernel", "Latency", "Tests", "Benchmarks", "Status"],
                summary_rows,
            )
        )
        if failures:
            print("\nFailures:")
            for failure in failures:
                print(f"  - {failure}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
