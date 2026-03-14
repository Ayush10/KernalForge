from __future__ import annotations

import argparse
import json
import math
import re
from html import escape
from pathlib import Path
from typing import Any

from summary_tables import (
    format_improvement,
    format_metric,
    format_ratio,
    status_from_improvement,
)

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"


def metric_direction(summary: dict[str, Any]) -> str:
    return str(summary.get("metric_direction", "higher")).lower()


def metric_unit(summary: dict[str, Any]) -> str:
    unit = summary.get("metric_unit")
    if unit:
        return str(unit)
    baseline = summary.get("baseline_result", {})
    return str(baseline.get("unit", ""))


def metric_label(summary: dict[str, Any]) -> str:
    label = summary.get("metric_label")
    if label:
        return str(label)
    unit = metric_unit(summary)
    return f"Throughput ({unit})" if unit else "Metric"


def result_metric_value(payload: dict[str, Any]) -> float | None:
    for key in ("metric_value", "throughput", "score", "avg_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def compute_delta_summary(summary: dict[str, Any]) -> tuple[float, float]:
    baseline = result_metric_value(summary["baseline_result"])
    final = result_metric_value(summary["final_result"])
    if baseline is None or final is None:
        return 0.0, 0.0

    raw_delta = final - baseline
    if baseline == 0:
        return raw_delta, 0.0

    if metric_direction(summary) == "lower":
        improvement_pct = ((baseline - final) / baseline) * 100.0
    else:
        improvement_pct = ((final - baseline) / baseline) * 100.0
    return raw_delta, improvement_pct


def render_run_summary_table(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_result"]
    final = summary["final_result"]
    direction = metric_direction(summary)
    unit = metric_unit(summary)
    metric_header = metric_label(summary)
    status = status_from_improvement(
        result_metric_value(baseline),
        result_metric_value(final),
        direction=direction,
    )

    headers = ["Kernel", metric_header, "Baseline", "vs Baseline", "Status"]
    values = [
        escape(str(summary.get("problem", "n/a"))),
        escape(format_metric(result_metric_value(final), unit)),
        escape(format_metric(result_metric_value(baseline), unit)),
        escape(
            format_improvement(
                result_metric_value(baseline),
                result_metric_value(final),
                direction=direction,
            )
        ),
        escape(status),
    ]

    if isinstance(final.get("speedup"), (int, float)):
        headers.insert(3, "vs Eager")
        values.insert(3, escape(format_ratio(final.get("speedup"))))

    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    value_html = "".join(f"<td>{value}</td>" for value in values)
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody><tr>{value_html}</tr></tbody>"
        "</table>"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def list_run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    run_dirs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.match(r"experiment_(\d+)_", path.name)
        number = int(match.group(1)) if match else -1
        return number, path.name

    return sorted(run_dirs, key=sort_key)


def find_run_dir(experiment_number: int) -> Path | None:
    prefix = f"experiment_{experiment_number:03d}_"
    for run_dir in list_run_dirs():
        if run_dir.name.startswith(prefix):
            return run_dir
    return None


def svg_before_after(summary: dict[str, Any]) -> str:
    baseline = result_metric_value(summary["baseline_result"]) or 0.0
    final = result_metric_value(summary["final_result"]) or 0.0
    unit = metric_unit(summary)

    width = 680
    height = 320
    margin_left = 80
    margin_bottom = 60
    margin_top = 30
    chart_height = height - margin_top - margin_bottom
    bar_width = 160
    max_value = max(baseline, final, 1.0)
    scale = chart_height / max_value

    bars = [
        ("Baseline", baseline, "#7c8aa5", margin_left + 80),
        ("Final", final, "#2f9e44", margin_left + 320),
    ]

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="320" role="img" aria-label="Before and after throughput chart">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="#c9d1d9" stroke-width="2" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#c9d1d9" stroke-width="2" />',
    ]

    for idx in range(6):
        y_value = max_value * idx / 5
        y = height - margin_bottom - (y_value * scale)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#eef2f6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#5b6576">{y_value:.1f}</text>'
        )

    for label, value, color, x in bars:
        bar_height = value * scale
        y = height - margin_bottom - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="10" fill="{color}" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle" font-size="14" fill="#1f2937">{value:.2f} {escape(unit)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="13" fill="#334155">{escape(label)}</text>'
        )

    parts.append(
        f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="13" fill="#475569">{escape(metric_label(summary))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_progress(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    metric_events = [
        event for event in events if result_metric_value(event) is not None
    ]
    if not metric_events:
        return "<p>No benchmarked candidate data was recorded for this experiment.</p>"

    width = 920
    height = 360
    margin_left = 80
    margin_right = 30
    margin_top = 30
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    baseline = result_metric_value(summary["baseline_result"]) or 0.0
    all_values = [baseline] + [
        result_metric_value(event) or 0.0 for event in metric_events
    ]
    min_value = min(all_values)
    max_value = max(all_values)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0
    padding = (max_value - min_value) * 0.15
    min_value -= padding
    max_value += padding

    def x_pos(index: int) -> float:
        if len(metric_events) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index / (len(metric_events) - 1)) * plot_width

    def y_pos(value: float) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return margin_top + (1.0 - ratio) * plot_height

    best_so_far: list[float] = []
    current_best = baseline
    keep_higher = metric_direction(summary) != "lower"
    for event in metric_events:
        value = result_metric_value(event) or 0.0
        if keep_higher:
            current_best = max(current_best, value)
        else:
            current_best = min(current_best, value)
        best_so_far.append(current_best)

    def polyline(points: list[tuple[float, float]], color: str, width_px: int, dash: str = "") -> str:
        joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="{width_px}"'
            f'{dash_attr} points="{joined}" />'
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="360" role="img" aria-label="Experiment progress chart">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
    ]

    for idx in range(6):
        value = min_value + ((max_value - min_value) * idx / 5)
        y = y_pos(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#eef2f6" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#5b6576">{value:.2f}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#c9d1d9" stroke-width="2" />'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#c9d1d9" stroke-width="2" />'
    )

    baseline_y = y_pos(baseline)
    parts.append(
        f'<line x1="{margin_left}" y1="{baseline_y:.1f}" x2="{width - margin_right}" y2="{baseline_y:.1f}" stroke="#9aa6b2" stroke-width="2" stroke-dasharray="6 6" />'
    )
    parts.append(
        f'<text x="{width - margin_right}" y="{baseline_y - 8:.1f}" text-anchor="end" font-size="12" fill="#6b7280">Baseline {baseline:.2f}</text>'
    )

    trial_points = [
        (x_pos(idx), y_pos(result_metric_value(event) or 0.0))
        for idx, event in enumerate(metric_events)
    ]
    best_points = [
        (x_pos(idx), y_pos(value))
        for idx, value in enumerate(best_so_far)
    ]
    parts.append(polyline(trial_points, "#93a4b7", 2))
    parts.append(polyline(best_points, "#2f9e44", 3))

    for idx, event in enumerate(metric_events):
        x = x_pos(idx)
        y = y_pos(result_metric_value(event) or 0.0)
        color = "#2f9e44" if event["result"] == "accepted" else "#6b7280"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" />')
        parts.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="11" fill="#475569">{event["iter"]}</text>'
        )

    parts.append(
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#475569">Iteration number</text>'
    )
    parts.append(
        f'<text x="{20}" y="{height / 2}" transform="rotate(-90, 20, {height / 2})" text-anchor="middle" font-size="13" fill="#475569">{escape(metric_label(summary))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_batch_table(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if not events:
        return "<p>No iteration events were recorded.</p>"

    metric_header = escape(metric_label(summary))
    rows: list[str] = []
    for event in events:
        rows.append(
            "<tr>"
            f"<td>{event.get('iter', 'n/a')}</td>"
            f"<td>{escape(str(event.get('timestamp', 'n/a')))}</td>"
            f"<td>{escape(str(event.get('stage', 'run')))}</td>"
            f"<td>{escape(str(event.get('result', 'n/a')))}</td>"
            f"<td>{escape(str(event.get('correct', 'n/a')))}</td>"
            f"<td>{escape(format_number(event.get('surrogate_score'), 3))}</td>"
            f"<td>{escape(format_number(result_metric_value(event)))}</td>"
            f"<td>{escape(format_number(event.get('avg_ms'), 3))}</td>"
            f"<td>{escape(str(event.get('summary', '')))}</td>"
            "</tr>"
        )

    return (
        "<table>"
        "<thead><tr>"
        "<th>Iter</th><th>Timestamp</th><th>Stage</th><th>Result</th><th>Correct</th>"
        f"<th>Surrogate</th><th>{metric_header}</th><th>Latency (ms)</th><th>Summary</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_summary_cards(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_result"]
    final = summary["final_result"]
    raw_delta, improvement_pct = compute_delta_summary(summary)
    unit = metric_unit(summary)
    cards = [
        ("Experiment", f"{summary['experiment_number']:03d}"),
        ("Suite", str(summary.get("suite", "n/a"))),
        ("Started", summary["started_at"]),
        ("Finished", summary["finished_at"]),
        ("Provider", summary["provider"]),
        ("Search", summary.get("search_mode", "n/a")),
        ("Runs", str(summary["iterations_completed"])),
        ("Benchmarked", str(summary["candidate_runs"])),
        ("Accepted", str(summary["accepted_count"])),
        ("Rejected", str(summary["rejected_count"])),
        ("Plan Errors", str(summary.get("plan_error_count", 0))),
        ("Static Blocks", str(summary.get("static_block_count", 0))),
        ("Test Fails", str(summary.get("test_fail_count", 0))),
        ("Screened Out", str(summary.get("screened_out_count", 0))),
        ("Resource Blocks", str(summary.get("resource_block_count", 0))),
        ("Baseline", f"{format_number(result_metric_value(baseline))} {unit}".strip()),
        ("Final", f"{format_number(result_metric_value(final))} {unit}".strip()),
        ("Delta", f"{raw_delta:+.2f} {unit}".strip()),
        (
            "Vs Baseline",
            format_improvement(
                result_metric_value(baseline),
                result_metric_value(final),
                direction=metric_direction(summary),
            ),
        ),
        ("Stop Reason", str(summary.get("plateau_reason", "n/a"))),
        ("Budget", f"{summary['planned_budget_minutes']:.0f} min"),
    ]
    if isinstance(final.get("speedup"), (int, float)):
        cards.append(("Vs Eager", format_ratio(final.get("speedup"))))
    if "planner_batch_size" in summary:
        cards.append(("Plan Batch", str(summary["planner_batch_size"])))
    if "materialize_top_k" in summary:
        cards.append(("Materialize", str(summary["materialize_top_k"])))
    if "benchmark_top_k" in summary:
        cards.append(("Benchmark TopK", str(summary["benchmark_top_k"])))
    return "".join(
        f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
        for label, value in cards
    )


def generate_report_for_run(run_dir: Path, output_path: Path | None = None) -> Path:
    summary = read_json(run_dir / "summary.json")
    events = read_jsonl(run_dir / "events.jsonl")
    output = output_path or (run_dir / "report.html")

    title = (
        f"Experiment {summary['experiment_number']:03d} | "
        f"{summary['problem']} | {summary['provider']}"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 0;
      padding: 32px;
      color: #1f2937;
      background: #f7fafc;
    }}
    h1, h2 {{
      margin: 0 0 14px 0;
    }}
    p {{
      line-height: 1.5;
    }}
    .muted {{
      color: #64748b;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 28px 0;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #dbe4ee;
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #64748b;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      color: #0f172a;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #dbe4ee;
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 22px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid #e7edf3;
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #475569;
      font-weight: 600;
      background: #f8fafc;
    }}
    code {{
      font-family: Consolas, monospace;
      background: #eff6ff;
      padding: 1px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="muted">
    Hourly experiment report for <code>{escape(summary['run_name'])}</code>.
    This report summarizes baseline vs final performance, iteration-by-iteration progress,
    and the full batch results table for the run.
  </p>

  <section class="grid">
    {render_summary_cards(summary)}
  </section>

  <section class="panel">
    <h2>Summary Table</h2>
    <p class="muted">
      Baseline and final performance for this experiment, with the net change made by the optimizer.
    </p>
    {render_run_summary_table(summary)}
  </section>

  <section class="panel">
    <h2>Before and After</h2>
    {svg_before_after(summary)}
  </section>

  <section class="panel">
    <h2>Progress Over Iterations</h2>
    <p class="muted">
      Gray line shows candidate metric values. Green line shows best-so-far values.
      Accepted candidates are green points.
    </p>
    {svg_progress(summary, events)}
  </section>

  <section class="panel">
    <h2>Batch Results</h2>
    <p class="muted">
      Each row is one attempted iteration with timestamp, outcome, and measured result.
    </p>
    {render_batch_table(summary, events)}
  </section>
</body>
</html>
"""

    output.write_text(html, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report for an experiment run."
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        help="Experiment number to render (default: latest run)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run directory to render instead of selecting by number",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the HTML report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.experiment is not None:
        run_dir = find_run_dir(args.experiment)
        if run_dir is None:
            print(f"Experiment {args.experiment:03d} not found.")
            return 1
    else:
        run_dirs = list_run_dirs()
        if not run_dirs:
            print("No experiment runs found.")
            return 1
        run_dir = run_dirs[-1]

    output = Path(args.output).resolve() if args.output else None
    report_path = generate_report_for_run(run_dir, output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
