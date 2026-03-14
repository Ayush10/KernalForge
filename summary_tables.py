from __future__ import annotations

from typing import Iterable


def render_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    if not headers:
        return ""

    widths = []
    for idx, header in enumerate(headers):
        cell_widths = [len(header)]
        for row in normalized_rows:
            if idx < len(row):
                cell_widths.append(len(row[idx]))
        widths.append(max(cell_widths))

    def border(left: str, middle: str, right: str, fill: str = "─") -> str:
        return left + middle.join(fill * (width + 2) for width in widths) + right

    def format_row(values: list[str]) -> str:
        padded = [
            f" {values[idx].ljust(widths[idx])} "
            for idx in range(len(headers))
        ]
        return "|" + "|".join(padded) + "|"

    lines = [border("+", "+", "+", "-"), format_row(headers), border("+", "+", "+", "-")]
    for row in normalized_rows:
        if len(row) < len(headers):
            row = row + ([""] * (len(headers) - len(row)))
        lines.append(format_row(row))
    lines.append(border("+", "+", "+", "-"))
    return "\n".join(lines)


def format_metric(value: object, unit: str = "", digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    number = f"{float(value):,.{digits}f}"
    return f"{number} {unit}".strip()


def format_ratio(value: object, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}x"


def compute_improvement(
    baseline: object,
    current: object,
    *,
    direction: str = "higher",
) -> tuple[float | None, float | None]:
    if not isinstance(baseline, (int, float)) or not isinstance(current, (int, float)):
        return None, None

    baseline_value = float(baseline)
    current_value = float(current)
    if baseline_value == 0:
        return None, None

    if direction == "lower":
        if current_value == 0:
            return None, None
        pct = ((baseline_value - current_value) / baseline_value) * 100.0
        ratio = baseline_value / current_value
        return pct, ratio

    pct = ((current_value - baseline_value) / baseline_value) * 100.0
    ratio = current_value / baseline_value
    return pct, ratio


def format_improvement(
    baseline: object,
    current: object,
    *,
    direction: str = "higher",
    digits: int = 2,
) -> str:
    pct, ratio = compute_improvement(baseline, current, direction=direction)
    if pct is None:
        return "n/a"

    pct_text = f"{pct:+.{digits}f}%"
    if ratio is None:
        return pct_text

    if direction == "lower":
        if abs(pct) < (0.5 * (10 ** (-digits))):
            return f"{pct_text} ({ratio:.2f}x)"
        qualifier = " faster" if pct > 0 else " slower"
        return f"{pct_text} ({ratio:.2f}x{qualifier})"
    return f"{pct_text} ({ratio:.2f}x)"


def status_from_improvement(
    baseline: object,
    current: object,
    *,
    direction: str = "higher",
    epsilon_pct: float = 0.01,
) -> str:
    pct, _ = compute_improvement(baseline, current, direction=direction)
    if pct is None:
        return "UNKNOWN"
    if abs(pct) < epsilon_pct:
        return "BASELINE"
    return "IMPROVED" if pct > 0 else "REGRESSED"
