from __future__ import annotations

import ast
import json
import re
from typing import Any

SUPPORTED_STRUCTURED_PROBLEMS = {"matmul", "matmul_relu"}
PROBLEM_TO_KERNEL = {
    "matmul": "matmul_kernel",
    "matmul_relu": "matmul_relu_kernel",
}

CONFIG_PRESETS: dict[str, list[int] | None] = {
    "autotune_default": None,
    "square_32": [32, 32],
    "square_64": [64, 64],
    "square_128": [128, 128],
    "wide_m_128x64": [128, 64],
    "wide_n_64x128": [64, 128],
}

ACCUMULATOR_DTYPES = {
    "float16": "torch.float16",
    "float32": "torch.float32",
}

MATMUL_OPS = {
    "addmm": "torch.addmm",
    "matmul_plus": "torch.matmul + add",
}


def supports_problem(problem_name: str) -> bool:
    return problem_name in SUPPORTED_STRUCTURED_PROBLEMS


def resolve_search_mode(problem_name: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "structured" if supports_problem(problem_name) else "freeform"
    if requested_mode == "structured" and not supports_problem(problem_name):
        raise ValueError(
            f"Structured search is not implemented for problem `{problem_name}`."
        )
    return requested_mode


def extract_json_object(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else text.strip()

    for payload in (candidate, _slice_first_json_object(candidate)):
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response did not contain a valid JSON object.")


def _slice_first_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _function_span(source: str, function_name: str) -> tuple[int, int]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
            return start, node.end_lineno
    raise ValueError(f"Could not find function `{function_name}` in kernel source.")


def extract_function_block(source: str, function_name: str) -> str:
    lines = source.splitlines()
    start, end = _function_span(source, function_name)
    return "\n".join(lines[start - 1 : end]).strip()


def normalize_plan(problem_name: str, raw_plan: dict[str, Any]) -> dict[str, Any]:
    if problem_name not in PROBLEM_TO_KERNEL:
        raise ValueError(f"Unsupported structured problem `{problem_name}`.")

    summary = str(raw_plan.get("summary", "")).strip() or "Structured matmul experiment."
    hypothesis = (
        str(raw_plan.get("hypothesis", "")).strip()
        or "This structured change may improve throughput."
    )
    config_preset = str(raw_plan.get("config_preset", "autotune_default")).strip()
    accumulator_dtype = str(raw_plan.get("accumulator_dtype", "float32")).strip()
    matmul_op = str(raw_plan.get("matmul_op", "addmm")).strip()

    if config_preset not in CONFIG_PRESETS:
        raise ValueError(
            "Invalid `config_preset`. "
            f"Expected one of: {', '.join(CONFIG_PRESETS)}"
        )
    if accumulator_dtype not in ACCUMULATOR_DTYPES:
        raise ValueError(
            "Invalid `accumulator_dtype`. "
            f"Expected one of: {', '.join(ACCUMULATOR_DTYPES)}"
        )
    if matmul_op not in MATMUL_OPS:
        raise ValueError(
            "Invalid `matmul_op`. "
            f"Expected one of: {', '.join(MATMUL_OPS)}"
        )

    plan = {
        "problem": problem_name,
        "kernel_name": PROBLEM_TO_KERNEL[problem_name],
        "summary": _sanitize_comment(summary),
        "hypothesis": _sanitize_comment(hypothesis),
        "config_preset": config_preset,
        "accumulator_dtype": accumulator_dtype,
        "matmul_op": matmul_op,
        "explicit_output_cast": bool(raw_plan.get("explicit_output_cast", True)),
    }
    return plan


def _sanitize_comment(text: str) -> str:
    return " ".join(text.replace("#", "").split())


def build_structured_prompt(
    *,
    problem_name: str,
    current_source: str,
    program_text: str,
    best_result: dict[str, Any],
    history_text: str,
) -> str:
    kernel_name = PROBLEM_TO_KERNEL[problem_name]
    target_function = extract_function_block(current_source, kernel_name)
    config_options = ", ".join(CONFIG_PRESETS)
    accumulator_options = ", ".join(ACCUMULATOR_DTYPES)
    matmul_ops = ", ".join(MATMUL_OPS)

    return f"""You are producing a structured optimization plan, not Python code.

Target problem: `{problem_name}`
Target kernel function: `{kernel_name}`
Current best throughput: {best_result['throughput']:.2f} {best_result['unit']}
Current best latency: {best_result['avg_ms']:.3f} ms

Research guidance:
{program_text}

Current target function:
```python
{target_function}
```

Recent experiment history:
{history_text}

Return exactly one JSON object with these keys:
- `summary`: short one-line description
- `hypothesis`: one sentence
- `config_preset`: one of [{config_options}]
- `accumulator_dtype`: one of [{accumulator_options}]
- `matmul_op`: one of [{matmul_ops}]
- `explicit_output_cast`: true or false

Rules:
- Do not return Python code.
- Do not add any keys outside this schema.
- Make one targeted change, not a rewrite.
- Keep the plan valid for the `{problem_name}` kernel family.
"""


def render_candidate_source(current_source: str, plan: dict[str, Any]) -> str:
    kernel_name = plan["kernel_name"]
    start, end = _function_span(current_source, kernel_name)
    lines = current_source.splitlines()
    replacement = render_target_function(plan).strip("\n").splitlines()
    updated_lines = lines[: start - 1] + replacement + lines[end:]
    return "\n".join(updated_lines) + "\n"


def render_target_function(plan: dict[str, Any]) -> str:
    kernel_name = plan["kernel_name"]
    relu = plan["problem"] == "matmul_relu"
    decorator = _render_decorator(plan["config_preset"])
    accumulator_dtype = ACCUMULATOR_DTYPES[plan["accumulator_dtype"]]
    update_expr = _render_update_expr(plan["matmul_op"])
    writeback_expr = _render_writeback_expr(
        relu=relu,
        explicit_output_cast=plan["explicit_output_cast"],
    )

    return f"""# Structured plan: {plan['summary']}
# Hypothesis: {plan['hypothesis']}
{decorator}
def {kernel_name}(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {{k}} != {{k2}}"
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype={accumulator_dtype})
        for tile_k in hl.tile(k):
            {update_expr}
        out[tile_m, tile_n] = {writeback_expr}
    return out
"""


def _render_decorator(config_preset: str) -> str:
    block_sizes = CONFIG_PRESETS[config_preset]
    if block_sizes is None:
        return "@helion.kernel()"
    return f"@helion.kernel(config=helion.Config(block_sizes={block_sizes}))"


def _render_update_expr(matmul_op: str) -> str:
    if matmul_op == "addmm":
        return "acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])"
    if matmul_op == "matmul_plus":
        return "acc = acc + torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])"
    raise ValueError(f"Unsupported matmul operation `{matmul_op}`.")


def _render_writeback_expr(*, relu: bool, explicit_output_cast: bool) -> str:
    expression = "acc"
    if relu:
        expression = f"torch.relu({expression})"
    if explicit_output_cast:
        expression = f"{expression}.to(x.dtype)"
    return expression
