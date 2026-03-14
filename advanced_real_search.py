from __future__ import annotations

import json
import re
from typing import Any

import real_problem_suite as suite

FOCUS_AREAS = {
    "shape_config",
    "memory_access",
    "redundant_compute",
    "loop_order",
    "casting",
    "fusion",
    "state_update",
    "host_setup",
}

STRUCTURAL_CHANGES = {
    "retune_shape_configs",
    "remove_redundant_ops",
    "cache_casted_values",
    "reorder_loops",
    "fuse_epilogue",
    "hoist_host_setup",
    "specialize_static_dims",
    "simplify_state_update",
    "reduce_temporary_allocations",
}

RISK_LEVELS = {"low", "medium", "high"}
EXPECTED_GAIN_LEVELS = {"small", "medium", "large"}
SEARCH_TRACKS = {"config_only", "structural_only"}
CONFIG_ONLY_CHANGES = {
    "retune_shape_configs",
    "cache_casted_values",
    "specialize_static_dims",
}
STRUCTURAL_PRIORITY = (
    "reduce_temporary_allocations",
    "simplify_state_update",
    "remove_redundant_ops",
    "reorder_loops",
    "fuse_epilogue",
    "hoist_host_setup",
)
DELTA_SAFE_STRUCTURAL_CHANGES = {
    "reduce_temporary_allocations",
    "simplify_state_update",
    "remove_redundant_ops",
    "reorder_loops",
}

SEEDED_PLAN_LIBRARY: dict[str, dict[str, list[dict[str, Any]]]] = {
    "gated_deltanet_chunk_fwd_h": {
        "config_only": [
            {
                "summary": "Retune V-block configs conservatively for K/V >= 100 shapes.",
                "hypothesis": "Smaller V tiles and lower stage counts should reduce shared-memory pressure without changing the recurrence.",
                "focus_area": "shape_config",
                "structural_changes": ["retune_shape_configs"],
                "config_strategy": "Keep the current kernel math fixed and reduce num_stages/num_warps on the 100 and 128 feature-width benchmark shapes before widening V blocks again.",
                "target_shapes": ["K=100,V=100", "K=128,V=128"],
                "risk": "low",
                "expected_gain": "medium",
            }
        ],
        "structural_only": [
            {
                "summary": "Reduce live intermediates in the state update path.",
                "hypothesis": "A leaner state-update sequence can cut register pressure and improve the large-shape recurrence kernels.",
                "focus_area": "state_update",
                "structural_changes": ["simplify_state_update"],
                "config_strategy": "Keep SHAPE_CONFIGS conservative while changing only the state-update ordering.",
                "target_shapes": ["all_benchmarks"],
                "risk": "medium",
                "expected_gain": "large",
            }
        ],
    },
    "gated_deltanet_recompute_w_u": {
        "config_only": [
            {
                "summary": "Retune K/V block families with low-stage configs only.",
                "hypothesis": "The kernel is sensitive to block-size and stage count on large benchmark shapes, so conservative config-only tuning should be safer than structural churn.",
                "focus_area": "shape_config",
                "structural_changes": ["retune_shape_configs"],
                "config_strategy": "Keep the current kernel form fixed and compare conservative K/V block-size choices with num_stages <= 2.",
                "target_shapes": ["all_benchmarks"],
                "risk": "low",
                "expected_gain": "medium",
            }
        ],
        "structural_only": [
            {
                "summary": "Reduce temporaries in the chunk GEMM path.",
                "hypothesis": "Collapsing transient tensors around the chunk-local GEMMs can lower register pressure and unblock stronger configs later.",
                "focus_area": "memory_access",
                "structural_changes": ["reduce_temporary_allocations"],
                "config_strategy": "Keep SHAPE_CONFIGS conservative and only change the local matrix materialization path.",
                "target_shapes": ["all_benchmarks"],
                "risk": "medium",
                "expected_gain": "large",
            }
        ],
    },
    "gated_deltanet_chunk_fwd_o": {
        "config_only": [
            {
                "summary": "Shrink query-tile configs for the large feature-width shapes.",
                "hypothesis": "Very small query blocks and low stage counts should reduce shared-memory failures on the 100/128 feature-width benchmarks.",
                "focus_area": "shape_config",
                "structural_changes": ["retune_shape_configs"],
                "config_strategy": "Limit num_stages to 1 and keep query tile sizes small on K/V >= 100 shapes.",
                "target_shapes": ["K=100,V=100", "K=128,V=128"],
                "risk": "low",
                "expected_gain": "medium",
            }
        ],
        "structural_only": [
            {
                "summary": "Reduce temporary matrices in the local/global attention merge.",
                "hypothesis": "A smaller live working set should improve the chunk output kernel without changing the mathematical formula.",
                "focus_area": "memory_access",
                "structural_changes": ["reduce_temporary_allocations"],
                "config_strategy": "Keep configs conservative while simplifying the local/global accumulation path.",
                "target_shapes": ["all_benchmarks"],
                "risk": "medium",
                "expected_gain": "large",
            }
        ],
    },
}


PLANNER_SYSTEM_PROMPT = """You are designing candidate search plans for a Helion kernel optimizer.
You are not writing code in this step.

Return exactly one JSON object with a top-level `plans` array.
Each plan must be a focused, testable optimization idea for the current problem.
Do not include prose outside the JSON.
"""


MATERIALIZER_SYSTEM_PROMPT = """You are an expert GPU kernel engineer optimizing a real Helion leaderboard kernel.
You will receive:
- the target task description and benchmark cases
- a structured optimization plan
- the current submission.py file
- the reference.py and task.py files

Return exactly one Python code block containing the full updated submission.py file.

Rules:
- Keep the file importable.
- Keep the entry point as `custom_kernel(data)`.
- The benchmark times `custom_kernel(data)` itself, so do not hide expensive work in the wrapper.
- Preserve correctness and CUDA-graph capturability.
- Implement the provided plan faithfully.
- Prefer one targeted optimization over a rewrite.
- Do not define nested functions or helper functions inside `@helion.kernel` bodies.
- Do not use unsupported Python statements inside `@helion.kernel` bodies such as nested defs, classes, lambdas, `with`, `try`, or `match`.
- Add a concise first-line comment summarizing the experiment.
- Do not include prose outside the Python code block.
"""


def sanitize_text(value: Any, fallback: str) -> str:
    text = str(value or "").replace("#", " ")
    text = " ".join(text.split())
    return text or fallback


def seeded_plans_for_problem(
    problem_name: str,
    search_track: str,
) -> list[dict[str, Any]]:
    seeded: list[dict[str, Any]] = []
    for plan in SEEDED_PLAN_LIBRARY.get(problem_name, {}).get(search_track, []):
        seeded.append(
            {
                **dict(plan),
                "problem": problem_name,
                "search_track": search_track,
            }
        )
    return seeded


def _renumber_plan_ids(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for idx, plan in enumerate(plans, start=1):
        renumbered.append({**plan, "plan_id": f"plan_{idx:02d}"})
    return renumbered


def merge_seeded_plans(
    *,
    problem_name: str,
    search_track: str,
    model_plans: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    seen_signatures: set[tuple[str, tuple[str, ...], str]] = set()
    merged: list[dict[str, Any]] = []

    for plan in seeded_plans_for_problem(problem_name, search_track) + model_plans:
        signature = (
            plan["focus_area"],
            tuple(sorted(plan["structural_changes"])),
            plan["search_track"],
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        merged.append(plan)

    return _renumber_plan_ids(merged[:batch_size])


def extract_json_payload(text: str) -> Any:
    candidates: list[str] = []
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    array_slice = _slice_bracket_block(stripped, "[", "]")
    if array_slice:
        candidates.append(array_slice)
    object_slice = _slice_bracket_block(stripped, "{", "}")
    if object_slice:
        candidates.append(object_slice)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Model response did not contain valid JSON.")


def _slice_bracket_block(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    end = text.rfind(close_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def normalize_plans(
    *,
    problem_name: str,
    raw_payload: Any,
    batch_size: int,
    search_track: str,
) -> list[dict[str, Any]]:
    if search_track not in SEARCH_TRACKS:
        raise ValueError(f"Unsupported search_track `{search_track}`.")
    if isinstance(raw_payload, dict):
        raw_plans = raw_payload.get("plans", [])
    elif isinstance(raw_payload, list):
        raw_plans = raw_payload
    else:
        raise ValueError("Planner payload must be a JSON object or array.")

    if not isinstance(raw_plans, list):
        raise ValueError("Planner payload did not contain a `plans` list.")

    normalized: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()

    for idx, raw_plan in enumerate(raw_plans):
        if not isinstance(raw_plan, dict):
            continue

        summary = sanitize_text(raw_plan.get("summary"), "Targeted kernel experiment.")
        hypothesis = sanitize_text(
            raw_plan.get("hypothesis"),
            "This change may reduce benchmark latency.",
        )
        focus_area = str(raw_plan.get("focus_area", "shape_config")).strip().lower()
        if focus_area not in FOCUS_AREAS:
            focus_area = "shape_config"

        structural_changes = raw_plan.get("structural_changes", [])
        if not isinstance(structural_changes, list):
            structural_changes = [structural_changes]
        normalized_changes = [
            str(item).strip().lower()
            for item in structural_changes
            if str(item).strip().lower() in STRUCTURAL_CHANGES
        ]
        if not normalized_changes:
            normalized_changes = ["retune_shape_configs"]

        config_strategy = sanitize_text(
            raw_plan.get("config_strategy"),
            "Retune shape-specific Helion config values.",
        )

        target_shapes = raw_plan.get("target_shapes", ["all_benchmarks"])
        if not isinstance(target_shapes, list):
            target_shapes = [target_shapes]
        normalized_targets = [
            sanitize_text(item, "all_benchmarks")
            for item in target_shapes
            if sanitize_text(item, "")
        ]
        if not normalized_targets:
            normalized_targets = ["all_benchmarks"]

        risk = str(raw_plan.get("risk", "medium")).strip().lower()
        if risk not in RISK_LEVELS:
            risk = "medium"

        expected_gain = str(raw_plan.get("expected_gain", "medium")).strip().lower()
        if expected_gain not in EXPECTED_GAIN_LEVELS:
            expected_gain = "medium"

        if search_track == "config_only":
            normalized_changes = [
                change for change in normalized_changes if change in CONFIG_ONLY_CHANGES
            ]
            if not normalized_changes:
                normalized_changes = ["retune_shape_configs"]
            if focus_area not in {"shape_config", "casting", "host_setup", "memory_access"}:
                focus_area = "shape_config"
            if risk == "high":
                risk = "medium"
        else:
            non_config_changes = [
                change for change in normalized_changes if change not in CONFIG_ONLY_CHANGES
            ]
            if problem_name.startswith("gated_deltanet_"):
                non_config_changes = [
                    change
                    for change in non_config_changes
                    if change in DELTA_SAFE_STRUCTURAL_CHANGES
                ]
            if not non_config_changes:
                focus_to_default = {
                    "state_update": "simplify_state_update",
                    "memory_access": "reduce_temporary_allocations",
                    "loop_order": "reorder_loops",
                    "redundant_compute": "remove_redundant_ops",
                    "fusion": "fuse_epilogue",
                }
                if problem_name.startswith("gated_deltanet_"):
                    focus_to_default["fusion"] = "reduce_temporary_allocations"
                non_config_changes = [
                    focus_to_default.get(focus_area, "remove_redundant_ops")
                ]
            prioritized = sorted(
                non_config_changes,
                key=lambda item: STRUCTURAL_PRIORITY.index(item)
                if item in STRUCTURAL_PRIORITY
                else len(STRUCTURAL_PRIORITY),
            )
            normalized_changes = prioritized[:1]
            config_strategy = (
                "Keep SHAPE_CONFIGS conservative while applying one structural change. "
                + config_strategy
            )

        signature = (focus_area, tuple(sorted(normalized_changes)), search_track)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        normalized.append(
            {
                "plan_id": f"plan_{idx + 1:02d}",
                "problem": problem_name,
                "summary": summary,
                "hypothesis": hypothesis,
                "focus_area": focus_area,
                "structural_changes": normalized_changes[:3],
                "config_strategy": config_strategy,
                "target_shapes": normalized_targets[:4],
                "risk": risk,
                "expected_gain": expected_gain,
                "search_track": search_track,
            }
        )

    if not normalized:
        raise ValueError("Planner did not produce any valid plans.")

    return normalized[:batch_size]


def build_plan_prompt(
    *,
    problem: suite.RealProblem,
    current_source: str,
    program_text: str,
    best_result: dict[str, Any],
    history_text: str,
    batch_size: int,
    search_track: str,
) -> str:
    task_yml = problem.task_yml_path.read_text(encoding="utf-8")
    seeded = seeded_plans_for_problem(problem.name, search_track)
    seeded_text = (
        json.dumps(seeded, indent=2, sort_keys=True) if seeded else "[]"
    )
    if search_track == "config_only":
        track_guidance = (
            "All plans must keep the current algorithmic structure fixed. "
            "Only propose safe tuning-style changes such as SHAPE_CONFIGS retunes, "
            "specialization, cast caching, or wrapper-cost reductions."
        )
        change_guidance = ", ".join(sorted(CONFIG_ONLY_CHANGES))
    else:
        track_guidance = (
            "All plans must use exactly one structural change from the allowed list. "
            "Keep SHAPE_CONFIGS conservative and avoid widening tiles or increasing "
            "num_stages aggressively in the same plan."
        )
        change_guidance = ", ".join(sorted(STRUCTURAL_CHANGES - CONFIG_ONLY_CHANGES))
    return f"""Design a batch of {batch_size} candidate optimization plans for `{problem.name}`.

Current best benchmark score: {best_result['score_ms']:.4f} ms geomean
Primary objective: lower geomean latency while preserving correctness.
Search track for this batch: `{search_track}`.
{track_guidance}

Research program:
{program_text}

Task description:
{problem.description}

task.yml:
```yaml
{task_yml}
```

Current submission.py:
```python
{current_source}
```

Recent experiment history:
{history_text}

Seeded plan families to consider:
```json
{seeded_text}
```

Return exactly one JSON object:
{{
  "plans": [
    {{
      "summary": "short one-line description",
      "hypothesis": "one sentence",
      "focus_area": "one of [{', '.join(sorted(FOCUS_AREAS))}]",
      "structural_changes": ["allowed changes: {change_guidance}"],
      "config_strategy": "how SHAPE_CONFIGS or Helion config should change",
      "target_shapes": ["all_benchmarks" or specific shape hints"],
      "risk": "one of [low, medium, high]",
      "expected_gain": "one of [small, medium, large]"
    }}
  ]
}}
"""


def build_materialization_prompt(
    *,
    problem: suite.RealProblem,
    current_source: str,
    program_text: str,
    plan: dict[str, Any],
    history_text: str,
) -> str:
    reference_source = problem.reference_path.read_text(encoding="utf-8")
    task_source = problem.task_path.read_text(encoding="utf-8")
    task_yml = problem.task_yml_path.read_text(encoding="utf-8")
    plan_json = json.dumps(plan, indent=2, sort_keys=True)

    return f"""## Target
Materialize the following structured optimization plan for `{problem.name}`.

## Research Program
{program_text}

## Structured Plan
```json
{plan_json}
```

## task.yml
```yaml
{task_yml}
```

## Current submission.py
```python
{current_source}
```

## reference.py
```python
{reference_source}
```

## task.py
```python
{task_source}
```

## Recent Experiment History
{history_text}

## Requirements
- Return the full updated `submission.py` file as a single Python code block.
- Keep `custom_kernel(data)` intact as the public entry point.
- Keep `custom_kernel(data)` minimal; benchmark time includes wrapper work.
- Preserve correctness for every listed test case.
- Preserve CUDA graph capturability.
- Keep changes aligned with the structured plan rather than introducing unrelated edits.
- Do not introduce nested defs, helper functions inside `@helion.kernel`, or unsupported statements inside kernel bodies.
- Do not use inline ternary expressions (`a if cond else b`) inside `@helion.kernel`.
- Do not create branch-dependent tensor rank changes inside `@helion.kernel`.
- If the plan search track is `config_only`, keep the existing algorithmic structure fixed.
- If the plan search track is `structural_only`, make exactly one structural change and keep configs conservative.
"""
