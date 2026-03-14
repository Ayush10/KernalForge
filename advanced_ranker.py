from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

DELTA_PROBLEMS = {
    "gated_deltanet_chunk_fwd_h",
    "gated_deltanet_chunk_fwd_o",
    "gated_deltanet_recompute_w_u",
}

FOCUS_WEIGHTS = {
    "shape_config": 0.72,
    "memory_access": 0.82,
    "redundant_compute": 0.9,
    "loop_order": 0.7,
    "casting": 0.62,
    "fusion": 0.66,
    "state_update": 0.8,
    "host_setup": 0.55,
}

CHANGE_WEIGHTS = {
    "retune_shape_configs": 0.32,
    "remove_redundant_ops": 0.42,
    "cache_casted_values": 0.24,
    "reorder_loops": 0.26,
    "fuse_epilogue": 0.22,
    "hoist_host_setup": 0.18,
    "specialize_static_dims": 0.2,
    "simplify_state_update": 0.35,
    "reduce_temporary_allocations": 0.25,
}

RISK_WEIGHTS = {"low": 0.18, "medium": 0.02, "high": -0.22}
GAIN_WEIGHTS = {"small": 0.06, "medium": 0.14, "large": 0.2}
SEARCH_TRACK_WEIGHTS = {"config_only": 0.1, "structural_only": 0.04}
CONFIG_RE = re.compile(r"helion\.Config\((.*?)\)", re.DOTALL)
FIELD_RE = {
    "num_warps": re.compile(r"num_warps\s*=\s*(\d+)"),
    "num_stages": re.compile(r"num_stages\s*=\s*(\d+)"),
    "block_sizes": re.compile(r"block_sizes\s*=\s*\[([^\]]*)\]"),
}


def classify_failure_message(message: str | None) -> str:
    lower = (message or "").lower()
    if not lower:
        return "unknown"
    if "shared memory" in lower and ("out of resource" in lower or "outofresources" in lower):
        return "shared_memory_oor"
    if "@jit functions should be defined in a python file" in lower:
        return "jit_file_error"
    if "triton codegen error" in lower or "inductorloweringerror" in lower:
        return "compile_error"
    if "ifexp is not supported" in lower or "unsupported syntax" in lower:
        return "unsupported_syntax"
    if "controlflowtensormismatch" in lower or "control flow tensor mismatch" in lower:
        return "control_flow_tensor_mismatch"
    if "invalid indexing type" in lower or "invalidindexingtype" in lower:
        return "invalid_indexing"
    if "shape mismatch" in lower or "incompatible" in lower and "shape" in lower:
        return "shape_mismatch"
    if "broadcast" in lower:
        return "broadcast_mismatch"
    if "syntax error" in lower:
        return "syntax_error"
    if "parse error" in lower:
        return "parse_error"
    if "provider" in lower:
        return "provider_error"
    if "test failed" in lower:
        return "correctness_failure"
    return "unknown"


def history_failure_counts(
    history: list[dict[str, Any]],
    problem_name: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in history:
        if record.get("suite") != "real" or record.get("problem") != problem_name:
            continue
        if record.get("result") == "accepted":
            continue
        kind = record.get("failure_kind")
        if not isinstance(kind, str) or not kind:
            kind = classify_failure_message(str(record.get("message", "")))
        counts[kind] += 1
    return counts


def extract_config_features(source: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for match in CONFIG_RE.findall(source):
        block_sizes_match = FIELD_RE["block_sizes"].search(match)
        if block_sizes_match:
            block_sizes = [
                int(item.strip())
                for item in block_sizes_match.group(1).split(",")
                if item.strip().isdigit()
            ]
        else:
            block_sizes = []

        field_values: dict[str, int] = {}
        for field_name in ("num_warps", "num_stages"):
            value_match = FIELD_RE[field_name].search(match)
            field_values[field_name] = int(value_match.group(1)) if value_match else 1

        configs.append(
            {
                "block_sizes": block_sizes,
                "num_warps": field_values["num_warps"],
                "num_stages": field_values["num_stages"],
            }
        )
    return configs


def estimate_resource_pressure(
    problem_name: str,
    config_features: list[dict[str, Any]],
) -> tuple[float, str]:
    if not config_features:
        return 0.0, "low"

    highest_pressure = 0.0
    for config in config_features:
        block_sizes = config["block_sizes"] or [8]
        max_block = max(block_sizes)
        block_sum = sum(block_sizes)
        pressure = max_block * max(config["num_stages"], 1)
        pressure *= 1.0 + (config["num_warps"] / 8.0)
        pressure += block_sum * 0.45

        if problem_name in DELTA_PROBLEMS:
            pressure += max_block * 0.9
            if config["num_stages"] >= 2:
                pressure += 18.0
            if config["num_warps"] >= 8:
                pressure += 16.0
            if problem_name == "gated_deltanet_chunk_fwd_o":
                pressure += max_block * 1.3
            elif problem_name == "gated_deltanet_chunk_fwd_h":
                pressure += max_block * 0.85
            else:
                pressure += max_block * 0.65
        elif problem_name == "causal_conv1d":
            pressure *= 0.55
        elif problem_name == "fp8_quant":
            pressure *= 0.35

        highest_pressure = max(highest_pressure, pressure)

    if highest_pressure >= 110:
        return round(highest_pressure, 2), "high"
    if highest_pressure >= 65:
        return round(highest_pressure, 2), "medium"
    return round(highest_pressure, 2), "low"


def _history_buckets(history: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    buckets = {
        "focus_area": defaultdict(list),
        "change": defaultdict(list),
        "signature": defaultdict(list),
        "summary": defaultdict(list),
    }
    for record in history:
        if record.get("suite") != "real":
            continue
        outcome = 1 if record.get("result") == "accepted" else 0
        focus = record.get("plan_focus_area")
        if isinstance(focus, str):
            buckets["focus_area"][focus].append(outcome)

        for change in record.get("plan_structural_changes", []) or []:
            if isinstance(change, str):
                buckets["change"][change].append(outcome)

        signature = record.get("plan_signature")
        if isinstance(signature, str):
            buckets["signature"][signature].append(outcome)

        summary = record.get("summary")
        if isinstance(summary, str):
            buckets["summary"][summary].append(outcome)
    return buckets


def _success_adjustment(outcomes: list[int]) -> float:
    if not outcomes:
        return 0.0
    success_rate = sum(outcomes) / len(outcomes)
    confidence = min(len(outcomes), 6) / 6.0
    return (success_rate - 0.5) * 0.55 * confidence


def score_plan(
    plan: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    seen_signatures: Counter[str] | None = None,
) -> dict[str, Any]:
    buckets = _history_buckets(history)
    problem_name = str(plan.get("problem", ""))
    focus_area = plan["focus_area"]
    changes = list(plan["structural_changes"])
    search_track = str(plan.get("search_track", "config_only"))
    signature = f"{focus_area}|{'/'.join(sorted(changes))}"
    summary = plan["summary"]
    failure_counts = history_failure_counts(history, problem_name)

    score = 0.0
    score += FOCUS_WEIGHTS.get(focus_area, 0.4)
    score += RISK_WEIGHTS.get(plan["risk"], 0.0)
    score += GAIN_WEIGHTS.get(plan["expected_gain"], 0.0)
    score += SEARCH_TRACK_WEIGHTS.get(search_track, 0.0)
    score += min(len(plan["target_shapes"]), 3) * 0.04
    if "all_benchmarks" in plan["target_shapes"]:
        score += 0.08
    if "shape" in plan["config_strategy"].lower():
        score += 0.06

    for change in changes:
        score += CHANGE_WEIGHTS.get(change, 0.0)

    score += _success_adjustment(buckets["focus_area"][focus_area])
    for change in changes:
        score += _success_adjustment(buckets["change"][change]) * 0.6

    duplicate_penalty = 0.0
    if buckets["signature"][signature]:
        duplicate_penalty -= 0.18
    if buckets["summary"][summary]:
        duplicate_penalty -= 0.08
    if seen_signatures is not None and seen_signatures[signature] > 0:
        duplicate_penalty -= 0.12
    score += duplicate_penalty

    if problem_name in DELTA_PROBLEMS:
        shared_oor = failure_counts["shared_memory_oor"]
        compile_failures = (
            failure_counts["compile_error"]
            + failure_counts["jit_file_error"]
            + failure_counts["invalid_indexing"]
        )
        syntax_failures = (
            failure_counts["unsupported_syntax"]
            + failure_counts["control_flow_tensor_mismatch"]
            + failure_counts["shape_mismatch"]
            + failure_counts["broadcast_mismatch"]
        )

        if search_track == "config_only":
            score += min(shared_oor, 4) * 0.05
            if compile_failures:
                score += min(compile_failures, 3) * 0.03
        else:
            score -= min(shared_oor, 4) * 0.06
            score -= min(compile_failures, 3) * 0.04

        if "fuse_epilogue" in changes and shared_oor:
            score -= 0.16
        if "reduce_temporary_allocations" in changes and shared_oor:
            score += 0.08
        if "simplify_state_update" in changes and syntax_failures:
            score += 0.05

    observed_count = (
        len(buckets["focus_area"][focus_area])
        + sum(len(buckets["change"][change]) for change in changes)
    )
    uncertainty = max(0.08, 0.95 - min(observed_count, 10) * 0.08)

    return {
        **plan,
        "plan_signature": signature,
        "surrogate_score": round(score, 4),
        "surrogate_uncertainty": round(uncertainty, 4),
        "surrogate_components": {
            "focus_area": focus_area,
            "change_count": len(changes),
            "history_observations": observed_count,
            "duplicate_penalty": round(duplicate_penalty, 4),
            "search_track": search_track,
            "shared_memory_oor_count": failure_counts["shared_memory_oor"],
        },
    }


def rank_plans(plans: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_signatures: Counter[str] = Counter()
    scored = [
        score_plan(plan, history=history, seen_signatures=seen_signatures)
        for plan in plans
    ]
    for item in scored:
        seen_signatures[item["plan_signature"]] += 1
    scored.sort(
        key=lambda item: (
            item["surrogate_score"],
            -item["surrogate_uncertainty"],
            item["summary"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(scored, start=1):
        item["surrogate_rank"] = rank
    return scored


def score_materialized_candidate(
    *,
    problem_name: str,
    source: str,
    baseline_source: str,
    plan: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_lines = [
        line.strip()
        for line in baseline_source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    candidate_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    duplicate_counter = Counter(candidate_lines)
    duplicate_lines = sum(count - 1 for count in duplicate_counter.values() if count > 1)

    baseline_duplicates = Counter(baseline_lines)
    baseline_duplicate_lines = sum(
        count - 1 for count in baseline_duplicates.values() if count > 1
    )

    line_delta = abs(len(candidate_lines) - len(baseline_lines))
    config_count = source.count("helion.Config(")
    specialize_count = source.count("hl.specialize(")
    config_features = extract_config_features(source)
    resource_pressure, resource_risk = estimate_resource_pressure(
        problem_name, config_features
    )
    failure_counts = history_failure_counts(history, problem_name)

    score = float(plan["surrogate_score"])
    score += min(config_count, 16) * 0.03
    score += min(specialize_count, 8) * 0.04
    score += max(0, baseline_duplicate_lines - duplicate_lines) * 0.05

    if "def custom_kernel" not in source:
        score -= 2.0
    if "SHAPE_CONFIGS" not in source:
        score -= 1.0
    if line_delta > 120:
        score -= 0.35
    elif line_delta > 60:
        score -= 0.18
    if resource_risk == "medium":
        score -= 0.14
    elif resource_risk == "high":
        score -= 0.32
    if failure_counts["shared_memory_oor"] and resource_risk != "low":
        score -= min(failure_counts["shared_memory_oor"], 4) * 0.08
    if failure_counts["jit_file_error"] and plan.get("search_track") == "structural_only":
        score -= 0.12
    if " if " in source and " else " in source:
        score -= 0.5

    return {
        "code_static_score": round(score, 4),
        "code_line_delta": line_delta,
        "code_duplicate_lines": duplicate_lines,
        "code_config_count": config_count,
        "code_specialize_count": specialize_count,
        "code_resource_pressure": resource_pressure,
        "code_resource_risk": resource_risk,
        "code_failure_priors": dict(failure_counts),
    }
