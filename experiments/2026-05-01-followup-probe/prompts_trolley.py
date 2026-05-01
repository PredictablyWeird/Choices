"""
Reconstruct trolley prompt configurations and edge prompts from saved data.

Given a saved preference graph and the (factor, nudge_type, target_group) tuple
implied by its directory layout, rebuild the exact prompt config that the
original pipeline used so we can re-issue prompts in a multi-turn follow-up.
"""

from __future__ import annotations

import json
from pathlib import Path

from choices.experiments.nudging.simple import (
    create_nudged_simple_config,
    format_target_group_for_path,
)
from choices.experiments.nudging.simple import generate_nudge_text


TROLLEY_FACTORS = (
    "age_group",
    "diet",
    "gender",
    "handedness",
    "nationality",
    "tech_view",
    "wealth",
)


def parse_target_from_dirname(dirname: str) -> str:
    """`20260121_201116_young` → `young`. `..._base` → `base`."""
    parts = dirname.split("_", 2)
    if len(parts) < 3:
        return parts[-1]
    return parts[-1]


def _restore_target_group(factor_name: str, target_group_path: str) -> str:
    """The runner sanitises slashes etc. when forming the dir name; un-sanitize
    by matching against the canonical factor values."""
    from choices.experiments.simple_rates import BINARY_FACTORS

    if factor_name not in BINARY_FACTORS:
        raise ValueError(f"Unknown factor {factor_name}")
    for value in BINARY_FACTORS[factor_name].values:
        if format_target_group_for_path(value) == target_group_path:
            return value
    return target_group_path


def build_prompt_config(
    factor_name: str,
    nudge_type: str,
    target_group: str | None,
    reasoning: str = "none",
):
    """
    Returns (prompt_config, system_prompt, full_prompt_fn).

    `full_prompt_fn(option_A, option_B)` returns the user-facing turn-1 string.
    """
    if target_group in (None, "base"):
        nudge_text_for_call = None
    else:
        nudge_text_for_call = generate_nudge_text(
            nudge_type, factor_name, target_group, None
        )

    _vars, prompt_config, _analysis = create_nudged_simple_config(
        factor_name=factor_name,
        nudge_type=nudge_type if target_group not in (None, "base") else "base",
        target_group=target_group if target_group not in (None, "base") else None,
        nudge_text=nudge_text_for_call,
        n_values_key="paper",  # matches the existing trolley batch configs
        reasoning=reasoning,
        setup="preference",  # matches the existing trolley batch configs
    )

    return prompt_config


def load_graph(graph_path: Path) -> dict:
    return json.loads(Path(graph_path).read_text())


def edges_in_graph(graph: dict) -> list[tuple[int, int]]:
    """Return list of (option_A_id, option_B_id) tuples in deterministic order."""
    keys = list(graph.get("edges", {}).keys())
    out = []
    for k in keys:
        # keys look like "(0, 5)" — eval is safe given structure but use json-safe parse
        s = k.strip().lstrip("(").rstrip(")")
        a, b = (int(x.strip()) for x in s.split(","))
        out.append((a, b))
    return out


def options_by_id(graph: dict) -> dict[int, dict]:
    return {opt["id"]: opt for opt in graph.get("options", [])}


def reasoning_mode_from_graph(graph: dict) -> str:
    sec = graph.get("simple_experiment_config", {}) or {}
    return sec.get("reasoning_mode", "none") or "none"
