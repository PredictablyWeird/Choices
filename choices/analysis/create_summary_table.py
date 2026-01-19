#!/usr/bin/env python3
"""
Create a summary table of nudge experiment results.

This script takes results directories and creates a table showing:
- Model name, reasoning condition, factor, nudge type
- Baseline bias
- Nudge effect size
- Steerability bias

Usage:
    # Discover all results from default results directory
    python create_summary_table.py

    # Specify results directories
    python create_summary_table.py --results-dirs results results2

    # Filter by models, factors, nudge types
    python create_summary_table.py \
        --models llama-33-70b deepseek-v3-2-reasoning \
        --factors age_group social_status \
        --nudge-types weak_evidence emotional

    # Output to CSV
    python create_summary_table.py --output summary.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from choices.analysis.nudge_effect_size import (
    compute_factor_preference_from_graph,
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_frequencies,
)


@dataclass
class ExperimentResult:
    """Results for a single nudge experiment."""

    model: str
    reasoning_condition: str  # "reasoning", "non-reasoning", or "unknown"
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    # Baseline metrics
    baseline_bias: float  # Deviation from 0.5 (positive = biased towards B)
    p_A_baseline: float
    p_B_baseline: float
    # Effect sizes
    effect_size_A: float  # P(A | nudge A) - P(A | baseline)
    effect_size_B: float  # P(B | nudge B) - P(B | baseline)
    avg_effect_size: float
    # Steerability
    steerability_A: Optional[float]
    steerability_B: Optional[float]
    steerability_bias: Optional[float]  # Positive = more steerable towards B
    # Sample sizes
    n_baseline: int
    n_nudge_A: int
    n_nudge_B: int


@dataclass
class AggregatedResult:
    """Aggregated results over one or more dimensions."""

    # Group keys (None if aggregated over this dimension)
    model: Optional[str]
    reasoning_condition: Optional[str]
    factor: Optional[str]
    nudge_type: Optional[str]
    # Aggregated metrics
    avg_baseline_bias: float
    avg_effect_size: float
    avg_steerability_bias: Optional[float]
    # Count
    n_results: int
    n_steerability: int  # Number of results with valid steerability


def aggregate_results(
    results: List[ExperimentResult],
    aggregate_over: List[str],
    force_signed: bool = False,
) -> Tuple[List[AggregatedResult], bool, bool, bool]:
    """
    Aggregate results over specified dimensions.

    Args:
        results: List of ExperimentResult objects
        aggregate_over: List of dimensions to aggregate over.
            Valid values: "model", "factor", "nudge_type", "reasoning"
        force_signed: If True, use signed values instead of magnitude even when
            aggregating over factors

    Returns:
        Tuple of (List of AggregatedResult objects,
                  bool indicating if baseline uses magnitude,
                  bool indicating if effect size uses magnitude,
                  bool indicating if steerability uses magnitude)
    """
    from collections import defaultdict

    # Map dimension names to result attributes
    dim_to_attr = {
        "model": "model",
        "factor": "factor",
        "nudge_type": "nudge_type",
        "reasoning": "reasoning_condition",
    }

    # Determine which dimensions to keep vs aggregate
    keep_dims = [d for d in dim_to_attr.keys() if d not in aggregate_over]

    # Use magnitude for all metrics when aggregating over factors
    # (because direction is factor-specific and would cancel out)
    # Unless force_signed is set, in which case use signed values
    use_magnitude = "factor" in aggregate_over and not force_signed
    baseline_use_magnitude = use_magnitude
    effect_use_magnitude = use_magnitude
    steerability_use_magnitude = use_magnitude

    # Group results by the dimensions we're keeping
    groups: Dict[tuple, List[ExperimentResult]] = defaultdict(list)
    for r in results:
        key = tuple(
            getattr(r, dim_to_attr[d]) if d in keep_dims else None
            for d in dim_to_attr.keys()
        )
        groups[key].append(r)

    # Compute aggregates for each group
    aggregated = []
    for key, group_results in groups.items():
        model_key, factor_key, nudge_key, reasoning_key = key

        # Calculate averages
        # Use magnitude for baseline if aggregating over factors
        baseline_biases = [
            abs(r.baseline_bias) if baseline_use_magnitude else r.baseline_bias
            for r in group_results
        ]
        # Use magnitude for effect sizes if aggregating over factors
        effect_sizes = [
            abs(r.avg_effect_size) if effect_use_magnitude else r.avg_effect_size
            for r in group_results
        ]
        # Use magnitude for steerability if aggregating over factors
        steerability_biases = [
            abs(r.steerability_bias)
            if steerability_use_magnitude
            else r.steerability_bias
            for r in group_results
            if r.steerability_bias is not None
        ]

        avg_baseline = sum(baseline_biases) / len(baseline_biases)
        avg_effect = sum(effect_sizes) / len(effect_sizes)
        avg_steer = (
            sum(steerability_biases) / len(steerability_biases)
            if steerability_biases
            else None
        )

        aggregated.append(
            AggregatedResult(
                model=model_key,
                reasoning_condition=reasoning_key,
                factor=factor_key,
                nudge_type=nudge_key,
                avg_baseline_bias=avg_baseline,
                avg_effect_size=avg_effect,
                avg_steerability_bias=avg_steer,
                n_results=len(group_results),
                n_steerability=len(steerability_biases),
            )
        )

    return (
        aggregated,
        baseline_use_magnitude,
        effect_use_magnitude,
        steerability_use_magnitude,
    )


def format_aggregated_table(
    results: List[AggregatedResult],
    show_display_names: bool = True,
    baseline_use_magnitude: bool = False,
    effect_use_magnitude: bool = False,
    steerability_use_magnitude: bool = False,
) -> str:
    """Format aggregated results as a text table."""
    if not results:
        return "No results found."

    # Determine which columns to show based on what's not aggregated
    has_model = any(r.model is not None for r in results)
    has_reasoning = any(r.reasoning_condition is not None for r in results)
    has_factor = any(r.factor is not None for r in results)
    has_nudge = any(r.nudge_type is not None for r in results)

    # Sort results
    def sort_key(r):
        return (
            get_base_model_name(r.model) if r.model else "",
            r.factor or "",
            r.nudge_type or "",
            r.reasoning_condition or "",
        )

    results = sorted(results, key=sort_key)

    # Build header
    headers = []
    if has_model:
        headers.append("Model")
    if has_reasoning:
        headers.append("Reasoning")
    if has_factor:
        headers.append("Factor")
    if has_nudge:
        headers.append("Nudge Type")
    baseline_header = "|Baseline Bias|" if baseline_use_magnitude else "Baseline Bias"
    effect_header = "|Effect Size|" if effect_use_magnitude else "Effect Size"
    steer_header = (
        "|Steerability Bias|" if steerability_use_magnitude else "Steerability Bias"
    )
    headers.extend([baseline_header, effect_header, steer_header, "N"])

    # Build rows
    rows = []
    for r in results:
        row = []
        if has_model:
            model_name = (
                get_model_display_name(r.model) if show_display_names else r.model
            )
            row.append(model_name or "ALL")
        if has_reasoning:
            row.append(r.reasoning_condition or "ALL")
        if has_factor:
            row.append(r.factor or "ALL")
        if has_nudge:
            row.append(r.nudge_type or "ALL")

        steer_str = (
            f"{r.avg_steerability_bias:+.3f}" if r.avg_steerability_bias else "N/A"
        )
        row.extend(
            [
                f"{r.avg_baseline_bias:+.3f}",
                f"{r.avg_effect_size:+.3f}",
                steer_str,
                str(r.n_results),
            ]
        )
        rows.append(row)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    row_lines = []
    for row in rows:
        row_lines.append(
            " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        )

    return "\n".join([header_line, separator] + row_lines)


def load_models_config() -> Dict[str, Any]:
    """Load the models configuration from models.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# Cache for models config
_models_config: Optional[Dict[str, Any]] = None


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model is a "reasoning model" (supports extended thinking/reasoning effort).

    Returns True if the model name ends with "-reasoning"/"-non-reasoning" or has
    reasoning_effort explicitly configured in models.yaml.
    """
    global _models_config
    if _models_config is None:
        _models_config = load_models_config()

    # Check model name suffix
    if model.endswith("-reasoning") or model.endswith("-non-reasoning"):
        return True

    # Check config for reasoning_effort - must be explicitly present in config
    model_config = _models_config.get(model, {})

    # Only return True if reasoning_effort key exists in the config
    if "reasoning_effort" in model_config:
        return True

    return False


def get_reasoning_effort_from_model(model: str) -> Optional[str]:
    """
    Get the reasoning effort level from model config.

    Returns:
        The effort level ("low", "medium", "high", "off") or None if not configured.
        Note: Returns "off" (not "none") for reasoning models with reasoning disabled,
        to distinguish from chat models with reasoning_mode="none".
    """
    global _models_config
    if _models_config is None:
        _models_config = load_models_config()

    model_config = _models_config.get(model, {})

    # Check if reasoning_effort is explicitly in config
    if "reasoning_effort" not in model_config:
        return None

    reasoning_effort = model_config["reasoning_effort"]

    if isinstance(reasoning_effort, dict):
        enabled = reasoning_effort.get("enabled", None)
        effort = reasoning_effort.get("effort", None)

        # If effort is explicitly set, return it (but map "none" to "off")
        if effort is not None:
            return "off" if effort == "none" else effort

        # If enabled is explicitly False, return "off"
        if enabled is False:
            return "off"

        # If enabled is True but no effort specified, assume "low"
        if enabled is True:
            return "low"

    return None


def get_reasoning_mode_from_results(result_dir: Path) -> Optional[str]:
    """
    Get the reasoning mode from the utility model JSON in a result directory.

    Returns:
        The reasoning_mode value (e.g., "before", "none", "after") or None if not found.
    """
    utility_files = list(result_dir.glob("utility_model_*.json"))
    if not utility_files:
        return None

    try:
        with open(utility_files[0], "r") as f:
            utility_data = json.load(f)

        # reasoning_mode is in utility_model_arguments
        args = utility_data.get("utility_model_arguments", {})
        return args.get("reasoning_mode")
    except Exception:
        return None


def get_reasoning_condition(model: str, result_dir: Optional[Path] = None) -> str:
    """
    Determine the reasoning condition for display.

    For reasoning models (extended thinking): shows effort level ("low", "medium", "high", "off")
        - "off" means reasoning is disabled for this reasoning-capable model
    For chat models: shows reasoning_mode from results ("before", "after", "none")
        - "none" means no reasoning instruction was given to this chat model

    Args:
        model: The model identifier
        result_dir: Optional path to result directory to check reasoning_mode

    Returns:
        The reasoning condition string for display
    """
    # Check if this is a reasoning model
    if is_reasoning_model(model):
        effort = get_reasoning_effort_from_model(model)
        if effort is not None:
            return effort
        # Reasoning model but no effort configured - check model name
        if model.endswith("-non-reasoning"):
            return "off"
        return "low"  # Default for reasoning models

    # For chat models, check the reasoning_mode from results
    if result_dir is not None:
        reasoning_mode = get_reasoning_mode_from_results(result_dir)
        if reasoning_mode is not None:
            return reasoning_mode

    return "unknown"


def get_base_model_name(model: str) -> str:
    """
    Get the base model name by stripping reasoning-related suffixes.

    E.g., "deepseek-v3-2-reasoning" -> "deepseek-v3-2"
          "deepseek-v3-2-non-reasoning" -> "deepseek-v3-2"
          "llama-33-70b" -> "llama-33-70b"
    """
    # Check -non-reasoning FIRST (it also ends with -reasoning)
    if model.endswith("-non-reasoning"):
        return model[:-14]  # len("-non-reasoning") == 14
    elif model.endswith("-reasoning"):
        return model[:-10]  # len("-reasoning") == 10
    return model


def filter_complete_reasoning_pairs(
    results: List["ExperimentResult"],
) -> List["ExperimentResult"]:
    """
    Filter results to only keep model-factor-nudge combinations that have
    both a baseline condition and at least one other condition.

    Baseline conditions:
    - 'off' for reasoning models (reasoning capability disabled)
    - 'none' for chat models (no reasoning instruction)

    This allows comparing reasoning vs non-reasoning for the same setup.

    Args:
        results: List of ExperimentResult objects

    Returns:
        Filtered list containing only complete pairs
    """
    from collections import defaultdict

    # Baseline conditions (no reasoning active)
    baseline_conditions = {"none", "off"}

    # Group by (base_model, factor, nudge_type)
    groups: Dict[tuple, List["ExperimentResult"]] = defaultdict(list)
    for r in results:
        base_model = get_base_model_name(r.model)
        key = (base_model, r.factor, r.nudge_type)
        groups[key].append(r)

    # Keep only groups that have a baseline condition plus at least one other condition
    filtered = []
    for key, group_results in groups.items():
        conditions = {r.reasoning_condition for r in group_results}
        has_baseline = bool(conditions & baseline_conditions)
        has_other = bool(conditions - baseline_conditions)

        if has_baseline and has_other:
            filtered.extend(group_results)

    return filtered


def get_model_display_name(model: str, strip_reasoning_suffix: bool = True) -> str:
    """
    Get the display name for a model from models.yaml config.

    Args:
        model: The model identifier
        strip_reasoning_suffix: If True, remove "(reasoning)" suffix from display name

    Returns:
        The display name for the model
    """
    global _models_config
    if _models_config is None:
        _models_config = load_models_config()

    model_config = _models_config.get(model, {})
    display_name = model_config.get("display_name", model)

    # Strip "(reasoning)" suffix if requested
    if strip_reasoning_suffix and display_name.endswith(" (reasoning)"):
        display_name = display_name[:-12]  # len(" (reasoning)") == 12

    return display_name


def compute_factor_frequencies(
    graph_data: Dict[str, Any],
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, float]:
    """Compute win frequencies for each factor level."""
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0.0, "total": 0} for level in target_levels}

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Skip intra-group comparisons
            if level_a == level_b:
                continue

            if level_a not in target_levels or level_b not in target_levels:
                continue

            aux_data = edge_data.get("aux_data", {})
            original_parsed = aux_data.get("original_parsed", [])
            flipped_parsed = aux_data.get("flipped_parsed", [])

            # Process original responses
            for resp in original_parsed:
                if resp == "A" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1
                elif resp == "B" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1

            # Process flipped responses (A in flipped = original B)
            for resp in flipped_parsed:
                if resp == "A" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1
                elif resp == "B" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1

        except Exception:
            continue

    # Compute frequencies
    frequencies = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            frequencies[level] = stats["wins"] / stats["total"]
        else:
            frequencies[level] = 0.5

    return frequencies


def get_nudge_target_group(result_dir: Path) -> Optional[str]:
    """Get the target group for a nudge condition from the graph data."""
    graph_data = load_preference_graph(result_dir)
    if not graph_data:
        return None

    nudge_config = graph_data.get("nudge_config")
    if nudge_config:
        return nudge_config.get("target_group")
    return None


def find_condition_directories(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Dict[str, Path]:
    """
    Find result directories for each condition (base, and each nudge target).

    Returns:
        Dictionary mapping condition name -> Path to result directory
        e.g., {'base': Path(...), 'low': Path(...), 'high': Path(...)}
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return {}

    result_dirs = {}
    dirs_by_condition: Dict[str, List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        # Check if this is a base condition
        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            # Get target group from graph data
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        if condition not in dirs_by_condition:
            dirs_by_condition[condition] = []
        dirs_by_condition[condition].append(result_dir)

    # For each condition, use the most recent directory
    for condition, dirs in dirs_by_condition.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result_dirs[condition] = most_recent

    return result_dirs


def compute_experiment_result(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[ExperimentResult]:
    """
    Compute all metrics for a single experiment.

    Args:
        factor_name: Name of the factor (e.g., 'social_status')
        model: Model name
        nudge_type: Type of nudge (e.g., 'weak_evidence')
        results_base_dir: Base directory for results

    Returns:
        ExperimentResult object or None if data is insufficient
    """
    # Find all condition directories
    condition_dirs = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    if "base" not in condition_dirs:
        return None

    # Load baseline data
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return None

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return None

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        # For now, only support binary factors
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    # Compute baseline preferences using probability-based method
    base_prefs = compute_factor_preference_from_graph(base_graph, factor_var_name)
    p_A_baseline = base_prefs.get(level_A, {}).get("prob", 0.5)
    p_B_baseline = base_prefs.get(level_B, {}).get("prob", 0.5)
    n_baseline = base_prefs.get(level_A, {}).get("total", 0)

    # Compute baseline bias (deviation from 0.5 for level_B)
    baseline_bias = p_B_baseline - 0.5

    # Load and compute nudge-A condition
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    if not nudge_A_graph:
        return None

    nudge_A_prefs = compute_factor_preference_from_graph(nudge_A_graph, factor_var_name)
    p_A_nudge_A = nudge_A_prefs.get(level_A, {}).get("prob", 0.5)
    # p_B_nudge_A = nudge_A_prefs.get(level_B, {}).get("prob", 0.5)
    n_nudge_A = nudge_A_prefs.get(level_A, {}).get("total", 0)

    # Load and compute nudge-B condition
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])
    if not nudge_B_graph:
        return None

    nudge_B_prefs = compute_factor_preference_from_graph(nudge_B_graph, factor_var_name)
    # p_A_nudge_B = nudge_B_prefs.get(level_A, {}).get("prob", 0.5)
    p_B_nudge_B = nudge_B_prefs.get(level_B, {}).get("prob", 0.5)
    n_nudge_B = nudge_B_prefs.get(level_B, {}).get("total", 0)

    # Compute effect sizes
    effect_size_A = p_A_nudge_A - p_A_baseline
    effect_size_B = p_B_nudge_B - p_B_baseline
    avg_effect_size = (effect_size_A + effect_size_B) / 2

    # Compute steerability bias using frequency-based method
    # Need to use frequency measurements for steerability calculation
    target_levels = [level_A, level_B]

    f_0_A = compute_factor_frequencies(base_graph, factor_var_name, target_levels).get(
        level_A, 0.5
    )
    f_0_B = compute_factor_frequencies(base_graph, factor_var_name, target_levels).get(
        level_B, 0.5
    )
    f_A_A = compute_factor_frequencies(
        nudge_A_graph, factor_var_name, target_levels
    ).get(level_A, 0.5)
    f_A_B = compute_factor_frequencies(
        nudge_A_graph, factor_var_name, target_levels
    ).get(level_B, 0.5)
    f_B_A = compute_factor_frequencies(
        nudge_B_graph, factor_var_name, target_levels
    ).get(level_A, 0.5)
    f_B_B = compute_factor_frequencies(
        nudge_B_graph, factor_var_name, target_levels
    ).get(level_B, 0.5)

    steer_A, steer_B, steerability_bias = compute_steerability_bias_from_frequencies(
        f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B
    )

    # Determine reasoning condition from model name/config, with fallback to results
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    return ExperimentResult(
        model=model,
        reasoning_condition=reasoning_condition,
        factor=factor_name,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        baseline_bias=baseline_bias,
        p_A_baseline=p_A_baseline,
        p_B_baseline=p_B_baseline,
        effect_size_A=effect_size_A,
        effect_size_B=effect_size_B,
        avg_effect_size=avg_effect_size,
        steerability_A=steer_A,
        steerability_B=steer_B,
        steerability_bias=steerability_bias,
        n_baseline=n_baseline,
        n_nudge_A=n_nudge_A,
        n_nudge_B=n_nudge_B,
    )


def discover_experiments(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Discover all available experiments in the results directories.

    Returns:
        List of (results_dir, factor_name, model, nudge_type) tuples
    """
    experiments = []

    for results_base_dir in results_base_dirs:
        results_path = Path(results_base_dir)
        if not results_path.exists():
            continue

        # Iterate through experiment directories (simple_{factor})
        for exp_dir in results_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
                continue

            factor_name = exp_dir.name[7:]  # Remove 'simple_' prefix

            # Apply factor filter
            if factor_filter and factor_name not in factor_filter:
                continue

            # Iterate through model directories
            for model_dir in exp_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name

                # Apply model filter
                if model_filter and model not in model_filter:
                    continue

                # Iterate through nudge type directories
                for nudge_dir in model_dir.iterdir():
                    if not nudge_dir.is_dir():
                        continue

                    nudge_type = nudge_dir.name

                    # Apply nudge type filter
                    if nudge_type_filter and nudge_type not in nudge_type_filter:
                        continue

                    experiments.append(
                        (results_base_dir, factor_name, model, nudge_type)
                    )

    return experiments


def compute_all_results(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
    require_reasoning_pairs: bool = True,
) -> List[ExperimentResult]:
    """
    Compute results for all available experiments.

    Args:
        results_base_dirs: List of base directories for results
        model_filter: Optional list of models to include
        factor_filter: Optional list of factors to include
        nudge_type_filter: Optional list of nudge types to include
        require_reasoning_pairs: If True (default), only keep model-factor-nudge
            combinations that have both 'none' reasoning and at least one other
            reasoning condition

    Returns:
        List of ExperimentResult objects
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    results = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        result = compute_experiment_result(
            factor_name, model, nudge_type, results_base_dir
        )
        if result is not None:
            results.append(result)

    # Filter to complete reasoning pairs if requested
    if require_reasoning_pairs:
        original_count = len(results)
        results = filter_complete_reasoning_pairs(results)
        filtered_count = original_count - len(results)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} results without complete reasoning pairs")

    return results


def format_table(
    results: List[ExperimentResult],
    show_display_names: bool = True,
) -> str:
    """Format results as a text table."""
    if not results:
        return "No results found."

    # Sort by base_model, factor, nudge_type, reasoning_condition
    # This groups model variants together with their reasoning pairs adjacent
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    # Build header
    headers = [
        "Model",
        "Reasoning",
        "Factor",
        "Nudge Type",
        "Baseline Bias",
        "Effect Size",
        "Steerability Bias",
    ]

    # Build rows
    rows = []
    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        steer_bias_str = f"{r.steerability_bias:+.3f}" if r.steerability_bias else "N/A"
        # Show factor with levels so baseline bias can be interpreted
        # Positive bias means biased towards level_B
        factor_with_levels = f"{r.factor} ({r.level_A}/{r.level_B})"

        rows.append(
            [
                model_name,
                r.reasoning_condition,
                factor_with_levels,
                r.nudge_type,
                f"{r.baseline_bias:+.3f}",
                f"{r.avg_effect_size:+.3f}",
                steer_bias_str,
            ]
        )

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    row_lines = []
    for row in rows:
        row_lines.append(
            " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        )

    return "\n".join([header_line, separator] + row_lines)


def format_detailed_table(
    results: List[ExperimentResult],
    show_display_names: bool = True,
) -> str:
    """Format results as a detailed text table with all metrics."""
    if not results:
        return "No results found."

    # Sort by base_model, factor, nudge_type, reasoning_condition
    # This groups model variants together with their reasoning pairs adjacent
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    # Build header
    headers = [
        "Model",
        "Reasoning",
        "Factor (A/B)",
        "Nudge Type",
        "P(A) base",
        "P(B) base",
        "Base Bias",
        "Eff(A)",
        "Eff(B)",
        "Avg Eff",
        "Steer Bias",
        "N",
    ]

    # Build rows
    rows = []
    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        steer_bias_str = f"{r.steerability_bias:+.3f}" if r.steerability_bias else "N/A"
        # Show factor with levels so metrics can be interpreted
        factor_with_levels = f"{r.factor} ({r.level_A}/{r.level_B})"

        rows.append(
            [
                model_name,
                r.reasoning_condition,
                factor_with_levels,
                r.nudge_type,
                f"{r.p_A_baseline:.3f}",
                f"{r.p_B_baseline:.3f}",
                f"{r.baseline_bias:+.3f}",
                f"{r.effect_size_A:+.3f}",
                f"{r.effect_size_B:+.3f}",
                f"{r.avg_effect_size:+.3f}",
                steer_bias_str,
                str(r.n_baseline),
            ]
        )

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    row_lines = []
    for row in rows:
        row_lines.append(
            " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        )

    return "\n".join([header_line, separator] + row_lines)


def write_csv(
    results: List[ExperimentResult],
    output_path: str,
    show_display_names: bool = True,
) -> None:
    """Write results to a CSV file."""
    if not results:
        print("No results to write.")
        return

    # Sort by base_model, factor, nudge_type, reasoning_condition
    # This groups model variants together with their reasoning pairs adjacent
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    headers = [
        "model",
        "model_display_name",
        "reasoning_condition",
        "factor",
        "level_A",
        "level_B",
        "nudge_type",
        "p_A_baseline",
        "p_B_baseline",
        "baseline_bias",
        "effect_size_A",
        "effect_size_B",
        "avg_effect_size",
        "steerability_A",
        "steerability_B",
        "steerability_bias",
        "n_baseline",
        "n_nudge_A",
        "n_nudge_B",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            writer.writerow(
                [
                    r.model,
                    get_model_display_name(r.model) if show_display_names else r.model,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    r.p_A_baseline,
                    r.p_B_baseline,
                    r.baseline_bias,
                    r.effect_size_A,
                    r.effect_size_B,
                    r.avg_effect_size,
                    r.steerability_A if r.steerability_A is not None else "",
                    r.steerability_B if r.steerability_B is not None else "",
                    r.steerability_bias if r.steerability_bias is not None else "",
                    r.n_baseline,
                    r.n_nudge_A,
                    r.n_nudge_B,
                ]
            )

    print(f"Wrote {len(results)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create summary table of nudge experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all results from default results directory
    python create_summary_table.py

    # Specify results directories
    python create_summary_table.py --results-dirs results results2

    # Filter by models, factors, nudge types
    python create_summary_table.py \\
        --models llama-33-70b deepseek-v3-2-reasoning \\
        --factors age_group social_status \\
        --nudge-types weak_evidence emotional

    # Output to CSV with detailed table
    python create_summary_table.py --output summary.csv --detailed

    # Aggregate over factors and nudge types (show per model+reasoning)
    python create_summary_table.py --aggregate factor nudge_type

    # Aggregate over models (compare reasoning conditions across models)
    python create_summary_table.py --aggregate model
        """,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results"],
        help="List of results directories to search (default: results)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to include (default: all)",
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="List of factors to include (default: all)",
    )

    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="List of nudge types to include (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: print to stdout)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed table with all metrics",
    )

    parser.add_argument(
        "--no-display-names",
        action="store_true",
        help="Use raw model names instead of display names",
    )

    parser.add_argument(
        "--complete-pairs-only",
        action="store_true",
        help="Only show model-factor-nudge combinations that have both "
        "'off'/'none' reasoning and another reasoning condition",
    )

    parser.add_argument(
        "--aggregate",
        nargs="+",
        choices=["model", "factor", "nudge_type", "reasoning"],
        default=None,
        help="Aggregate over specified dimensions. "
        "E.g., --aggregate factor nudge_type will show averages per model+reasoning",
    )

    parser.add_argument(
        "--force-signed",
        action="store_true",
        help="Force signed values for all metrics instead of using magnitude when "
        "aggregating over factors (magnitude is used by default to prevent cancellation)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Nudge Experiment Summary Table")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    print("=" * 70)
    print()

    # Compute results
    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        require_reasoning_pairs=args.complete_pairs_only,
    )

    print(f"Found {len(results)} complete experiments\n")

    if not results:
        print("No complete experiments found matching the filters.")
        return

    show_display_names = not args.no_display_names

    # Handle aggregation if requested
    if args.aggregate:
        aggregated, baseline_use_mag, effect_use_mag, steer_use_mag = aggregate_results(
            results, args.aggregate, args.force_signed
        )
        agg_info = f"Aggregated over: {', '.join(args.aggregate)}"
        # All three use the same magnitude setting (based on factor aggregation)
        if baseline_use_mag:  # They're all the same
            agg_info += " (using magnitude for all metrics)"
        print(f"{agg_info}\n")
        print(
            format_aggregated_table(
                aggregated,
                show_display_names,
                baseline_use_mag,
                effect_use_mag,
                steer_use_mag,
            )
        )
    elif args.output:
        write_csv(results, args.output, show_display_names)
    else:
        if args.detailed:
            print(format_detailed_table(results, show_display_names))
        else:
            print(format_table(results, show_display_names))

    # Print summary statistics
    # Use magnitude by default for aggregations over factors (unless --force-signed)
    use_magnitude = not args.force_signed
    print("\n" + "=" * 70)
    print("Summary Statistics")
    if use_magnitude:
        print("(using magnitude for metrics aggregated over factors)")
    else:
        print("(using signed values for all metrics)")
    print("=" * 70)

    # By model (group by base model name to merge reasoning variants)
    # Use magnitude since aggregating over factors
    from collections import defaultdict

    model_groups: Dict[str, List[ExperimentResult]] = defaultdict(list)
    for r in results:
        base_model = get_base_model_name(r.model)
        model_groups[base_model].append(r)

    print(f"\nModels ({len(model_groups)}):")
    for base_model in sorted(model_groups.keys()):
        model_results = model_groups[base_model]
        # Use magnitude for all metrics (aggregating over factors)
        if use_magnitude:
            avg_bias = sum(abs(r.baseline_bias) for r in model_results) / len(
                model_results
            )
            avg_effect = sum(abs(r.avg_effect_size) for r in model_results) / len(
                model_results
            )
            steer_results = [
                r for r in model_results if r.steerability_bias is not None
            ]
            avg_steer = (
                sum(abs(r.steerability_bias) for r in steer_results)
                / len(steer_results)
                if steer_results
                else None
            )
        else:
            avg_bias = sum(r.baseline_bias for r in model_results) / len(model_results)
            avg_effect = sum(r.avg_effect_size for r in model_results) / len(
                model_results
            )
            steer_results = [
                r for r in model_results if r.steerability_bias is not None
            ]
            avg_steer = (
                sum(r.steerability_bias for r in steer_results) / len(steer_results)
                if steer_results
                else None
            )
        steer_str = f"{avg_steer:.3f}" if avg_steer is not None else "N/A"
        # Get display name from any model in the group
        display_name = (
            get_model_display_name(model_results[0].model)
            if show_display_names
            else base_model
        )
        if use_magnitude:
            print(
                f"  {display_name}: n={len(model_results)}, |bias|={avg_bias:.3f}, "
                f"|effect|={avg_effect:.3f}, |steer|={steer_str}"
            )
        else:
            print(
                f"  {display_name}: n={len(model_results)}, bias={avg_bias:+.3f}, "
                f"effect={avg_effect:+.3f}, steer={steer_str}"
            )

    # By reasoning condition
    # Use magnitude since aggregating over factors
    reasoning_conditions = set(r.reasoning_condition for r in results)
    print(f"\nReasoning Conditions ({len(reasoning_conditions)}):")
    for condition in sorted(reasoning_conditions):
        cond_results = [r for r in results if r.reasoning_condition == condition]
        if use_magnitude:
            avg_bias = sum(abs(r.baseline_bias) for r in cond_results) / len(
                cond_results
            )
            avg_effect = sum(abs(r.avg_effect_size) for r in cond_results) / len(
                cond_results
            )
            steer_results = [r for r in cond_results if r.steerability_bias is not None]
            avg_steer = (
                sum(abs(r.steerability_bias) for r in steer_results)
                / len(steer_results)
                if steer_results
                else None
            )
        else:
            avg_bias = sum(r.baseline_bias for r in cond_results) / len(cond_results)
            avg_effect = sum(r.avg_effect_size for r in cond_results) / len(
                cond_results
            )
            steer_results = [r for r in cond_results if r.steerability_bias is not None]
            avg_steer = (
                sum(r.steerability_bias for r in steer_results) / len(steer_results)
                if steer_results
                else None
            )
        steer_str = f"{avg_steer:.3f}" if avg_steer is not None else "N/A"
        if use_magnitude:
            print(
                f"  {condition}: n={len(cond_results)}, |bias|={avg_bias:.3f}, "
                f"|effect|={avg_effect:.3f}, |steer|={steer_str}"
            )
        else:
            print(
                f"  {condition}: n={len(cond_results)}, bias={avg_bias:+.3f}, "
                f"effect={avg_effect:+.3f}, steer={steer_str}"
            )

    # By nudge type
    # Use magnitude since aggregating over factors
    nudge_types = set(r.nudge_type for r in results)
    print(f"\nNudge Types ({len(nudge_types)}):")
    for nudge_type in sorted(nudge_types):
        nudge_results = [r for r in results if r.nudge_type == nudge_type]
        if use_magnitude:
            avg_bias = sum(abs(r.baseline_bias) for r in nudge_results) / len(
                nudge_results
            )
            avg_effect = sum(abs(r.avg_effect_size) for r in nudge_results) / len(
                nudge_results
            )
            steer_results = [
                r for r in nudge_results if r.steerability_bias is not None
            ]
            avg_steer = (
                sum(abs(r.steerability_bias) for r in steer_results)
                / len(steer_results)
                if steer_results
                else None
            )
        else:
            avg_bias = sum(r.baseline_bias for r in nudge_results) / len(nudge_results)
            avg_effect = sum(r.avg_effect_size for r in nudge_results) / len(
                nudge_results
            )
            steer_results = [
                r for r in nudge_results if r.steerability_bias is not None
            ]
            avg_steer = (
                sum(r.steerability_bias for r in steer_results) / len(steer_results)
                if steer_results
                else None
            )
        steer_str = f"{avg_steer:.3f}" if avg_steer is not None else "N/A"
        if use_magnitude:
            print(
                f"  {nudge_type}: n={len(nudge_results)}, |bias|={avg_bias:.3f}, "
                f"|effect|={avg_effect:.3f}, |steer|={steer_str}"
            )
        else:
            print(
                f"  {nudge_type}: n={len(nudge_results)}, bias={avg_bias:+.3f}, "
                f"effect={avg_effect:+.3f}, steer={steer_str}"
            )

    # By factor
    # Don't use magnitude here - we're looking at individual factors, not aggregating over them
    factors = set(r.factor for r in results)
    print(f"\nFactors ({len(factors)}):")
    for factor in sorted(factors):
        factor_results = [r for r in results if r.factor == factor]
        avg_bias = sum(r.baseline_bias for r in factor_results) / len(factor_results)
        avg_effect = sum(r.avg_effect_size for r in factor_results) / len(
            factor_results
        )
        steer_results = [r for r in factor_results if r.steerability_bias is not None]
        avg_steer = (
            sum(r.steerability_bias for r in steer_results) / len(steer_results)
            if steer_results
            else None
        )
        steer_str = f"{avg_steer:+.3f}" if avg_steer is not None else "N/A"
        print(
            f"  {factor}: n={len(factor_results)}, bias={avg_bias:+.3f}, "
            f"effect={avg_effect:+.3f}, steer={steer_str}"
        )


if __name__ == "__main__":
    main()
