#!/usr/bin/env python3
"""
Plot baseline bias vs sign-adjusted steerability bias (MSB) scatter plot.

This script creates a scatter plot showing the relationship between
baseline bias magnitude and MSB, where MSB is sign-adjusted so that
positive values indicate MSB is in the same direction as the baseline bias.

Only data points with statistically significant baseline bias are included.

Usage:
    # Basic usage - all data points
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2

    # Group by model, factor, or nudge type
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups model
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups factor
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups nudge

    # Specify results directories
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --results-dirs results results2

    # Combine options
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups model --results-dirs results
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_counts,
)
from choices.analysis.utils import (
    get_model_color,
    get_model_display_name,
    get_reasoning_condition,
)

# Default significance level (95% confidence)
DEFAULT_ALPHA = 0.05


@dataclass
class DataPoint:
    """Data point for the scatter plot."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    # Baseline metrics
    baseline_bias: float  # f_0(B) - 0.5 (positive = biased towards B)
    baseline_bias_magnitude: float  # |baseline_bias|
    baseline_significant: bool  # Whether baseline bias is statistically significant
    # Steerability metrics
    steerability_A: float  # How much nudging towards A increases A's odds
    steerability_B: float  # How much nudging towards B increases B's odds
    steerability_A_significant: bool  # Whether steerability_A differs from 0
    steerability_B_significant: bool  # Whether steerability_B differs from 0
    steerability_bias: float  # MSB (steerability_B - steerability_A)
    sign_adjusted_msb: float  # MSB adjusted so positive = same direction as baseline
    msb_significant: bool  # Whether MSB is statistically significant
    # Sample info
    n_samples: int


def binomial_test_vs_half(
    proportion: float,
    n: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float | bool]:
    """
    Test if a proportion differs significantly from 0.5 using exact binomial test.

    Args:
        proportion: Observed proportion
        n: Total number of trials
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with p_value, ci_low, ci_high, and is_significant
    """
    if n <= 0:
        return {
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 1.0,
            "is_significant": False,
        }

    successes = int(round(proportion * n))
    result = stats.binomtest(successes, n, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=1 - alpha)

    return {
        "p_value": result.pvalue,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "is_significant": result.pvalue < alpha,
    }


def test_steerability_significance(
    c_base_target: float,
    c_base_other: float,
    c_nudge_target: float,
    c_nudge_other: float,
    steerability: float,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float | bool]:
    """
    Test if a single steerability value differs significantly from 0 using Wald test.

    steerability = log(r_nudge) - log(r_base)
                 = log(c_nudge_target/c_nudge_other) - log(c_base_target/c_base_other)

    Args:
        c_base_target, c_base_other: Baseline counts
        c_nudge_target, c_nudge_other: Counts when nudged towards target
        steerability: The computed steerability value
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with p_value, se, z_score, and is_significant
    """
    # Apply Haldane-Anscombe correction
    c_base_target_adj = c_base_target + 0.5
    c_base_other_adj = c_base_other + 0.5
    c_nudge_target_adj = c_nudge_target + 0.5
    c_nudge_other_adj = c_nudge_other + 0.5

    # Var(log(a/b)) ≈ 1/a + 1/b
    var_log_ratio_base = 1.0 / c_base_target_adj + 1.0 / c_base_other_adj
    var_log_ratio_nudge = 1.0 / c_nudge_target_adj + 1.0 / c_nudge_other_adj

    var_steer = var_log_ratio_base + var_log_ratio_nudge
    se_steer = np.sqrt(var_steer)

    if se_steer > 0:
        z_score = steerability / se_steer
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0

    return {
        "p_value": p_value,
        "se": se_steer,
        "z_score": z_score,
        "is_significant": p_value < alpha,
    }


def test_msb_significance(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    msb: float,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float | bool]:
    """
    Test if MSB (steerability bias) differs significantly from 0 using Wald test.

    Uses log-odds ratio variance approximation:
    Var(log(a/b)) ≈ 1/a + 1/b

    MSB = steerability_B - steerability_A
        = [log(c_B_B/c_B_A) - log(c_0_B/c_0_A)] - [log(c_A_A/c_A_B) - log(c_0_A/c_0_B)]

    Variance is computed assuming independence between nudge conditions
    (the baseline terms partially cancel in the variance calculation).

    Args:
        c_0_A, c_0_B: Baseline counts
        c_A_A, c_A_B: Counts when nudged towards A
        c_B_A, c_B_B: Counts when nudged towards B
        msb: The computed MSB value
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with p_value, se, z_score, and is_significant
    """
    # Apply Haldane-Anscombe correction for variance calculation
    c_0_A_adj = c_0_A + 0.5
    c_0_B_adj = c_0_B + 0.5
    c_A_A_adj = c_A_A + 0.5
    c_A_B_adj = c_A_B + 0.5
    c_B_A_adj = c_B_A + 0.5
    c_B_B_adj = c_B_B + 0.5

    # Variance of each log-odds term
    # steerability_A = log(c_A_A/c_A_B) - log(c_0_A/c_0_B)
    # steerability_B = log(c_B_B/c_B_A) - log(c_0_B/c_0_A)
    # MSB = steerability_B - steerability_A

    # Var(log(a/b)) ≈ 1/a + 1/b
    var_log_ratio_nudge_A = 1.0 / c_A_A_adj + 1.0 / c_A_B_adj
    var_log_ratio_nudge_B = 1.0 / c_B_B_adj + 1.0 / c_B_A_adj
    var_log_ratio_base = 1.0 / c_0_A_adj + 1.0 / c_0_B_adj

    # Total variance (treating nudge conditions as independent)
    # Baseline terms appear in both steerability_A and steerability_B with opposite signs
    # so they contribute 2 * var_log_ratio_base to total variance
    var_msb = var_log_ratio_nudge_A + var_log_ratio_nudge_B + 2 * var_log_ratio_base
    se_msb = np.sqrt(var_msb)

    # Wald test: z = MSB / SE(MSB)
    if se_msb > 0:
        z_score = msb / se_msb
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0

    return {
        "p_value": p_value,
        "se": se_msb,
        "z_score": z_score,
        "is_significant": p_value < alpha,
    }


def compute_factor_frequencies_with_counts(
    graph_data: Dict,
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute win frequencies and sample counts for each factor level.

    Returns:
        Dictionary mapping level -> {"freq": float, "n": int, "wins": int}
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0, "total": 0} for level in target_levels}

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

    # Compute frequencies with counts
    result = {}
    for level, stats_data in level_stats.items():
        if stats_data["total"] > 0:
            result[level] = {
                "freq": stats_data["wins"] / stats_data["total"],
                "n": stats_data["total"],
                "wins": stats_data["wins"],
            }
        else:
            result[level] = {"freq": 0.5, "n": 0, "wins": 0}

    return result


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


def compute_data_point(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[DataPoint]:
    """
    Compute baseline bias and steerability metrics for a single experiment.

    Args:
        factor_name: Name of the factor
        model: Model name
        nudge_type: Type of nudge
        results_base_dir: Base directory for results

    Returns:
        DataPoint object or None if data is insufficient
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
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    # Load nudge condition data
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_A_graph or not nudge_B_graph:
        return None

    # Compute frequencies for each condition
    target_levels = [level_A, level_B]

    base_stats = compute_factor_frequencies_with_counts(
        base_graph, factor_var_name, target_levels
    )
    nudge_A_stats = compute_factor_frequencies_with_counts(
        nudge_A_graph, factor_var_name, target_levels
    )
    nudge_B_stats = compute_factor_frequencies_with_counts(
        nudge_B_graph, factor_var_name, target_levels
    )

    # Get frequencies, sample sizes, and win counts
    f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
    n_0_B = base_stats.get(level_B, {}).get("n", 0)
    c_0_A = base_stats.get(level_A, {}).get("wins", 0)
    c_0_B = base_stats.get(level_B, {}).get("wins", 0)

    c_A_A = nudge_A_stats.get(level_A, {}).get("wins", 0)
    c_A_B = nudge_A_stats.get(level_B, {}).get("wins", 0)

    c_B_A = nudge_B_stats.get(level_A, {}).get("wins", 0)
    c_B_B = nudge_B_stats.get(level_B, {}).get("wins", 0)

    # Compute baseline bias: deviation from 50% for level_B
    # Positive means biased towards level_B
    baseline_bias = f_0_B - 0.5

    # Test significance of baseline bias
    baseline_test = binomial_test_vs_half(f_0_B, n_0_B)
    baseline_significant = baseline_test["is_significant"]

    # Compute steerability bias (MSB) using counts with Haldane-Anscombe correction
    steer_A, steer_B, steerability_bias = compute_steerability_bias_from_counts(
        c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B
    )

    if steerability_bias is None:
        return None

    # Test significance of individual steerabilities
    # steer_A = log(c_A_A/c_A_B) - log(c_0_A/c_0_B)
    steer_A_test = test_steerability_significance(c_0_A, c_0_B, c_A_A, c_A_B, steer_A)
    steer_A_significant = steer_A_test["is_significant"]

    # steer_B = log(c_B_B/c_B_A) - log(c_0_B/c_0_A)
    steer_B_test = test_steerability_significance(c_0_B, c_0_A, c_B_B, c_B_A, steer_B)
    steer_B_significant = steer_B_test["is_significant"]

    # Test significance of MSB
    msb_test = test_msb_significance(
        c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B, steerability_bias
    )
    msb_significant = msb_test["is_significant"]

    # Compute sign-adjusted MSB:
    # If baseline_bias > 0 (biased towards B), positive MSB means same direction
    # If baseline_bias < 0 (biased towards A), we flip MSB sign
    if baseline_bias >= 0:
        sign_adjusted_msb = steerability_bias
    else:
        sign_adjusted_msb = -steerability_bias

    # Determine reasoning condition
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    return DataPoint(
        model=model,
        reasoning_condition=reasoning_condition,
        factor=factor_name,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        baseline_bias=baseline_bias,
        baseline_bias_magnitude=abs(baseline_bias),
        baseline_significant=baseline_significant,
        steerability_A=steer_A,
        steerability_B=steer_B,
        steerability_A_significant=steer_A_significant,
        steerability_B_significant=steer_B_significant,
        steerability_bias=steerability_bias,
        sign_adjusted_msb=sign_adjusted_msb,
        msb_significant=msb_significant,
        n_samples=n_0_B,
    )


def compute_all_data_points(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
    require_significant_baseline: bool = True,
) -> List[DataPoint]:
    """
    Compute data points for all available experiments.

    Args:
        results_base_dirs: List of base directories for results
        model_filter: Optional list of models to include
        factor_filter: Optional list of factors to include
        nudge_type_filter: Optional list of nudge types to include
        require_significant_baseline: If True, only include points with significant baseline bias

    Returns:
        List of DataPoint objects
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    data_points = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        point = compute_data_point(factor_name, model, nudge_type, results_base_dir)
        if point is not None:
            if require_significant_baseline and not point.baseline_significant:
                continue
            data_points.append(point)

    return data_points


# Color palette for factors
FACTOR_COLORS = {
    "gender": "#E63946",  # red
    "wealth": "#457B9D",  # blue
    "age_group": "#2A9D8F",  # teal
    "social_status": "#E9C46A",  # yellow/gold
    "ethnicity": "#9B5DE5",  # purple
}

# Extended factor colors
_EXTRA_FACTOR_COLORS = [
    "#264653",
    "#e76f51",
    "#8338ec",
    "#ff006e",
    "#3a86ff",
]

_dynamic_factor_colors: Dict[str, str] = {}


def get_factor_color(factor: str) -> str:
    """Get color for a factor, auto-assigning from palette if not predefined."""
    if factor in FACTOR_COLORS:
        return FACTOR_COLORS[factor]

    if factor not in _dynamic_factor_colors:
        idx = len(_dynamic_factor_colors) % len(_EXTRA_FACTOR_COLORS)
        _dynamic_factor_colors[factor] = _EXTRA_FACTOR_COLORS[idx]

    return _dynamic_factor_colors[factor]


# Color palette for nudge types
NUDGE_COLORS = {
    "survey_preference": "#E63946",  # red
    "always_save": "#457B9D",  # blue
    "utilitarian": "#2A9D8F",  # teal
    "deontological": "#E9C46A",  # yellow/gold
    "weak_evidence": "#9B5DE5",  # purple
    "emotional": "#F4A261",  # orange
    "expert_opinion": "#00BBF9",  # cyan
    "social_proof": "#F15BB5",  # pink
}

_EXTRA_NUDGE_COLORS = [
    "#264653",
    "#e76f51",
    "#8338ec",
    "#ff006e",
    "#3a86ff",
]

_dynamic_nudge_colors: Dict[str, str] = {}


def get_nudge_color(nudge_type: str) -> str:
    """Get color for a nudge type, auto-assigning from palette if not predefined."""
    if nudge_type in NUDGE_COLORS:
        return NUDGE_COLORS[nudge_type]

    if nudge_type not in _dynamic_nudge_colors:
        idx = len(_dynamic_nudge_colors) % len(_EXTRA_NUDGE_COLORS)
        _dynamic_nudge_colors[nudge_type] = _EXTRA_NUDGE_COLORS[idx]

    return _dynamic_nudge_colors[nudge_type]


def create_scatter_plot(
    data_points: List[DataPoint],
    groups: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    n_bins: int = 10,
):
    """
    Create a scatter plot of baseline bias magnitude vs sign-adjusted MSB.

    Args:
        data_points: List of DataPoint objects
        groups: Grouping variable ('model', 'factor', or 'nudge')
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size as (width, height)
        n_bins: Number of bins for the binned average overlay (default: 10)

    Returns:
        Tuple of (figure, axes) or None if no data
    """
    if not data_points:
        print("No data points to plot!")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Define helper functions for grouping
    def get_group_model(dp: DataPoint):
        return (dp.model, dp.reasoning_condition)

    def get_color_model(g):
        return get_model_color(g[0])

    def get_label_model(g):
        return f"{get_model_display_name(g[0])} ({g[1]})"

    def get_group_factor(dp: DataPoint):
        return dp.factor

    def get_label_factor(g):
        return g.replace("_", " ").title()

    def get_group_nudge(dp: DataPoint):
        return dp.nudge_type

    def get_label_nudge(g):
        return g.replace("_", " ").title()

    def get_group_none(dp: DataPoint):
        return None

    def get_color_none(g):
        return "#457B9D"  # Default blue

    def get_label_none(g):
        return None

    # Get unique groups for coloring
    if groups == "model":
        unique_groups = list(
            dict.fromkeys((dp.model, dp.reasoning_condition) for dp in data_points)
        )
        get_group = get_group_model
        get_color = get_color_model
        get_label = get_label_model
    elif groups == "factor":
        unique_groups = list(dict.fromkeys(dp.factor for dp in data_points))
        get_group = get_group_factor
        get_color = get_factor_color
        get_label = get_label_factor
    elif groups == "nudge":
        unique_groups = list(dict.fromkeys(dp.nudge_type for dp in data_points))
        get_group = get_group_nudge
        get_color = get_nudge_color
        get_label = get_label_nudge
    else:
        unique_groups = None
        get_group = get_group_none
        get_color = get_color_none
        get_label = get_label_none

    # Plot data points
    if unique_groups:
        for group in unique_groups:
            group_points = [dp for dp in data_points if get_group(dp) == group]
            x = [dp.baseline_bias_magnitude for dp in group_points]
            y = [dp.sign_adjusted_msb for dp in group_points]

            ax.scatter(
                x,
                y,
                c=get_color(group),
                label=get_label(group),
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )
    else:
        x = [dp.baseline_bias_magnitude for dp in data_points]
        y = [dp.sign_adjusted_msb for dp in data_points]

        ax.scatter(
            x,
            y,
            c=get_color(None),
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Add reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Compute and overlay binned averages (across all groups)
    if len(data_points) > 0:
        x_all = np.array([dp.baseline_bias_magnitude for dp in data_points])
        y_all = np.array([dp.sign_adjusted_msb for dp in data_points])

        # Create bins from 0 to 0.5
        bin_edges = np.linspace(0, 0.5, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Compute average y for each bin
        bin_means = []
        bin_counts = []
        for i in range(n_bins):
            mask = (x_all >= bin_edges[i]) & (x_all < bin_edges[i + 1])
            # Include the right edge for the last bin
            if i == n_bins - 1:
                mask = (x_all >= bin_edges[i]) & (x_all <= bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(y_all[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_means.append(np.nan)
                bin_counts.append(0)

        bin_means = np.array(bin_means)
        bin_counts = np.array(bin_counts)

        # Plot binned averages as a step plot (histogram style)
        # Only plot bins that have data
        valid_mask = ~np.isnan(bin_means)
        if np.any(valid_mask):
            # Plot as bars
            ax.bar(
                bin_centers[valid_mask],
                bin_means[valid_mask],
                width=bin_width * 0.9,
                alpha=0.3,
                color="black",
                edgecolor="black",
                linewidth=1.5,
                label=f"Binned avg (n={len(data_points)})",
                zorder=1,
            )

    # Set axis limits
    ax.set_xlim(0, 0.5)

    # Labels
    ax.set_xlabel("Baseline Bias Magnitude (|f₀(B) - 0.5|)", fontsize=14)
    ax.set_ylabel(
        "Sign-Adjusted MSB\n(+ = MSB in same direction as baseline)", fontsize=14
    )

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        ax.set_title(
            "Baseline Bias vs Steerability Bias\n(Only Significant Baseline Biases)",
            fontsize=16,
            fontweight="bold",
        )

    # Legend - always show (includes binned average)
    handles, _ = ax.get_legend_handles_labels()
    if handles and len(handles) <= 16:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            framealpha=0.9,
        )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved plot to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Create scatter plot of baseline bias magnitude vs sign-adjusted MSB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - all data points
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2

    # Group by model, factor, or nudge type
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups model
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups factor
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --groups nudge

    # Specify results directories
    uv run python -m choices.analysis.plot_baseline_vs_steerability_v2 --results-dirs results results2
        """,
    )

    parser.add_argument(
        "--groups",
        type=str,
        choices=["model", "factor", "nudge"],
        default=None,
        help="Group data points by this variable (model, factor, or nudge)",
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
        "--include-insignificant",
        action="store_true",
        help="Include data points with insignificant baseline bias",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: baseline_vs_msb.pdf)",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 8],
        help="Figure size (width height)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for the binned average overlay (default: 10)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.groups}" if args.groups else ""
        output_path = f"baseline_vs_msb{suffix}.pdf"

    # Print header
    print("=" * 70)
    print("Baseline Bias vs Sign-Adjusted MSB Scatter Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    if args.groups:
        print(f"Grouping by: {args.groups}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    require_significant = not args.include_insignificant
    print(f"Require significant baseline: {require_significant}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Compute data points (with significant baseline for plotting)
    data_points = compute_all_data_points(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        require_significant_baseline=require_significant,
    )

    print(f"Found {len(data_points)} data points with significant baseline bias\n")

    # Also compute all data points to get stats for non-significant baseline cases
    all_data_points = compute_all_data_points(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        require_significant_baseline=False,  # Include all
    )

    # Filter for non-significant baseline cases
    nonsig_baseline_points = [
        dp for dp in all_data_points if not dp.baseline_significant
    ]

    if nonsig_baseline_points:
        print("=" * 70)
        print("Statistics for Samples with Non-Significant Baseline Bias")
        print("=" * 70)
        print(
            f"  Total samples with non-significant baseline: {len(nonsig_baseline_points)}"
        )
        print()

        # Fraction of cases with significant MSB
        sig_msb_points = [dp for dp in nonsig_baseline_points if dp.msb_significant]
        sig_msb_count = len(sig_msb_points)
        sig_msb_fraction = sig_msb_count / len(nonsig_baseline_points)
        print(
            f"  Fraction with significant MSB: {sig_msb_count}/{len(nonsig_baseline_points)} "
            f"({100 * sig_msb_fraction:.1f}%)"
        )

        # Average absolute value of MSB for significant cases
        if sig_msb_points:
            avg_abs_msb_sig = np.mean(
                [abs(dp.steerability_bias) for dp in sig_msb_points]
            )
            print(f"  Average |MSB| for significant cases: {avg_abs_msb_sig:.4f}")
        print()

        # Backfire statistics for non-significant baseline cases
        # A backfire = nudging towards X resulted in significantly negative steerability_X
        backfire_A = sum(
            1
            for dp in nonsig_baseline_points
            if dp.steerability_A < 0 and dp.steerability_A_significant
        )
        backfire_B = sum(
            1
            for dp in nonsig_baseline_points
            if dp.steerability_B < 0 and dp.steerability_B_significant
        )
        # Total backfires (each data point can have 0, 1, or 2 backfires)
        total_nudges = 2 * len(nonsig_baseline_points)  # 2 nudge directions per point
        total_backfires = backfire_A + backfire_B
        print(
            f"  Significant backfires: {total_backfires}/{total_nudges} "
            f"({100 * total_backfires / total_nudges:.1f}%)"
        )
        print()

        # Percentile values of |MSB| in steps of 10%
        abs_msb_values = np.array(
            [abs(dp.steerability_bias) for dp in nonsig_baseline_points]
        )
        percentiles = np.arange(0, 101, 10)
        percentile_values = np.percentile(abs_msb_values, percentiles)

        print("  Percentiles of |MSB| (absolute value, irrespective of significance):")
        print("  " + "-" * 40)
        for p, val in zip(percentiles, percentile_values):
            print(f"    {p:3d}th percentile: {val:.4f}")
        print("=" * 70)
        print()

    # Backfire statistics for significant baseline cases
    sig_baseline_points = [dp for dp in all_data_points if dp.baseline_significant]

    if sig_baseline_points:
        print("=" * 70)
        print("Backfire Statistics for Samples with Significant Baseline Bias")
        print("=" * 70)
        print(f"  Total samples with significant baseline: {len(sig_baseline_points)}")
        print()

        # Overall backfire rate
        # A backfire = nudging towards X resulted in significantly negative steerability_X
        backfire_A = sum(
            1
            for dp in sig_baseline_points
            if dp.steerability_A < 0 and dp.steerability_A_significant
        )
        backfire_B = sum(
            1
            for dp in sig_baseline_points
            if dp.steerability_B < 0 and dp.steerability_B_significant
        )
        total_nudges = 2 * len(sig_baseline_points)
        total_backfires = backfire_A + backfire_B
        print(
            f"  Overall significant backfires: {total_backfires}/{total_nudges} "
            f"({100 * total_backfires / total_nudges:.1f}%)"
        )

        # Backfire when nudging towards dominant option
        # Dominant = option model is biased towards (B if baseline_bias > 0, A if < 0)
        backfire_towards_dominant = 0
        nudges_towards_dominant = len(sig_baseline_points)
        for dp in sig_baseline_points:
            if dp.baseline_bias > 0:
                # B is dominant, check if nudging towards B backfired
                if dp.steerability_B < 0 and dp.steerability_B_significant:
                    backfire_towards_dominant += 1
            else:
                # A is dominant, check if nudging towards A backfired
                if dp.steerability_A < 0 and dp.steerability_A_significant:
                    backfire_towards_dominant += 1
        print(
            f"  Backfire nudging towards dominant: {backfire_towards_dominant}/{nudges_towards_dominant} "
            f"({100 * backfire_towards_dominant / nudges_towards_dominant:.1f}%)"
        )

        # Backfire when nudging towards less preferred option
        backfire_towards_less_preferred = 0
        nudges_towards_less_preferred = len(sig_baseline_points)
        for dp in sig_baseline_points:
            if dp.baseline_bias > 0:
                # B is dominant, so A is less preferred
                # Check if nudging towards A backfired
                if dp.steerability_A < 0 and dp.steerability_A_significant:
                    backfire_towards_less_preferred += 1
            else:
                # A is dominant, so B is less preferred
                # Check if nudging towards B backfired
                if dp.steerability_B < 0 and dp.steerability_B_significant:
                    backfire_towards_less_preferred += 1
        print(
            f"  Backfire nudging towards less preferred: {backfire_towards_less_preferred}/{nudges_towards_less_preferred} "
            f"({100 * backfire_towards_less_preferred / nudges_towards_less_preferred:.1f}%)"
        )
        print("=" * 70)
        print()

    if not data_points:
        print("No data points found matching the criteria.")
        return

    # Create plot
    result = create_scatter_plot(
        data_points=data_points,
        groups=args.groups,
        output_path=output_path,
        title=args.title,
        figsize=tuple(args.figsize),
        n_bins=args.n_bins,
    )

    if result is None:
        return

    # Print data summary
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    print(
        f"{'Model':<25} {'Factor':<12} {'Nudge':<18} "
        f"{'Base Bias':>10} {'MSB':>10} {'Adj MSB':>10}"
    )
    print("-" * 95)
    for dp in sorted(data_points, key=lambda x: (x.model, x.factor, x.nudge_type)):
        print(
            f"{get_model_display_name(dp.model):<25} "
            f"{dp.factor:<12} "
            f"{dp.nudge_type:<18} "
            f"{dp.baseline_bias:>+10.3f} "
            f"{dp.steerability_bias:>+10.3f} "
            f"{dp.sign_adjusted_msb:>+10.3f}"
        )

    # Print statistics by group if grouping is enabled
    if args.groups:
        print("\n" + "=" * 70)
        print(f"Statistics by {args.groups.title()}")
        print("=" * 70)

        if args.groups == "model":
            groups = {}
            for dp in data_points:
                key = (dp.model, dp.reasoning_condition)
                if key not in groups:
                    groups[key] = []
                groups[key].append(dp)

            for (model, reasoning), points in sorted(groups.items()):
                avg_baseline = np.mean([dp.baseline_bias_magnitude for dp in points])
                avg_adj_msb = np.mean([dp.sign_adjusted_msb for dp in points])
                positive_count = sum(1 for dp in points if dp.sign_adjusted_msb > 0)
                print(
                    f"  {get_model_display_name(model)} ({reasoning}): "
                    f"n={len(points)}, "
                    f"avg |baseline|={avg_baseline:.3f}, "
                    f"avg adj_MSB={avg_adj_msb:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

        elif args.groups == "factor":
            groups = {}
            for dp in data_points:
                if dp.factor not in groups:
                    groups[dp.factor] = []
                groups[dp.factor].append(dp)

            for factor, points in sorted(groups.items()):
                avg_baseline = np.mean([dp.baseline_bias_magnitude for dp in points])
                avg_adj_msb = np.mean([dp.sign_adjusted_msb for dp in points])
                positive_count = sum(1 for dp in points if dp.sign_adjusted_msb > 0)
                print(
                    f"  {factor}: "
                    f"n={len(points)}, "
                    f"avg |baseline|={avg_baseline:.3f}, "
                    f"avg adj_MSB={avg_adj_msb:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

        elif args.groups == "nudge":
            groups = {}
            for dp in data_points:
                if dp.nudge_type not in groups:
                    groups[dp.nudge_type] = []
                groups[dp.nudge_type].append(dp)

            for nudge_type, points in sorted(groups.items()):
                avg_baseline = np.mean([dp.baseline_bias_magnitude for dp in points])
                avg_adj_msb = np.mean([dp.sign_adjusted_msb for dp in points])
                positive_count = sum(1 for dp in points if dp.sign_adjusted_msb > 0)
                print(
                    f"  {nudge_type}: "
                    f"n={len(points)}, "
                    f"avg |baseline|={avg_baseline:.3f}, "
                    f"avg adj_MSB={avg_adj_msb:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

    # Overall statistics
    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    avg_baseline = np.mean([dp.baseline_bias_magnitude for dp in data_points])
    avg_adj_msb = np.mean([dp.sign_adjusted_msb for dp in data_points])
    positive_count = sum(1 for dp in data_points if dp.sign_adjusted_msb > 0)
    x_all = [dp.baseline_bias_magnitude for dp in data_points]
    y_all = [dp.sign_adjusted_msb for dp in data_points]
    corr, p_value = stats.pearsonr(x_all, y_all)

    print(f"  Total data points: {len(data_points)}")
    print(f"  Average |baseline bias|: {avg_baseline:.3f}")
    print(f"  Average sign-adjusted MSB: {avg_adj_msb:+.3f}")
    print(
        f"  Positive sign-adjusted MSB: {positive_count}/{len(data_points)} "
        f"({100*positive_count/len(data_points):.1f}%)"
    )
    print(f"  Correlation (r): {corr:.3f} (p = {p_value:.4f})")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
