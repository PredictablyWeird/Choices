#!/usr/bin/env python3
"""
Surface Form Analysis: Compare normal nudges vs random information baseline nudges.

This script analyzes whether nudge effectiveness comes from the actual content
(e.g., survey data, evidence) or just the surface form (presenting any information
that points to a specific option).

For each (model, factor, nudge_type, option) combination, this script plots:
- x-axis: f_0(c) = baseline frequency of choosing option c (no nudge)
- y-axis: f_c(c) = frequency of choosing option c when nudged towards c

Two conditions are shown (by default):
- Normal nudge: Uses actual nudge content (e.g., real survey preference)
- Baseline nudge: Uses random/nonsensical information with same surface form

If nudges work primarily through surface form (not content), baseline nudges
should be nearly as effective as normal nudges.

Baseline nudges are identified by the "_baseline" suffix on nudge type names
(e.g., "survey_preference_baseline" pairs with "survey_preference").

Usage:
    # Basic usage - search multiple results directories
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline

    # Show only baseline nudge effects
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --condition baseline

    # Show only normal nudge effects
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --condition normal

    # Group by model while showing both conditions
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --groups model
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    get_nudge_target_group,
    load_preference_graph,
)
from choices.analysis.plot_baseline_vs_nudged_frequency import (
    get_factor_color,
    get_nudge_color,
    get_reasoning_color,
)
from choices.analysis.analyze_simple_nudging_results import (
    two_proportion_z_test,
)
from choices.analysis.steerability_metric import compute_odds
from choices.analysis.utils import (
    compute_factor_frequencies_with_counts,
    get_model_color,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)

# Default significance level (95% confidence)
DEFAULT_ALPHA = 0.05


@dataclass
class SurfaceFormDataPoint:
    """Data point for a single nudge condition (normal or baseline)."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str  # The base nudge type (without "_baseline" suffix)
    option: str  # The option being tracked (e.g., "male", "female")
    other_option: str  # The other option
    condition: str  # "normal" or "baseline"
    # Frequencies
    f_0: float  # Baseline frequency (no nudge)
    f_c: float  # Frequency when nudged towards this option
    effect_size: float  # f_c - f_0
    steerability: float  # ln odds steerability: s(d) = ln(r_d(d)) - ln(r_0(d))
    # Significance (two-proportion z-test comparing nudge to baseline)
    is_significant: bool  # True if effect differs significantly from 0
    p_value: float  # p-value from the test
    # Sample sizes
    n_base: int
    n_nudged: int


def compute_steerability_from_freq(
    f_0: float, n_base: int, f_c: float, n_nudged: int
) -> float:
    """
    Compute ln odds steerability: s(d) = ln(r_c(d)) - ln(r_0(d)).

    Uses Haldane-Anscombe correction via compute_odds.
    """
    c_0_d = round(f_0 * n_base)
    c_0_dbar = n_base - c_0_d
    c_c_d = round(f_c * n_nudged)
    c_c_dbar = n_nudged - c_c_d
    r_0 = compute_odds(c_0_d, c_0_dbar)
    r_c = compute_odds(c_c_d, c_c_dbar)
    return math.log(r_c) - math.log(r_0)


# Condition colors for the two nudge types
CONDITION_COLORS = {
    "normal": "#E63946",  # red - actual nudge content
    "baseline": "#457B9D",  # blue - random information
}

# Condition markers
CONDITION_MARKERS = {
    "normal": "o",  # circle
    "baseline": "s",  # square
}


def discover_nudge_pairs(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> Dict[Tuple[str, str, str], Dict[str, List[str]]]:
    """
    Discover pairs of normal nudges and their baseline counterparts.

    Returns:
        Dictionary mapping (factor, model, base_nudge_type) ->
            {"normal": [results_dir, ...], "baseline": [results_dir, ...]}
    """
    # Collect all nudge experiments across directories
    # Key: (factor, model, nudge_type), Value: list of results_dirs where found
    all_experiments: Dict[Tuple[str, str, str], List[str]] = {}

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

                    key = (factor_name, model, nudge_type)
                    if key not in all_experiments:
                        all_experiments[key] = []
                    all_experiments[key].append(results_base_dir)

    # Now pair up normal nudges with their baseline counterparts
    pairs: Dict[Tuple[str, str, str], Dict[str, List[str]]] = {}

    for (factor, model, nudge_type), dirs in all_experiments.items():
        # Check if this is a baseline nudge
        if nudge_type.endswith("_baseline"):
            base_nudge = nudge_type[:-9]  # Remove "_baseline" suffix

            # Apply nudge type filter to base name
            if nudge_type_filter and base_nudge not in nudge_type_filter:
                continue

            key = (factor, model, base_nudge)
            if key not in pairs:
                pairs[key] = {"normal": [], "baseline": []}
            pairs[key]["baseline"].extend(dirs)
        else:
            # Apply nudge type filter
            if nudge_type_filter and nudge_type not in nudge_type_filter:
                continue

            key = (factor, model, nudge_type)
            if key not in pairs:
                pairs[key] = {"normal": [], "baseline": []}
            pairs[key]["normal"].extend(dirs)

    # Filter to keep only complete pairs (both normal and baseline exist)
    complete_pairs = {
        key: dirs for key, dirs in pairs.items() if dirs["normal"] and dirs["baseline"]
    }

    return complete_pairs


def find_condition_directories_multi(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dirs: List[str],
) -> List[Dict[str, Path]]:
    """
    Find result directories for each condition across multiple result directories.

    Groups directories by reasoning_mode to handle cases where
    the same model/factor/nudge_type has results with different reasoning settings.

    Returns:
        List of dictionaries, each mapping condition name -> Path to result directory.
    """
    experiment_name = f"simple_{factor_name}"

    # Group directories by (condition, reasoning_mode)
    dirs_by_condition_and_reasoning: Dict[Tuple[str, str], List[Path]] = {}

    for results_base_dir in results_base_dirs:
        base_path = Path(results_base_dir) / experiment_name / model / nudge_type

        if not base_path.exists():
            continue

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

            # Get reasoning_mode from the utility model JSON
            reasoning_mode = get_reasoning_mode_from_results(result_dir)
            if reasoning_mode is None:
                reasoning_mode = "unknown"

            key = (condition, reasoning_mode)
            if key not in dirs_by_condition_and_reasoning:
                dirs_by_condition_and_reasoning[key] = []
            dirs_by_condition_and_reasoning[key].append(result_dir)

    # For each (condition, reasoning_mode), use the most recent directory
    # Then group by reasoning_mode to build complete experiments
    experiments_by_reasoning: Dict[str, Dict[str, Path]] = {}

    for (condition, reasoning_mode), dirs in dirs_by_condition_and_reasoning.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)

        if reasoning_mode not in experiments_by_reasoning:
            experiments_by_reasoning[reasoning_mode] = {}
        experiments_by_reasoning[reasoning_mode][condition] = most_recent

    return list(experiments_by_reasoning.values())


def compute_data_points_for_pair(
    factor_name: str,
    model: str,
    base_nudge_type: str,
    normal_dirs: List[str],
    baseline_dirs: List[str],
) -> List[SurfaceFormDataPoint]:
    """
    Compute surface form comparison data points for a nudge pair.

    Args:
        factor_name: Name of the factor
        model: Model name
        base_nudge_type: Base nudge type name (without "_baseline")
        normal_dirs: Result directories containing normal nudge results
        baseline_dirs: Result directories containing baseline nudge results

    Returns:
        List of SurfaceFormDataPoint objects (one per option per condition)
    """
    # Find condition directories for normal nudge
    normal_experiments = find_condition_directories_multi(
        factor_name, model, base_nudge_type, normal_dirs
    )

    # Find condition directories for baseline nudge
    baseline_nudge_type = f"{base_nudge_type}_baseline"
    baseline_experiments = find_condition_directories_multi(
        factor_name, model, baseline_nudge_type, baseline_dirs
    )

    if not normal_experiments or not baseline_experiments:
        return []

    data_points = []

    # For each reasoning mode that has both normal and baseline data
    for normal_cond_dirs in normal_experiments:
        if "base" not in normal_cond_dirs:
            continue

        # Load baseline (no-nudge) data from normal experiment
        base_graph = load_preference_graph(normal_cond_dirs["base"])
        if not base_graph:
            continue

        # Get factor info
        factor_var_name = get_factor_name_from_graph(base_graph)
        if not factor_var_name:
            continue

        factor_levels = get_factor_levels_from_graph(base_graph)
        if len(factor_levels) != 2:
            continue

        level_A, level_B = factor_levels[0], factor_levels[1]

        # Check we have nudge conditions for both levels in normal experiment
        if level_A not in normal_cond_dirs or level_B not in normal_cond_dirs:
            continue

        # Determine reasoning condition
        reasoning_condition = get_reasoning_condition(model, normal_cond_dirs["base"])

        # Find matching baseline experiment (same reasoning mode)
        matching_baseline = None
        for baseline_cond_dirs in baseline_experiments:
            if "base" not in baseline_cond_dirs:
                continue
            baseline_reasoning = get_reasoning_condition(
                model, baseline_cond_dirs["base"]
            )
            if baseline_reasoning == reasoning_condition:
                # Check this baseline experiment has both level conditions
                if level_A in baseline_cond_dirs and level_B in baseline_cond_dirs:
                    matching_baseline = baseline_cond_dirs
                    break

        if not matching_baseline:
            continue

        # Compute frequencies for base (no-nudge) condition
        base_stats = compute_factor_frequencies_with_counts(
            base_graph, factor_var_name, [level_A, level_B]
        )

        # Load and compute normal nudge conditions
        normal_A_graph = load_preference_graph(normal_cond_dirs[level_A])
        normal_B_graph = load_preference_graph(normal_cond_dirs[level_B])

        if not normal_A_graph or not normal_B_graph:
            continue

        normal_A_stats = compute_factor_frequencies_with_counts(
            normal_A_graph, factor_var_name, [level_A, level_B]
        )
        normal_B_stats = compute_factor_frequencies_with_counts(
            normal_B_graph, factor_var_name, [level_A, level_B]
        )

        # Load and compute baseline nudge conditions
        baseline_A_graph = load_preference_graph(matching_baseline[level_A])
        baseline_B_graph = load_preference_graph(matching_baseline[level_B])

        if not baseline_A_graph or not baseline_B_graph:
            continue

        baseline_A_stats = compute_factor_frequencies_with_counts(
            baseline_A_graph, factor_var_name, [level_A, level_B]
        )
        baseline_B_stats = compute_factor_frequencies_with_counts(
            baseline_B_graph, factor_var_name, [level_A, level_B]
        )

        # Create data points for option A - NORMAL nudge
        f_0_A = base_stats.get(level_A, {}).get("freq", 0.5)
        n_base_A = base_stats.get(level_A, {}).get("n", 0)

        f_c_normal_A = normal_A_stats.get(level_A, {}).get("freq", 0.5)
        n_normal_A = normal_A_stats.get(level_A, {}).get("n", 0)

        # Test significance for normal nudge A
        test_normal_A = two_proportion_z_test(
            f_0_A, n_base_A, f_c_normal_A, n_normal_A, DEFAULT_ALPHA
        )

        data_points.append(
            SurfaceFormDataPoint(
                model=model,
                reasoning_condition=reasoning_condition,
                factor=factor_name,
                nudge_type=base_nudge_type,
                option=level_A,
                other_option=level_B,
                condition="normal",
                f_0=f_0_A,
                f_c=f_c_normal_A,
                effect_size=f_c_normal_A - f_0_A,
                steerability=compute_steerability_from_freq(
                    f_0_A, n_base_A, f_c_normal_A, n_normal_A
                ),
                is_significant=test_normal_A["is_significant"],
                p_value=test_normal_A["p_value"],
                n_base=n_base_A,
                n_nudged=n_normal_A,
            )
        )

        # Create data points for option A - BASELINE nudge
        f_c_baseline_A = baseline_A_stats.get(level_A, {}).get("freq", 0.5)
        n_baseline_A = baseline_A_stats.get(level_A, {}).get("n", 0)

        # Test significance for baseline nudge A
        test_baseline_A = two_proportion_z_test(
            f_0_A, n_base_A, f_c_baseline_A, n_baseline_A, DEFAULT_ALPHA
        )

        data_points.append(
            SurfaceFormDataPoint(
                model=model,
                reasoning_condition=reasoning_condition,
                factor=factor_name,
                nudge_type=base_nudge_type,
                option=level_A,
                other_option=level_B,
                condition="baseline",
                f_0=f_0_A,
                f_c=f_c_baseline_A,
                effect_size=f_c_baseline_A - f_0_A,
                steerability=compute_steerability_from_freq(
                    f_0_A, n_base_A, f_c_baseline_A, n_baseline_A
                ),
                is_significant=test_baseline_A["is_significant"],
                p_value=test_baseline_A["p_value"],
                n_base=n_base_A,
                n_nudged=n_baseline_A,
            )
        )

        # Create data points for option B - NORMAL nudge
        f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
        n_base_B = base_stats.get(level_B, {}).get("n", 0)

        f_c_normal_B = normal_B_stats.get(level_B, {}).get("freq", 0.5)
        n_normal_B = normal_B_stats.get(level_B, {}).get("n", 0)

        # Test significance for normal nudge B
        test_normal_B = two_proportion_z_test(
            f_0_B, n_base_B, f_c_normal_B, n_normal_B, DEFAULT_ALPHA
        )

        data_points.append(
            SurfaceFormDataPoint(
                model=model,
                reasoning_condition=reasoning_condition,
                factor=factor_name,
                nudge_type=base_nudge_type,
                option=level_B,
                other_option=level_A,
                condition="normal",
                f_0=f_0_B,
                f_c=f_c_normal_B,
                effect_size=f_c_normal_B - f_0_B,
                steerability=compute_steerability_from_freq(
                    f_0_B, n_base_B, f_c_normal_B, n_normal_B
                ),
                is_significant=test_normal_B["is_significant"],
                p_value=test_normal_B["p_value"],
                n_base=n_base_B,
                n_nudged=n_normal_B,
            )
        )

        # Create data points for option B - BASELINE nudge
        f_c_baseline_B = baseline_B_stats.get(level_B, {}).get("freq", 0.5)
        n_baseline_B = baseline_B_stats.get(level_B, {}).get("n", 0)

        # Test significance for baseline nudge B
        test_baseline_B = two_proportion_z_test(
            f_0_B, n_base_B, f_c_baseline_B, n_baseline_B, DEFAULT_ALPHA
        )

        data_points.append(
            SurfaceFormDataPoint(
                model=model,
                reasoning_condition=reasoning_condition,
                factor=factor_name,
                nudge_type=base_nudge_type,
                option=level_B,
                other_option=level_A,
                condition="baseline",
                f_0=f_0_B,
                f_c=f_c_baseline_B,
                effect_size=f_c_baseline_B - f_0_B,
                steerability=compute_steerability_from_freq(
                    f_0_B, n_base_B, f_c_baseline_B, n_baseline_B
                ),
                is_significant=test_baseline_B["is_significant"],
                p_value=test_baseline_B["p_value"],
                n_base=n_base_B,
                n_nudged=n_baseline_B,
            )
        )

    return data_points


def compute_all_data_points(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
    reasoning_conditions_filter: Optional[List[str]] = None,
    condition_filter: Optional[List[str]] = None,
) -> List[SurfaceFormDataPoint]:
    """
    Compute data points for all available nudge pairs.

    Args:
        results_base_dirs: List of base directories for results
        model_filter: Optional list of models to include
        factor_filter: Optional list of factors to include
        nudge_type_filter: Optional list of nudge types to include
        reasoning_conditions_filter: Optional list of reasoning conditions to include
        condition_filter: Optional list of conditions to include ("normal", "baseline")

    Returns:
        List of SurfaceFormDataPoint objects
    """
    pairs = discover_nudge_pairs(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    data_points = []
    for (factor, model, base_nudge), dirs in pairs.items():
        points = compute_data_points_for_pair(
            factor, model, base_nudge, dirs["normal"], dirs["baseline"]
        )
        for point in points:
            # Apply reasoning condition filter
            if reasoning_conditions_filter:
                if point.reasoning_condition not in reasoning_conditions_filter:
                    continue
            # Apply condition filter (normal vs baseline)
            if condition_filter:
                if point.condition not in condition_filter:
                    continue
            data_points.append(point)

    return data_points


def create_grouped_bar_chart(
    data_points: List[SurfaceFormDataPoint],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a grouped bar chart comparing normal vs irrelevant nudge effect magnitudes by model.

    Shows |effect| for normal and irrelevant nudges side by side for each model,
    with the irrelevant/normal ratio annotated.
    """
    if not data_points:
        print("No data points to plot!")
        return None

    # Group by (model, reasoning_condition)
    model_data: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for dp in data_points:
        key = (dp.model, dp.reasoning_condition)
        if key not in model_data:
            model_data[key] = {"normal": [], "baseline": []}
        model_data[key][dp.condition].append(abs(dp.steerability))

    # Compute averages and prepare plot data
    models = []
    normal_means = []
    baseline_means = []
    ratios = []

    # Custom sort: group by model, then non-reasoning before reasoning
    # Reasoning order: off/none (non-reasoning) before low/before (reasoning)
    def sort_key(item):
        model, reasoning = item[0]
        # Map reasoning conditions to sort order (non-reasoning=0, reasoning=1)
        reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(reasoning, 2)
        return (model, reasoning_order)

    for (model, reasoning), effects in sorted(model_data.items(), key=sort_key):
        if effects["normal"] and effects["baseline"]:
            models.append(f"{get_model_display_name(model)}\n({reasoning})")
            n_mean = np.mean(effects["normal"])
            b_mean = np.mean(effects["baseline"])
            normal_means.append(n_mean)
            baseline_means.append(b_mean)
            ratios.append(b_mean / n_mean if n_mean > 0 else 0)

    if not models:
        print("No complete model data found!")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.35

    _bars1 = ax.bar(
        x - width / 2,
        normal_means,
        width,
        label="Informative",
        color="#E63946",
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )
    _bars2 = ax.bar(
        x + width / 2,
        baseline_means,
        width,
        label="Irrelevant",
        color="#457B9D",
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    # Styling
    ax.set_ylabel("Average |Steerability|", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc="upper right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(normal_means + baseline_means) * 1.15)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved bar chart to: {output_path}")

    return fig, ax


def create_scatter_plot(
    data_points: List[SurfaceFormDataPoint],
    groups: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    n_bins: int = 10,
    show_diagonal: bool = True,
):
    """
    Create a scatter plot of baseline frequency (f_0) vs nudged frequency (f_c).

    By default, groups by condition (normal vs baseline nudge) to show both
    on the same plot. Can also group by model, factor, nudge type, or reasoning.

    Args:
        data_points: List of SurfaceFormDataPoint objects
        groups: Grouping variable ('model', 'factor', 'nudge', 'reasoning', or None for 'condition')
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size as (width, height)
        n_bins: Number of bins for the binned average overlay
        show_diagonal: If True, show the y=x diagonal line

    Returns:
        Tuple of (figure, axes) or None if no data
    """
    if not data_points:
        print("No data points to plot!")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Define helper functions for grouping
    def get_group_condition(dp: SurfaceFormDataPoint):
        return dp.condition

    def get_color_condition(g):
        return CONDITION_COLORS.get(g, "#808080")

    def get_marker_condition(g):
        return CONDITION_MARKERS.get(g, "o")

    def get_label_condition(g):
        if g == "normal":
            return "Normal Nudge (actual content)"
        elif g == "baseline":
            return "Baseline Nudge (random info)"
        return g

    def get_group_model(dp: SurfaceFormDataPoint):
        return (dp.model, dp.reasoning_condition)

    def get_color_model(g):
        return get_model_color(g[0])

    def get_label_model(g):
        return f"{get_model_display_name(g[0])} ({g[1]})"

    def get_group_factor(dp: SurfaceFormDataPoint):
        return dp.factor

    def get_label_factor(g):
        return g.replace("_", " ").title()

    def get_group_nudge(dp: SurfaceFormDataPoint):
        return dp.nudge_type

    def get_label_nudge(g):
        return g.replace("_", " ").title()

    def get_group_reasoning(dp: SurfaceFormDataPoint):
        return dp.reasoning_condition

    def get_label_reasoning(g):
        return g

    def get_marker_default(g):
        return "o"

    # Determine grouping mode
    # Default: group by condition (normal vs baseline)
    if groups == "model":
        unique_groups = list(
            dict.fromkeys((dp.model, dp.reasoning_condition) for dp in data_points)
        )
        get_group = get_group_model
        get_color = get_color_model
        get_marker = get_marker_default
        get_label = get_label_model
    elif groups == "factor":
        unique_groups = list(dict.fromkeys(dp.factor for dp in data_points))
        get_group = get_group_factor
        get_color = get_factor_color
        get_marker = get_marker_default
        get_label = get_label_factor
    elif groups == "nudge":
        unique_groups = list(dict.fromkeys(dp.nudge_type for dp in data_points))
        get_group = get_group_nudge
        get_color = get_nudge_color
        get_marker = get_marker_default
        get_label = get_label_nudge
    elif groups == "reasoning":
        unique_groups = list(
            dict.fromkeys(dp.reasoning_condition for dp in data_points)
        )
        get_group = get_group_reasoning
        get_color = get_reasoning_color
        get_marker = get_marker_default
        get_label = get_label_reasoning
    else:
        # Default: group by condition (normal vs baseline)
        unique_groups = list(dict.fromkeys(dp.condition for dp in data_points))
        get_group = get_group_condition
        get_color = get_color_condition
        get_marker = get_marker_condition
        get_label = get_label_condition

    # Plot data points (x = f_0, y = f_c)
    for group in unique_groups:
        group_points = [dp for dp in data_points if get_group(dp) == group]
        x = [dp.f_0 for dp in group_points]
        y = [dp.f_c for dp in group_points]

        ax.scatter(
            x,
            y,
            c=get_color(group),
            marker=get_marker(group),
            label=get_label(group),
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Add diagonal line (y = x) - no nudge effect
    if show_diagonal:
        ax.plot(
            [0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="y = x (no effect)"
        )

    # Compute and overlay binned averages per condition
    conditions = list(dict.fromkeys(dp.condition for dp in data_points))

    for condition in conditions:
        condition_points = [dp for dp in data_points if dp.condition == condition]
        if not condition_points:
            continue

        x_all = np.array([dp.f_0 for dp in condition_points])
        y_all = np.array([dp.f_c for dp in condition_points])

        # Create bins from 0 to 1
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute average y for each bin
        bin_means = []
        for i in range(n_bins):
            mask = (x_all >= bin_edges[i]) & (x_all < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (x_all >= bin_edges[i]) & (x_all <= bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(y_all[mask]))
            else:
                bin_means.append(np.nan)

        bin_means = np.array(bin_means)

        # Plot binned averages as a line
        valid_mask = ~np.isnan(bin_means)
        if np.any(valid_mask):
            color = CONDITION_COLORS.get(condition, "#808080")
            ax.plot(
                bin_centers[valid_mask],
                bin_means[valid_mask],
                color=color,
                linestyle="-",
                linewidth=2.5,
                alpha=0.9,
                zorder=10,
            )
            ax.scatter(
                bin_centers[valid_mask],
                bin_means[valid_mask],
                c=color,
                s=60,
                zorder=11,
                edgecolors="white",
                linewidths=1.5,
                marker=CONDITION_MARKERS.get(condition, "o"),
            )

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel("Baseline Frequency f₀(c) (no nudge)", fontsize=14)
    ax.set_ylabel("Nudged Frequency fₓ(c) (nudged towards c)", fontsize=14)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        ax.set_title(
            "Surface Form Analysis: Baseline Preference vs Nudged Frequency",
            fontsize=16,
            fontweight="bold",
        )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
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

    # Make plot square
    ax.set_aspect("equal", adjustable="box")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved plot to: {output_path}")

    return fig, ax


def print_statistics(
    data_points: List[SurfaceFormDataPoint],
    groups: Optional[str] = None,
):
    """Print comprehensive statistics for the surface form analysis."""
    if not data_points:
        print("No data points to analyze.")
        return

    # Separate by condition
    normal_points = [dp for dp in data_points if dp.condition == "normal"]
    baseline_points = [dp for dp in data_points if dp.condition == "baseline"]

    # Data summary table (with significance markers: * = p<0.05)
    print("\n" + "=" * 130)
    print("Data Summary (* = p<0.05)")
    print("=" * 130)
    print(
        f"{'Model':<20} {'Reas':<6} {'Factor':<12} {'Nudge':<18} "
        f"{'Option':<10} {'Cond':<10} {'f_0':>8} {'f_c':>8} {'Effect':>10} {'Steer':>10}"
    )
    print("-" * 140)
    for dp in sorted(
        data_points,
        key=lambda x: (
            x.model,
            x.reasoning_condition,
            x.factor,
            x.nudge_type,
            x.option,
            x.condition,
        ),
    ):
        # Format effect with significance marker
        effect_str = f"{dp.effect_size:+.3f}{'*' if dp.is_significant else ''}"
        steer_str = f"{dp.steerability:+.3f}"
        print(
            f"{get_model_display_name(dp.model):<20} "
            f"{dp.reasoning_condition:<6} "
            f"{dp.factor:<12} "
            f"{dp.nudge_type:<18} "
            f"{dp.option:<10} "
            f"{dp.condition:<10} "
            f"{dp.f_0:>8.3f} "
            f"{dp.f_c:>8.3f} "
            f"{effect_str:>10} "
            f"{steer_str:>10}"
        )

    # Statistics by condition
    print("\n" + "=" * 115)
    print("Statistics by Condition")
    print("=" * 115)

    for condition, points in [("normal", normal_points), ("baseline", baseline_points)]:
        if not points:
            continue
        avg_f0 = np.mean([dp.f_0 for dp in points])
        avg_fc = np.mean([dp.f_c for dp in points])
        avg_effect = np.mean([dp.effect_size for dp in points])
        positive_count = sum(1 for dp in points if dp.effect_size > 0)
        sig_count = sum(1 for dp in points if dp.is_significant)
        print(
            f"  {condition.capitalize()}: "
            f"n={len(points)}, "
            f"avg f₀={avg_f0:.3f}, "
            f"avg f_c={avg_fc:.3f}, "
            f"avg effect={avg_effect:+.3f}, "
            f"positive={positive_count}/{len(points)}, "
            f"significant={sig_count}/{len(points)} ({100*sig_count/len(points):.0f}%)"
        )

    # Statistics by additional group if specified
    if groups:
        print("\n" + "=" * 115)
        print(f"Statistics by {groups.title()}")
        print("=" * 115)

        def print_group_stats(group_name, points):
            # Separate by condition
            normal_pts = [dp for dp in points if dp.condition == "normal"]
            baseline_pts = [dp for dp in points if dp.condition == "baseline"]

            print(f"  {group_name}:")

            # Normal condition stats
            if normal_pts:
                avg_effect_n = np.mean([dp.effect_size for dp in normal_pts])
                avg_mag_n = np.mean([abs(dp.effect_size) for dp in normal_pts])
                avg_steer_mag_n = np.mean([abs(dp.steerability) for dp in normal_pts])
                sig_n = sum(1 for dp in normal_pts if dp.is_significant)
                print(
                    f"    Normal:   n={len(normal_pts):>3}, "
                    f"avg effect={avg_effect_n:+.3f}, "
                    f"avg |effect|={avg_mag_n:.3f}, "
                    f"avg |steer|={avg_steer_mag_n:.3f}, "
                    f"sig={sig_n}/{len(normal_pts)} ({100*sig_n/len(normal_pts):.0f}%)"
                )
            else:
                avg_mag_n = None
                avg_steer_mag_n = None

            # Baseline condition stats
            if baseline_pts:
                avg_effect_b = np.mean([dp.effect_size for dp in baseline_pts])
                avg_mag_b = np.mean([abs(dp.effect_size) for dp in baseline_pts])
                avg_steer_mag_b = np.mean([abs(dp.steerability) for dp in baseline_pts])
                sig_b = sum(1 for dp in baseline_pts if dp.is_significant)
                print(
                    f"    Baseline: n={len(baseline_pts):>3}, "
                    f"avg effect={avg_effect_b:+.3f}, "
                    f"avg |effect|={avg_mag_b:.3f}, "
                    f"avg |steer|={avg_steer_mag_b:.3f}, "
                    f"sig={sig_b}/{len(baseline_pts)} ({100*sig_b/len(baseline_pts):.0f}%)"
                )
            else:
                avg_mag_b = None
                avg_steer_mag_b = None

            # Ratio of baseline to normal magnitude
            if avg_mag_n is not None and avg_mag_b is not None and avg_mag_n > 0:
                ratio = avg_mag_b / avg_mag_n
                print(f"    Baseline/Normal magnitude ratio: {ratio:.2f}x")
            if (
                avg_steer_mag_n is not None
                and avg_steer_mag_b is not None
                and avg_steer_mag_n > 0
            ):
                steer_ratio = avg_steer_mag_b / avg_steer_mag_n
                print(f"    Baseline/Normal steerability ratio: {steer_ratio:.2f}x")

        if groups == "model":
            group_dict = {}
            for dp in data_points:
                key = (dp.model, dp.reasoning_condition)
                if key not in group_dict:
                    group_dict[key] = []
                group_dict[key].append(dp)

            for (model, reasoning), points in sorted(group_dict.items()):
                print_group_stats(
                    f"{get_model_display_name(model)} ({reasoning})", points
                )

        # Per model x nudge type breakdown (for paper table)
        if groups == "model":
            print("\n" + "-" * 115)
            print("Per Model x Nudge Type Steerability (for table)")
            print("-" * 115)

            model_nudge_data: Dict[tuple, Dict[str, list]] = {}
            for dp in data_points:
                key = (dp.model, dp.reasoning_condition, dp.nudge_type)
                if key not in model_nudge_data:
                    model_nudge_data[key] = {"normal": [], "baseline": []}
                cond = "normal" if dp.condition == "normal" else "baseline"
                model_nudge_data[key][cond].append(abs(dp.steerability))

            nudge_types_ordered = sorted(set(dp.nudge_type for dp in data_points))
            model_keys_ordered = sorted(
                set((dp.model, dp.reasoning_condition) for dp in data_points),
                key=lambda x: (
                    x[0],
                    {"off": 0, "none": 0, "low": 1, "before": 1}.get(x[1], 2),
                ),
            )

            header = "{:<30}".format("Model")
            for nt in nudge_types_ordered:
                header += " {:>20}".format(nt)
            print(header)

            for model, reasoning in model_keys_ordered:
                display = "{} ({})".format(get_model_display_name(model), reasoning)
                row = "{:<30}".format(display)
                for nt in nudge_types_ordered:
                    key = (model, reasoning, nt)
                    if key in model_nudge_data and model_nudge_data[key]["normal"]:
                        n_avg = np.mean(model_nudge_data[key]["normal"])
                        b_avg = np.mean(model_nudge_data[key]["baseline"])
                        delta = n_avg - b_avg
                        cell = "{:.2f} ({:+.2f})".format(n_avg, delta)
                    else:
                        cell = "-"
                    row += " {:>20}".format(cell)
                print(row)

        elif groups == "factor":
            group_dict = {}
            for dp in data_points:
                if dp.factor not in group_dict:
                    group_dict[dp.factor] = []
                group_dict[dp.factor].append(dp)

            for factor, points in sorted(group_dict.items()):
                print_group_stats(factor, points)

        elif groups == "nudge":
            group_dict = {}
            for dp in data_points:
                if dp.nudge_type not in group_dict:
                    group_dict[dp.nudge_type] = []
                group_dict[dp.nudge_type].append(dp)

            for nudge_type, points in sorted(group_dict.items()):
                print_group_stats(nudge_type, points)

        elif groups == "reasoning":
            group_dict = {}
            for dp in data_points:
                if dp.reasoning_condition not in group_dict:
                    group_dict[dp.reasoning_condition] = []
                group_dict[dp.reasoning_condition].append(dp)

            for reasoning, points in sorted(group_dict.items()):
                print_group_stats(reasoning, points)

    # Overall statistics
    print("\n" + "=" * 115)
    print("Overall Statistics")
    print("=" * 115)

    x_all = [dp.f_0 for dp in data_points]
    y_all = [dp.f_c for dp in data_points]
    effects = [dp.effect_size for dp in data_points]
    steers = [dp.steerability for dp in data_points]

    avg_f0 = np.mean(x_all)
    avg_fc = np.mean(y_all)
    avg_effect = np.mean(effects)
    avg_steer = np.mean(steers)
    avg_steer_mag = np.mean([abs(s) for s in steers])
    positive_count = sum(1 for e in effects if e > 0)
    sig_count = sum(1 for dp in data_points if dp.is_significant)

    # Correlation
    corr, p_value = stats.pearsonr(x_all, y_all)

    # T-test for effect size different from 0
    t_stat, t_pvalue = stats.ttest_1samp(effects, 0)

    print(f"  Total data points: {len(data_points)}")
    print(f"  Average baseline frequency (f₀): {avg_f0:.3f}")
    print(f"  Average nudged frequency (f_c): {avg_fc:.3f}")
    print(f"  Average effect size (f_c - f₀): {avg_effect:+.3f}")
    print(
        f"  Average steerability (ln odds): {avg_steer:+.3f}, avg |steer|: {avg_steer_mag:.3f}"
    )
    print(
        f"  Positive effects: {positive_count}/{len(data_points)} "
        f"({100*positive_count/len(data_points):.1f}%)"
    )
    print(
        f"  Significant effects (p<0.05): {sig_count}/{len(data_points)} "
        f"({100*sig_count/len(data_points):.1f}%)"
    )
    print(f"  Correlation (f₀ vs f_c): r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Effect size t-test: t = {t_stat:.3f} (p = {t_pvalue:.4f})")

    # Comparison between normal and baseline (if both present)
    if normal_points and baseline_points:
        print("\n" + "-" * 115)
        print("Normal vs Baseline Comparison")
        print("-" * 115)

        normal_effects = [dp.effect_size for dp in normal_points]
        baseline_effects = [dp.effect_size for dp in baseline_points]
        normal_steers = [dp.steerability for dp in normal_points]
        baseline_steers = [dp.steerability for dp in baseline_points]
        normal_sig = sum(1 for dp in normal_points if dp.is_significant)
        baseline_sig = sum(1 for dp in baseline_points if dp.is_significant)

        avg_normal = np.mean(normal_effects)
        avg_baseline = np.mean(baseline_effects)
        avg_mag_normal = np.mean([abs(e) for e in normal_effects])
        avg_mag_baseline = np.mean([abs(e) for e in baseline_effects])
        avg_steer_mag_normal = np.mean([abs(s) for s in normal_steers])
        avg_steer_mag_baseline = np.mean([abs(s) for s in baseline_steers])

        print(
            f"  Normal nudge avg effect: {avg_normal:+.3f}, avg |effect|: {avg_mag_normal:.3f}, avg |steer|: {avg_steer_mag_normal:.3f}"
        )
        print(
            f"  Normal nudge significant: {normal_sig}/{len(normal_points)} "
            f"({100*normal_sig/len(normal_points):.0f}%)"
        )
        print(
            f"  Baseline nudge avg effect: {avg_baseline:+.3f}, avg |effect|: {avg_mag_baseline:.3f}, avg |steer|: {avg_steer_mag_baseline:.3f}"
        )
        print(
            f"  Baseline nudge significant: {baseline_sig}/{len(baseline_points)} "
            f"({100*baseline_sig/len(baseline_points):.0f}%)"
        )

        # Magnitude ratio
        if avg_mag_normal > 0:
            ratio = avg_mag_baseline / avg_mag_normal
            print(f"  Baseline/Normal magnitude ratio: {ratio:.2f}x")
        if avg_steer_mag_normal > 0:
            steer_ratio = avg_steer_mag_baseline / avg_steer_mag_normal
            print(f"  Baseline/Normal steerability ratio: {steer_ratio:.2f}x")

        print(
            f"  Effect difference (normal - baseline): {avg_normal - avg_baseline:+.3f}"
        )

        # Independent t-test comparing effect sizes
        t_compare, p_compare = stats.ttest_ind(normal_effects, baseline_effects)
        print(f"  Independent t-test: t = {t_compare:.3f} (p = {p_compare:.4f})")

        # Interpretation
        if p_compare < 0.05:
            if avg_normal > avg_baseline:
                print(
                    "  -> Normal nudges are significantly MORE effective than baseline nudges."
                )
            else:
                print(
                    "  -> Baseline nudges are significantly MORE effective than normal nudges."
                )
        else:
            print("  -> No significant difference between normal and baseline nudges.")

        # Filtered analysis: only cases where informative nudge had substantial effect
        # This addresses the issue that models with many small effects have misleading ratios
        print("\n" + "-" * 115)
        print(
            "Filtered Analysis: Steerability Ratios by Model (only cases where |normal steerability| > threshold)"
        )
        print("-" * 115)
        print(
            "(When informative nudges have little steerability, both conditions look similar, inflating ratios toward 1.0)"
        )

        # Group data points by (model, reasoning, factor, nudge_type, option) to pair normal vs baseline
        pairs: Dict[tuple, Dict[str, float]] = {}
        for dp in data_points:
            key = (
                dp.model,
                dp.reasoning_condition,
                dp.factor,
                dp.nudge_type,
                dp.option,
            )
            if key not in pairs:
                pairs[key] = {}
            pairs[key][dp.condition] = abs(dp.steerability)

        # Filter to complete pairs only
        complete_pairs = [
            (k, v) for k, v in pairs.items() if "normal" in v and "baseline" in v
        ]

        thresholds = [0.0, 0.05, 0.10]

        for threshold in thresholds:
            print(f"\n  Threshold: |normal steerability| > {threshold}")

            # Group by model
            model_stats: Dict[tuple, Dict[str, list]] = {}
            for (
                model,
                reasoning,
                factor,
                nudge_type,
                option,
            ), effects in complete_pairs:
                key = (model, reasoning)
                if key not in model_stats:
                    model_stats[key] = {"normal": [], "baseline": [], "diffs": []}

                normal_eff = effects["normal"]
                baseline_eff = effects["baseline"]

                # Only include if normal effect exceeds threshold
                if normal_eff > threshold:
                    model_stats[key]["normal"].append(normal_eff)
                    model_stats[key]["baseline"].append(baseline_eff)
                    model_stats[key]["diffs"].append(normal_eff - baseline_eff)

            # Print header
            print(
                f"  {'Model':<30} {'N':<5} {'Avg Norm':<10} {'Avg Irrel':<10} {'Diff':<10} {'Ratio':<8}"
            )

            # Sort: group by model, non-reasoning before reasoning
            def sort_key(item):
                model, reasoning = item[0]
                reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(
                    reasoning, 2
                )
                return (model, reasoning_order)

            for (model, reasoning), model_data in sorted(
                model_stats.items(), key=sort_key
            ):
                if not model_data["normal"]:
                    continue
                n = len(model_data["normal"])
                avg_normal = sum(model_data["normal"]) / n
                avg_baseline = sum(model_data["baseline"]) / n
                avg_diff = sum(model_data["diffs"]) / n
                ratio = avg_baseline / avg_normal if avg_normal > 0 else 0

                display_name = f"{get_model_display_name(model)} ({reasoning})"
                print(
                    f"  {display_name:<30} {n:<5} {avg_normal:<10.3f} {avg_baseline:<10.3f} {avg_diff:+10.3f} {ratio:.2f}x"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze surface form effects: compare normal nudges vs random information baseline nudges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - search multiple results directories (shows both conditions)
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline

    # Show only baseline nudge effects
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --condition baseline

    # Show only normal nudge effects
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --condition normal

    # Group by model while showing both conditions
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --groups model

    # Group by model but only show baseline nudges
    uv run python -m choices.analysis.surface_form_analysis --results-dirs results_main0 results_baseline --groups model --condition baseline
        """,
    )

    parser.add_argument(
        "--groups",
        type=str,
        choices=["model", "factor", "nudge", "reasoning"],
        default=None,
        help="Group data points by this variable (default: condition)",
    )

    parser.add_argument(
        "--condition",
        type=str,
        nargs="+",
        choices=["normal", "baseline"],
        default=None,
        help="Filter to show only specific condition(s): 'normal', 'baseline', or both (default: both)",
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
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include (e.g., 'before', 'none', 'after')",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: surface_form_analysis.pdf)",
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

    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Don't show the y=x diagonal line",
    )

    parser.add_argument(
        "--bar-chart",
        action="store_true",
        help="Create grouped bar chart instead of scatter plot",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = ""
        if args.groups:
            suffix += f"_{args.groups}"
        if args.condition and len(args.condition) == 1:
            suffix += f"_{args.condition[0]}"
        output_path = f"surface_form_analysis{suffix}.pdf"

    # Print header
    print("=" * 110)
    print("Surface Form Analysis: Normal vs Baseline Nudge Effects")
    print("=" * 110)
    print(f"Results directories: {args.results_dirs}")
    if args.groups:
        print(f"Grouping by: {args.groups}")
    else:
        print("Grouping by: condition (default)")
    if args.condition:
        print(f"Condition filter: {args.condition}")
    else:
        print("Showing: both normal and baseline conditions")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning_conditions:
        print(f"Reasoning conditions filter: {args.reasoning_conditions}")
    print(f"Output: {output_path}")
    print("=" * 110)
    print()

    # Discover and display available pairs
    pairs = discover_nudge_pairs(
        args.results_dirs, args.models, args.factors, args.nudge_types
    )

    print(f"Found {len(pairs)} nudge pairs with both normal and baseline results:\n")
    for (factor, model, nudge), dirs in sorted(pairs.items()):
        print(f"  {factor}/{model}/{nudge}")
        print(f"    Normal from: {dirs['normal']}")
        print(f"    Baseline from: {dirs['baseline']}")

    # Compute data points
    data_points = compute_all_data_points(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        reasoning_conditions_filter=args.reasoning_conditions,
        condition_filter=args.condition,
    )

    print(f"\nComputed {len(data_points)} data points\n")

    if not data_points:
        print("No data points found matching the criteria.")
        return

    # Create plot
    if args.bar_chart:
        result = create_grouped_bar_chart(
            data_points=data_points,
            output_path=output_path,
            title=args.title,
            figsize=tuple(args.figsize),
        )
    else:
        result = create_scatter_plot(
            data_points=data_points,
            groups=args.groups,
            output_path=output_path,
            title=args.title,
            figsize=tuple(args.figsize),
            n_bins=args.n_bins,
            show_diagonal=not args.no_diagonal,
        )

    if result is None:
        return

    # Print statistics
    print_statistics(data_points, args.groups)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
