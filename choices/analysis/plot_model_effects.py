#!/usr/bin/env python3
"""
Plot nudge effects for models across factors.

Two modes:
1. Single model mode (--model specified): Shows all factors for one model
2. Multi-model mode (no --model): Shows best factor per model based on --selection

Shows each row with:
- Green diamond marking baseline preference
- Different symbols for each nudge type effect
- Top region: effects when nudging towards A
- Bottom region: effects when nudging towards B

Usage:
    # Single model: basic usage with model and results directories
    uv run python -m choices.analysis.plot_model_effects \
        --model claude-3-5-sonnet-latest \
        --results-dirs results results_anthropic

    # Single model: specify reasoning condition
    uv run python -m choices.analysis.plot_model_effects \
        --model deepseek-v3-2-reasoning \
        --results-dirs results \
        --reasoning low

    # Multi-model: show best factor per model by steerability bias (default)
    uv run python -m choices.analysis.plot_model_effects \
        --results-dirs results \
        --reasoning none

    # Multi-model: select by effect size
    uv run python -m choices.analysis.plot_model_effects \
        --results-dirs results \
        --reasoning none \
        --selection effect

    # Multi-model: select by steerability
    uv run python -m choices.analysis.plot_model_effects \
        --results-dirs results \
        --reasoning none \
        --selection steerability

    # Show significance markers (grey for non-significant)
    uv run python -m choices.analysis.plot_model_effects \
        --model claude-3-5-sonnet-latest \
        --results-dirs results \
        --significance

    # Save to file
    uv run python -m choices.analysis.plot_model_effects \
        --model gpt-4o-mini \
        --results-dirs results \
        --output model_effects.pdf
"""

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import (
    get_model_display_name,
    get_nudge_marker,
    is_reasoning_model,
)


def freq_to_log_odds(freq: float, pseudo_n: float = 100.0) -> float:
    """
    Convert frequency to log odds with Haldane-Anscombe correction.

    Uses pseudo-counts to handle frequencies at or near 0 and 1.
    The correction adds 0.5 to both wins and losses before computing odds.

    Args:
        freq: Frequency (probability) in [0, 1]
        pseudo_n: Pseudo sample size for correction (default 100)

    Returns:
        Log10 odds ratio
    """
    pseudo_wins = freq * pseudo_n
    pseudo_losses = (1 - freq) * pseudo_n
    odds = (pseudo_wins + 0.5) / (pseudo_losses + 0.5)
    return math.log10(odds)


def log_odds_to_freq(log_odds: float) -> float:
    """
    Convert log odds back to frequency.

    Args:
        log_odds: Log10 odds ratio

    Returns:
        Frequency (probability) in [0, 1]
    """
    odds = 10**log_odds
    return odds / (1 + odds)


def geometric_mean_freq(frequencies: List[float]) -> float:
    """
    Compute the geometric mean of frequencies by averaging in log odds space.

    Args:
        frequencies: List of frequencies in [0, 1]

    Returns:
        Geometric mean frequency
    """
    if not frequencies:
        return 0.5
    log_odds_values = [freq_to_log_odds(f) for f in frequencies]
    mean_log_odds = sum(log_odds_values) / len(log_odds_values)
    return log_odds_to_freq(mean_log_odds)


def collect_data_by_factor(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by factor.

    For each factor, collects data points from all nudge types.

    Returns:
        Dictionary mapping factor -> {
            'nudge_data': list of {
                'nudge_type': str,
                'f_A_B': float,  # freq of B when nudged towards A
                'f_B_B': float,  # freq of B when nudged towards B
                'sig_A': bool,   # significance flag for nudge towards A
                'sig_B': bool,   # significance flag for nudge towards B
            },
            'f_0_B': float,      # baseline freq of B (should be same for all nudges)
            'level_A': str,
            'level_B': str,
        }
    """
    data_by_factor: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "nudge_data": [],
            "f_0_B": None,
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        data_by_factor[r.factor]["nudge_data"].append(
            {
                "nudge_type": r.nudge_type,
                "f_A_B": r.f_A_B,
                "f_B_B": r.f_B_B,
                "sig_A": r.sig_A,
                "sig_B": r.sig_B,
            }
        )
        # Store baseline (should be consistent across nudge types for same factor)
        if data_by_factor[r.factor]["f_0_B"] is None:
            data_by_factor[r.factor]["f_0_B"] = r.f_0_B
        # Store level names
        if data_by_factor[r.factor]["level_A"] is None:
            data_by_factor[r.factor]["level_A"] = r.level_A
            data_by_factor[r.factor]["level_B"] = r.level_B

    return dict(data_by_factor)


def select_best_factor_per_model(
    results: List[FrequencyResult],
    selection: str = "steer_bias",
) -> Dict[Tuple[str, str], Tuple[str, float, Optional[float]]]:
    """
    For each (model, reasoning_condition), select the factor with the largest metric.

    Args:
        results: List of FrequencyResult objects
        selection: Metric to use for selection:
            - "steer_bias": abs of average steerability bias (preserves sign before abs)
            - "abs_steer_bias": average of absolute steerability bias values
            - "steerability": average of absolute steerability values
            - "effect": average of absolute effect size values

    Returns:
        Dictionary mapping (model, reasoning_condition) -> (best_factor, abs_value, signed_value)
        signed_value is the original signed average for steer_bias, None for other selections
    """
    # Group results by (model, reasoning_condition, factor)
    by_model_reason_factor: Dict[Tuple[str, str, str], List[FrequencyResult]] = (
        defaultdict(list)
    )
    for r in results:
        by_model_reason_factor[(r.model, r.reasoning_condition, r.factor)].append(r)

    # Compute average metric for each (model, reasoning_condition, factor)
    # Store both absolute value (for selection) and signed value (for display)
    model_reason_factor_metrics: Dict[
        Tuple[str, str, str], Tuple[float, Optional[float]]
    ] = {}
    for (model, reasoning, factor), factor_results in by_model_reason_factor.items():
        if selection == "steer_bias":
            # Average steerability bias first, then take absolute value
            values = [
                r.steerability_bias
                for r in factor_results
                if r.steerability_bias is not None
            ]
            if values:
                avg_bias = sum(values) / len(values)
                model_reason_factor_metrics[(model, reasoning, factor)] = (
                    abs(avg_bias),
                    avg_bias,
                )
            continue
        elif selection == "abs_steer_bias":
            # Average of absolute steerability bias values
            values = [
                abs(r.steerability_bias)
                for r in factor_results
                if r.steerability_bias is not None
            ]
        elif selection == "steerability":
            # Average magnitude of steerability (average of A and B)
            values = [
                abs(r.avg_steerability)
                for r in factor_results
                if r.avg_steerability is not None
            ]
        elif selection == "effect":
            # Average magnitude of effect size
            values = [abs(r.abs_effect) for r in factor_results]
        else:
            raise ValueError(f"Unknown selection metric: {selection}")

        if values:
            avg_val = sum(values) / len(values)
            model_reason_factor_metrics[(model, reasoning, factor)] = (avg_val, None)

    # For each (model, reasoning_condition), find the factor with the largest metric
    best_by_model_reason: Dict[Tuple[str, str], Tuple[str, float, Optional[float]]] = {}
    model_reason_keys = set(
        (model, reasoning) for model, reasoning, _ in model_reason_factor_metrics.keys()
    )

    for model, reasoning in model_reason_keys:
        best_factor = None
        best_abs_value = -1
        best_signed_value = None
        for (m, r, factor), (
            abs_val,
            signed_val,
        ) in model_reason_factor_metrics.items():
            if m == model and r == reasoning and abs_val > best_abs_value:
                best_abs_value = abs_val
                best_signed_value = signed_val
                best_factor = factor
        if best_factor is not None:
            best_by_model_reason[(model, reasoning)] = (
                best_factor,
                best_abs_value,
                best_signed_value,
            )

    return best_by_model_reason


def collect_data_for_multi_model(
    results: List[FrequencyResult],
    model_reason_factor_tuples: List[Tuple[str, str, str]],
) -> Dict[Tuple[str, str, str], Dict[str, any]]:
    """
    Collect frequency data for specific (model, reasoning, factor) tuples.

    Args:
        results: List of FrequencyResult objects
        model_reason_factor_tuples: List of (model, reasoning, factor) tuples

    Returns:
        Dictionary mapping (model, reasoning, factor) -> {
            'nudge_data': list of nudge data dicts,
            'f_0_B': baseline frequency,
            'level_A': str,
            'level_B': str,
        }
    """
    tuples_set = set(model_reason_factor_tuples)
    data: Dict[Tuple[str, str, str], Dict[str, any]] = defaultdict(
        lambda: {
            "nudge_data": [],
            "f_0_B": None,
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        key = (r.model, r.reasoning_condition, r.factor)
        if key not in tuples_set:
            continue

        data[key]["nudge_data"].append(
            {
                "nudge_type": r.nudge_type,
                "f_A_B": r.f_A_B,
                "f_B_B": r.f_B_B,
                "sig_A": r.sig_A,
                "sig_B": r.sig_B,
            }
        )
        if data[key]["f_0_B"] is None:
            data[key]["f_0_B"] = r.f_0_B
        if data[key]["level_A"] is None:
            data[key]["level_A"] = r.level_A
            data[key]["level_B"] = r.level_B

    return dict(data)


def create_multi_model_effects_plot(
    data_by_model_reason_factor: Dict[Tuple[str, str, str], Dict[str, any]],
    reasoning_conditions: List[str],
    selection: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, None),
    show_significance: bool = False,
    show_geom_mean: bool = False,
) -> plt.Figure:
    """
    Create scatter plot showing nudge effects for multiple models.

    Each row shows the best factor for a (model, reasoning_condition) combination.

    Args:
        data_by_model_reason_factor: Dict mapping (model, reasoning, factor) -> data
        reasoning_conditions: List of reasoning conditions included
        selection: Selection metric used
        output_path: Optional path to save the figure
        figsize: Figure size (width, height). Height auto-calculated if None.
        show_significance: If True, show non-significant points in grey
        show_geom_mean: If True, show geometric mean markers

    Returns:
        The matplotlib Figure object
    """
    # Sort by model display name, then reasoning condition
    rows = sorted(
        data_by_model_reason_factor.keys(),
        key=lambda x: (get_model_display_name(x[0]), x[1]),
    )
    n_rows = len(rows)

    if n_rows == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of rows
    height = figsize[1] if figsize[1] else max(4, n_rows * 1.2)

    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # Colors - based on nudge direction
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker
    color_nonsig = "#A0A0A0"  # Grey - non-significant points

    # Y offset for points from center line
    y_offset = 0.08

    # Y positions for each row
    y_positions = np.arange(n_rows)

    # Collect all nudge types for legend
    all_nudge_types = set()
    for row_data in data_by_model_reason_factor.values():
        for nd in row_data["nudge_data"]:
            all_nudge_types.add(nd["nudge_type"])
    all_nudge_types = sorted(all_nudge_types)

    # Process each row
    for i, (model, reasoning, factor) in enumerate(rows):
        row_data = data_by_model_reason_factor[(model, reasoning, factor)]
        y_pos = y_positions[i]
        f_0_B = row_data["f_0_B"]

        # Draw horizontal line for this row
        ax.axhline(
            y=y_pos,
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

        # Draw vertical dashed line at baseline
        ax.plot(
            [f_0_B, f_0_B],
            [y_pos - 0.35, y_pos + 0.35],
            color=color_baseline,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            zorder=8,
        )

        # Draw green diamond at baseline
        ax.scatter(
            [f_0_B],
            [y_pos],
            color=color_baseline,
            marker="D",
            s=120,
            edgecolors="black",
            linewidths=1.5,
            zorder=9,
        )

        # Plot each nudge type with its own symbol
        for nd in row_data["nudge_data"]:
            nudge_type = nd["nudge_type"]
            marker = get_nudge_marker(nudge_type)

            # Nudge towards A (above center line) - RED
            y_A = y_pos - y_offset
            if show_significance and not nd["sig_A"]:
                point_color_A = color_nonsig
            else:
                point_color_A = color_nudge_A

            ax.scatter(
                [nd["f_A_B"]],
                [y_A],
                color=point_color_A,
                marker=marker,
                s=80,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

            # Nudge towards B (below center line) - BLUE
            y_B = y_pos + y_offset
            if show_significance and not nd["sig_B"]:
                point_color_B = color_nonsig
            else:
                point_color_B = color_nudge_B

            ax.scatter(
                [nd["f_B_B"]],
                [y_B],
                color=point_color_B,
                marker=marker,
                s=80,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

        # Compute and draw geometric mean markers
        if show_geom_mean:
            f_A_B_values = [nd["f_A_B"] for nd in row_data["nudge_data"]]
            f_B_B_values = [nd["f_B_B"] for nd in row_data["nudge_data"]]

            bar_height = 0.25

            if f_A_B_values:
                geom_mean_A = geometric_mean_freq(f_A_B_values)
                ax.plot(
                    [geom_mean_A, geom_mean_A],
                    [y_pos - bar_height, y_pos],
                    color=color_nudge_A,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=6,
                )

            if f_B_B_values:
                geom_mean_B = geometric_mean_freq(f_B_B_values)
                ax.plot(
                    [geom_mean_B, geom_mean_B],
                    [y_pos, y_pos + bar_height],
                    color=color_nudge_B,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=6,
                )

    # Add reference line at 0.5 (no preference)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

    # Configure axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    # Title
    selection_labels = {
        "steer_bias": "Steerability Bias",
        "abs_steer_bias": "|Steerability Bias|",
        "steerability": "Steerability",
        "effect": "Effect Size",
    }
    if len(reasoning_conditions) == 1:
        reasoning_str = f"Reasoning: {reasoning_conditions[0]}"
    else:
        reasoning_str = f"Reasoning: {', '.join(reasoning_conditions)}"
    ax.set_title(
        f"Nudge Effects by Model (best factor by {selection_labels.get(selection, selection)})\n"
        f"{reasoning_str}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add labels on left and right sides
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    label_offset = x_range * 0.02

    # Check if we have multiple reasoning conditions to show in labels
    show_reasoning_in_label = len(reasoning_conditions) > 1

    for i, (model, reasoning, factor) in enumerate(rows):
        y_pos = y_positions[i]
        rd = data_by_model_reason_factor[(model, reasoning, factor)]
        level_A = rd.get("level_A") or "A"
        level_B = rd.get("level_B") or "B"
        model_display = get_model_display_name(model)

        # Model name (with reasoning if multiple conditions)
        if show_reasoning_in_label:
            model_label = f"{model_display} ({reasoning})"
        else:
            model_label = model_display

        ax.text(
            x_min - label_offset,
            y_pos - 0.25,
            model_label,
            ha="right",
            va="center",
            fontsize=9,
            fontstyle="italic",
            color="#333333",
            clip_on=False,
        )

        # Left label (level A)
        ax.text(
            x_min - label_offset,
            y_pos + 0.1,
            level_A,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#E63946",
            clip_on=False,
        )

        # Right label (level B)
        ax.text(
            x_max + label_offset,
            y_pos,
            level_B,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#457B9D",
            clip_on=False,
        )

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_baseline,
            markeredgecolor="black",
            markersize=10,
            label="Baseline f₀(B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_nudge_A,
            markeredgecolor="white",
            markersize=8,
            label="Nudge towards A",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_nudge_B,
            markeredgecolor="white",
            markersize=8,
            label="Nudge towards B",
        ),
    ]

    if show_geom_mean:
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=color_nudge_A,
                    linewidth=2,
                    solid_capstyle="round",
                    label="Geom. mean (→A)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color_nudge_B,
                    linewidth=2,
                    solid_capstyle="round",
                    label="Geom. mean (→B)",
                ),
            ]
        )

    for nudge_type in all_nudge_types:
        marker = get_nudge_marker(nudge_type)
        display_name = nudge_type.replace("_", " ").title()
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="#666666",
                markeredgecolor="white",
                markersize=8,
                label=display_name,
            )
        )

    if show_significance:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_nonsig,
                markersize=8,
                label="Non-significant",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        framealpha=0.9,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Invert y-axis so first row is at top
    ax.invert_yaxis()

    # Adjust margins to make room for legend on right
    plt.subplots_adjust(left=0.15, right=0.75)
    plt.tight_layout(rect=[0.10, 0.02, 0.78, 0.98])

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def create_model_effects_plot(
    data_by_factor: Dict[str, Dict[str, any]],
    model_name: str,
    reasoning_condition: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, None),
    show_significance: bool = False,
    show_geom_mean: bool = False,
) -> plt.Figure:
    """
    Create scatter plot showing nudge effects for a single model.

    For each factor (row):
    - Green diamond marks baseline preference
    - Each nudge type is shown with a different symbol
    - Points above row center: effects when nudging towards A
    - Points below row center: effects when nudging towards B

    Args:
        data_by_factor: Dictionary mapping factor -> data
        model_name: Display name for the model
        reasoning_condition: Reasoning condition string
        output_path: Optional path to save the figure
        figsize: Figure size (width, height). Height auto-calculated if None.
        show_significance: If True, show non-significant points in grey
        show_geom_mean: If True, show geometric mean markers

    Returns:
        The matplotlib Figure object
    """
    factors = sorted(data_by_factor.keys())
    n_factors = len(factors)

    if n_factors == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of factors
    height = figsize[1] if figsize[1] else max(4, n_factors * 1.2)

    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # Colors - based on nudge direction
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker
    color_nonsig = "#A0A0A0"  # Grey - non-significant points

    # Y offset for points from center line (smaller = closer to line)
    y_offset = 0.08

    # Y positions for each factor
    y_positions = np.arange(n_factors)

    # Collect all nudge types for legend
    all_nudge_types = set()
    for factor_data in data_by_factor.values():
        for nd in factor_data["nudge_data"]:
            all_nudge_types.add(nd["nudge_type"])
    all_nudge_types = sorted(all_nudge_types)

    # Process each factor
    for i, factor in enumerate(factors):
        factor_data = data_by_factor[factor]
        y_pos = y_positions[i]
        f_0_B = factor_data["f_0_B"]

        # Draw baseline marker (green diamond with dashed line)
        ax.axhline(
            y=y_pos,
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

        # Draw vertical dashed line at baseline
        ax.plot(
            [f_0_B, f_0_B],
            [y_pos - 0.35, y_pos + 0.35],
            color=color_baseline,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            zorder=8,
        )

        # Draw green diamond at baseline
        ax.scatter(
            [f_0_B],
            [y_pos],
            color=color_baseline,
            marker="D",
            s=120,
            edgecolors="black",
            linewidths=1.5,
            zorder=9,
            label="Baseline" if i == 0 else None,
        )

        # Plot each nudge type with its own symbol
        for nd in factor_data["nudge_data"]:
            nudge_type = nd["nudge_type"]
            marker = get_nudge_marker(nudge_type)

            # Nudge towards A (above center line, visually on top) - RED
            # Note: y-axis is inverted, so subtract offset to go up
            y_A = y_pos - y_offset
            if show_significance and not nd["sig_A"]:
                point_color_A = color_nonsig
            else:
                point_color_A = color_nudge_A

            ax.scatter(
                [nd["f_A_B"]],
                [y_A],
                color=point_color_A,
                marker=marker,
                s=80,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

            # Nudge towards B (below center line, visually on bottom) - BLUE
            # Note: y-axis is inverted, so add offset to go down
            y_B = y_pos + y_offset
            if show_significance and not nd["sig_B"]:
                point_color_B = color_nonsig
            else:
                point_color_B = color_nudge_B

            ax.scatter(
                [nd["f_B_B"]],
                [y_B],
                color=point_color_B,
                marker=marker,
                s=80,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
            )

        # Compute and draw geometric mean markers (computed in log odds space)
        if show_geom_mean:
            f_A_B_values = [nd["f_A_B"] for nd in factor_data["nudge_data"]]
            f_B_B_values = [nd["f_B_B"] for nd in factor_data["nudge_data"]]

            bar_height = 0.25  # Height of the vertical bar

            if f_A_B_values:
                geom_mean_A = geometric_mean_freq(f_A_B_values)
                # Draw vertical bar for geometric mean of nudge towards A
                # Positioned above the center line (y-axis is inverted, so subtract)
                ax.plot(
                    [geom_mean_A, geom_mean_A],
                    [y_pos - bar_height, y_pos],
                    color=color_nudge_A,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=6,
                )

            if f_B_B_values:
                geom_mean_B = geometric_mean_freq(f_B_B_values)
                # Draw vertical bar for geometric mean of nudge towards B
                # Positioned below the center line (y-axis is inverted, so add)
                ax.plot(
                    [geom_mean_B, geom_mean_B],
                    [y_pos, y_pos + bar_height],
                    color=color_nudge_B,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=6,
                )

    # Add reference line at 0.5 (no preference)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

    # Configure axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n_factors)
    ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    # Title
    ax.set_title(
        f"Nudge Effects: {model_name} ({reasoning_condition})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add factor labels with level names on left and right sides
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    label_offset = x_range * 0.02

    for i, factor in enumerate(factors):
        y_pos = y_positions[i]
        fd = data_by_factor[factor]
        level_A = fd.get("level_A") or "A"
        level_B = fd.get("level_B") or "B"

        # Left label (level A)
        ax.text(
            x_min - label_offset,
            y_pos,
            level_A,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#E63946",
            clip_on=False,
        )

        # Right label (level B)
        ax.text(
            x_max + label_offset,
            y_pos,
            level_B,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#457B9D",
            clip_on=False,
        )

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_baseline,
            markeredgecolor="black",
            markersize=10,
            label="Baseline f₀(B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_nudge_A,
            markeredgecolor="white",
            markersize=8,
            label="Nudge towards A",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_nudge_B,
            markeredgecolor="white",
            markersize=8,
            label="Nudge towards B",
        ),
    ]

    # Add geometric mean legend entries if enabled
    if show_geom_mean:
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=color_nudge_A,
                    linewidth=2,
                    solid_capstyle="round",
                    label="Geom. mean (→A)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=color_nudge_B,
                    linewidth=2,
                    solid_capstyle="round",
                    label="Geom. mean (→B)",
                ),
            ]
        )

    # Add nudge type symbols (shown in grey/neutral to indicate shape only)
    for nudge_type in all_nudge_types:
        marker = get_nudge_marker(nudge_type)
        display_name = nudge_type.replace("_", " ").title()
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="#666666",
                markeredgecolor="white",
                markersize=8,
                label=display_name,
            )
        )

    if show_significance:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_nonsig,
                markersize=8,
                label="Non-significant",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        framealpha=0.9,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Invert y-axis so first factor is at top
    ax.invert_yaxis()

    # Adjust margins to make room for legend on right
    plt.subplots_adjust(left=0.12, right=0.75)
    plt.tight_layout(rect=[0.08, 0.02, 0.78, 0.98])

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def get_default_reasoning_condition(model: str) -> str:
    """
    Get the default reasoning condition for a model.

    Returns 'off' for reasoning models, 'none' for chat models.
    """
    if is_reasoning_model(model):
        return "off"
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="Plot nudge effects for models across factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model mode
    uv run python -m choices.analysis.plot_model_effects \\
        --model claude-3-5-sonnet-latest \\
        --results-dirs results

    # Multi-model mode (best factor per model by steerability bias)
    uv run python -m choices.analysis.plot_model_effects \\
        --results-dirs results \\
        --reasoning none

    # Multi-model mode with different selection
    uv run python -m choices.analysis.plot_model_effects \\
        --results-dirs results \\
        --reasoning none \\
        --selection effect
        """,
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (if not specified, shows best factor per model)",
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to search",
    )

    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=None,
        help="Reasoning condition(s) to filter by (optional, can specify multiple for multi-model mode)",
    )

    parser.add_argument(
        "--selection",
        type=str,
        choices=["steer_bias", "abs_steer_bias", "steerability", "effect"],
        default="steer_bias",
        help="Selection metric for multi-model mode: steer_bias (default), abs_steer_bias, steerability, effect",
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="List of factors to include (default: all discovered)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, None],
        help="Figure size (width height). Height auto-calculated if not provided.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )

    parser.add_argument(
        "--significance",
        action="store_true",
        help="Show non-significant data points in grey",
    )

    parser.add_argument(
        "--geom-mean",
        action="store_true",
        help="Show geometric mean of nudge effects as vertical bars",
    )

    args = parser.parse_args()

    # Determine mode
    single_model_mode = args.model is not None

    # Determine reasoning condition(s)
    reasoning_conditions = args.reasoning
    if reasoning_conditions is None:
        if single_model_mode:
            reasoning_conditions = [get_default_reasoning_condition(args.model)]
        # For multi-model mode, None means "all" - will be handled later

    # For single model mode, only use first reasoning condition
    if single_model_mode and reasoning_conditions and len(reasoning_conditions) > 1:
        print("Warning: Multiple reasoning conditions given for single model mode.")
        print(f"Using only the first: {reasoning_conditions[0]}")
        reasoning_conditions = [reasoning_conditions[0]]

    # Determine output path
    if args.output:
        output_path = args.output
    elif single_model_mode:
        safe_model = args.model.replace("/", "_").replace(":", "_")
        output_path = f"model_effects_{safe_model}.pdf"
    else:
        output_path = f"model_effects_multi_{args.selection}.pdf"

    # Print header
    print("=" * 70)
    print("Model Effects Plot")
    print("=" * 70)
    if single_model_mode:
        print("Mode: Single model")
        print(f"Model: {args.model}")
    else:
        print("Mode: Multi-model (best factor per model)")
        print(f"Selection: {args.selection}")
    if reasoning_conditions:
        print(f"Reasoning condition(s): {', '.join(reasoning_conditions)}")
    else:
        print("Reasoning condition(s): all")
    print(f"Results directories: {args.results_dirs}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    print(f"Significance: {'Yes' if args.significance else 'No'}")
    print(f"Geom. mean: {'Yes' if args.geom_mean else 'No'}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Compute frequency results
    print("Computing frequency results...")
    if single_model_mode:
        results = compute_all_results(
            args.results_dirs,
            model_filter=[args.model],
            factor_filter=args.factors,
        )
    else:
        results = compute_all_results(
            args.results_dirs,
            factor_filter=args.factors,
        )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s)")

    # Filter by reasoning condition(s) if specified
    if reasoning_conditions:
        results = [r for r in results if r.reasoning_condition in reasoning_conditions]
        print(
            f"After reasoning filter ({', '.join(reasoning_conditions)}): {len(results)} result(s)"
        )
    else:
        # For multi-model mode without reasoning filter, get all unique conditions
        reasoning_conditions = sorted(set(r.reasoning_condition for r in results))
        print(f"Using all reasoning conditions: {', '.join(reasoning_conditions)}")

    if not results:
        print("No results found after filtering.")
        return

    print()

    figsize = (args.figsize[0], args.figsize[1] if len(args.figsize) > 1 else None)

    if single_model_mode:
        # Single model mode: show all factors for one model
        data_by_factor = collect_data_by_factor(results)

        print("Data Summary by Factor:")
        print("-" * 70)
        for factor in sorted(data_by_factor.keys()):
            fd = data_by_factor[factor]
            n_nudges = len(fd["nudge_data"])
            nudge_types = [nd["nudge_type"] for nd in fd["nudge_data"]]
            print(
                f"  {factor:<15} baseline={fd['f_0_B']:.3f}  "
                f"nudge types ({n_nudges}): {', '.join(nudge_types)}"
            )
        print()

        model_display = get_model_display_name(args.model)

        fig = create_model_effects_plot(
            data_by_factor,
            model_name=model_display,
            reasoning_condition=reasoning_conditions[0],
            output_path=output_path,
            figsize=figsize,
            show_significance=args.significance,
            show_geom_mean=args.geom_mean,
        )
    else:
        # Multi-model mode: show best factor per (model, reasoning) combination
        best_by_model_reason = select_best_factor_per_model(results, args.selection)

        print(f"Best factor per (model, reasoning) by {args.selection}:")
        print("-" * 70)
        for model, reasoning in sorted(
            best_by_model_reason.keys(),
            key=lambda x: (get_model_display_name(x[0]), x[1]),
        ):
            factor, abs_value, signed_value = best_by_model_reason[(model, reasoning)]
            if len(reasoning_conditions) > 1:
                label = f"{get_model_display_name(model)} ({reasoning})"
            else:
                label = get_model_display_name(model)
            # Show signed value if available, otherwise show absolute value
            if signed_value is not None:
                print(f"  {label:<40} {factor:<15} ({signed_value:+.3f})")
            else:
                print(f"  {label:<40} {factor:<15} ({abs_value:.3f})")
        print()

        # Collect data for the selected (model, reasoning, factor) tuples
        model_reason_factor_tuples = [
            (model, reasoning, factor)
            for (model, reasoning), (factor, _, _) in best_by_model_reason.items()
        ]
        data_by_model_reason_factor = collect_data_for_multi_model(
            results, model_reason_factor_tuples
        )

        fig = create_multi_model_effects_plot(
            data_by_model_reason_factor,
            reasoning_conditions=reasoning_conditions,
            selection=args.selection,
            output_path=output_path,
            figsize=figsize,
            show_significance=args.significance,
            show_geom_mean=args.geom_mean,
        )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
