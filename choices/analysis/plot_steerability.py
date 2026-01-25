#!/usr/bin/env python3
"""
Plot steerability distributions with violin plots for each factor.

For each factor, shows two horizontal violin plots:
- Left: distribution of f(B) across all models/nudge types when nudging towards A
- Right: distribution of f(B) across all models/nudge types when nudging towards B

The average baseline preference f_0(B) is visually indicated on both plots.

Usage:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --factors gender age_group wealth

    # Save to file
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --output steerability_violins.pdf

    # Show in log odds space
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --log-odds

    # Show median and IQR instead of mean
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --percentiles

    # Show effects relative to baseline
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --relative

    # Combine: relative effects in log odds space
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --log-odds --relative

    # Show one row per model (forces log odds space)
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --rows models

    # Show one row per nudge type (forces log odds space)
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --rows nudges
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
    discover_experiments,
)


def freq_to_log_odds(
    freq: float,
    pseudo_n: float = 100.0,
) -> float:
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
    # Convert frequency to pseudo-counts
    pseudo_wins = freq * pseudo_n
    pseudo_losses = (1 - freq) * pseudo_n

    # Apply Haldane-Anscombe correction
    odds = (pseudo_wins + 0.5) / (pseudo_losses + 0.5)

    return math.log10(odds)


def transform_data_to_log_odds(
    data_by_factor: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Transform all frequency data to log odds space.

    Args:
        data_by_factor: Dictionary mapping factor -> frequency data

    Returns:
        Same structure with frequencies converted to log odds
    """
    transformed = {}
    for factor, factor_data in data_by_factor.items():
        transformed[factor] = {
            "f_A_B": [freq_to_log_odds(f) for f in factor_data["f_A_B"]],
            "f_B_B": [freq_to_log_odds(f) for f in factor_data["f_B_B"]],
            "f_0_B": [freq_to_log_odds(f) for f in factor_data["f_0_B"]],
            "level_A": factor_data.get("level_A"),
            "level_B": factor_data.get("level_B"),
        }
    return transformed


def transform_data_to_relative(
    data_by_factor: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Transform data to be relative to baseline (subtract baseline from each nudge condition).

    For each experiment, computes:
    - f_A_B - f_0_B (effect of nudging towards A)
    - f_B_B - f_0_B (effect of nudging towards B)
    - f_0_B remains as 0 (reference point)

    This works in both frequency space (giving frequency differences) and
    log odds space (giving log odds ratios).

    Args:
        data_by_factor: Dictionary mapping factor -> data (frequency or log odds)

    Returns:
        Same structure with values relative to baseline
    """
    transformed = {}
    for factor, factor_data in data_by_factor.items():
        # Compute relative values (subtract corresponding baseline)
        relative_f_A_B = [
            f_A - f_0 for f_A, f_0 in zip(factor_data["f_A_B"], factor_data["f_0_B"])
        ]
        relative_f_B_B = [
            f_B - f_0 for f_B, f_0 in zip(factor_data["f_B_B"], factor_data["f_0_B"])
        ]
        # Baseline becomes zero (reference point)
        relative_f_0_B = [0.0] * len(factor_data["f_0_B"])

        transformed[factor] = {
            "f_A_B": relative_f_A_B,
            "f_B_B": relative_f_B_B,
            "f_0_B": relative_f_0_B,
            "level_A": factor_data.get("level_A"),
            "level_B": factor_data.get("level_B"),
        }
    return transformed


def collect_data_by_factor(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by factor.

    Returns:
        Dictionary mapping factor -> {
            'f_A_B': list of f_A(B) values (freq of B when nudged towards A),
            'f_B_B': list of f_B(B) values (freq of B when nudged towards B),
            'f_0_B': list of f_0(B) values (baseline freq of B),
            'level_A': name of level A (e.g., 'poor'),
            'level_B': name of level B (e.g., 'rich'),
        }
    """
    data_by_factor: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        data_by_factor[r.factor]["f_A_B"].append(r.f_A_B)
        data_by_factor[r.factor]["f_B_B"].append(r.f_B_B)
        data_by_factor[r.factor]["f_0_B"].append(r.f_0_B)
        # Store level names (they should be consistent within a factor)
        if data_by_factor[r.factor]["level_A"] is None:
            data_by_factor[r.factor]["level_A"] = r.level_A
            data_by_factor[r.factor]["level_B"] = r.level_B

    return dict(data_by_factor)


def collect_data_by_model(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by model.

    Returns:
        Dictionary mapping model -> {
            'f_A_B': list of f_A(B) values across all factors/nudge types,
            'f_B_B': list of f_B(B) values across all factors/nudge types,
            'f_0_B': list of f_0(B) values (baseline),
            'level_A': None (not applicable for model grouping),
            'level_B': None (not applicable for model grouping),
        }
    """
    from choices.analysis.utils import get_model_display_name

    data_by_model: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        # Use display name for cleaner labels
        model_key = get_model_display_name(r.model)
        data_by_model[model_key]["f_A_B"].append(r.f_A_B)
        data_by_model[model_key]["f_B_B"].append(r.f_B_B)
        data_by_model[model_key]["f_0_B"].append(r.f_0_B)

    return dict(data_by_model)


def collect_data_by_nudge_type(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by nudge type.

    Returns:
        Dictionary mapping nudge_type -> {
            'f_A_B': list of f_A(B) values across all factors/models,
            'f_B_B': list of f_B(B) values across all factors/models,
            'f_0_B': list of f_0(B) values (baseline),
            'level_A': None (not applicable for nudge type grouping),
            'level_B': None (not applicable for nudge type grouping),
        }
    """
    data_by_nudge: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        # Format nudge type for display
        nudge_key = r.nudge_type.replace("_", " ").title()
        data_by_nudge[nudge_key]["f_A_B"].append(r.f_A_B)
        data_by_nudge[nudge_key]["f_B_B"].append(r.f_B_B)
        data_by_nudge[nudge_key]["f_0_B"].append(r.f_0_B)

    return dict(data_by_nudge)


def format_factor_label(factor: str, level_A: str, level_B: str) -> str:
    """Format factor name with level labels."""
    factor_display = factor.replace("_", " ").title()
    return f"{factor_display}\n({level_A} vs {level_B})"


def _split_violin_halves(parts, side: str, y_position: float):
    """
    Modify violin plot to show only one half (left or right).

    Args:
        parts: The violinplot collection
        side: "left" or "right"
        y_position: The y position of the violin
    """
    for vp in parts["bodies"]:
        # Get the path vertices
        paths = vp.get_paths()
        if not paths:
            continue
        path = paths[0]
        vertices = path.vertices.copy()

        # For horizontal violins, we clip on y-axis
        # side="left" means keep points below y_position
        # side="right" means keep points above y_position
        if side == "left":
            # Keep only the lower half (below center line)
            vertices[vertices[:, 1] > y_position, 1] = y_position
        else:
            # Keep only the upper half (above center line)
            vertices[vertices[:, 1] < y_position, 1] = y_position

        path.vertices = vertices


def create_steerability_violin_plot(
    data_by_row: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, None),
    log_odds: bool = False,
    percentiles: bool = False,
    relative: bool = False,
    row_type: str = "factors",
) -> plt.Figure:
    """
    Create violin plot showing steerability distributions.

    For each row, shows a split violin:
    - Left half (below row center): distribution when nudged towards A
    - Right half (above row center): distribution when nudged towards B

    Args:
        data_by_row: Dictionary mapping row key -> frequency data
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size (width, height). If height is None, auto-calculated.
        log_odds: If True, data is in log odds space
        percentiles: If True, show median and 25/75 percentiles instead of mean
        relative: If True, values are relative to baseline
        row_type: Type of rows - "factors", "models", or "nudges"

    Returns:
        The matplotlib Figure object
    """
    rows = sorted(data_by_row.keys())
    n_rows = len(rows)

    if n_rows == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of rows
    height = figsize[1] if figsize[1] else max(4, n_rows * 1.5)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # Colors
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker

    # Y positions for each row
    y_positions = np.arange(n_rows)

    # Process each row separately to create split violins
    for i, row_key in enumerate(rows):
        row_data = data_by_row[row_key]
        y_pos = y_positions[i]

        # Create violin for nudge towards A (left/lower half)
        if len(row_data["f_A_B"]) >= 2:
            parts_A = ax.violinplot(
                [row_data["f_A_B"]],
                positions=[y_pos],
                vert=False,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )

            # Style and clip to left half
            for pc in parts_A["bodies"]:
                pc.set_facecolor(color_nudge_A)
                pc.set_edgecolor(color_nudge_A)
                pc.set_alpha(0.3)  # Lighter background

            _split_violin_halves(parts_A, "left", y_pos)

        # Create violin for nudge towards B (right/upper half)
        if len(row_data["f_B_B"]) >= 2:
            parts_B = ax.violinplot(
                [row_data["f_B_B"]],
                positions=[y_pos],
                vert=False,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )

            # Style and clip to right half
            for pc in parts_B["bodies"]:
                pc.set_facecolor(color_nudge_B)
                pc.set_edgecolor(color_nudge_B)
                pc.set_alpha(0.3)  # Lighter background

            _split_violin_halves(parts_B, "right", y_pos)

        # Add individual data points as dots
        # Scatter points for nudge A (below center line)
        n_A = len(row_data["f_A_B"])
        jitter_A = np.random.uniform(-0.25, -0.05, n_A)
        ax.scatter(
            row_data["f_A_B"],
            y_pos + jitter_A,
            color=color_nudge_A,
            alpha=0.7,
            s=25,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Scatter points for nudge B (above center line)
        n_B = len(row_data["f_B_B"])
        jitter_B = np.random.uniform(0.05, 0.25, n_B)
        ax.scatter(
            row_data["f_B_B"],
            y_pos + jitter_B,
            color=color_nudge_B,
            alpha=0.7,
            s=25,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Add central tendency markers for nudge conditions
        if percentiles:
            # Use median and show 25/75 percentiles
            center_nudge_A = np.median(row_data["f_A_B"])
            center_nudge_B = np.median(row_data["f_B_B"])
            p25_nudge_A = np.percentile(row_data["f_A_B"], 25)
            p75_nudge_A = np.percentile(row_data["f_A_B"], 75)
            p25_nudge_B = np.percentile(row_data["f_B_B"], 25)
            p75_nudge_B = np.percentile(row_data["f_B_B"], 75)
        else:
            # Use mean
            center_nudge_A = np.mean(row_data["f_A_B"])
            center_nudge_B = np.mean(row_data["f_B_B"])

        # Central marker for nudge A (in lower half)
        # Add black outline for visibility (drawn first, behind)
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color="black",
            marker="|",
            s=550,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color=color_nudge_A,
            marker="|",
            s=500,
            linewidths=2.5,
            zorder=7,
        )

        # Central marker for nudge B (in upper half)
        # Add black outline for visibility (drawn first, behind)
        ax.scatter(
            [center_nudge_B],
            [y_pos + 0.15],
            color="black",
            marker="|",
            s=550,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_B],
            [y_pos + 0.15],
            color=color_nudge_B,
            marker="|",
            s=500,
            linewidths=2.5,
            zorder=7,
        )

        # Add percentile markers if enabled
        if percentiles:
            # 25th and 75th percentile markers for nudge A - more visible
            ax.scatter(
                [p25_nudge_A, p75_nudge_A],
                [y_pos - 0.15, y_pos - 0.15],
                color=color_nudge_A,
                marker="|",
                s=250,
                linewidths=2.5,
                zorder=6,
            )

            # 25th and 75th percentile markers for nudge B - more visible
            ax.scatter(
                [p25_nudge_B, p75_nudge_B],
                [y_pos + 0.15, y_pos + 0.15],
                color=color_nudge_B,
                marker="|",
                s=250,
                linewidths=2.5,
                zorder=6,
            )

            # Connect percentiles with a horizontal line (IQR) - thicker
            ax.plot(
                [p25_nudge_A, p75_nudge_A],
                [y_pos - 0.15, y_pos - 0.15],
                color=color_nudge_A,
                linewidth=2.5,
                alpha=0.8,
                zorder=5,
            )
            ax.plot(
                [p25_nudge_B, p75_nudge_B],
                [y_pos + 0.15, y_pos + 0.15],
                color=color_nudge_B,
                linewidth=2.5,
                alpha=0.8,
                zorder=5,
            )

        # Add baseline marker (median or mean f_0(B)) - skip in relative mode (always 0)
        if not relative:
            if percentiles:
                center_baseline = np.median(row_data["f_0_B"])
            else:
                center_baseline = np.mean(row_data["f_0_B"])

            # Draw vertical line at baseline spanning the row (high zorder to be visible)
            ax.plot(
                [center_baseline, center_baseline],
                [y_pos - 0.35, y_pos + 0.35],
                color=color_baseline,
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=8,
            )

            # Add diamond marker at center (highest zorder to always be on top)
            ax.scatter(
                [center_baseline],
                [y_pos],
                color=color_baseline,
                marker="D",
                s=100,
                edgecolors="black",
                linewidths=1.5,
                zorder=9,
            )

    # Add reference line at appropriate value
    # - relative mode: 0 (no effect)
    # - log odds mode: 0 (equal odds)
    # - frequency mode: 0.5 (no preference)
    if relative:
        ref_value = 0.0
    elif log_odds:
        ref_value = 0.0
    else:
        ref_value = 0.5
    ax.axvline(
        x=ref_value, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1
    )

    # Remove y-axis tick labels - options will be shown on sides instead
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(rows))

    # Build x-axis label based on mode
    if relative and log_odds:
        ax.set_xlabel("Δ Log₁₀ Odds (relative to baseline)", fontsize=12)
    elif relative:
        ax.set_xlabel("Δ Frequency (relative to baseline)", fontsize=12)
    elif log_odds:
        ax.set_xlabel("Log₁₀ Odds of Choosing B", fontsize=12)
    else:
        ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        # Build space label
        if relative and log_odds:
            space_label = "(Relative Log Odds)"
        elif relative:
            space_label = "(Relative Frequency)"
        elif log_odds:
            space_label = "(Log Odds Space)"
        else:
            space_label = "(Frequency Space)"

        # Build row type label
        if row_type == "factors":
            row_label = "Factor"
            dist_label = "models and nudge types"
        elif row_type == "models":
            row_label = "Model"
            dist_label = "factors and nudge types"
        else:  # nudges
            row_label = "Nudge Type"
            dist_label = "factors and models"

        ax.set_title(
            f"Steerability by {row_label} {space_label}\n(Distribution across {dist_label})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Set x-axis limits
    if log_odds or relative:
        # Auto-scale for log odds or relative mode, with some padding
        all_values = []
        for rd in data_by_row.values():
            all_values.extend(rd["f_A_B"])
            all_values.extend(rd["f_B_B"])
            all_values.extend(rd["f_0_B"])
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
    else:
        ax.set_xlim(-0.05, 1.05)

    # Add option labels on left and right sides of each row (only for factors)
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    label_offset = x_range * 0.02  # Small offset from plot edge

    if row_type == "factors":
        for i, row_key in enumerate(rows):
            y_pos = y_positions[i]
            rd = data_by_row[row_key]
            # Get level names from data (extracted from FrequencyResult)
            level_A = rd.get("level_A") or "A"
            level_B = rd.get("level_B") or "B"

            # Left label (level A - towards which nudging decreases f(B))
            ax.text(
                x_min - label_offset,
                y_pos,
                level_A,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#E63946",  # Same as nudge A color
                clip_on=False,  # Allow text outside plot area
            )

            # Right label (level B - towards which nudging increases f(B))
            ax.text(
                x_max + label_offset,
                y_pos,
                level_B,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#457B9D",  # Same as nudge B color
                clip_on=False,  # Allow text outside plot area
            )
    else:
        # For models/nudges, show row labels on the left side
        for i, row_key in enumerate(rows):
            y_pos = y_positions[i]
            ax.text(
                x_min - label_offset,
                y_pos,
                row_key,
                ha="right",
                va="center",
                fontsize=10,
                clip_on=False,
            )

    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    central_label = "Median" if percentiles else "Mean"
    legend_elements = [
        Patch(facecolor=color_nudge_A, alpha=0.3, label="Nudged towards A"),
        Patch(facecolor=color_nudge_B, alpha=0.3, label="Nudged towards B"),
        Line2D(
            [0],
            [0],
            marker="|",
            color=color_nudge_A,
            markersize=14,
            linewidth=0,
            markeredgewidth=2.5,
            label=f"{central_label} (nudged)",
        ),
    ]

    # Only show baseline in legend when not in relative mode
    if not relative:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=color_baseline,
                markeredgecolor="black",
                markersize=10,
                label=f"{central_label} Baseline f₀(B)",
            )
        )

    if percentiles:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=2.5,
                alpha=0.8,
                label="IQR (25-75%)",
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add light horizontal grid lines to separate rows
    for i in range(n_rows - 1):
        ax.axhline(
            y=i + 0.5,
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

    # Invert y-axis so first row is at top
    ax.invert_yaxis()

    # Adjust margins to make room for option labels on left and right
    plt.subplots_adjust(left=0.12, right=0.88)
    plt.tight_layout(rect=[0.08, 0, 0.92, 1])

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create violin plots showing steerability distributions by factor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability \\
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability \\
        --results-dirs results \\
        --factors gender age_group wealth
        """,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to search",
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="List of factors to include (default: all discovered)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to include (default: all discovered)",
    )

    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="List of nudge types to include (default: all discovered)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: steerability_violins.pdf)",
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
        default=[12, None],
        help="Figure size (width height). Height auto-calculated if not provided.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )

    parser.add_argument(
        "--log-odds",
        action="store_true",
        help="Show plot in log odds space instead of frequency space",
    )

    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="Show median and IQR (25-75%%) instead of mean",
    )

    parser.add_argument(
        "--relative",
        action="store_true",
        help="Compute effects relative to baseline (subtract baseline from each condition)",
    )

    parser.add_argument(
        "--rows",
        type=str,
        choices=["factors", "models", "nudges"],
        default="factors",
        help="What to show as rows: factors (default), models, or nudges. "
        "Log odds space is forced for models/nudges.",
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output if args.output else "steerability_violins.pdf"

    # Force log odds and relative mode for non-factor rows
    use_log_odds = args.log_odds
    use_relative = args.relative
    if args.rows in ("models", "nudges"):
        if not use_log_odds:
            print(f"Note: Forcing log odds space for --rows={args.rows}")
            use_log_odds = True
        if not use_relative:
            print(f"Note: Forcing relative mode for --rows={args.rows}")
            use_relative = True

    # Print header
    print("=" * 70)
    print("Steerability Violin Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    print(f"Rows: {args.rows}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    print(f"Space: {'Log Odds' if use_log_odds else 'Frequency'}")
    print(f"Relative: {'Yes' if use_relative else 'No'}")
    print(f"Statistics: {'Median + IQR' if args.percentiles else 'Mean'}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Discover and compute results
    print("Discovering experiments...")
    experiments = discover_experiments(
        args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if not experiments:
        print("No experiments found matching the filters.")
        return

    print(f"Found {len(experiments)} experiment(s)")
    print()

    # Compute frequency results
    print("Computing frequency results...")
    results = compute_all_results(
        args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if not results:
        print("No results found.")
        return

    print(f"Computed {len(results)} result(s)")
    print()

    # Collect data based on row type
    if args.rows == "factors":
        data_by_row = collect_data_by_factor(results)
        row_type_label = "Factor"
    elif args.rows == "models":
        data_by_row = collect_data_by_model(results)
        row_type_label = "Model"
    else:  # nudges
        data_by_row = collect_data_by_nudge_type(results)
        row_type_label = "Nudge Type"

    # Print summary (always in frequency space for clarity)
    print(f"Data Summary by {row_type_label} (Frequency Space):")
    print("-" * 70)
    for row_key in sorted(data_by_row.keys()):
        rd = data_by_row[row_key]
        n_experiments = len(rd["f_0_B"])
        avg_baseline = np.mean(rd["f_0_B"])
        avg_nudge_A = np.mean(rd["f_A_B"])
        avg_nudge_B = np.mean(rd["f_B_B"])
        print(
            f"  {row_key:<25} n={n_experiments:>3}  "
            f"Baseline={avg_baseline:.3f}  "
            f"Nudge→A={avg_nudge_A:.3f}  "
            f"Nudge→B={avg_nudge_B:.3f}"
        )
    print()

    # Transform to log odds if needed (do this before relative transform)
    if use_log_odds:
        print("Transforming data to log odds space...")
        data_by_row = transform_data_to_log_odds(data_by_row)
        print()

    # Transform to relative values if requested (after log odds transform)
    if use_relative:
        print("Computing effects relative to baseline...")
        data_by_row = transform_data_to_relative(data_by_row)
        print()

    # Create plot
    figsize = (args.figsize[0], args.figsize[1] if args.figsize[1] else None)
    fig = create_steerability_violin_plot(
        data_by_row,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        log_odds=use_log_odds,
        percentiles=args.percentiles,
        relative=use_relative,
        row_type=args.rows,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
