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
from choices.analysis.utils import FACTOR_LEVELS


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
        }
    return transformed


def collect_data_by_factor(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect frequency data grouped by factor.

    Returns:
        Dictionary mapping factor -> {
            'f_A_B': list of f_A(B) values (freq of B when nudged towards A),
            'f_B_B': list of f_B(B) values (freq of B when nudged towards B),
            'f_0_B': list of f_0(B) values (baseline freq of B),
        }
    """
    data_by_factor: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"f_A_B": [], "f_B_B": [], "f_0_B": []}
    )

    for r in results:
        data_by_factor[r.factor]["f_A_B"].append(r.f_A_B)
        data_by_factor[r.factor]["f_B_B"].append(r.f_B_B)
        data_by_factor[r.factor]["f_0_B"].append(r.f_0_B)

    return dict(data_by_factor)


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
    data_by_factor: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, None),
    log_odds: bool = False,
    percentiles: bool = False,
) -> plt.Figure:
    """
    Create violin plot showing steerability distributions for each factor.

    For each factor, shows a split violin:
    - Left half (below row center): distribution when nudged towards A
    - Right half (above row center): distribution when nudged towards B

    Args:
        data_by_factor: Dictionary mapping factor -> frequency data
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size (width, height). If height is None, auto-calculated.
        log_odds: If True, data is in log odds space
        percentiles: If True, show median and 25/75 percentiles instead of mean

    Returns:
        The matplotlib Figure object
    """
    factors = sorted(data_by_factor.keys())
    n_factors = len(factors)

    if n_factors == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of factors
    height = figsize[1] if figsize[1] else max(4, n_factors * 1.5)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # Colors
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker

    # Y positions for each factor
    y_positions = np.arange(n_factors)

    # Process each factor separately to create split violins
    for i, factor in enumerate(factors):
        factor_data = data_by_factor[factor]
        y_pos = y_positions[i]

        # Create violin for nudge towards A (left/lower half)
        if len(factor_data["f_A_B"]) >= 2:
            parts_A = ax.violinplot(
                [factor_data["f_A_B"]],
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
        if len(factor_data["f_B_B"]) >= 2:
            parts_B = ax.violinplot(
                [factor_data["f_B_B"]],
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
        n_A = len(factor_data["f_A_B"])
        jitter_A = np.random.uniform(-0.25, -0.05, n_A)
        ax.scatter(
            factor_data["f_A_B"],
            y_pos + jitter_A,
            color=color_nudge_A,
            alpha=0.7,
            s=25,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Scatter points for nudge B (above center line)
        n_B = len(factor_data["f_B_B"])
        jitter_B = np.random.uniform(0.05, 0.25, n_B)
        ax.scatter(
            factor_data["f_B_B"],
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
            center_nudge_A = np.median(factor_data["f_A_B"])
            center_nudge_B = np.median(factor_data["f_B_B"])
            p25_nudge_A = np.percentile(factor_data["f_A_B"], 25)
            p75_nudge_A = np.percentile(factor_data["f_A_B"], 75)
            p25_nudge_B = np.percentile(factor_data["f_B_B"], 25)
            p75_nudge_B = np.percentile(factor_data["f_B_B"], 75)
        else:
            # Use mean
            center_nudge_A = np.mean(factor_data["f_A_B"])
            center_nudge_B = np.mean(factor_data["f_B_B"])

        # Central marker for nudge A (in lower half)
        # Add black outline for visibility (drawn first, behind)
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color="black",
            marker="|",
            s=350,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color=color_nudge_A,
            marker="|",
            s=300,
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
            s=350,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_B],
            [y_pos + 0.15],
            color=color_nudge_B,
            marker="|",
            s=300,
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

        # Add baseline marker (median or mean f_0(B))
        if percentiles:
            center_baseline = np.median(factor_data["f_0_B"])
        else:
            center_baseline = np.mean(factor_data["f_0_B"])

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

    # Add reference line at 0.5 (no preference) or 0 (log odds)
    ref_value = 0.0 if log_odds else 0.5
    ax.axvline(
        x=ref_value, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1
    )

    # Create factor labels
    factor_labels = []
    for factor in factors:
        levels = FACTOR_LEVELS.get(factor, (None, None))
        if levels[0] and levels[1]:
            factor_labels.append(format_factor_label(factor, levels[0], levels[1]))
        else:
            factor_labels.append(factor.replace("_", " ").title())

    # Set labels and title
    ax.set_yticks(y_positions)
    ax.set_yticklabels(factor_labels, fontsize=11)

    if log_odds:
        ax.set_xlabel("Log₁₀ Odds of Choosing B", fontsize=12)
    else:
        ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        space_label = "(Log Odds Space)" if log_odds else "(Frequency Space)"
        ax.set_title(
            f"Steerability by Factor {space_label}\n(Distribution across models and nudge types)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Set x-axis limits
    if log_odds:
        # Auto-scale for log odds, with some padding
        all_values = []
        for factor_data in data_by_factor.values():
            all_values.extend(factor_data["f_A_B"])
            all_values.extend(factor_data["f_B_B"])
            all_values.extend(factor_data["f_0_B"])
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            padding = (max_val - min_val) * 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
    else:
        ax.set_xlim(-0.05, 1.05)

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
            markersize=12,
            linewidth=0,
            markeredgewidth=2.5,
            label=f"{central_label} (nudged)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_baseline,
            markeredgecolor="black",
            markersize=10,
            label=f"{central_label} Baseline f₀(B)",
        ),
    ]

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

    # Add light horizontal grid lines to separate factors
    for i in range(n_factors - 1):
        ax.axhline(
            y=i + 0.5,
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

    # Invert y-axis so first factor is at top
    ax.invert_yaxis()

    plt.tight_layout()

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

    args = parser.parse_args()

    # Determine output path
    output_path = args.output if args.output else "steerability_violins.pdf"

    # Print header
    print("=" * 70)
    print("Steerability Violin Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    print(f"Space: {'Log Odds' if args.log_odds else 'Frequency'}")
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

    # Collect data by factor
    data_by_factor = collect_data_by_factor(results)

    # Print summary (always in frequency space for clarity)
    print("Data Summary by Factor (Frequency Space):")
    print("-" * 70)
    for factor in sorted(data_by_factor.keys()):
        factor_data = data_by_factor[factor]
        n_experiments = len(factor_data["f_0_B"])
        avg_baseline = np.mean(factor_data["f_0_B"])
        avg_nudge_A = np.mean(factor_data["f_A_B"])
        avg_nudge_B = np.mean(factor_data["f_B_B"])
        print(
            f"  {factor:<15} n={n_experiments:>3}  "
            f"Baseline={avg_baseline:.3f}  "
            f"Nudge→A={avg_nudge_A:.3f}  "
            f"Nudge→B={avg_nudge_B:.3f}"
        )
    print()

    # Transform to log odds if requested
    if args.log_odds:
        print("Transforming data to log odds space...")
        data_by_factor = transform_data_to_log_odds(data_by_factor)
        print()

    # Create plot
    figsize = (args.figsize[0], args.figsize[1] if args.figsize[1] else None)
    fig = create_steerability_violin_plot(
        data_by_factor,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        log_odds=args.log_odds,
        percentiles=args.percentiles,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
