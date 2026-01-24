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
"""

import argparse
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
                pc.set_alpha(0.6)

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
                pc.set_alpha(0.6)

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

        # Add mean markers for nudge conditions
        mean_nudge_A = np.mean(factor_data["f_A_B"])
        mean_nudge_B = np.mean(factor_data["f_B_B"])

        # Mean marker for nudge A (in lower half)
        ax.scatter(
            [mean_nudge_A],
            [y_pos - 0.15],
            color=color_nudge_A,
            marker="|",
            s=200,
            linewidths=3,
            zorder=6,
        )

        # Mean marker for nudge B (in upper half)
        ax.scatter(
            [mean_nudge_B],
            [y_pos + 0.15],
            color=color_nudge_B,
            marker="|",
            s=200,
            linewidths=3,
            zorder=6,
        )

        # Add baseline marker (average f_0(B))
        avg_baseline = np.mean(factor_data["f_0_B"])

        # Draw vertical line at baseline spanning the row
        ax.plot(
            [avg_baseline, avg_baseline],
            [y_pos - 0.35, y_pos + 0.35],
            color=color_baseline,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            zorder=4,
        )

        # Add diamond marker at center
        ax.scatter(
            [avg_baseline],
            [y_pos],
            color=color_baseline,
            marker="D",
            s=100,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    # Add reference line at 0.5 (no preference)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

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
    ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        ax.set_title(
            "Steerability by Factor\n(Distribution across models and nudge types)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Set x-axis limits
    ax.set_xlim(-0.05, 1.05)

    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor=color_nudge_A, alpha=0.6, label="Nudged towards A"),
        Patch(facecolor=color_nudge_B, alpha=0.6, label="Nudged towards B"),
        Line2D(
            [0],
            [0],
            marker="|",
            color=color_nudge_A,
            markersize=12,
            linewidth=0,
            markeredgewidth=3,
            label="Mean (nudged)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=color_baseline,
            markeredgecolor="black",
            markersize=10,
            label="Mean Baseline f₀(B)",
        ),
    ]
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

    # Print summary
    print("Data Summary by Factor:")
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

    # Create plot
    figsize = (args.figsize[0], args.figsize[1] if args.figsize[1] else None)
    fig = create_steerability_violin_plot(
        data_by_factor,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
