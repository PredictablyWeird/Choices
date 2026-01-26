#!/usr/bin/env python3
"""
Plot steerability using AMCE-based effects as points.

For each factor, shows effect_A and effect_B as points based on AMCE scores.
The plot is in log odds space (AMCE is already on log-odds scale).

Usage:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability_amce --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results \
        --factors gender age_group wealth

    # Save to file
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results \
        --output steerability_amce_factors.pdf

    # Show one row per model
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results \
        --rows models

    # Show one row per nudge type
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results \
        --rows nudges

    # Filter by reasoning conditions
    uv run python -m choices.analysis.plot_steerability_amce \
        --results-dirs results \
        --reasoning-conditions none before after
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary_amce import (
    AMCEResult,
    compute_all_results,
    discover_experiments,
)
from choices.analysis.utils import get_model_display_name


def collect_data_by_factor(
    results: List[AMCEResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect AMCE effect data grouped by factor.

    Returns:
        Dictionary mapping factor -> {
            'effect_A': list of effect_A values,
            'effect_B': list of effect_B values,
            'amce_0': list of amce_0 values (baseline AMCE),
            'level_A': name of level A,
            'level_B': name of level B,
        }
    """
    data_by_factor: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "effect_A": [],
            "effect_B": [],
            "amce_0": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        data_by_factor[r.factor]["effect_A"].append(r.effect_A)
        data_by_factor[r.factor]["effect_B"].append(r.effect_B)
        data_by_factor[r.factor]["amce_0"].append(r.amce_0)
        # Store level names (they should be consistent within a factor)
        if data_by_factor[r.factor]["level_A"] is None:
            data_by_factor[r.factor]["level_A"] = r.level_A
            data_by_factor[r.factor]["level_B"] = r.level_B

    return dict(data_by_factor)


def collect_data_by_model(
    results: List[AMCEResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect AMCE effect data grouped by model and reasoning condition.

    Returns:
        Dictionary mapping "model (reasoning)" -> {
            'effect_A': list of effect_A values across all factors/nudge types,
            'effect_B': list of effect_B values across all factors/nudge types,
            'amce_0': list of amce_0 values (baseline),
            'level_A': None (not applicable for model grouping),
            'level_B': None (not applicable for model grouping),
        }
    """
    data_by_model: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "effect_A": [],
            "effect_B": [],
            "amce_0": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        # Include reasoning condition in key to separate different conditions
        display_name = get_model_display_name(r.model)
        model_key = f"{display_name} ({r.reasoning_condition})"
        data_by_model[model_key]["effect_A"].append(r.effect_A)
        data_by_model[model_key]["effect_B"].append(r.effect_B)
        data_by_model[model_key]["amce_0"].append(r.amce_0)

    return dict(data_by_model)


def collect_data_by_nudge_type(
    results: List[AMCEResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect AMCE effect data grouped by nudge type.

    Returns:
        Dictionary mapping nudge_type -> {
            'effect_A': list of effect_A values across all factors/models,
            'effect_B': list of effect_B values across all factors/models,
            'amce_0': list of amce_0 values (baseline),
            'level_A': None (not applicable for nudge type grouping),
            'level_B': None (not applicable for nudge type grouping),
        }
    """
    data_by_nudge: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "effect_A": [],
            "effect_B": [],
            "amce_0": [],
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        # Format nudge type for display
        nudge_key = r.nudge_type.replace("_", " ").title()
        data_by_nudge[nudge_key]["effect_A"].append(r.effect_A)
        data_by_nudge[nudge_key]["effect_B"].append(r.effect_B)
        data_by_nudge[nudge_key]["amce_0"].append(r.amce_0)

    return dict(data_by_nudge)


def format_factor_label(factor: str, level_A: str, level_B: str) -> str:
    """Format factor name with level labels."""
    factor_display = factor.replace("_", " ").title()
    return f"{factor_display}\n({level_A} vs {level_B})"


def create_steerability_amce_plot(
    data_by_row: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, None),
    row_type: str = "factors",
) -> plt.Figure:
    """
    Create point plot showing AMCE-based steerability effects.

    For each row, shows:
    - Points for effect_A (effect of nudging towards A)
    - Points for effect_B (effect of nudging towards B)

    Args:
        data_by_row: Dictionary mapping row key -> AMCE effect data
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size (width, height). If height is None, auto-calculated.
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
    color_effect_A = "#E63946"  # Red - effect of nudging towards A
    color_effect_B = "#457B9D"  # Blue - effect of nudging towards B

    # Y positions for each row
    y_positions = np.arange(n_rows)

    # Process each row separately
    for i, row_key in enumerate(rows):
        row_data = data_by_row[row_key]
        y_pos = y_positions[i]

        # Add individual data points as dots
        # Scatter points for effect A (below center line)
        n_A = len(row_data["effect_A"])
        if n_A > 0:
            jitter_A = np.random.uniform(-0.25, -0.05, n_A)
            ax.scatter(
                row_data["effect_A"],
                y_pos + jitter_A,
                color=color_effect_A,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Scatter points for effect B (above center line)
        n_B = len(row_data["effect_B"])
        if n_B > 0:
            jitter_B = np.random.uniform(0.05, 0.25, n_B)
            ax.scatter(
                row_data["effect_B"],
                y_pos + jitter_B,
                color=color_effect_B,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Add central tendency markers
        if n_A > 0:
            center_effect_A = np.mean(row_data["effect_A"])
            # Central marker for effect A (in lower half)
            # Add black outline for visibility (drawn first, behind)
            ax.scatter(
                [center_effect_A],
                [y_pos - 0.15],
                color="black",
                marker="|",
                s=550,
                linewidths=3,
                zorder=6,
            )
            ax.scatter(
                [center_effect_A],
                [y_pos - 0.15],
                color=color_effect_A,
                marker="|",
                s=500,
                linewidths=2.5,
                zorder=7,
            )

        if n_B > 0:
            center_effect_B = np.mean(row_data["effect_B"])
            # Central marker for effect B (in upper half)
            # Add black outline for visibility (drawn first, behind)
            ax.scatter(
                [center_effect_B],
                [y_pos + 0.15],
                color="black",
                marker="|",
                s=550,
                linewidths=3,
                zorder=6,
            )
            ax.scatter(
                [center_effect_B],
                [y_pos + 0.15],
                color=color_effect_B,
                marker="|",
                s=500,
                linewidths=2.5,
                zorder=7,
            )

    # Add reference line at 0 (no effect) - more prominent as center line
    ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

    # Remove y-axis tick labels - options will be shown on sides instead
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(rows))

    # Build x-axis label
    ax.set_xlabel("Nudge Effect Size (Log Odds)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        # Build row type label
        if row_type == "factors":
            row_label = "Factor"
            dist_label = "models and nudge types"
        elif row_type == "models":
            row_label = "Model"
            dist_label = "factors and nudge types"
        elif row_type == "nudges":
            row_label = "Nudge Type"
            dist_label = "factors and models"
        else:
            row_label = "Group"
            dist_label = "experiments"

        ax.set_title(
            f"Steerability by {row_label} (Effect Sizes)\n(Distribution across {dist_label})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Set x-axis limits centered at 0 with symmetric padding
    all_values = []
    for rd in data_by_row.values():
        all_values.extend(rd["effect_A"])
        all_values.extend(rd["effect_B"])
    if all_values:
        max_abs_val = max(abs(min(all_values)), abs(max(all_values)))
        padding = max_abs_val * 0.1 if max_abs_val > 0 else 0.1
        ax.set_xlim(-max_abs_val - padding, max_abs_val + padding)
    else:
        ax.set_xlim(-1, 1)

    # Add option labels on left and right sides of each row (only for factors)
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    label_offset = x_range * 0.02  # Small offset from plot edge

    if row_type == "factors":
        for i, row_key in enumerate(rows):
            y_pos = y_positions[i]
            rd = data_by_row[row_key]
            # Get level names from data
            level_A = rd.get("level_A") or "A"
            level_B = rd.get("level_B") or "B"

            # Left label (level A - towards which nudging decreases preference for B)
            ax.text(
                x_min - label_offset,
                y_pos,
                level_A,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color_effect_A,
                clip_on=False,  # Allow text outside plot area
            )

            # Right label (level B - towards which nudging increases preference for B)
            ax.text(
                x_max + label_offset,
                y_pos,
                level_B,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color_effect_B,
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
    from matplotlib.lines import Line2D

    # Use different legend labels when aggregating across factors
    if row_type == "factors":
        label_A = "Effect of nudging towards A"
        label_B = "Effect of nudging towards B"
    else:
        # When aggregating, A = less preferred at baseline, B = more preferred
        label_A = "Effect A (nudge away from baseline pref.)"
        label_B = "Effect B (nudge towards baseline pref.)"

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_effect_A,
            markeredgecolor="white",
            markersize=8,
            label=label_A,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_effect_B,
            markeredgecolor="white",
            markersize=8,
            label=label_B,
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color=color_effect_A,
            markersize=14,
            linewidth=0,
            markeredgewidth=2.5,
            label="Mean (effect)",
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
        description="Create point plots showing AMCE-based steerability effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability_amce --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability_amce \\
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability_amce \\
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
        help="Output file path (default: steerability_amce_<rows>.pdf)",
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
        "--rows",
        type=str,
        choices=["factors", "models", "nudges"],
        default="factors",
        help="What to show as rows: factors (default), models, or nudges",
    )

    parser.add_argument(
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include (e.g., 'before', 'none', 'after', 'low', 'medium', 'high')",
    )

    args = parser.parse_args()

    # Determine output path (append row type to default filename)
    if args.output:
        output_path = args.output
    else:
        output_path = f"steerability_amce_{args.rows}.pdf"

    # Print header
    print("=" * 70)
    print("Steerability AMCE Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    print(f"Rows: {args.rows}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning_conditions:
        print(f"Reasoning filter: {args.reasoning_conditions}")
    print("Space: Log Odds (AMCE)")
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

    # Compute AMCE results
    print("Computing AMCE results...")
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

    # Filter by reasoning condition if specified
    if args.reasoning_conditions:
        results = [
            r for r in results if r.reasoning_condition in args.reasoning_conditions
        ]
        print(f"After reasoning filter: {len(results)} result(s)")

    if not results:
        print("No results after filtering.")
        return

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

    # Print summary
    print(f"Data Summary by {row_type_label} (Effect Sizes):")
    print("-" * 70)
    for row_key in sorted(data_by_row.keys()):
        rd = data_by_row[row_key]
        n_experiments = len(rd["effect_A"])
        avg_effect_A = np.mean(rd["effect_A"]) if rd["effect_A"] else 0.0
        avg_effect_B = np.mean(rd["effect_B"]) if rd["effect_B"] else 0.0
        print(
            f"  {row_key:<25} n={n_experiments:>3}  "
            f"Effect_A={avg_effect_A:+.3f}  "
            f"Effect_B={avg_effect_B:+.3f}"
        )
    print()

    # Create plot
    figsize = (args.figsize[0], args.figsize[1] if args.figsize[1] else None)
    fig = create_steerability_amce_plot(
        data_by_row,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        row_type=args.rows,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
