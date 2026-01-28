#!/usr/bin/env python3
"""
Plot steerability effects (v2) - simplified and streamlined.

Two modes:
1. Single model mode (--model specified): Shows steerability for all factors
2. Single factor mode (--factor specified): Shows steerability for all models

Features:
- Always uses log-odds steerability
- Always includes baseline preference column
- Shows mean steerability bars (thick) plus marker for nudge type with largest bias
- No title (use --title to override)

Usage:
    # Single model: show all factors for one model
    uv run python -m choices.analysis.plot_model_effects_v2 \
        --model claude-3-5-sonnet-latest \
        --results-dirs results

    # Single model: filter to specific factors
    uv run python -m choices.analysis.plot_model_effects_v2 \
        --model claude-3-5-sonnet-latest \
        --results-dirs results \
        --factors wealth "framing effect"

    # Single factor: show all models for one factor
    uv run python -m choices.analysis.plot_model_effects_v2 \
        --factor wealth \
        --results-dirs results \
        --reasoning none

    # Single factor: filter to specific models
    uv run python -m choices.analysis.plot_model_effects_v2 \
        --factor wealth \
        --results-dirs results \
        --models gpt-4o claude-3-5-sonnet-latest
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import (
    get_model_display_name,
    get_nudge_display_name,
    get_nudge_marker,
    is_reasoning_model,
)


def get_default_reasoning_condition(model: str) -> str:
    """Get the default reasoning condition for a model."""
    if is_reasoning_model(model):
        return "off"
    return "none"


def collect_data_for_single_model(
    results: List[FrequencyResult],
    model: str,
    reasoning: str,
) -> Dict[str, Dict[str, any]]:
    """
    Collect data for a single model, grouped by factor.

    Returns:
        Dictionary mapping factor -> {
            'nudge_data': list of nudge data dicts,
            'f_0_B': baseline frequency,
            'level_A': str,
            'level_B': str,
        }
    """
    data: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "nudge_data": [],
            "f_0_B": None,
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        if r.model != model or r.reasoning_condition != reasoning:
            continue

        data[r.factor]["nudge_data"].append(
            {
                "nudge_type": r.nudge_type,
                "steerability_A": r.steerability_A,
                "steerability_B": r.steerability_B,
                "steerability_bias": r.steerability_bias,
            }
        )
        if data[r.factor]["f_0_B"] is None:
            data[r.factor]["f_0_B"] = r.f_0_B
        if data[r.factor]["level_A"] is None:
            data[r.factor]["level_A"] = r.level_A
            data[r.factor]["level_B"] = r.level_B

    return dict(data)


def collect_data_for_single_factor(
    results: List[FrequencyResult],
    factor: str,
) -> Dict[Tuple[str, str], Dict[str, any]]:
    """
    Collect data for a single factor, grouped by (model, reasoning).

    Returns:
        Dictionary mapping (model, reasoning) -> {
            'nudge_data': list of nudge data dicts,
            'f_0_B': baseline frequency,
            'level_A': str,
            'level_B': str,
        }
    """
    data: Dict[Tuple[str, str], Dict[str, any]] = defaultdict(
        lambda: {
            "nudge_data": [],
            "f_0_B": None,
            "level_A": None,
            "level_B": None,
        }
    )

    for r in results:
        if r.factor != factor:
            continue

        key = (r.model, r.reasoning_condition)
        data[key]["nudge_data"].append(
            {
                "nudge_type": r.nudge_type,
                "steerability_A": r.steerability_A,
                "steerability_B": r.steerability_B,
                "steerability_bias": r.steerability_bias,
            }
        )
        if data[key]["f_0_B"] is None:
            data[key]["f_0_B"] = r.f_0_B
        if data[key]["level_A"] is None:
            data[key]["level_A"] = r.level_A
            data[key]["level_B"] = r.level_B

    return dict(data)


def find_most_biased_nudge(nudge_data: List[Dict]) -> Optional[Dict]:
    """Find the nudge type with the largest magnitude steerability bias."""
    valid = [nd for nd in nudge_data if nd.get("steerability_bias") is not None]
    if not valid:
        return None
    return max(valid, key=lambda nd: abs(nd["steerability_bias"]))


def abbreviate_label(label: str) -> str:
    """Abbreviate long labels for display."""
    if label is None:
        return label
    # Abbreviate left-handed/right-handed
    label = label.replace("left-handed", "left-h.")
    label = label.replace("right-handed", "right-h.")
    label = label.replace("Left-handed", "Left-h.")
    label = label.replace("Right-handed", "Right-h.")
    return label


def create_steerability_plot(
    row_data: Dict[str, Dict[str, any]],
    row_labels: List[str],
    row_keys: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, None),
    show_legend: bool = True,
    title: Optional[str] = None,
    single_model_mode: bool = False,
) -> plt.Figure:
    """
    Create the steerability plot with baseline column.

    Args:
        row_data: Dict mapping row_key -> data dict
        row_labels: Display labels for each row (used in multi-model mode)
        row_keys: Keys to look up in row_data
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        show_legend: Whether to show legend
        title: Optional title
        single_model_mode: If True, rows are factors and use option labels instead

    Returns:
        The matplotlib Figure object
    """
    n_rows = len(row_keys)

    if n_rows == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of rows
    height = figsize[1] if figsize[1] else max(3, n_rows * 0.8)

    # Create figure with baseline column
    fig, (ax, ax_baseline) = plt.subplots(
        1, 2, figsize=(figsize[0], height), gridspec_kw={"width_ratios": [4, 1]}
    )

    # Colors
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker

    # Bar settings
    bar_linewidth = 4
    bar_height = 0.55  # Total height, centered on y_pos

    # Y positions for each row
    y_positions = np.arange(n_rows)

    # Collect all nudge types for legend
    all_nudge_types = set()

    # Process each row
    for i, (key, label) in enumerate(zip(row_keys, row_labels)):
        rd = row_data[key]
        y_pos = y_positions[i]

        # Draw horizontal line for this row (stronger/more visible)
        ax.axhline(
            y=y_pos,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.5,
            zorder=2,
        )

        # Compute mean steerability values
        steer_A_values = [
            nd.get("steerability_A")
            for nd in rd["nudge_data"]
            if nd.get("steerability_A") is not None
        ]
        steer_B_values = [
            nd.get("steerability_B")
            for nd in rd["nudge_data"]
            if nd.get("steerability_B") is not None
        ]

        # Draw mean bars (centered on y_pos)
        if steer_A_values:
            mean_A = -sum(steer_A_values) / len(steer_A_values)  # Negate for plotting
            ax.plot(
                [mean_A, mean_A],
                [y_pos - bar_height / 2, y_pos + bar_height / 2],
                color=color_nudge_A,
                linewidth=bar_linewidth,
                solid_capstyle="round",
                zorder=6,
            )

        if steer_B_values:
            mean_B = sum(steer_B_values) / len(steer_B_values)
            ax.plot(
                [mean_B, mean_B],
                [y_pos - bar_height / 2, y_pos + bar_height / 2],
                color=color_nudge_B,
                linewidth=bar_linewidth,
                solid_capstyle="round",
                zorder=6,
            )

        # Find and plot the most biased nudge type
        most_biased = find_most_biased_nudge(rd["nudge_data"])
        if most_biased:
            nudge_type = most_biased["nudge_type"]
            all_nudge_types.add(nudge_type)
            marker = get_nudge_marker(nudge_type)

            # Plot marker for steerability towards A (slightly above center)
            steer_A = most_biased.get("steerability_A")
            if steer_A is not None:
                ax.scatter(
                    [-steer_A],  # Negate for plotting
                    [y_pos - 0.15],
                    color=color_nudge_A,
                    marker=marker,
                    s=50,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=7,
                )

            # Plot marker for steerability towards B (slightly below center)
            steer_B = most_biased.get("steerability_B")
            if steer_B is not None:
                ax.scatter(
                    [steer_B],
                    [y_pos + 0.15],
                    color=color_nudge_B,
                    marker=marker,
                    s=50,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=7,
                )

    # Add reference line at 0
    ax.axvline(x=0.0, color="gray", linestyle="-", linewidth=1.5, alpha=0.7, zorder=1)

    # Auto-scale x-axis based on data
    all_steer_values = []
    for rd in row_data.values():
        for nd in rd["nudge_data"]:
            if nd.get("steerability_A") is not None:
                all_steer_values.append(-nd["steerability_A"])
            if nd.get("steerability_B") is not None:
                all_steer_values.append(nd["steerability_B"])
    if all_steer_values:
        x_min = min(all_steer_values)
        x_max = max(all_steer_values)
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    else:
        ax.set_xlim(-1.0, 1.0)

    # Add light background fills for left (red) and right (blue) halves - per row with gaps
    x_min_bg, x_max_bg = ax.get_xlim()
    row_bg_height = (
        0.42  # Height of each row's background (less than 0.5 to create gaps)
    )
    for i in range(n_rows):
        y_pos = y_positions[i]
        # Light red for left side (steerability towards A)
        ax.fill_between(
            [x_min_bg, 0],
            y_pos - row_bg_height,
            y_pos + row_bg_height,
            color=color_nudge_A,
            alpha=0.08,
            zorder=0,
        )
        # Light blue for right side (steerability towards B)
        ax.fill_between(
            [0, x_max_bg],
            y_pos - row_bg_height,
            y_pos + row_bg_height,
            color=color_nudge_B,
            alpha=0.08,
            zorder=0,
        )

    ax.set_xlabel("Log Odds Effect", fontsize=11)

    # Add vertical grid lines
    ax.xaxis.grid(True, linestyle=":", linewidth=1.0, alpha=0.5, zorder=0)

    # Configure y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n_rows)

    # Add row labels
    x_min_ax, x_max_ax = ax.get_xlim()
    x_range_ax = x_max_ax - x_min_ax
    label_offset = x_range_ax * 0.02

    if single_model_mode:
        # Single model mode: option A labels on left (red), option B labels on right (blue)
        for i, key in enumerate(row_keys):
            y_pos = y_positions[i]
            rd = row_data[key]
            level_A = abbreviate_label(rd.get("level_A") or "A")
            level_B = abbreviate_label(rd.get("level_B") or "B")

            # Left label: option A in bold red
            ax.text(
                x_min_ax - label_offset,
                y_pos,
                level_A,
                ha="right",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color_nudge_A,
                clip_on=False,
            )

            # Right label: option B in bold blue (will be between steerability and baseline)
            ax.text(
                x_max_ax + label_offset,
                y_pos,
                level_B,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color_nudge_B,
                clip_on=False,
            )
    else:
        # Multi-model mode: model names on left, header labels at top
        for i, label in enumerate(row_labels):
            y_pos = y_positions[i]
            ax.text(
                x_min_ax - label_offset,
                y_pos,
                label,
                ha="right",
                va="center",
                fontsize=11,
                color="#333333",
                clip_on=False,
            )

        # Add header labels for options (get from first row) with arrows
        if row_keys:
            first_rd = row_data[row_keys[0]]
            level_A = abbreviate_label(first_rd.get("level_A") or "A")
            level_B = abbreviate_label(first_rd.get("level_B") or "B")

            header_y = -0.6
            left_x = x_min_ax + 0.05 * x_range_ax
            right_x = x_max_ax - 0.05 * x_range_ax

            ax.text(
                left_x,
                header_y,
                f"A ← {level_A}",
                ha="left",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color_nudge_A,
                clip_on=False,
            )
            ax.text(
                right_x,
                header_y,
                f"{level_B} → B",
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color_nudge_B,
                clip_on=False,
            )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    # Style main axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Invert y-axis so first row is at top
    ax.invert_yaxis()

    # Draw baseline bar column
    for i, key in enumerate(row_keys):
        y_pos = y_positions[i]
        rd = row_data[key]
        f_0_B = rd["f_0_B"]

        if f_0_B is not None:
            # Draw horizontal bar from 0.5 (neutral) towards baseline
            ax_baseline.barh(
                y_pos,
                f_0_B - 0.5,
                left=0.5,
                height=0.35,
                color=color_baseline,
                alpha=0.7,
                edgecolor="none",
            )

            # Add text label
            ax_baseline.text(
                f_0_B + 0.02 if f_0_B >= 0.5 else f_0_B - 0.02,
                y_pos,
                f"{f_0_B:.2f}",
                ha="left" if f_0_B >= 0.5 else "right",
                va="center",
                fontsize=9,
                color=color_baseline,
                fontweight="bold",
            )

    # Style baseline axis
    ax_baseline.set_xlim(0, 1)
    ax_baseline.set_ylim(ax.get_ylim())
    ax_baseline.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_baseline.spines["top"].set_visible(False)
    ax_baseline.spines["right"].set_visible(False)
    ax_baseline.spines["left"].set_visible(False)
    ax_baseline.spines["bottom"].set_visible(False)
    ax_baseline.set_yticks([])
    ax_baseline.set_xticks([])
    ax_baseline.set_xlabel("Baseline\nPreference", fontsize=10)

    # Create legend with two groups
    if show_legend:
        from matplotlib.patches import Patch

        legend_elements = [
            # Group 1: Mean steerability bars
            Line2D(
                [0],
                [0],
                color=color_nudge_A,
                linewidth=4,
                solid_capstyle="round",
                label="Mean (→A)",
            ),
            Line2D(
                [0],
                [0],
                color=color_nudge_B,
                linewidth=4,
                solid_capstyle="round",
                label="Mean (→B)",
            ),
            # Spacer and header for influence types
            Patch(facecolor="none", edgecolor="none", label=""),
            Patch(facecolor="none", edgecolor="none", label=r"$\bf{Influence}$"),
        ]

        # Group 2: Nudge type symbols
        for nudge_type in sorted(all_nudge_types):
            marker = get_nudge_marker(nudge_type)
            display_name = get_nudge_display_name(nudge_type)
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

        # Position legend further right in single model mode due to B labels
        legend_anchor = (1.55, 1) if single_model_mode else (1.35, 1)
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=legend_anchor,
            fontsize=10,
            framealpha=0.9,
        )

    # Adjust subplot spacing - more space between plots in single model mode for B labels
    if single_model_mode:
        fig.subplots_adjust(left=0.15, right=0.60 if show_legend else 0.75, wspace=0.55)
    else:
        fig.subplots_adjust(left=0.18, right=0.70 if show_legend else 0.85, wspace=0.15)

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot steerability effects (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model mode
    uv run python -m choices.analysis.plot_model_effects_v2 \\
        --model claude-3-5-sonnet-latest \\
        --results-dirs results

    # Single factor mode
    uv run python -m choices.analysis.plot_model_effects_v2 \\
        --factor wealth \\
        --results-dirs results \\
        --reasoning none
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model",
        type=str,
        help="Single model mode: show all factors for this model",
    )
    mode_group.add_argument(
        "--factor",
        type=str,
        help="Single factor mode: show all models for this factor",
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
        help="Filter by reasoning condition(s)",
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Filter to specific factors (single model mode)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific models (single factor mode)",
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
        default=[8, None],
        help="Figure size (width height)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot",
    )

    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide the legend",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title (default: no title)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.model:
        safe_model = args.model.replace("/", "_").replace(":", "_")
        output_path = f"steerability_{safe_model}.pdf"
    else:
        safe_factor = args.factor.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_path = f"steerability_{safe_factor}.pdf"

    # Print header
    print("=" * 60)
    print("Steerability Plot (v2)")
    print("=" * 60)
    if args.model:
        print(f"Mode: Single model ({args.model})")
    else:
        print(f"Mode: Single factor ({args.factor})")
    print(f"Results directories: {args.results_dirs}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()

    # Compute frequency results
    print("Computing frequency results...")
    if args.model:
        results = compute_all_results(
            args.results_dirs,
            model_filter=[args.model],
            factor_filter=args.factors,
        )
    else:
        results = compute_all_results(
            args.results_dirs,
            model_filter=args.models,
            factor_filter=[args.factor],
        )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s)")

    # Filter by reasoning condition(s)
    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]
        print(f"After reasoning filter: {len(results)} result(s)")

    if not results:
        print("No results found after filtering.")
        return

    figsize = (args.figsize[0], args.figsize[1] if len(args.figsize) > 1 else None)

    if args.model:
        # Single model mode: rows are factors
        # Get the actual model name from results (may differ slightly from args.model)
        models_in_results = set(r.model for r in results)
        if not models_in_results:
            print("No results found for this model.")
            return
        actual_model = list(models_in_results)[0]  # Should be only one after filtering

        # Determine reasoning condition
        if args.reasoning:
            # Filter to specified reasoning conditions and pick the first one found
            reasoning_in_results = set(r.reasoning_condition for r in results)
            valid_reasoning = [r for r in args.reasoning if r in reasoning_in_results]
            if not valid_reasoning:
                print(
                    f"No results with reasoning {args.reasoning}. Available: {reasoning_in_results}"
                )
                return
            reasoning = valid_reasoning[0]
        else:
            reasoning = get_default_reasoning_condition(actual_model)

        # Filter to this reasoning condition
        results = [r for r in results if r.reasoning_condition == reasoning]

        data = collect_data_for_single_model(results, actual_model, reasoning)

        if not data:
            print(f"No data found for model={actual_model}, reasoning={reasoning}.")
            return

        row_keys = sorted(data.keys())
        row_labels = row_keys  # Factor names as labels

        print(f"\nFactors: {len(row_keys)}")
        for f in row_keys:
            print(f"  - {f}")
        print()

        fig = create_steerability_plot(
            row_data=data,
            row_labels=row_labels,
            row_keys=row_keys,
            output_path=output_path,
            figsize=figsize,
            show_legend=not args.no_legend,
            title=args.title,
            single_model_mode=True,
        )

    else:
        # Single factor mode: rows are models
        data = collect_data_for_single_factor(results, args.factor)

        if not data:
            print("No data found for this factor.")
            return

        # Check which models appear only once (single reasoning condition)
        model_counts = defaultdict(int)
        for model, reasoning in data.keys():
            model_counts[model] += 1

        # Sort by model display name, then reasoning
        row_keys = sorted(
            data.keys(),
            key=lambda x: (get_model_display_name(x[0]), x[1]),
        )

        # Create labels - hide reasoning if model only appears once
        row_labels = []
        for model, reasoning in row_keys:
            model_display = get_model_display_name(model)
            if model_counts[model] > 1:
                row_labels.append(f"{model_display} ({reasoning})")
            else:
                row_labels.append(model_display)

        print(f"\nModels: {len(row_keys)}")
        for label in row_labels:
            print(f"  - {label}")
        print()

        fig = create_steerability_plot(
            row_data=data,
            row_labels=row_labels,
            row_keys=row_keys,
            output_path=output_path,
            figsize=figsize,
            show_legend=not args.no_legend,
            title=args.title,
        )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
