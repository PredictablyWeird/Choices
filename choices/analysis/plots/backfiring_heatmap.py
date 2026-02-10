#!/usr/bin/env python3
"""
Generate heatmap plots of backfiring rates.

Backfiring occurs when a nudge towards option X actually *decreases* the
frequency of choosing X compared to the baseline. Each experiment has two
nudge directions (A and B), so the backfire rate per experiment is in [0, 1].

Two of {model, factor, nudge} are chosen as axes; the third is aggregated over.
A single heatmap is produced showing the backfire rate in each cell.

Usage:
    # Model (x) vs Factor (y), aggregating over nudge types
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor

    # With filters
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor \\
        --results-dirs results results_anthropic

    # Only count statistically significant backfires
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor --sig-only
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.plots.larger_group import (
    VALID_AXES,
    get_aspect_value,
)
from choices.analysis.utils import PLOTS_OUTPUT_DIR


def build_backfire_data(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    sig_only: bool = False,
    display_names: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a 2D array of backfire rates for the heatmap.

    Each experiment contributes 2 nudge directions. The backfire rate per cell
    is the fraction of nudges that backfired.

    Args:
        results: List of FrequencyResult objects
        x_aspect: Aspect for x-axis (columns)
        y_aspect: Aspect for y-axis (rows)
        sig_only: If True, only count backfires that are also statistically significant
        display_names: Whether to use display names for models

    Returns:
        (data_array, x_labels, y_labels) where data values are in [0, 1]
    """
    # Each cell accumulates (backfire_count, total_nudges)
    cell_counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

    for r in results:
        x_val = get_aspect_value(r, x_aspect, display_names)
        y_val = get_aspect_value(r, y_aspect, display_names)
        key = (x_val, y_val)

        if sig_only:
            bf_a = int(r.backfire_A and r.sig_A)
            bf_b = int(r.backfire_B and r.sig_B)
        else:
            bf_a = int(r.backfire_A)
            bf_b = int(r.backfire_B)

        cell_counts[key][0] += bf_a + bf_b
        cell_counts[key][1] += 2  # 2 nudge directions per experiment

    # Get sorted unique labels
    x_labels = sorted(set(k[0] for k in cell_counts.keys()))
    y_labels = sorted(set(k[1] for k in cell_counts.keys()))

    # Build data array (rows = y, cols = x)
    data = np.full((len(y_labels), len(x_labels)), np.nan)
    for (x_val, y_val), (bf_count, total) in cell_counts.items():
        xi = x_labels.index(x_val)
        yi = y_labels.index(y_val)
        data[yi, xi] = bf_count / total if total > 0 else np.nan

    return data, x_labels, y_labels


def plot_backfire_heatmap(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    sig_only: bool = False,
    display_names: bool = True,
    decimals: int = 2,
    subtitle: str = "",
    output_path: Optional[str] = None,
) -> None:
    """Create a single heatmap of backfire rates."""
    if not results:
        print("No results to plot.")
        return

    data, x_labels, y_labels = build_backfire_data(
        results, x_aspect, y_aspect, sig_only=sig_only, display_names=display_names
    )

    if data.size == 0 or np.all(np.isnan(data)):
        print("No valid data to plot.")
        return

    # Color range: 0 to max observed, sequential colormap
    valid = data[~np.isnan(data)]
    vmax = max(valid.max(), 0.1)  # at least 0.1 for visual clarity

    fig_w = 3 + len(x_labels) * 1.2
    fig_h = 2 + len(y_labels) * 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fmt = f".{decimals}f"

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Backfire Rate"},
        mask=np.isnan(data),
    )

    sig_label = "Significant " if sig_only else ""
    ax.set_title(
        f"{sig_label}Backfire Rate ({subtitle})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(x_aspect.capitalize(), fontsize=11)
    ax.set_ylabel(y_aspect.capitalize(), fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps of backfiring rates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Model (x) vs Factor (y), aggregating over nudge types
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor

    # Only significant backfires
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor --sig-only

    # Nudge (x) vs Factor (y), aggregating over models
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes nudge factor
        """,
    )

    parser.add_argument(
        "--axes",
        nargs=2,
        required=True,
        choices=["model", "factor", "nudge"],
        metavar=("X_AXIS", "Y_AXIS"),
        help="Two aspects for the heatmap axes. Choose from: model, factor, nudge. "
        "The third aspect is aggregated over.",
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
        "--reasoning",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include "
        "(e.g., 'low', 'medium', 'high', 'off', 'before', 'after', 'none')",
    )

    parser.add_argument(
        "--sig-only",
        action="store_true",
        help="Only count backfires that are also statistically significant",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (e.g., backfire.png). "
        f"If not set, saves to {PLOTS_OUTPUT_DIR}/ with an auto-generated name.",
    )

    parser.add_argument(
        "--no-display-names",
        action="store_true",
        help="Use raw model names instead of display names",
    )

    parser.add_argument(
        "--decimals",
        "-d",
        type=int,
        default=2,
        help="Number of decimal places for cell annotations (default: 2)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save as PDF instead of PNG",
    )

    args = parser.parse_args()

    x_aspect, y_aspect = args.axes
    if x_aspect == y_aspect:
        parser.error("The two axes must be different aspects.")

    aggregated_aspect = (VALID_AXES - {x_aspect, y_aspect}).pop()

    sig_label = " (sig-only)" if args.sig_only else ""
    print(
        f"Backfire Rate{sig_label}: x={x_aspect}, y={y_aspect} "
        f"(aggregating over {aggregated_aspect})"
    )
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning:
        print(f"Reasoning filter: {args.reasoning}")

    # Compute results
    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    # Apply reasoning filter
    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]

    print(f"Found {len(results)} experiments")

    if not results:
        print("No experiments found matching the filters.")
        return

    display_names = not args.no_display_names

    # Determine subtitle
    agg_values = sorted(
        set(get_aspect_value(r, aggregated_aspect, display_names) for r in results)
    )
    if len(agg_values) == 1:
        subtitle = f"{aggregated_aspect}: {agg_values[0]}"
    else:
        subtitle = f"aggregated over {aggregated_aspect} (n={len(agg_values)})"

    # Determine output path
    ext = "pdf" if args.pdf else "png"
    sig_suffix = "_sig" if args.sig_only else ""
    output_path = args.output
    if output_path is None:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            PLOTS_OUTPUT_DIR,
            f"backfire{sig_suffix}_{x_aspect}_vs_{y_aspect}.{ext}",
        )

    plot_backfire_heatmap(
        results,
        x_aspect,
        y_aspect,
        sig_only=args.sig_only,
        display_names=display_names,
        decimals=args.decimals,
        subtitle=subtitle,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
