#!/usr/bin/env python3
"""
Generate heatmap plots of backfiring rates.

Backfiring occurs when a nudge towards option X actually *decreases* the
frequency of choosing X compared to the baseline. Each experiment has two
nudge directions (A and B), so the backfire rate per experiment is in [0, 1].

Two of {model, factor, nudge} are chosen as axes; the third is aggregated over.
A single heatmap is produced showing the backfire rate in each cell.

With --bidirectional, each cell also shows a directional breakdown:
- If factor is an axis: (towards A, towards B) per factor
- If factor is aggregated: (towards baseline pref, away from baseline pref)

Usage:
    # Model (x) vs Factor (y), aggregating over nudge types
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor

    # With filters
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor \\
        --results-dirs results results_anthropic

    # Only count statistically significant backfires
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor --sig-only

    # Bidirectional breakdown
    uv run python -m choices.analysis.plots.backfiring_heatmap --axes model factor --bidirectional
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.plots.larger_group import (
    VALID_AXES,
    get_aspect_display_name,
    get_aspect_value,
)
from choices.analysis.utils import PLOTS_OUTPUT_DIR


def _get_backfire_counts(r: FrequencyResult, sig_only: bool) -> Tuple[int, int]:
    """Return (bf_a, bf_b) backfire indicators for a result."""
    if sig_only:
        return int(r.backfire_A and r.sig_A), int(r.backfire_B and r.sig_B)
    return int(r.backfire_A), int(r.backfire_B)


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

    Returns:
        (data_array, x_labels, y_labels) where data values are in [0, 1]
    """
    # Each cell accumulates (backfire_count, total_nudges)
    cell_counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

    for r in results:
        x_val = get_aspect_value(r, x_aspect, display_names)
        y_val = get_aspect_value(r, y_aspect, display_names)
        key = (x_val, y_val)
        bf_a, bf_b = _get_backfire_counts(r, sig_only)
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


def build_backfire_data_bidirectional(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    sig_only: bool = False,
    display_names: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    bool,
    Dict[str, Tuple[str, str]],
]:
    """
    Build 2D arrays of backfire rates with directional breakdown.

    Returns:
        (data_main, data_dir1, data_dir2, x_labels, y_labels,
         factor_is_axis, factor_level_map)

    If factor IS an axis (Case 2):
        dir1 = backfire when nudging towards A
        dir2 = backfire when nudging towards B
        factor_level_map maps factor_display_name -> (level_A, level_B)

    If factor is NOT an axis (Case 1):
        dir1 = backfire when nudging towards baseline preference
        dir2 = backfire when nudging away from baseline preference
        factor_level_map is empty
    """
    factor_is_axis = "factor" in (x_aspect, y_aspect)

    # key -> [bf_total, total, bf_d1, total_d1, bf_d2, total_d2]
    cell_counts: Dict[Tuple[str, str], List[int]] = defaultdict(
        lambda: [0, 0, 0, 0, 0, 0]
    )
    # factor display name -> (level_A, level_B)
    factor_level_map: Dict[str, Tuple[str, str]] = {}

    for r in results:
        x_val = get_aspect_value(r, x_aspect, display_names)
        y_val = get_aspect_value(r, y_aspect, display_names)
        key = (x_val, y_val)
        bf_a, bf_b = _get_backfire_counts(r, sig_only)

        # Main rate
        cell_counts[key][0] += bf_a + bf_b
        cell_counts[key][1] += 2

        if factor_is_axis:
            # Case 2: dir1 = towards A, dir2 = towards B
            cell_counts[key][2] += bf_a
            cell_counts[key][3] += 1
            cell_counts[key][4] += bf_b
            cell_counts[key][5] += 1
            # Track factor levels for labels
            factor_label = get_aspect_value(r, "factor", display_names)
            if factor_label not in factor_level_map:
                factor_level_map[factor_label] = (r.level_A, r.level_B)
        else:
            # Case 1: dir1 = towards baseline pref, dir2 = away from baseline pref
            baseline_prefers_B = r.f_0_B > 0.5
            if baseline_prefers_B:
                bf_towards, bf_away = bf_b, bf_a
            else:
                bf_towards, bf_away = bf_a, bf_b
            cell_counts[key][2] += bf_towards
            cell_counts[key][3] += 1
            cell_counts[key][4] += bf_away
            cell_counts[key][5] += 1

    x_labels = sorted(set(k[0] for k in cell_counts))
    y_labels = sorted(set(k[1] for k in cell_counts))

    n_y, n_x = len(y_labels), len(x_labels)
    data_main = np.full((n_y, n_x), np.nan)
    data_d1 = np.full((n_y, n_x), np.nan)
    data_d2 = np.full((n_y, n_x), np.nan)

    for (x_val, y_val), c in cell_counts.items():
        xi = x_labels.index(x_val)
        yi = y_labels.index(y_val)
        data_main[yi, xi] = c[0] / c[1] if c[1] else np.nan
        data_d1[yi, xi] = c[2] / c[3] if c[3] else np.nan
        data_d2[yi, xi] = c[4] / c[5] if c[5] else np.nan

    return (
        data_main,
        data_d1,
        data_d2,
        x_labels,
        y_labels,
        factor_is_axis,
        factor_level_map,
    )


def _build_annot_array(
    data_main: np.ndarray,
    data_d1: np.ndarray,
    data_d2: np.ndarray,
    decimals: int,
) -> np.ndarray:
    """Build a string annotation array with main rate and directional breakdown."""
    n_y, n_x = data_main.shape
    annot = np.empty((n_y, n_x), dtype=object)
    fmt = f".{decimals}%"
    for yi in range(n_y):
        for xi in range(n_x):
            m = data_main[yi, xi]
            d1 = data_d1[yi, xi]
            d2 = data_d2[yi, xi]
            if np.isnan(m):
                annot[yi, xi] = ""
            else:
                main_str = f"{m:{fmt}}"
                d1_str = f"{d1:{fmt}}" if not np.isnan(d1) else "–"
                d2_str = f"{d2:{fmt}}" if not np.isnan(d2) else "–"
                annot[yi, xi] = f"{main_str}\n({d1_str}, {d2_str})"
    return annot


def _add_factor_level_suffixes(
    labels: List[str], factor_level_map: Dict[str, Tuple[str, str]]
) -> List[str]:
    """Append 'A vs B' to factor labels."""
    new_labels = []
    for label in labels:
        if label in factor_level_map:
            a, b = factor_level_map[label]
            new_labels.append(f"{label}\n({a} vs {b})")
        else:
            new_labels.append(label)
    return new_labels


def _plot_heatmap_common(
    ax,
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    x_aspect: str,
    y_aspect: str,
    sig_only: bool,
    subtitle: str,
    decimals: int,
    annot=True,
    fmt: Optional[str] = None,
    vmax: Optional[float] = None,
    no_title: bool = False,
) -> None:
    """Shared heatmap rendering logic."""
    valid = data[~np.isnan(data)]
    if vmax is None:
        vmax = max(valid.max(), 0.1) if len(valid) > 0 else 0.1

    if fmt is None:
        fmt = f".{decimals}%"

    sns.heatmap(
        data,
        ax=ax,
        annot=annot,
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

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    if not no_title:
        sig_label = "Significant " if sig_only else ""
        ax.set_title(
            f"{sig_label}Backfire Rate ({subtitle})",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
    ax.set_xlabel(get_aspect_display_name(x_aspect), fontsize=11)
    ax.set_ylabel(get_aspect_display_name(y_aspect), fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="y", rotation=0)


def plot_backfire_heatmap(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    sig_only: bool = False,
    bidirectional: bool = False,
    display_names: bool = True,
    decimals: int = 0,
    subtitle: str = "",
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    no_title: bool = False,
) -> None:
    """Create a single heatmap of backfire rates."""
    if not results:
        print("No results to plot.")
        return

    if bidirectional:
        (
            data,
            data_d1,
            data_d2,
            x_labels,
            y_labels,
            factor_is_axis,
            factor_level_map,
        ) = build_backfire_data_bidirectional(
            results, x_aspect, y_aspect, sig_only=sig_only, display_names=display_names
        )

        # Build custom annotation strings
        annot = _build_annot_array(data, data_d1, data_d2, decimals)

        # Add factor level suffixes to the appropriate axis labels
        if factor_is_axis:
            if x_aspect == "factor":
                x_labels = _add_factor_level_suffixes(x_labels, factor_level_map)
            else:
                y_labels = _add_factor_level_suffixes(y_labels, factor_level_map)
    else:
        data, x_labels, y_labels = build_backfire_data(
            results, x_aspect, y_aspect, sig_only=sig_only, display_names=display_names
        )
        annot = True

    if data.size == 0 or np.all(np.isnan(data)):
        print("No valid data to plot.")
        return

    if figsize is None:
        fig_w = 3 + len(x_labels) * 1.2
        fig_h = 2 + len(y_labels) * 0.7
        # Extra height for bidirectional annotations
        if bidirectional:
            fig_h = 2.5 + len(y_labels) * 0.9
        figsize = (fig_w, fig_h)
    fig, ax = plt.subplots(figsize=figsize)

    if bidirectional:
        _plot_heatmap_common(
            ax,
            data,
            x_labels,
            y_labels,
            x_aspect,
            y_aspect,
            sig_only,
            subtitle,
            decimals,
            annot=annot,
            fmt="",
            no_title=no_title,
        )
    else:
        _plot_heatmap_common(
            ax,
            data,
            x_labels,
            y_labels,
            x_aspect,
            y_aspect,
            sig_only,
            subtitle,
            decimals,
            no_title=no_title,
        )

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
        "--bidirectional",
        action="store_true",
        help="Show directional breakdown in each cell. "
        "If factor is an axis: (towards A, towards B). "
        "If factor is aggregated: (towards baseline pref, away from baseline pref).",
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
        default=0,
        help="Number of decimal places for cell annotations (default: 0)",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches as WIDTH HEIGHT (default: auto-computed)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save as PDF instead of PNG",
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Suppress the plot title",
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
    agg_display = get_aspect_display_name(aggregated_aspect)
    if len(agg_values) == 1:
        subtitle = f"{agg_display}: {agg_values[0]}"
    else:
        subtitle = f"aggregated over {agg_display} (n={len(agg_values)})"

    # Determine output path
    ext = "pdf" if args.pdf else "png"
    sig_suffix = "_sig" if args.sig_only else ""
    bidir_suffix = "_bidir" if args.bidirectional else ""
    output_path = args.output
    if output_path is None:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            PLOTS_OUTPUT_DIR,
            f"backfire{sig_suffix}{bidir_suffix}_{x_aspect}_vs_{y_aspect}.{ext}",
        )

    figsize = tuple(args.figsize) if args.figsize else None

    plot_backfire_heatmap(
        results,
        x_aspect,
        y_aspect,
        sig_only=args.sig_only,
        bidirectional=args.bidirectional,
        display_names=display_names,
        decimals=args.decimals,
        subtitle=subtitle,
        output_path=output_path,
        figsize=figsize,
        no_title=args.no_title,
    )


if __name__ == "__main__":
    main()
