#!/usr/bin/env python3
"""
Generate heatmap plots of larger-group preference rates (P(Large)).

When nudge is NOT an axis, creates two side-by-side heatmaps:
- P_0(Large): Rate of choosing the larger group in the baseline condition
- P_AB(Large): Rate of choosing the larger group in the nudged conditions

When nudge IS an axis, creates a single heatmap with a "none" row/column
for the baseline P_0(Large) followed by per-nudge-type P(Large) values,
separated by a small gap.

Two of {model, factor, nudge} are chosen as axes; the third is aggregated over.

Usage:
    # Model (x) vs Factor (y), aggregating over nudge types
    uv run python -m choices.analysis.plots.larger_group --axes model factor

    # Factor (x) vs Nudge (y), aggregating over models
    uv run python -m choices.analysis.plots.larger_group --axes factor nudge

    # With filters
    uv run python -m choices.analysis.plots.larger_group --axes model factor \\
        --results-dirs results results_anthropic \\
        --nudge-types user_preference

    # Save to file
    uv run python -m choices.analysis.plots.larger_group --axes model factor -o heatmaps.png
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import (
    PLOTS_OUTPUT_DIR,
    get_base_model_name,
    get_model_display_name,
    get_nudge_display_name,
)

# Aspect keys
VALID_AXES = {"model", "factor", "nudge"}

# Label for the baseline row/column when nudge is an axis
BASELINE_LABEL = "Baseline"


def get_aspect_value(
    result: FrequencyResult, aspect: str, display_names: bool = True
) -> str:
    """Get the value of an aspect (model, factor, nudge) from a FrequencyResult."""
    if aspect == "model":
        if display_names:
            name = get_model_display_name(result.model)
        else:
            name = get_base_model_name(result.model)
        # Append reasoning condition
        return f"{name} ({result.reasoning_condition})"
    elif aspect == "factor":
        if display_names:
            return result.factor.replace("_", " ").title()
        return result.factor
    elif aspect == "nudge":
        if display_names:
            return get_nudge_display_name(result.nudge_type)
        return result.nudge_type
    else:
        raise ValueError(f"Unknown aspect: {aspect}")


def build_heatmap_data(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    metric: str,
    display_names: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a 2D array for the heatmap.

    Args:
        results: List of FrequencyResult objects
        x_aspect: Aspect for x-axis (columns)
        y_aspect: Aspect for y-axis (rows)
        metric: Which metric to plot ("p0" for P_0(Large), "pab" for P_AB(Large))
        display_names: Whether to use display names for models

    Returns:
        (data_array, x_labels, y_labels)
    """
    # Group results by (x_value, y_value)
    cell_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for r in results:
        x_val = get_aspect_value(r, x_aspect, display_names)
        y_val = get_aspect_value(r, y_aspect, display_names)

        if metric == "p0":
            val = r.larger_group_rate_base
        elif metric == "pab":
            # Average of nudge-A and nudge-B conditions
            if r.larger_group_rate_A is not None and r.larger_group_rate_B is not None:
                val = (r.larger_group_rate_A + r.larger_group_rate_B) / 2
            else:
                val = None
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if val is not None:
            cell_values[(x_val, y_val)].append(val)

    # Get sorted unique labels
    x_labels = sorted(set(k[0] for k in cell_values.keys()))
    y_labels = sorted(set(k[1] for k in cell_values.keys()))

    # Build data array (rows = y, cols = x)
    data = np.full((len(y_labels), len(x_labels)), np.nan)
    for (x_val, y_val), vals in cell_values.items():
        xi = x_labels.index(x_val)
        yi = y_labels.index(y_val)
        data[yi, xi] = np.mean(vals)

    return data, x_labels, y_labels


def _compute_color_range(*data_arrays: np.ndarray) -> Tuple[float, float]:
    """Compute a shared symmetric color range centered on 0.5."""
    all_vals = np.concatenate([d[~np.isnan(d)] for d in data_arrays if d.size > 0])
    if len(all_vals) == 0:
        return 0.0, 1.0
    max_dev = max(abs(all_vals.max() - 0.5), abs(all_vals.min() - 0.5), 0.05)
    return 0.5 - max_dev, 0.5 + max_dev


def _style_heatmap_ax(ax, x_aspect: str, y_aspect: str) -> None:
    """Apply common styling to a heatmap axis."""
    ax.set_xlabel(x_aspect.capitalize(), fontsize=11)
    ax.set_ylabel(y_aspect.capitalize(), fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="y", rotation=0)


def plot_two_heatmaps(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    display_names: bool = True,
    decimals: int = 2,
    subtitle: str = "",
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create side-by-side heatmaps for P_0(Large) and P_AB(Large).
    Used when nudge is NOT one of the axes.
    """
    # Build data for both metrics
    data_p0, x_labels_p0, y_labels_p0 = build_heatmap_data(
        results, x_aspect, y_aspect, "p0", display_names
    )
    data_pab, x_labels_pab, y_labels_pab = build_heatmap_data(
        results, x_aspect, y_aspect, "pab", display_names
    )

    # Use union of labels for consistent axes
    x_labels = sorted(set(x_labels_p0) | set(x_labels_pab))
    y_labels = sorted(set(y_labels_p0) | set(y_labels_pab))

    # Rebuild data with consistent labels
    def reindex(data, old_x, old_y, new_x, new_y):
        out = np.full((len(new_y), len(new_x)), np.nan)
        for oi, oy in enumerate(old_y):
            for oj, ox in enumerate(old_x):
                ni = new_y.index(oy)
                nj = new_x.index(ox)
                out[ni, nj] = data[oi, oj]
        return out

    data_p0 = reindex(data_p0, x_labels_p0, y_labels_p0, x_labels, y_labels)
    data_pab = reindex(data_pab, x_labels_pab, y_labels_pab, x_labels, y_labels)

    vmin, vmax = _compute_color_range(data_p0, data_pab)

    if figsize is None:
        figsize = (7 + len(x_labels) * 1.2, 2 + len(y_labels) * 0.7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fmt = f".{decimals}f"
    cmap = "RdBu_r"

    # Plot P_0(Large)
    sns.heatmap(
        data_p0,
        ax=ax1,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.5,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "P(Larger Group)"},
        mask=np.isnan(data_p0),
    )
    ax1.set_title(r"$P_0$(Large) — Baseline", fontsize=13, fontweight="bold")
    _style_heatmap_ax(ax1, x_aspect, y_aspect)

    # Plot P_AB(Large)
    sns.heatmap(
        data_pab,
        ax=ax2,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.5,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "P(Larger Group)"},
        mask=np.isnan(data_pab),
    )
    ax2.set_title(r"$P_{AB}$(Large) — Nudged", fontsize=13, fontweight="bold")
    _style_heatmap_ax(ax2, x_aspect, y_aspect)

    fig.suptitle(
        f"Larger-Group Preference ({subtitle})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_nudge_axis_heatmap(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    display_names: bool = True,
    decimals: int = 2,
    subtitle: str = "",
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a single heatmap with a baseline 'none' row/column and per-nudge-type
    rows/columns. Used when nudge IS one of the axes.
    """
    # Determine which axis is nudge
    nudge_is_x = x_aspect == "nudge"
    other_aspect = y_aspect if nudge_is_x else x_aspect

    # Collect per-nudge P_AB data and baseline P_0 data
    # P_AB keyed by (other_val, nudge_type)
    pab_cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    # P_0 keyed by other_val (aggregated over nudge types since baseline is shared)
    p0_cells: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        other_val = get_aspect_value(r, other_aspect, display_names)
        nudge_val = (
            get_nudge_display_name(r.nudge_type) if display_names else r.nudge_type
        )

        # Baseline
        if r.larger_group_rate_base is not None:
            p0_cells[other_val].append(r.larger_group_rate_base)

        # Nudged
        if r.larger_group_rate_A is not None and r.larger_group_rate_B is not None:
            pab = (r.larger_group_rate_A + r.larger_group_rate_B) / 2
            pab_cells[(other_val, nudge_val)].append(pab)

    # Build labels
    other_labels = sorted(set(p0_cells.keys()) | set(k[0] for k in pab_cells.keys()))
    nudge_labels = sorted(set(k[1] for k in pab_cells.keys()))
    # Full nudge axis: baseline first, then nudge types
    full_nudge_labels = [BASELINE_LABEL] + nudge_labels

    n_other = len(other_labels)
    n_nudge = len(full_nudge_labels)

    if nudge_is_x:
        # rows = other (y), cols = nudge (x)
        data = np.full((n_other, n_nudge), np.nan)
        for oi, ov in enumerate(other_labels):
            # Baseline column
            if ov in p0_cells:
                data[oi, 0] = np.mean(p0_cells[ov])
            # Nudge columns
            for ni, nv in enumerate(nudge_labels):
                if (ov, nv) in pab_cells:
                    data[oi, ni + 1] = np.mean(pab_cells[(ov, nv)])
        # x_labels = full_nudge_labels
        # y_labels = other_labels
        # gap_pos = 1  # vertical line after column 1
    else:
        # rows = nudge (y), cols = other (x)
        data = np.full((n_nudge, n_other), np.nan)
        for oi, ov in enumerate(other_labels):
            # Baseline row
            if ov in p0_cells:
                data[0, oi] = np.mean(p0_cells[ov])
            # Nudge rows
            for ni, nv in enumerate(nudge_labels):
                if (ov, nv) in pab_cells:
                    data[ni + 1, oi] = np.mean(pab_cells[(ov, nv)])
        # x_labels = other_labels
        # y_labels = full_nudge_labels
        # gap_pos = 1  # horizontal line after row 1

    # Split data into baseline and nudge parts
    if nudge_is_x:
        data_base = data[:, :1]  # (n_other, 1)
        data_nudge = data[:, 1:]  # (n_other, n_nudge_types)
        base_x_labels = [BASELINE_LABEL]
        nudge_x_labels = nudge_labels
        base_y_labels = other_labels
        nudge_y_labels = other_labels
    else:
        data_base = data[:1, :]  # (1, n_other)
        data_nudge = data[1:, :]  # (n_nudge_types, n_other)
        base_x_labels = other_labels
        nudge_x_labels = other_labels
        base_y_labels = [BASELINE_LABEL]
        nudge_y_labels = nudge_labels

    vmin, vmax = _compute_color_range(data)

    fmt = f".{decimals}f"
    cmap = "RdBu_r"
    heatmap_kws = dict(
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.5,
        linewidths=0.5,
        linecolor="white",
    )

    # Gap ratio between baseline and nudge panels
    gap_ratio = 0.05

    if nudge_is_x:
        # Columns: [baseline(1)] [gap] [nudge types] [colorbar]
        n_base_cols = 1
        n_nudge_cols = len(nudge_labels)
        width_ratios = [n_base_cols, gap_ratio, n_nudge_cols, 0.4]
        fig_w = 3 + (n_base_cols + n_nudge_cols) * 1.2
        fig_h = 2 + n_other * 0.7
        fig = plt.figure(figsize=figsize or (fig_w, fig_h))
        gs = GridSpec(1, 4, figure=fig, width_ratios=width_ratios, wspace=0.05)
        ax_base = fig.add_subplot(gs[0, 0])
        ax_nudge = fig.add_subplot(gs[0, 2])
        ax_cbar = fig.add_subplot(gs[0, 3])

        # Baseline panel
        sns.heatmap(
            data_base,
            ax=ax_base,
            xticklabels=base_x_labels,
            yticklabels=base_y_labels,
            cbar=False,
            mask=np.isnan(data_base),
            **heatmap_kws,
        )
        ax_base.set_ylabel(other_aspect.capitalize(), fontsize=11)
        ax_base.set_xlabel("", fontsize=11)
        ax_base.set_xticklabels(ax_base.get_xticklabels(), rotation=45, ha="right")
        ax_base.tick_params(axis="y", rotation=0)

        # Nudge panel
        sns.heatmap(
            data_nudge,
            ax=ax_nudge,
            xticklabels=nudge_x_labels,
            yticklabels=[],
            cbar_ax=ax_cbar,
            cbar_kws={"label": "P(Larger Group)"},
            mask=np.isnan(data_nudge),
            **heatmap_kws,
        )
        ax_nudge.set_xlabel("Nudge", fontsize=11)
        ax_nudge.set_ylabel("")
        ax_nudge.set_xticklabels(ax_nudge.get_xticklabels(), rotation=45, ha="right")

    else:
        # Rows: [baseline(1)] [gap] [nudge types], plus colorbar on right
        n_base_rows = 1
        n_nudge_rows = len(nudge_labels)
        height_ratios = [n_base_rows, gap_ratio, n_nudge_rows]
        fig_w = 3 + n_other * 1.2
        fig_h = 2 + (n_base_rows + n_nudge_rows) * 0.7
        fig = plt.figure(figsize=figsize or (fig_w, fig_h))
        gs = GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=[1, 0.04],
            hspace=0.05,
            wspace=0.15,
        )
        ax_base = fig.add_subplot(gs[0, 0])
        ax_nudge = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[:, 1])

        # Baseline panel (top)
        sns.heatmap(
            data_base,
            ax=ax_base,
            xticklabels=[],
            yticklabels=base_y_labels,
            cbar=False,
            mask=np.isnan(data_base),
            **heatmap_kws,
        )
        ax_base.set_xlabel("")
        ax_base.set_ylabel("")
        ax_base.tick_params(axis="y", rotation=0)

        # Nudge panel (bottom)
        sns.heatmap(
            data_nudge,
            ax=ax_nudge,
            xticklabels=nudge_x_labels,
            yticklabels=nudge_y_labels,
            cbar_ax=ax_cbar,
            cbar_kws={"label": "P(Larger Group)"},
            mask=np.isnan(data_nudge),
            **heatmap_kws,
        )
        ax_nudge.set_xlabel(other_aspect.capitalize(), fontsize=11)
        ax_nudge.set_ylabel("Nudge", fontsize=11)
        ax_nudge.set_xticklabels(ax_nudge.get_xticklabels(), rotation=45, ha="right")
        ax_nudge.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Larger-Group Preference ({subtitle})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_heatmaps(
    results: List[FrequencyResult],
    x_aspect: str,
    y_aspect: str,
    aggregated_aspect: str,
    display_names: bool = True,
    output_path: Optional[str] = None,
    decimals: int = 2,
    subtitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Dispatch to the appropriate plotting function based on whether nudge is an axis.
    """
    if not results:
        print("No results to plot.")
        return

    all_vals = [r.larger_group_rate for r in results if r.larger_group_rate is not None]
    if not all_vals:
        print("No valid data to plot.")
        return

    sub = subtitle or f"aggregated over {aggregated_aspect}"

    if "nudge" in (x_aspect, y_aspect):
        plot_nudge_axis_heatmap(
            results,
            x_aspect,
            y_aspect,
            display_names=display_names,
            decimals=decimals,
            subtitle=sub,
            output_path=output_path,
            figsize=figsize,
        )
    else:
        plot_two_heatmaps(
            results,
            x_aspect,
            y_aspect,
            display_names=display_names,
            decimals=decimals,
            subtitle=sub,
            output_path=output_path,
            figsize=figsize,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps of larger-group preference rates P(Large).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Model (x) vs Factor (y), aggregating over nudge types
    uv run python -m choices.analysis.plots.larger_group --axes model factor

    # Factor (x) vs Nudge (y), aggregating over models
    uv run python -m choices.analysis.plots.larger_group --axes factor nudge

    # With filters and output
    uv run python -m choices.analysis.plots.larger_group --axes model factor \\
        --results-dirs results results_anthropic -o heatmaps.png
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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (e.g., heatmaps.png). "
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

    args = parser.parse_args()

    x_aspect, y_aspect = args.axes
    if x_aspect == y_aspect:
        parser.error("The two axes must be different aspects.")

    aggregated_aspect = (VALID_AXES - {x_aspect, y_aspect}).pop()

    print(f"Axes: x={x_aspect}, y={y_aspect} (aggregating over {aggregated_aspect})")
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning:
        print(f"Reasoning filter: {args.reasoning}")

    # Compute results (reusing the same pipeline as create_summary)
    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    # Apply reasoning filter
    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]

    # Filter out results without larger group data
    results_with_lg = [r for r in results if r.larger_group_rate is not None]

    print(
        f"Found {len(results)} experiments ({len(results_with_lg)} with larger-group data)"
    )

    if not results_with_lg:
        print("No experiments with larger-group data found.")
        return

    display_names = not args.no_display_names

    # Determine subtitle: if aggregated aspect has a single unique value, show it
    agg_values = sorted(
        set(
            get_aspect_value(r, aggregated_aspect, display_names)
            for r in results_with_lg
        )
    )
    if len(agg_values) == 1:
        subtitle = f"{aggregated_aspect}: {agg_values[0]}"
    else:
        subtitle = f"aggregated over {aggregated_aspect} (n={len(agg_values)})"

    # Determine output path
    ext = "pdf" if args.pdf else "png"
    output_path = args.output
    if output_path is None:
        import os

        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            PLOTS_OUTPUT_DIR,
            f"larger_group_{x_aspect}_vs_{y_aspect}.{ext}",
        )

    figsize = tuple(args.figsize) if args.figsize else None

    plot_heatmaps(
        results_with_lg,
        x_aspect,
        y_aspect,
        aggregated_aspect,
        display_names=display_names,
        output_path=output_path,
        decimals=args.decimals,
        subtitle=subtitle,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
