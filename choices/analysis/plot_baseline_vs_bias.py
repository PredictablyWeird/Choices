#!/usr/bin/env python3
"""
Plot steerability asymmetry for cases without significant baseline preference.

Usage:
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1

    # Custom figsize and no title
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1 \
        --figsize 6 4 --no-title
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.steerability_metric import freq_to_log_odds
from choices.analysis.utils import get_model_display_name


@dataclass
class NormalizedResult:
    """Result normalized so baseline preference is always >= 0.5."""

    baseline_pref: float  # f_0(preferred) - always >= 0.5
    steerability_towards: float  # Steerability when nudging towards baseline pref
    steerability_against: float  # Steerability when nudging against baseline pref
    steerability_asym: (
        float  # normalized (towards - against) / (|towards| + |against| + eps)
    )
    sig_baseline: bool  # Whether baseline preference is significant
    sig_asym: bool  # Whether steerability asymmetry is significant
    model: str
    model_display: str
    reasoning_condition: str
    factor: str
    nudge_type: str


def compute_steerability(f_nudged: float, f_baseline: float) -> float:
    """Compute steerability as change in log odds."""
    return freq_to_log_odds(f_nudged) - freq_to_log_odds(f_baseline)


def normalize_result(r: FrequencyResult) -> NormalizedResult:
    """Normalize a result so baseline preference is always >= 0.5."""
    if r.f_0_B >= 0.5:
        baseline_pref = r.f_0_B
        steer_towards = compute_steerability(r.f_B_B, r.f_0_B)
        steer_against = compute_steerability(1 - r.f_A_B, 1 - r.f_0_B)
    else:
        baseline_pref = 1 - r.f_0_B
        steer_towards = compute_steerability(1 - r.f_A_B, 1 - r.f_0_B)
        steer_against = compute_steerability(r.f_B_B, r.f_0_B)

    # Compute normalized asymmetry using the new formula
    eps = 0.01
    asym = (steer_towards - steer_against) / (
        abs(steer_towards) + abs(steer_against) + eps
    )

    return NormalizedResult(
        baseline_pref=baseline_pref,
        steerability_towards=steer_towards,
        steerability_against=steer_against,
        steerability_asym=asym,
        sig_baseline=r.sig_baseline_B,
        sig_asym=r.sig_asym,
        model=r.model,
        model_display=get_model_display_name(r.model),
        reasoning_condition=r.reasoning_condition,
        factor=r.factor,
        nudge_type=r.nudge_type,
    )


def create_nonsig_analysis_plot(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_title: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    show_mean: bool = True,
    show_median: bool = False,
    n_bins: int = 15,
) -> plt.Figure:
    """
    Create analysis plot for cases WITHOUT significant baseline preference.

    Shows stacked histogram of |steerability bias| with bars split by:
    - Significance (significant vs non-significant)
    - Reasoning condition (reasoning vs non-reasoning)

    Args:
        normalized_results: List of normalized results (filtered to non-sig baseline)
        output_path: Optional path to save the figure
        title: Optional custom title
        show_title: Whether to show title (default True)
        figsize: Figure size
        show_mean: Show mean line
        show_median: Show median line
        n_bins: Number of histogram bins

    Returns:
        The matplotlib Figure object
    """
    if not normalized_results:
        print("No data to plot.")
        return None

    # Categorize results into 4 groups
    REASONING_CONDITIONS = {"low", "medium", "high", "before", "after"}

    sig_reasoning = []
    sig_non_reasoning = []
    nonsig_reasoning = []
    nonsig_non_reasoning = []

    for r in normalized_results:
        abs_bias = abs(r.steerability_asym)
        is_reasoning = r.reasoning_condition in REASONING_CONDITIONS

        if r.sig_asym:
            if is_reasoning:
                sig_reasoning.append(abs_bias)
            else:
                sig_non_reasoning.append(abs_bias)
        else:
            if is_reasoning:
                nonsig_reasoning.append(abs_bias)
            else:
                nonsig_non_reasoning.append(abs_bias)

    # Compute overall statistics
    all_abs_biases = np.abs([r.steerability_asym for r in normalized_results])
    n_total = len(normalized_results)
    n_sig = len(sig_reasoning) + len(sig_non_reasoning)
    n_nonsig = len(nonsig_reasoning) + len(nonsig_non_reasoning)
    pct_sig = 100 * n_sig / n_total
    pct_nonsig = 100 * n_nonsig / n_total

    # Check if we have both reasoning and non-reasoning conditions
    has_reasoning = len(sig_reasoning) + len(nonsig_reasoning) > 0
    has_non_reasoning = len(sig_non_reasoning) + len(nonsig_non_reasoning) > 0
    split_by_reasoning = has_reasoning and has_non_reasoning

    fig, ax = plt.subplots(figsize=figsize)

    # Compute bin edges
    bin_edges = np.linspace(0, max(all_abs_biases) * 1.05, n_bins + 1)

    if split_by_reasoning:
        # 4-way split: sig/nonsig × reasoning/non-reasoning
        color_reasoning = "#009E73"  # Green
        color_non_reasoning = "#E69F00"  # Orange

        hist_nonsig_nonreas, _ = np.histogram(nonsig_non_reasoning, bins=bin_edges)
        hist_nonsig_reas, _ = np.histogram(nonsig_reasoning, bins=bin_edges)
        hist_sig_nonreas, _ = np.histogram(sig_non_reasoning, bins=bin_edges)
        hist_sig_reas, _ = np.histogram(sig_reasoning, bins=bin_edges)

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bottom layer: non-significant, non-reasoning
        ax.bar(
            bin_centers,
            hist_nonsig_nonreas,
            width=bin_width * 0.9,
            color=color_non_reasoning,
            edgecolor="black",
            linewidth=0.5,
        )

        # Second layer: non-significant, reasoning
        ax.bar(
            bin_centers,
            hist_nonsig_reas,
            width=bin_width * 0.9,
            bottom=hist_nonsig_nonreas,
            color=color_reasoning,
            edgecolor="black",
            linewidth=0.5,
        )

        # Third layer: significant, non-reasoning
        ax.bar(
            bin_centers,
            hist_sig_nonreas,
            width=bin_width * 0.9,
            bottom=hist_nonsig_nonreas + hist_nonsig_reas,
            color=color_non_reasoning,
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
        )

        # Top layer: significant, reasoning
        ax.bar(
            bin_centers,
            hist_sig_reas,
            width=bin_width * 0.9,
            bottom=hist_nonsig_nonreas + hist_nonsig_reas + hist_sig_nonreas,
            color=color_reasoning,
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
        )

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color_reasoning, edgecolor="white", label="Reasoning"),
            Patch(
                facecolor=color_non_reasoning, edgecolor="white", label="Non-reasoning"
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="//",
                label=f"Significant ({pct_sig:.1f}%)",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                label=f"Non-significant ({pct_nonsig:.1f}%)",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9
        )

    else:
        # 2-way split: sig/nonsig only
        sig_asyms = [abs(r.steerability_asym) for r in normalized_results if r.sig_asym]
        nonsig_asyms = [
            abs(r.steerability_asym) for r in normalized_results if not r.sig_asym
        ]

        hist_nonsig, _ = np.histogram(nonsig_asyms, bins=bin_edges)
        hist_sig, _ = np.histogram(sig_asyms, bins=bin_edges)

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(
            bin_centers,
            hist_nonsig,
            width=bin_width * 0.9,
            color="#D0D0D0",
            edgecolor="white",
            linewidth=0.5,
            label=f"Non-significant ({pct_nonsig:.1f}%)",
        )

        ax.bar(
            bin_centers,
            hist_sig,
            width=bin_width * 0.9,
            bottom=hist_nonsig,
            color="#7B68EE",
            edgecolor="white",
            linewidth=0.5,
            label=f"Significant ({pct_sig:.1f}%)",
        )

    # Add mean/median lines
    if show_mean:
        mean_val = np.mean(all_abs_biases)
        mean_color = "#D55E00"  # Dark orange
        ax.axvline(mean_val, color=mean_color, linestyle="--", linewidth=2.5)
        y_max = ax.get_ylim()[1]
        ax.text(
            mean_val + 0.02,
            y_max * 0.95,
            f"Mean: {mean_val:.2f}",
            color=mean_color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )

    if show_median:
        median_val = np.median(all_abs_biases)
        median_color = "#0072B2"  # Blue
        ax.axvline(median_val, color=median_color, linestyle="--", linewidth=2.5)
        y_max = ax.get_ylim()[1]
        ax.text(
            median_val + 0.02,
            y_max * 0.85,
            f"Median: {median_val:.2f}",
            color=median_color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )

    ax.set_xlabel("|Steerability Asymmetry|", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    if show_title:
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        else:
            ax.set_title(
                f"Steerability Asymmetry Without Baseline Preference (n={n_total})",
                fontsize=12,
                fontweight="bold",
            )

    # Add legend for non-split case (split case handles its own legend above)
    if not split_by_reasoning:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot steerability asymmetry for non-significant baseline cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m choices.analysis.plot_baseline_vs_bias \\
        --results-dirs results_main0 results_main1

    uv run python -m choices.analysis.plot_baseline_vs_bias \\
        --results-dirs results_main0 results_main1 \\
        --figsize 6 4 --no-title
        """,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to search",
    )
    parser.add_argument(
        "--factors", nargs="+", default=None, help="List of factors to include"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="List of models to include"
    )
    parser.add_argument(
        "--nudge-types", nargs="+", default=None, help="List of nudge types to include"
    )
    parser.add_argument(
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="Filter by reasoning conditions (e.g., 'none', 'off', 'before', 'low')",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path"
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    parser.add_argument(
        "--no-title", action="store_true", help="Suppress the plot title"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        help="Figure size (width height)",
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument(
        "--n-bins", type=int, default=15, help="Number of histogram bins"
    )
    parser.add_argument(
        "--show-median", action="store_true", help="Show median line on plot"
    )
    parser.add_argument("--no-mean", action="store_true", help="Hide mean line on plot")

    args = parser.parse_args()

    output_path = args.output or "nonsig_baseline_analysis.pdf"

    print("=" * 70)
    print("Steerability Asymmetry Analysis (Non-significant Baseline)")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning_conditions:
        print(f"Reasoning filter: {args.reasoning_conditions}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

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

    print(f"Found {len(results)} result(s)")

    if args.reasoning_conditions:
        results = [
            r for r in results if r.reasoning_condition in args.reasoning_conditions
        ]
        print(f"After reasoning filter: {len(results)} result(s)")

    if not results:
        print("No results after filtering.")
        return

    print("Normalizing results...")
    normalized = [normalize_result(r) for r in results]
    print(f"Normalized {len(normalized)} result(s)")

    # Filter to non-significant baseline cases only
    nonsig_baseline = [r for r in normalized if not r.sig_baseline]
    print(f"Cases without significant baseline preference: {len(nonsig_baseline)}")
    print()

    if not nonsig_baseline:
        print("No cases without significant baseline preference.")
        return

    # Stats for non-significant baseline cases
    nonsig_asyms = np.array([r.steerability_asym for r in nonsig_baseline])
    nonsig_sig_asym = sum(1 for r in nonsig_baseline if r.sig_asym)
    print("Summary Statistics:")
    print("-" * 50)
    print(f"  N experiments: {len(nonsig_baseline)}")
    print(
        f"  |Asym|: mean={np.mean(np.abs(nonsig_asyms)):.3f}, "
        f"median={np.median(np.abs(nonsig_asyms)):.3f}"
    )
    print(
        f"  With significant steerability asymmetry: {nonsig_sig_asym} "
        f"({100*nonsig_sig_asym/len(nonsig_baseline):.1f}%)"
    )
    print()

    # Create plot
    figsize = tuple(args.figsize) if args.figsize else (8, 6)
    fig = create_nonsig_analysis_plot(
        nonsig_baseline,
        output_path=output_path,
        title=args.title,
        show_title=not args.no_title,
        figsize=figsize,
        show_mean=not args.no_mean,
        show_median=args.show_median,
        n_bins=args.n_bins,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
