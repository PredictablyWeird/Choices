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
from choices.analysis.metrics import compute_asym, freq_to_log_odds
from choices.analysis.utils import get_model_display_name, PLOTS_OUTPUT_DIR


@dataclass
class NormalizedResult:
    """Result normalized so baseline preference is always >= 0.5."""

    baseline_pref: float  # f_0(preferred) - always >= 0.5
    steerability_towards: float  # Steerability when nudging towards baseline pref
    steerability_against: float  # Steerability when nudging against baseline pref
    steerability_asym: float  # towards - against (non-normalized)
    sig_baseline: bool  # Whether baseline preference is significant
    sig_asym: bool  # Whether steerability asymmetry is significant
    sig_any_nudge: bool  # Whether at least one nudge effect is significant
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

    # Compute asymmetry: positive = easier towards baseline preference
    asym = compute_asym(steer_against, steer_towards)

    return NormalizedResult(
        baseline_pref=baseline_pref,
        steerability_towards=steer_towards,
        steerability_against=steer_against,
        steerability_asym=asym,
        sig_baseline=r.sig_baseline_B,
        sig_asym=r.sig_asym,
        sig_any_nudge=r.sig_A or r.sig_B,
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
    show_line: Optional[str] = None,
    n_bins: int = 15,
) -> plt.Figure:
    """
    Create analysis plot for cases WITHOUT significant baseline preference.

    Shows stacked histogram of |steerability bias| with bars split by:
    - Significance (significant vs non-significant)
    - Reasoning condition (reasoning vs non-reasoning)

    Hatching patterns:
    - "/" for significant asymmetry

    Args:
        normalized_results: List of normalized results (filtered to non-sig baseline)
        output_path: Optional path to save the figure
        title: Optional custom title
        show_title: Whether to show title (default True)
        figsize: Figure size
        show_line: Which line to show: 'mean', 'median', or None (default)
        n_bins: Number of histogram bins

    Returns:
        The matplotlib Figure object
    """
    from matplotlib.patches import Patch

    if not normalized_results:
        print("No data to plot.")
        return None

    # Categorize results by reasoning condition and sig_asym only
    REASONING_CONDITIONS = {"low", "medium", "high", "before", "after"}

    # Categories: (reasoning, sig_asym)
    categories = {
        (False, False): [],  # non-reasoning, non-sig
        (False, True): [],  # non-reasoning, sig_asym
        (True, False): [],  # reasoning, non-sig
        (True, True): [],  # reasoning, sig_asym
    }

    for r in normalized_results:
        abs_bias = abs(r.steerability_asym)
        is_reasoning = r.reasoning_condition in REASONING_CONDITIONS
        key = (is_reasoning, r.sig_asym)
        categories[key].append(abs_bias)

    # Compute overall statistics
    all_abs_biases = np.abs([r.steerability_asym for r in normalized_results])
    n_total = len(normalized_results)
    n_sig_asym = sum(1 for r in normalized_results if r.sig_asym)
    n_nonsig_asym = n_total - n_sig_asym
    # n_sig_nudge = sum(1 for r in normalized_results if r.sig_any_nudge)
    # n_sig_both = sum(1 for r in normalized_results if r.sig_asym and r.sig_any_nudge)
    pct_sig_asym = 100 * n_sig_asym / n_total
    pct_nonsig_asym = 100 * n_nonsig_asym / n_total

    # Check if we have both reasoning and non-reasoning conditions
    has_reasoning = any(len(v) > 0 for k, v in categories.items() if k[0])
    has_non_reasoning = any(len(v) > 0 for k, v in categories.items() if not k[0])
    split_by_reasoning = has_reasoning and has_non_reasoning

    fig, ax = plt.subplots(figsize=figsize)

    # Compute bin edges
    bin_edges = np.linspace(0, max(all_abs_biases) * 1.05, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define hatch patterns (dense for visibility)
    def get_hatch(sig_asym: bool) -> str:
        if sig_asym:
            return "///"
        else:
            return ""

    if split_by_reasoning:
        color_reasoning = "#009E73"  # Green
        color_non_reasoning = "#E69F00"  # Orange

        # Stack order: non-reasoning first (bottom), then reasoning
        # Within each: non-sig first, then sig
        stack_order = [
            # Non-reasoning
            ((False, False), color_non_reasoning),
            ((False, True), color_non_reasoning),
            # Reasoning
            ((True, False), color_reasoning),
            ((True, True), color_reasoning),
        ]

        bottom = np.zeros(len(bin_centers))
        for key, color in stack_order:
            data = categories[key]
            if not data:
                continue
            hist, _ = np.histogram(data, bins=bin_edges)
            is_reasoning, sig_asym = key
            hatch = get_hatch(sig_asym)
            ax.bar(
                bin_centers,
                hist,
                width=bin_width * 0.9,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            bottom = bottom + hist

        # Create custom legend
        legend_elements = [
            Patch(facecolor=color_reasoning, edgecolor="white", label="Reasoning"),
            Patch(
                facecolor=color_non_reasoning, edgecolor="white", label="Non-reasoning"
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label=f"Significant ({pct_sig_asym:.1f}%)",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                label=f"Non-significant ({pct_nonsig_asym:.1f}%)",
            ),
        ]

        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9
        )

    else:
        # Non-split case: just stack by sig_asym
        stack_order = [False, True]  # non-sig first, then sig

        bottom = np.zeros(len(bin_centers))
        for sig_asym in stack_order:
            data = [
                abs(r.steerability_asym)
                for r in normalized_results
                if r.sig_asym == sig_asym
            ]
            if not data:
                continue
            hist, _ = np.histogram(data, bins=bin_edges)
            hatch = get_hatch(sig_asym)
            color = "#7B68EE" if sig_asym else "#D0D0D0"
            ax.bar(
                bin_centers,
                hist,
                width=bin_width * 0.9,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            bottom = bottom + hist

        # Create legend
        legend_elements = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label=f"Significant ({pct_sig_asym:.1f}%)",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                label=f"Non-significant ({pct_nonsig_asym:.1f}%)",
            ),
        ]

        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9
        )

    # Always show mean line with text annotation
    mean_val = np.mean(all_abs_biases)
    ax.axvline(mean_val, color="#D55E00", linestyle="--", linewidth=2.5)
    # Add bold orange text showing mean value
    y_max = ax.get_ylim()[1]
    ax.text(
        mean_val + 0.02,
        y_max * 0.95,
        f"Mean: {mean_val:.2f}",
        color="#D55E00",
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Optionally also show median line if requested
    if show_line == "median":
        median_val = np.median(all_abs_biases)
        ax.axvline(median_val, color="#0072B2", linestyle="--", linewidth=2.5)

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
        "--show-line",
        type=str,
        choices=["mean", "median"],
        default=None,
        help="Show a vertical line for mean or median (default: none)",
    )
    args = parser.parse_args()

    output_path = args.output or f"{PLOTS_OUTPUT_DIR}/nonsig_baseline_analysis.pdf"

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
    n_total = len(nonsig_baseline)
    n_sig_asym = sum(1 for r in nonsig_baseline if r.sig_asym)
    n_sig_nudge = sum(1 for r in nonsig_baseline if r.sig_any_nudge)
    n_sig_both = sum(1 for r in nonsig_baseline if r.sig_asym and r.sig_any_nudge)
    n_sig_any = sum(1 for r in nonsig_baseline if r.sig_asym or r.sig_any_nudge)
    print("Summary Statistics:")
    print("-" * 50)
    print(f"  N experiments: {n_total}")
    print(
        f"  |Asym|: mean={np.mean(np.abs(nonsig_asyms)):.3f}, "
        f"median={np.median(np.abs(nonsig_asyms)):.3f}"
    )
    print(
        f"  With significant steerability asymmetry: {n_sig_asym} "
        f"({100*n_sig_asym/n_total:.1f}%)"
    )
    print(
        f"  With significant nudge effect (any): {n_sig_nudge} "
        f"({100*n_sig_nudge/n_total:.1f}%)"
    )
    print(
        f"  With any significant effect: {n_sig_any} " f"({100*n_sig_any/n_total:.1f}%)"
    )
    print(
        f"  With both (intersection): {n_sig_both} " f"({100*n_sig_both/n_total:.1f}%)"
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
        show_line=args.show_line,
        n_bins=args.n_bins,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
