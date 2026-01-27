#!/usr/bin/env python3
"""
Scatter plot: Baseline Preference vs Steerability Bias.

This plot directly shows whether baseline preferences predict steerability
asymmetries. Each point represents one experiment (model × factor × nudge type).

Key questions this plot answers:
1. Is there a correlation between baseline preference and steerability bias?
2. At baseline ≈ 0.5 (no preference), is there still systematic bias?
3. Are there "intrinsic" steerability asymmetries independent of preferences?

Usage:
    # Basic usage
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1

    # Filter by reasoning conditions
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1 \
        --reasoning-conditions none off

    # Color by model
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1 \
        --color-by model

    # Color by factor
    uv run python -m choices.analysis.plot_baseline_vs_bias \
        --results-dirs results_main0 results_main1 \
        --color-by factor
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    steerability_bias: float  # towards - against (positive = easier towards pref)
    sig_baseline: bool  # Whether baseline preference is significant
    sig_bias: bool  # Whether steerability bias is significant
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

    return NormalizedResult(
        baseline_pref=baseline_pref,
        steerability_towards=steer_towards,
        steerability_against=steer_against,
        steerability_bias=steer_towards - steer_against,
        sig_baseline=r.sig_baseline_B,
        sig_bias=r.sig_bias,
        model=r.model,
        model_display=get_model_display_name(r.model),
        reasoning_condition=r.reasoning_condition,
        factor=r.factor,
        nudge_type=r.nudge_type,
    )


def create_scatter_plot(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
    color_by: Optional[str] = None,
    show_regression: bool = True,
    show_ci_band: bool = True,
    annotate_intercept: bool = True,
) -> plt.Figure:
    """
    Create scatter plot of baseline preference vs steerability bias.

    Args:
        normalized_results: List of normalized results
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size
        color_by: Color points by 'model', 'factor', 'nudge_type', or None
        show_regression: Show regression line
        show_ci_band: Show 95% CI band around regression
        annotate_intercept: Annotate the y-intercept (bias at baseline=0.5)

    Returns:
        The matplotlib Figure object
    """
    if not normalized_results:
        print("No data to plot.")
        return None

    baselines = np.array([r.baseline_pref for r in normalized_results])
    biases = np.array([r.steerability_bias for r in normalized_results])

    fig, ax = plt.subplots(figsize=figsize)

    # Determine coloring
    if color_by == "model":
        categories = [
            f"{r.model_display} ({r.reasoning_condition})" for r in normalized_results
        ]
    elif color_by == "factor":
        categories = [r.factor for r in normalized_results]
    elif color_by == "nudge_type":
        categories = [r.nudge_type for r in normalized_results]
    else:
        categories = None

    if categories:
        unique_cats = sorted(set(categories))
        # Use a colormap
        cmap = plt.colormaps["tab10" if len(unique_cats) <= 10 else "tab20"]
        colors = {cat: cmap(i / len(unique_cats)) for i, cat in enumerate(unique_cats)}

        for cat in unique_cats:
            mask = np.array([c == cat for c in categories])
            ax.scatter(
                baselines[mask],
                biases[mask],
                alpha=0.6,
                s=40,
                color=colors[cat],
                label=cat,
                edgecolors="white",
                linewidths=0.5,
            )

        # Legend outside plot
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            framealpha=0.9,
        )
    else:
        ax.scatter(
            baselines,
            biases,
            alpha=0.5,
            s=40,
            color="#7B68EE",
            edgecolors="white",
            linewidths=0.5,
        )

    # Regression analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(baselines, biases)

    if show_regression:
        x_line = np.linspace(0.5, 1.0, 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            "r-",
            linewidth=2.5,
            zorder=10,
            label=f"Regression: r={r_value:.3f}, p={p_value:.2g}",
        )

        if show_ci_band:
            # Compute prediction interval
            n = len(baselines)
            x_mean = np.mean(baselines)
            ss_x = np.sum((baselines - x_mean) ** 2)
            se_y = np.sqrt(
                np.sum((biases - (slope * baselines + intercept)) ** 2) / (n - 2)
            )

            # Standard error of prediction at each x
            se_pred = se_y * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)

            # 95% CI
            t_crit = stats.t.ppf(0.975, n - 2)
            ax.fill_between(
                x_line,
                y_line - t_crit * se_pred,
                y_line + t_crit * se_pred,
                color="red",
                alpha=0.15,
                zorder=5,
            )

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=1)

    # Annotate intercept (bias at baseline = 0.5)
    if annotate_intercept and show_regression:
        intercept_at_05 = slope * 0.5 + intercept
        # Test if intercept differs from 0
        # Use bootstrap to get CI for intercept at x=0.5
        n_bootstrap = 1000
        bootstrap_intercepts = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(baselines), size=len(baselines), replace=True)
            b_slope, b_intercept, _, _, _ = stats.linregress(
                baselines[idx], biases[idx]
            )
            bootstrap_intercepts.append(b_slope * 0.5 + b_intercept)

        ci_low = np.percentile(bootstrap_intercepts, 2.5)
        ci_high = np.percentile(bootstrap_intercepts, 97.5)
        sig_marker = "*" if ci_low > 0 or ci_high < 0 else ""

        ax.plot(
            0.5,
            intercept_at_05,
            "ro",
            markersize=10,
            zorder=15,
            markeredgecolor="black",
        )
        ax.annotate(
            f"Bias at 0.5: {intercept_at_05:.2f}\n95% CI: [{ci_low:.2f}, {ci_high:.2f}]{sig_marker}",
            xy=(0.5, intercept_at_05),
            xytext=(0.55, intercept_at_05 + 0.3),
            fontsize=10,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9
            ),
        )

    # Labels
    ax.set_xlabel(
        "Baseline Preference\n(frequency of preferred option at baseline)", fontsize=12
    )
    ax.set_ylabel(
        "Steerability Bias\n(steerability towards pref. − against pref.)", fontsize=12
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            "Does Baseline Preference Predict Steerability Asymmetry?",
            fontsize=14,
            fontweight="bold",
        )

    # Axis limits
    ax.set_xlim(0.45, 1.02)

    # Interpretation annotations
    ax.annotate(
        "Positive bias:\nEasier to steer towards\nbaseline preference",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Add regression stats in corner if not using legend for colors
    if not categories:
        stats_text = (
            f"n = {len(baselines)}\n"
            f"r = {r_value:.3f}\n"
            f"p = {p_value:.3g}\n"
            f"slope = {slope:.3f}"
        )
        ax.annotate(
            stats_text,
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def create_no_baseline_analysis(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
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
    # NON_REASONING_CONDITIONS = {"none", "off"}

    sig_reasoning = []
    sig_non_reasoning = []
    nonsig_reasoning = []
    nonsig_non_reasoning = []

    for r in normalized_results:
        abs_bias = abs(r.steerability_bias)
        is_reasoning = r.reasoning_condition in REASONING_CONDITIONS

        if r.sig_bias:
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
    all_abs_biases = np.abs([r.steerability_bias for r in normalized_results])
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
        # Color: orange = reasoning, green = non-reasoning
        # Hatching: // with black = significant, solid = non-significant
        color_reasoning = "#E69F00"  # Orange
        color_non_reasoning = "#009E73"  # Green

        hist_nonsig_nonreas, _ = np.histogram(nonsig_non_reasoning, bins=bin_edges)
        hist_nonsig_reas, _ = np.histogram(nonsig_reasoning, bins=bin_edges)
        hist_sig_nonreas, _ = np.histogram(sig_non_reasoning, bins=bin_edges)
        hist_sig_reas, _ = np.histogram(sig_reasoning, bins=bin_edges)

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bottom layer: non-significant, non-reasoning (green, solid)
        ax.bar(
            bin_centers,
            hist_nonsig_nonreas,
            width=bin_width * 0.9,
            color=color_non_reasoning,
            edgecolor="black",
            linewidth=0.5,
        )

        # Second layer: non-significant, reasoning (orange, solid)
        ax.bar(
            bin_centers,
            hist_nonsig_reas,
            width=bin_width * 0.9,
            bottom=hist_nonsig_nonreas,
            color=color_reasoning,
            edgecolor="black",
            linewidth=0.5,
        )

        # Third layer: significant, non-reasoning (green, // hatch with black)
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

        # Top layer: significant, reasoning (orange, // hatch with black)
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
        # Color = reasoning condition, Hatching = significance
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
        sig_biases = [
            abs(r.steerability_bias) for r in normalized_results if r.sig_bias
        ]
        nonsig_biases = [
            abs(r.steerability_bias) for r in normalized_results if not r.sig_bias
        ]

        hist_nonsig, _ = np.histogram(nonsig_biases, bins=bin_edges)
        hist_sig, _ = np.histogram(sig_biases, bins=bin_edges)

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
        # Add text annotation next to the line
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

    ax.set_xlabel("|Steerability Bias|", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

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


def create_comparison_plot(
    sig_baseline_results: List[NormalizedResult],
    nonsig_baseline_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Create comparison plot: significant vs non-significant baseline preference.

    Shows side-by-side comparison of |steerability bias| distribution.

    Args:
        sig_baseline_results: Results with significant baseline preference
        nonsig_baseline_results: Results without significant baseline preference
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size

    Returns:
        The matplotlib Figure object
    """
    if not sig_baseline_results and not nonsig_baseline_results:
        print("No data to plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    sig_biases = (
        np.abs([r.steerability_bias for r in sig_baseline_results])
        if sig_baseline_results
        else []
    )
    nonsig_biases = (
        np.abs([r.steerability_bias for r in nonsig_baseline_results])
        if nonsig_baseline_results
        else []
    )

    # Box plot comparison
    data = []
    labels = []
    colors = []

    if nonsig_baseline_results:
        data.append(nonsig_biases)
        labels.append(f"No Baseline Pref.\n(n={len(nonsig_biases)})")
        colors.append("#A0A0A0")

    if sig_baseline_results:
        data.append(sig_biases)
        labels.append(f"Sig. Baseline Pref.\n(n={len(sig_biases)})")
        colors.append("#457B9D")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points with jitter
    for i, (d, color) in enumerate(zip(data, colors), 1):
        jitter = np.random.uniform(-0.15, 0.15, len(d))
        ax.scatter(
            np.full(len(d), i) + jitter, d, color=color, alpha=0.4, s=20, zorder=3
        )

    # Statistical test
    if len(sig_biases) > 0 and len(nonsig_biases) > 0:
        t_stat, p_val = stats.mannwhitneyu(
            sig_biases, nonsig_biases, alternative="two-sided"
        )
        ax.annotate(
            f"Mann-Whitney U test:\np = {p_val:.3g}",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    ax.set_ylabel("|Steerability Bias|", fontsize=12)
    ax.set_title(
        title or "Bias Magnitude: With vs Without Baseline Preference",
        fontsize=14,
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
        description="Scatter plot: Baseline preference vs steerability bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m choices.analysis.plot_baseline_vs_bias \\
        --results-dirs results_main0 results_main1

    uv run python -m choices.analysis.plot_baseline_vs_bias \\
        --results-dirs results_main0 results_main1 \\
        --reasoning-conditions none off \\
        --color-by model
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
        "--color-by",
        choices=["model", "factor", "nudge_type"],
        default=None,
        help="Color points by this attribute",
    )
    parser.add_argument(
        "--no-regression", action="store_true", help="Hide regression line"
    )
    parser.add_argument(
        "--no-ci", action="store_true", help="Hide confidence interval band"
    )
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Don't annotate intercept at baseline=0.5",
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
        "--sig-baseline-only",
        action="store_true",
        help="Only include cases with significant baseline preference",
    )
    parser.add_argument(
        "--show-nonsig-analysis",
        action="store_true",
        help="Show additional analysis for non-significant baseline cases",
    )

    args = parser.parse_args()

    output_path = args.output or "baseline_vs_bias.pdf"

    print("=" * 70)
    print("Baseline Preference vs Steerability Bias")
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
    if args.color_by:
        print(f"Color by: {args.color_by}")
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

    # Split by baseline significance
    sig_baseline = [r for r in normalized if r.sig_baseline]
    nonsig_baseline = [r for r in normalized if not r.sig_baseline]
    print(f"  With significant baseline pref: {len(sig_baseline)}")
    print(f"  Without significant baseline pref: {len(nonsig_baseline)}")
    print()

    # Choose which results to use for main plot
    if args.sig_baseline_only:
        plot_results = sig_baseline
        print("Using only cases with significant baseline preference for main plot.")
    else:
        plot_results = normalized
        print("Using all cases for main plot.")
    print()

    if not plot_results:
        print("No results to plot after filtering.")
        return

    # Statistics
    baselines = np.array([r.baseline_pref for r in plot_results])
    biases = np.array([r.steerability_bias for r in plot_results])

    print("Summary Statistics (for main plot):")
    print("-" * 50)
    print(f"  N experiments: {len(plot_results)}")
    print(
        f"  Baseline preference: mean={np.mean(baselines):.3f}, "
        f"std={np.std(baselines):.3f}"
    )
    print(
        f"  Steerability bias:   mean={np.mean(biases):.3f}, "
        f"std={np.std(biases):.3f}"
    )

    # Test if mean bias differs from 0
    t_stat, t_pval = stats.ttest_1samp(biases, 0)
    print(f"  Bias ≠ 0: t={t_stat:.3f}, p={t_pval:.3g}")

    # Correlation
    r, p = stats.pearsonr(baselines, biases)
    print(f"  Correlation (baseline vs bias): r={r:.3f}, p={p:.3g}")

    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(baselines, biases)
    print(
        f"  Regression: slope={slope:.3f} (SE={std_err:.3f}), intercept={intercept:.3f}"
    )
    print(f"  Bias at baseline=0.5: {slope * 0.5 + intercept:.3f}")
    print()

    # Stats for non-significant baseline cases
    if nonsig_baseline:
        print("Cases WITHOUT significant baseline preference:")
        print("-" * 50)
        nonsig_biases = np.array([r.steerability_bias for r in nonsig_baseline])
        nonsig_sig_bias = sum(1 for r in nonsig_baseline if r.sig_bias)
        print(f"  N experiments: {len(nonsig_baseline)}")
        print(
            f"  |Bias|: mean={np.mean(np.abs(nonsig_biases)):.3f}, "
            f"median={np.median(np.abs(nonsig_biases)):.3f}"
        )
        print(
            f"  With significant steerability bias: {nonsig_sig_bias} "
            f"({100*nonsig_sig_bias/len(nonsig_baseline):.1f}%)"
        )
        print()

    # Create main scatter plot
    figsize = tuple(args.figsize) if args.figsize else (10, 7)
    fig = create_scatter_plot(
        plot_results,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        color_by=args.color_by,
        show_regression=not args.no_regression,
        show_ci_band=not args.no_ci,
        annotate_intercept=not args.no_intercept and not args.sig_baseline_only,
    )

    # Create additional analysis plots if requested
    if args.show_nonsig_analysis and nonsig_baseline:
        # Plot for non-significant baseline cases
        nonsig_output = (
            output_path.replace(".pdf", "_nonsig_analysis.pdf")
            if output_path
            else "nonsig_baseline_analysis.pdf"
        )
        create_no_baseline_analysis(
            nonsig_baseline,
            output_path=nonsig_output,
        )

        # Comparison plot
        comparison_output = (
            output_path.replace(".pdf", "_comparison.pdf")
            if output_path
            else "baseline_comparison.pdf"
        )
        create_comparison_plot(
            sig_baseline,
            nonsig_baseline,
            output_path=comparison_output,
        )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
