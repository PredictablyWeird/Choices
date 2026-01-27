#!/usr/bin/env python3
"""
Plot steerability as a function of baseline preference.

Creates a two-line plot showing how steerability differs when nudging
towards vs against the baseline preference, as a function of baseline
preference strength.

Key insight: If nudges merely amplify existing preferences, we expect:
- Higher steerability when nudging towards baseline preference
- Lower steerability when nudging against baseline preference
- This asymmetry should grow with baseline preference strength

If there are "intrinsic" steerability asymmetries, we'd see them even
when baseline preference is near 0.5 (no clear preference).

Usage:
    # Basic usage
    uv run python -m choices.analysis.plot_steerability_by_baseline \
        --results-dirs results_main0 results_main1

    # Filter by reasoning conditions
    uv run python -m choices.analysis.plot_steerability_by_baseline \
        --results-dirs results_main0 results_main1 \
        --reasoning-conditions none off

    # Use binned x-axis instead of continuous
    uv run python -m choices.analysis.plot_steerability_by_baseline \
        --results-dirs results_main0 results_main1 \
        --bins 5

    # Save to file
    uv run python -m choices.analysis.plot_steerability_by_baseline \
        --results-dirs results_main0 results_main1 \
        --output steerability_by_baseline.pdf
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


@dataclass
class NormalizedResult:
    """Result normalized so baseline preference is always >= 0.5."""

    baseline_pref: float  # f_0(preferred) - always >= 0.5
    steerability_towards: float  # Steerability when nudging towards baseline pref
    steerability_against: float  # Steerability when nudging against baseline pref
    sig_towards: bool  # Significance of nudge towards
    sig_against: bool  # Significance of nudge against
    sig_baseline: bool  # Whether baseline preference is significant
    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str


def compute_steerability(f_nudged: float, f_baseline: float) -> float:
    """
    Compute steerability as change in log odds.

    steerability = log_odds(f_nudged) - log_odds(f_baseline)

    Positive = nudging increased frequency of target option.
    """
    return freq_to_log_odds(f_nudged) - freq_to_log_odds(f_baseline)


def normalize_result(r: FrequencyResult) -> NormalizedResult:
    """
    Normalize a result so baseline preference is always towards B (>= 0.5).

    If baseline prefers A (f_0_B < 0.5), we swap labels:
    - "preferred" becomes A
    - "steerability towards" = steerability towards A
    - "steerability against" = steerability towards B
    """
    if r.f_0_B >= 0.5:
        # B is preferred at baseline
        baseline_pref = r.f_0_B
        # Steerability towards B (the preferred option)
        steer_towards = compute_steerability(r.f_B_B, r.f_0_B)
        # Steerability towards A (against preference) - measure freq of A
        steer_against = compute_steerability(1 - r.f_A_B, 1 - r.f_0_B)
        sig_towards = r.sig_B
        sig_against = r.sig_A
    else:
        # A is preferred at baseline - swap everything
        baseline_pref = 1 - r.f_0_B  # f_0(A) = 1 - f_0(B)
        # Steerability towards A (the preferred option)
        steer_towards = compute_steerability(1 - r.f_A_B, 1 - r.f_0_B)
        # Steerability towards B (against preference)
        steer_against = compute_steerability(r.f_B_B, r.f_0_B)
        sig_towards = r.sig_A
        sig_against = r.sig_B

    return NormalizedResult(
        baseline_pref=baseline_pref,
        steerability_towards=steer_towards,
        steerability_against=steer_against,
        sig_towards=sig_towards,
        sig_against=sig_against,
        sig_baseline=r.sig_baseline_B,
        model=r.model,
        reasoning_condition=r.reasoning_condition,
        factor=r.factor,
        nudge_type=r.nudge_type,
    )


def create_line_plot(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    n_bins: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_scatter: bool = True,
) -> plt.Figure:
    """
    Create a two-line plot showing steerability vs baseline preference.
    """
    if not normalized_results:
        print("No data to plot.")
        return None

    # Extract data
    baselines = np.array([r.baseline_pref for r in normalized_results])
    steer_towards = np.array([r.steerability_towards for r in normalized_results])
    steer_against = np.array([r.steerability_against for r in normalized_results])

    # Colors
    color_towards = "#457B9D"  # Blue - nudging with preference
    color_against = "#E63946"  # Red - nudging against preference

    fig, ax = plt.subplots(figsize=figsize)

    if n_bins is not None:
        # Binned version
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baselines, quantiles)
        bin_edges = np.unique(bin_edges)

        bin_centers, mean_towards, mean_against = [], [], []
        se_towards, se_against, n_per_bin = [], [], []

        for i in range(len(bin_edges) - 1):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            if i == len(bin_edges) - 2:
                mask = (baselines >= lower) & (baselines <= upper)
            else:
                mask = (baselines >= lower) & (baselines < upper)

            if mask.sum() > 0:
                bin_centers.append((lower + upper) / 2)
                mean_towards.append(np.mean(steer_towards[mask]))
                mean_against.append(np.mean(steer_against[mask]))
                se_towards.append(
                    stats.sem(steer_towards[mask]) if mask.sum() > 1 else 0
                )
                se_against.append(
                    stats.sem(steer_against[mask]) if mask.sum() > 1 else 0
                )
                n_per_bin.append(mask.sum())

        bin_centers = np.array(bin_centers)
        mean_towards = np.array(mean_towards)
        mean_against = np.array(mean_against)
        se_towards = np.array(se_towards)
        se_against = np.array(se_against)

        ax.fill_between(
            bin_centers,
            mean_towards - 1.96 * se_towards,
            mean_towards + 1.96 * se_towards,
            color=color_towards,
            alpha=0.2,
        )
        ax.fill_between(
            bin_centers,
            mean_against - 1.96 * se_against,
            mean_against + 1.96 * se_against,
            color=color_against,
            alpha=0.2,
        )

        ax.plot(
            bin_centers,
            mean_towards,
            "o-",
            color=color_towards,
            linewidth=2,
            markersize=8,
            label="Nudge towards baseline pref.",
        )
        ax.plot(
            bin_centers,
            mean_against,
            "o-",
            color=color_against,
            linewidth=2,
            markersize=8,
            label="Nudge against baseline pref.",
        )

        for i, (x, n) in enumerate(zip(bin_centers, n_per_bin)):
            y_pos = min(mean_towards[i], mean_against[i]) - 0.15
            ax.annotate(f"n={n}", (x, y_pos), ha="center", fontsize=8, color="gray")
    else:
        # Smooth continuous version
        if show_scatter:
            ax.scatter(baselines, steer_towards, color=color_towards, alpha=0.3, s=20)
            ax.scatter(baselines, steer_against, color=color_against, alpha=0.3, s=20)

        n_smooth_bins = 20
        smooth_edges = np.linspace(baselines.min(), baselines.max(), n_smooth_bins + 1)

        smooth_x, smooth_towards, smooth_against = [], [], []
        smooth_se_towards, smooth_se_against = [], []

        for i in range(n_smooth_bins):
            lower, upper = smooth_edges[i], smooth_edges[i + 1]
            if i == n_smooth_bins - 1:
                mask = (baselines >= lower) & (baselines <= upper)
            else:
                mask = (baselines >= lower) & (baselines < upper)

            if mask.sum() >= 3:
                smooth_x.append((lower + upper) / 2)
                smooth_towards.append(np.mean(steer_towards[mask]))
                smooth_against.append(np.mean(steer_against[mask]))
                smooth_se_towards.append(stats.sem(steer_towards[mask]))
                smooth_se_against.append(stats.sem(steer_against[mask]))

        smooth_x = np.array(smooth_x)
        smooth_towards = np.array(smooth_towards)
        smooth_against = np.array(smooth_against)
        smooth_se_towards = np.array(smooth_se_towards)
        smooth_se_against = np.array(smooth_se_against)

        ax.fill_between(
            smooth_x,
            smooth_towards - 1.96 * smooth_se_towards,
            smooth_towards + 1.96 * smooth_se_towards,
            color=color_towards,
            alpha=0.2,
        )
        ax.fill_between(
            smooth_x,
            smooth_against - 1.96 * smooth_se_against,
            smooth_against + 1.96 * smooth_se_against,
            color=color_against,
            alpha=0.2,
        )

        ax.plot(
            smooth_x,
            smooth_towards,
            "-",
            color=color_towards,
            linewidth=2.5,
            label="Nudge towards baseline pref.",
        )
        ax.plot(
            smooth_x,
            smooth_against,
            "-",
            color=color_against,
            linewidth=2.5,
            label="Nudge against baseline pref.",
        )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel(
        "Baseline Preference Strength\n(frequency of preferred option)", fontsize=12
    )
    ax.set_ylabel("Steerability\n(Δ log odds)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            "Steerability by Baseline Preference\n"
            "(Does nudging amplify existing preferences?)",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def create_scatter_plot(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Create a scatter plot showing steerability bias vs baseline preference.
    """
    if not normalized_results:
        print("No data to plot.")
        return None

    baselines = np.array([r.baseline_pref for r in normalized_results])
    steer_towards = np.array([r.steerability_towards for r in normalized_results])
    steer_against = np.array([r.steerability_against for r in normalized_results])
    steer_bias = steer_towards - steer_against

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(baselines, steer_bias, alpha=0.5, s=30, color="#7B68EE")

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        baselines, steer_bias
    )
    x_line = np.linspace(baselines.min(), baselines.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line, "r-", linewidth=2, label=f"r={r_value:.3f}, p={p_value:.3g}"
    )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel(
        "Baseline Preference Strength\n(frequency of preferred option)", fontsize=12
    )
    ax.set_ylabel("Steerability Bias\n(towards pref. minus against pref.)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            "Steerability Bias vs Baseline Preference", fontsize=14, fontweight="bold"
        )

    ax.annotate(
        "Above 0: Easier to steer\ntowards baseline preference",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def create_combined_plot(
    normalized_results: List[NormalizedResult],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    n_bins: Optional[int] = None,
    figsize: Tuple[float, float] = (14, 5),
    show_scatter: bool = True,
) -> plt.Figure:
    """
    Create a combined plot with both views side by side.
    """
    if not normalized_results:
        print("No data to plot.")
        return None

    baselines = np.array([r.baseline_pref for r in normalized_results])
    steer_towards = np.array([r.steerability_towards for r in normalized_results])
    steer_against = np.array([r.steerability_against for r in normalized_results])
    steer_bias = steer_towards - steer_against

    color_towards = "#457B9D"
    color_against = "#E63946"
    color_bias = "#7B68EE"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # === Left panel: Two-line plot ===
    if n_bins is not None:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baselines, quantiles)
        bin_edges = np.unique(bin_edges)

        bin_centers, mean_towards, mean_against = [], [], []
        se_towards, se_against, n_per_bin = [], [], []

        for i in range(len(bin_edges) - 1):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            if i == len(bin_edges) - 2:
                mask = (baselines >= lower) & (baselines <= upper)
            else:
                mask = (baselines >= lower) & (baselines < upper)

            if mask.sum() > 0:
                bin_centers.append((lower + upper) / 2)
                mean_towards.append(np.mean(steer_towards[mask]))
                mean_against.append(np.mean(steer_against[mask]))
                se_towards.append(
                    stats.sem(steer_towards[mask]) if mask.sum() > 1 else 0
                )
                se_against.append(
                    stats.sem(steer_against[mask]) if mask.sum() > 1 else 0
                )
                n_per_bin.append(mask.sum())

        bin_centers = np.array(bin_centers)
        mean_towards = np.array(mean_towards)
        mean_against = np.array(mean_against)
        se_towards = np.array(se_towards)
        se_against = np.array(se_against)

        ax1.fill_between(
            bin_centers,
            mean_towards - 1.96 * se_towards,
            mean_towards + 1.96 * se_towards,
            color=color_towards,
            alpha=0.2,
        )
        ax1.fill_between(
            bin_centers,
            mean_against - 1.96 * se_against,
            mean_against + 1.96 * se_against,
            color=color_against,
            alpha=0.2,
        )

        ax1.plot(
            bin_centers,
            mean_towards,
            "o-",
            color=color_towards,
            linewidth=2,
            markersize=8,
            label="Nudge towards pref.",
        )
        ax1.plot(
            bin_centers,
            mean_against,
            "o-",
            color=color_against,
            linewidth=2,
            markersize=8,
            label="Nudge against pref.",
        )

        for i, (x, n) in enumerate(zip(bin_centers, n_per_bin)):
            y_pos = min(mean_towards[i], mean_against[i]) - 0.1
            ax1.annotate(f"n={n}", (x, y_pos), ha="center", fontsize=8, color="gray")
    else:
        if show_scatter:
            ax1.scatter(baselines, steer_towards, color=color_towards, alpha=0.3, s=15)
            ax1.scatter(baselines, steer_against, color=color_against, alpha=0.3, s=15)

        n_smooth_bins = 15
        smooth_edges = np.linspace(baselines.min(), baselines.max(), n_smooth_bins + 1)

        smooth_x, smooth_towards, smooth_against = [], [], []
        smooth_se_towards, smooth_se_against = [], []

        for i in range(n_smooth_bins):
            lower, upper = smooth_edges[i], smooth_edges[i + 1]
            mask = (
                (baselines >= lower) & (baselines <= upper)
                if i == n_smooth_bins - 1
                else (baselines >= lower) & (baselines < upper)
            )
            if mask.sum() >= 3:
                smooth_x.append((lower + upper) / 2)
                smooth_towards.append(np.mean(steer_towards[mask]))
                smooth_against.append(np.mean(steer_against[mask]))
                smooth_se_towards.append(stats.sem(steer_towards[mask]))
                smooth_se_against.append(stats.sem(steer_against[mask]))

        smooth_x = np.array(smooth_x)
        smooth_towards = np.array(smooth_towards)
        smooth_against = np.array(smooth_against)
        smooth_se_towards = np.array(smooth_se_towards)
        smooth_se_against = np.array(smooth_se_against)

        ax1.fill_between(
            smooth_x,
            smooth_towards - 1.96 * smooth_se_towards,
            smooth_towards + 1.96 * smooth_se_towards,
            color=color_towards,
            alpha=0.2,
        )
        ax1.fill_between(
            smooth_x,
            smooth_against - 1.96 * smooth_se_against,
            smooth_against + 1.96 * smooth_se_against,
            color=color_against,
            alpha=0.2,
        )

        ax1.plot(
            smooth_x,
            smooth_towards,
            "-",
            color=color_towards,
            linewidth=2.5,
            label="Nudge towards pref.",
        )
        ax1.plot(
            smooth_x,
            smooth_against,
            "-",
            color=color_against,
            linewidth=2.5,
            label="Nudge against pref.",
        )

    ax1.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Baseline Preference", fontsize=11)
    ax1.set_ylabel("Steerability (Δ log odds)", fontsize=11)
    ax1.set_title("Steerability by Direction", fontsize=12, fontweight="bold")
    ax1.set_xlim(0.45, 1.0)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # === Right panel: Scatter plot of bias ===
    ax2.scatter(baselines, steer_bias, alpha=0.5, s=25, color=color_bias)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        baselines, steer_bias
    )
    x_line = np.linspace(baselines.min(), baselines.max(), 100)
    y_line = slope * x_line + intercept
    ax2.plot(
        x_line, y_line, "r-", linewidth=2, label=f"r={r_value:.3f}, p={p_value:.2g}"
    )

    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Baseline Preference", fontsize=11)
    ax2.set_ylabel("Steerability Bias", fontsize=11)
    ax2.set_title("Bias vs Baseline", fontsize=12, fontweight="bold")
    ax2.set_xlim(0.45, 1.0)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.3)

    ax2.annotate(
        "Above 0: Easier to steer\ntowards baseline pref.",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    else:
        fig.suptitle(
            "Does Nudging Amplify Existing Preferences?",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot steerability as a function of baseline preference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m choices.analysis.plot_steerability_by_baseline \\
        --results-dirs results_main0 results_main1

    uv run python -m choices.analysis.plot_steerability_by_baseline \\
        --results-dirs results_main0 results_main1 \\
        --reasoning-conditions none off
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
        help="List of factors to include (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to include (default: all)",
    )
    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="List of nudge types to include (default: all)",
    )
    parser.add_argument(
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="Filter by reasoning conditions (e.g., 'none', 'off', 'before', 'low')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: steerability_by_baseline.pdf)",
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of bins for x-axis (default: continuous)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["line", "scatter", "combined"],
        default="combined",
        help="Type of plot: 'line', 'scatter', 'combined' (default)",
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Hide individual data points in line plot",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )
    parser.add_argument(
        "--sig-baseline-only",
        action="store_true",
        help="Only include cases with significant baseline preference",
    )

    args = parser.parse_args()

    output_path = args.output or f"steerability_by_baseline_{args.plot_type}.pdf"

    print("=" * 70)
    print("Steerability by Baseline Preference")
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
    if args.sig_baseline_only:
        print("Sig baseline only: Yes")
    print(f"Plot type: {args.plot_type}")
    print(f"Bins: {args.bins or 'continuous'}")
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

    # Filter by baseline significance if requested
    if args.sig_baseline_only:
        n_before = len(normalized)
        normalized = [r for r in normalized if r.sig_baseline]
        print(
            f"After sig-baseline filter: {len(normalized)} result(s) (removed {n_before - len(normalized)})"
        )

    if not normalized:
        print("No results after filtering.")
        return

    print()

    baselines = np.array([r.baseline_pref for r in normalized])
    steer_towards = np.array([r.steerability_towards for r in normalized])
    steer_against = np.array([r.steerability_against for r in normalized])
    steer_bias = steer_towards - steer_against

    print("Summary Statistics:")
    print("-" * 50)
    print(
        f"  Baseline preference: mean={np.mean(baselines):.3f}, "
        f"min={np.min(baselines):.3f}, max={np.max(baselines):.3f}"
    )
    print(
        f"  Steer towards pref:  mean={np.mean(steer_towards):.3f}, "
        f"std={np.std(steer_towards):.3f}"
    )
    print(
        f"  Steer against pref:  mean={np.mean(steer_against):.3f}, "
        f"std={np.std(steer_against):.3f}"
    )
    print(
        f"  Steerability bias:   mean={np.mean(steer_bias):.3f}, "
        f"std={np.std(steer_bias):.3f}"
    )

    r, p = stats.pearsonr(baselines, steer_bias)
    print(f"  Correlation (baseline vs bias): r={r:.3f}, p={p:.3g}")
    print()

    if args.plot_type == "line":
        figsize = tuple(args.figsize) if args.figsize else (10, 6)
        fig = create_line_plot(
            normalized,
            output_path=output_path,
            title=args.title,
            n_bins=args.bins,
            figsize=figsize,
            show_scatter=not args.no_scatter,
        )
    elif args.plot_type == "scatter":
        figsize = tuple(args.figsize) if args.figsize else (10, 6)
        fig = create_scatter_plot(
            normalized, output_path=output_path, title=args.title, figsize=figsize
        )
    else:
        figsize = tuple(args.figsize) if args.figsize else (14, 5)
        fig = create_combined_plot(
            normalized,
            output_path=output_path,
            title=args.title,
            n_bins=args.bins,
            figsize=figsize,
            show_scatter=not args.no_scatter,
        )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
