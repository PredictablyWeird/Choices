#!/usr/bin/env python3
"""
Plot nudge effect sizes by nudge type.

Shows the average effect magnitude for each type of nudge, aggregated across
all models and factors.

Effect size = P(target | nudge towards target) - P(target | baseline)
This measures how much the nudge shifts preferences toward the target option.

By default, uses magnitude (absolute value) of effect sizes, which shows
how much the nudge can shift preferences regardless of direction.
Use --signed to show signed effect sizes instead.

Usage:
    uv run python -m choices.analysis.plot_nudge_effects_by_type
    uv run python -m choices.analysis.plot_nudge_effects_by_type --output nudge_effects.pdf
    uv run python -m choices.analysis.plot_nudge_effects_by_type --signed  # Use signed effect sizes
    uv run python -m choices.analysis.plot_nudge_effects_by_type --results-dir results --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.nudge_effect_size import (
    aggregate_effect_sizes_by_nudge_type,
    compute_all_effect_sizes,
)


# Nudge type display names and categories for better visualization
NUDGE_CATEGORIES = {
    "evidence": {
        "nudges": [
            "weak_evidence",
            "strong_evidence",
            "survey_preference",
            "expert_recommendation",
        ],
        "color": "#2E86AB",  # Steel blue
        "label": "Evidence-based",
    },
    "pressure": {
        "nudges": ["emotional", "identity", "user_preference", "social_norm"],
        "color": "#A23B72",  # Magenta
        "label": "Pressure-based",
    },
    "direct": {
        "nudges": ["always_save", "moral_imperative"],
        "color": "#F18F01",  # Orange
        "label": "Direct instruction",
    },
    "other": {
        "nudges": ["few_shot_3", "few_shot_5"],
        "color": "#C73E1D",  # Red
        "label": "Few-shot examples",
    },
}

# Nice display names for nudge types
NUDGE_DISPLAY_NAMES = {
    "weak_evidence": "Weak Evidence",
    "strong_evidence": "Strong Evidence",
    "survey_preference": "Survey Preference",
    "expert_recommendation": "Expert Rec.",
    "emotional": "Emotional",
    "identity": "Identity",
    "user_preference": "User Preference",
    "social_norm": "Social Norm",
    "always_save": "Direct Instruction",
    "moral_imperative": "Moral Imperative",
    "few_shot_3": "Few-shot (3)",
    "few_shot_5": "Few-shot (5)",
}


def get_category_for_nudge(nudge_type: str) -> str:
    """Get the category name for a nudge type."""
    for category, info in NUDGE_CATEGORIES.items():
        if nudge_type in info["nudges"]:
            return category
    return "other"


def get_color_for_nudge(nudge_type: str) -> str:
    """Get the color for a nudge type based on its category."""
    category = get_category_for_nudge(nudge_type)
    return NUDGE_CATEGORIES.get(category, NUDGE_CATEGORIES["other"])["color"]


def plot_nudge_effect_sizes(
    results_base_dir: str = "results",
    output_path: str | None = None,
    show: bool = False,
    figsize: tuple = (12, 7),
    use_signed: bool = False,
) -> None:
    """
    Create a bar plot showing average effect size by nudge type.

    Args:
        results_base_dir: Base directory for results
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively
        figsize: Figure size (width, height) in inches
        use_signed: If False (default), use magnitude (absolute value) of effect sizes.
                    If True, use signed effect sizes.
    """
    # Compute all effect sizes
    print("Computing effect sizes from results...")
    effect_sizes = compute_all_effect_sizes(results_base_dir)

    if not effect_sizes:
        print("No effect size data found. Make sure experiments have been run.")
        return

    print(f"Found {len(effect_sizes)} experiment results")

    # Aggregate by nudge type (magnitude by default)
    aggregated = aggregate_effect_sizes_by_nudge_type(
        effect_sizes, use_magnitude=not use_signed
    )

    if not aggregated:
        print("No aggregated data available.")
        return

    # Sort nudge types by category and then by effect size within category
    def sort_key(nudge_type):
        category_order = {"evidence": 0, "pressure": 1, "direct": 2, "other": 3}
        category = get_category_for_nudge(nudge_type)
        return (
            category_order.get(category, 4),
            -aggregated[nudge_type]["avg_effect_size"],
        )

    sorted_nudge_types = sorted(aggregated.keys(), key=sort_key)

    # Prepare data for plotting
    nudge_names = [NUDGE_DISPLAY_NAMES.get(nt, nt) for nt in sorted_nudge_types]
    avg_effects = [aggregated[nt]["avg_effect_size"] for nt in sorted_nudge_types]
    std_effects = [aggregated[nt]["std_effect_size"] for nt in sorted_nudge_types]
    n_experiments = [aggregated[nt]["n_experiments"] for nt in sorted_nudge_types]
    colors = [get_color_for_nudge(nt) for nt in sorted_nudge_types]

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars
    x = np.arange(len(nudge_names))
    bars = ax.bar(
        x,
        avg_effects,
        yerr=std_effects,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        capsize=4,
        alpha=0.85,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.7},
    )

    # Add value labels on bars
    for i, (bar, n) in enumerate(zip(bars, n_experiments)):
        height = bar.get_height()
        label_y = (
            height + std_effects[i] + 0.005
            if height >= 0
            else height - std_effects[i] - 0.02
        )
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"{height:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, label_y),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )
        # Add sample size below bar
        ax.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, -0.01),
            ha="center",
            va="top",
            fontsize=7,
            color="#666666",
        )

    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(nudge_names, rotation=35, ha="right", fontsize=10)

    # Labels depend on whether using magnitude or signed values
    if use_signed:
        ylabel = "Average Effect Size\n(signed change in selection probability)"
        title = (
            "Nudge Effect Size by Type (Signed)\n(averaged across models and factors)"
        )
    else:
        ylabel = "Average Effect Magnitude\n(|change in selection probability|)"
        title = "Nudge Effect Magnitude by Type\n(averaged across models and factors)"

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Nudge Type", fontsize=11)

    # Add zero line
    ax.axhline(y=0, color="#333333", linestyle="-", linewidth=0.8, alpha=0.5)

    # Set y-axis limits with some padding
    # Lower bound: only go below 0 if there are actually negative values
    max_val = max(avg_effects) + max(std_effects) + 0.05
    min_effect_with_error = min(e - s for e, s in zip(avg_effects, std_effects))
    if min_effect_with_error >= 0:
        min_val = -0.02  # Small padding below 0 for the zero line visibility
    else:
        min_val = min_effect_with_error - 0.02
    ax.set_ylim(min_val, max_val)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add title
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Add legend for categories
    from matplotlib.patches import Patch

    legend_elements = []
    seen_categories = set()
    for nt in sorted_nudge_types:
        cat = get_category_for_nudge(nt)
        if cat not in seen_categories:
            seen_categories.add(cat)
            cat_info = NUDGE_CATEGORIES[cat]
            legend_elements.append(
                Patch(facecolor=cat_info["color"], label=cat_info["label"])
            )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.95,
        fontsize=9,
    )

    # Adjust layout
    plt.tight_layout()

    # Add summary statistics as text
    total_experiments = sum(n_experiments)
    unique_models = len(set(es.model for es in effect_sizes))
    unique_factors = len(set(es.factor for es in effect_sizes))

    summary_text = (
        f"Total: {total_experiments} experiments, "
        f"{unique_models} models, {unique_factors} factors"
    )
    fig.text(
        0.99,
        0.01,
        summary_text,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    # Save or show
    if output_path:
        output_file = Path(output_path)
        plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {output_file}")

    if show:
        plt.show()

    if not output_path and not show:
        # Default: save to current directory
        default_output = "nudge_effect_sizes_by_type.pdf"
        plt.savefig(default_output, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {default_output}")

    plt.close()


def print_summary_table(
    results_base_dir: str = "results", use_signed: bool = False
) -> None:
    """Print a summary table of effect sizes by nudge type."""
    effect_sizes = compute_all_effect_sizes(results_base_dir)

    if not effect_sizes:
        print("No effect size data found.")
        return

    aggregated = aggregate_effect_sizes_by_nudge_type(
        effect_sizes, use_magnitude=not use_signed
    )

    # Sort by average effect size
    sorted_nudges = sorted(
        aggregated.items(), key=lambda x: x[1]["avg_effect_size"], reverse=True
    )

    metric_label = "Effect Size (Signed)" if use_signed else "Effect Magnitude"

    print("\n" + "=" * 70)
    print(f"NUDGE {metric_label.upper()} SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Nudge Type':<25} {'Avg ' + metric_label[:6]:>12} {'Std Dev':>12} {'N':>8}"
    )
    print("-" * 70)

    for nudge_type, stats in sorted_nudges:
        display_name = NUDGE_DISPLAY_NAMES.get(nudge_type, nudge_type)
        # Use + sign only for signed values
        fmt = "+11.1%" if use_signed else "11.1%"
        print(
            f"{display_name:<25} "
            f"{stats['avg_effect_size']:{fmt}} "
            f"{stats['std_effect_size']:>12.1%} "
            f"{stats['n_experiments']:>8}"
        )

    print("-" * 70)

    # Overall statistics
    if use_signed:
        all_effects = [es.avg_effect_size for es in effect_sizes]
    else:
        all_effects = [es.avg_effect_magnitude for es in effect_sizes]
    overall_mean = sum(all_effects) / len(all_effects)
    fmt = "+11.1%" if use_signed else "11.1%"
    print(f"\n{'Overall Mean':<25} {overall_mean:{fmt}}")
    print(f"{'Total Experiments':<25} {len(effect_sizes):>12}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot nudge effect sizes by nudge type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m choices.analysis.plot_nudge_effects_by_type
    python -m choices.analysis.plot_nudge_effects_by_type --output effects.pdf --show
    python -m choices.analysis.plot_nudge_effects_by_type --results-dir results --summary
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory for results (default: results)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the plot (default: nudge_effect_sizes_by_type.pdf)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table to console",
    )

    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use signed effect sizes instead of magnitude (default: magnitude)",
    )

    parser.add_argument(
        "--figsize",
        type=str,
        default="12,7",
        help="Figure size as 'width,height' in inches (default: 12,7)",
    )

    args = parser.parse_args()

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(",")))

    if args.summary:
        print_summary_table(args.results_dir, use_signed=args.signed)

    plot_nudge_effect_sizes(
        results_base_dir=args.results_dir,
        output_path=args.output,
        show=args.show,
        figsize=figsize,
        use_signed=args.signed,
    )


if __name__ == "__main__":
    main()
