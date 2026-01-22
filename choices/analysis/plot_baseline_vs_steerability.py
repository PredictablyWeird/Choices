#!/usr/bin/env python3
"""
Plot baseline bias vs steerability bias scatter plot.

This script creates a scatter plot showing the relationship between
baseline bias (without contextual influence) and steerability bias,
with (model, nudge_type) combinations as different points.

Usage:
    # Discover all models and nudge types from results directories
    python plot_baseline_vs_steerability.py --category gender

    # Specify results directories
    python plot_baseline_vs_steerability.py --category gender \
        --results-dirs results results2

    # Filter by models and/or nudge types
    python plot_baseline_vs_steerability.py --category gender \
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \
        --nudge-types survey_preference weak_evidence
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt

from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_counts,
)
from choices.analysis.utils import (
    compute_factor_frequencies_with_counts,
    get_factor_levels,
    get_model_color,
    get_model_display_name,
    get_nudge_marker,
    get_reasoning_condition,
)


def discover_experiments_for_category(
    results_base_dirs: List[str],
    category: str,
    model_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Discover all available experiments for a specific category.

    Args:
        results_base_dirs: List of base directories for results
        category: The category/factor to filter for
        model_filter: Optional list of models to include (None = all)
        nudge_type_filter: Optional list of nudge types to include (None = all)

    Returns:
        List of (results_dir, model, nudge_type) tuples
    """
    experiments = []
    experiment_name = f"simple_{category}"

    for results_base_dir in results_base_dirs:
        results_path = Path(results_base_dir)
        if not results_path.exists():
            continue

        # Look for the experiment directory
        exp_dir = results_path / experiment_name
        if not exp_dir.exists() or not exp_dir.is_dir():
            continue

        # Iterate through model directories
        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name

            # Apply model filter
            if model_filter and model not in model_filter:
                continue

            # Iterate through nudge type directories
            for nudge_dir in model_dir.iterdir():
                if not nudge_dir.is_dir():
                    continue

                nudge_type = nudge_dir.name

                # Apply nudge type filter
                if nudge_type_filter and nudge_type not in nudge_type_filter:
                    continue

                experiments.append((results_base_dir, model, nudge_type))

    return experiments


def get_available_models_and_nudges(
    results_base_dirs: List[str],
    category: str,
) -> Tuple[Set[str], Set[str]]:
    """
    Get all available models and nudge types for a category.

    Returns:
        Tuple of (set of models, set of nudge types)
    """
    models = set()
    nudge_types = set()

    experiments = discover_experiments_for_category(results_base_dirs, category)
    for _, model, nudge_type in experiments:
        models.add(model)
        nudge_types.add(nudge_type)

    return models, nudge_types


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load results from a simple_nudging experiment directory."""
    results_path = Path(results_dir)
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        return None

    with open(graph_files[0], "r") as f:
        graph_data = json.load(f)

    return graph_data


def find_base_result_directory(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[str]:
    """Find the base (no-nudge) result directory."""
    experiment_name = f"simple_{factor_name}"

    # First, try to find base in the nudge directory (new location)
    nudge_path = Path(results_base_dir) / experiment_name / model / nudge_type
    if nudge_path.exists():
        base_dirs = [
            d for d in nudge_path.iterdir() if d.is_dir() and d.name.endswith("_base")
        ]
        if base_dirs:
            most_recent = max(base_dirs, key=lambda d: d.stat().st_mtime)
            return str(most_recent)

    # Fall back to legacy "base" directory
    base_path = Path(results_base_dir) / experiment_name / model / "base"
    if not base_path.exists():
        return None

    result_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not result_dirs:
        return None

    most_recent = max(result_dirs, key=lambda d: d.stat().st_mtime)
    return str(most_recent)


def find_nudge_result_directory(
    factor_name: str,
    model: str,
    nudge_type: str,
    target_group: str,
    results_base_dir: str = "results",
) -> Optional[str]:
    """Find the result directory for a specific nudge condition."""
    experiment_name = f"simple_{factor_name}"
    nudge_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not nudge_path.exists():
        return None

    # Find directories with matching target_group
    for result_dir in nudge_path.iterdir():
        if not result_dir.is_dir():
            continue
        if result_dir.name.endswith("_base"):
            continue

        # Load nudge config to check target_group
        graph_files = list(result_dir.glob("preference_graph_*.json"))
        if not graph_files:
            continue

        with open(graph_files[0], "r") as f:
            graph_data = json.load(f)

        nudge_config = graph_data.get("nudge_config", {})
        if nudge_config.get("target_group") == target_group:
            return str(result_dir)

    return None


def compute_baseline_and_steerability_bias(
    category: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[Dict[str, float]]:
    """
    Compute baseline bias and steerability bias for a model/category/nudge combination.

    Returns:
        Dictionary with 'baseline_bias' and 'steerability_bias' or None if data not available.
    """
    level_A, level_B = get_factor_levels(category)
    if level_A is None:
        print(f"  Warning: Unknown category '{category}'")
        return None

    target_levels = [level_A, level_B]

    # Load base condition
    base_dir = find_base_result_directory(category, model, nudge_type, results_base_dir)
    if not base_dir:
        print(f"  Warning: No base results for {model}/{category}/{nudge_type}")
        return None

    base_data = load_results(base_dir)
    if not base_data:
        print(
            f"  Warning: Could not load base data for {model}/{category}/{nudge_type}"
        )
        return None

    # Load nudge conditions
    nudge_A_dir = find_nudge_result_directory(
        category, model, nudge_type, level_A, results_base_dir
    )
    nudge_B_dir = find_nudge_result_directory(
        category, model, nudge_type, level_B, results_base_dir
    )

    if not nudge_A_dir or not nudge_B_dir:
        print(
            f"  Warning: Missing nudge results for {model}/{category}/{nudge_type} "
            f"(A={nudge_A_dir is not None}, B={nudge_B_dir is not None})"
        )
        return None

    nudge_A_data = load_results(nudge_A_dir)
    nudge_B_data = load_results(nudge_B_dir)

    if not nudge_A_data or not nudge_B_data:
        print(
            f"  Warning: Could not load nudge data for {model}/{category}/{nudge_type}"
        )
        return None

    # Compute frequencies and counts
    base_stats = compute_factor_frequencies_with_counts(
        base_data, category, target_levels
    )
    nudge_A_stats = compute_factor_frequencies_with_counts(
        nudge_A_data, category, target_levels
    )
    nudge_B_stats = compute_factor_frequencies_with_counts(
        nudge_B_data, category, target_levels
    )

    f_0_A = base_stats.get(level_A, {}).get("freq", 0.5)
    f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
    c_0_A = base_stats.get(level_A, {}).get("wins", 0)
    c_0_B = base_stats.get(level_B, {}).get("wins", 0)

    f_A_A = nudge_A_stats.get(level_A, {}).get("freq", 0.5)
    f_A_B = nudge_A_stats.get(level_B, {}).get("freq", 0.5)
    c_A_A = nudge_A_stats.get(level_A, {}).get("wins", 0)
    c_A_B = nudge_A_stats.get(level_B, {}).get("wins", 0)

    f_B_A = nudge_B_stats.get(level_A, {}).get("freq", 0.5)
    f_B_B = nudge_B_stats.get(level_B, {}).get("freq", 0.5)
    c_B_A = nudge_B_stats.get(level_A, {}).get("wins", 0)
    c_B_B = nudge_B_stats.get(level_B, {}).get("wins", 0)

    # Compute baseline bias: deviation from 50% for level_B
    # Positive means biased towards level_B (consistent with steerability_bias)
    baseline_bias = f_0_B - 0.5

    # Compute steerability bias using counts with Haldane-Anscombe correction
    # Positive steerability_bias means more steerable towards B
    steer_A, steer_B, steerability_bias = compute_steerability_bias_from_counts(
        c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B
    )

    if steerability_bias is None:
        # Find which frequencies are at the boundary (0 or 1)
        boundary_issues = []
        if f_0_A < 0.01 or f_0_A > 0.99:
            boundary_issues.append(f"f_0({level_A})={f_0_A:.1%}")
        if f_0_B < 0.01 or f_0_B > 0.99:
            boundary_issues.append(f"f_0({level_B})={f_0_B:.1%}")
        if f_A_A < 0.01 or f_A_A > 0.99:
            boundary_issues.append(f"f_{level_A}({level_A})={f_A_A:.1%}")
        if f_A_B < 0.01 or f_A_B > 0.99:
            boundary_issues.append(f"f_{level_A}({level_B})={f_A_B:.1%}")
        if f_B_A < 0.01 or f_B_A > 0.99:
            boundary_issues.append(f"f_{level_B}({level_A})={f_B_A:.1%}")
        if f_B_B < 0.01 or f_B_B > 0.99:
            boundary_issues.append(f"f_{level_B}({level_B})={f_B_B:.1%}")

        if boundary_issues:
            print(
                f"  Warning: Saturated frequencies for {model}/{category}/{nudge_type}: "
                f"{', '.join(boundary_issues)}"
            )
        else:
            print(
                f"  Warning: Could not compute steerability bias for {model}/{category}/{nudge_type}"
            )
        return None

    return {
        "baseline_bias": baseline_bias,
        "steerability_bias": steerability_bias,
        "f_0_A": f_0_A,
        "f_0_B": f_0_B,
        "steerability_A": steer_A,
        "steerability_B": steer_B,
        "level_A": level_A,
        "level_B": level_B,
    }


def create_scatter_plot(
    category: str,
    experiments: List[Tuple[str, str, str]],
    results_base_dirs: List[str],
    output_path: str = None,
    title: str = None,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (10, 8),
):
    """
    Create a scatter plot of baseline bias vs steerability bias.

    Points represent (model, nudge_type) combinations for a single category.
    Color indicates model, marker indicates nudge type.

    Args:
        category: The category/factor to plot
        experiments: List of (results_dir, model, nudge_type) tuples to plot
        results_base_dirs: List of base result directories (for fallback lookups)
        output_path: Optional path to save the figure
        title: Optional custom title
        show_legend: Whether to show the legend
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (figure, data_points) or None if no data found
    """
    # Collect data points
    data_points = []

    for results_base_dir, model, nudge_type in experiments:
        print(f"Processing: {model} / {category} / {nudge_type}")
        result = compute_baseline_and_steerability_bias(
            category, model, nudge_type, results_base_dir
        )

        if result:
            # Find the base result directory to get reasoning condition
            base_dir = find_base_result_directory(
                category, model, nudge_type, results_base_dir
            )
            base_dir_path = Path(base_dir) if base_dir else None
            reasoning_condition = get_reasoning_condition(model, base_dir_path)

            data_points.append(
                {
                    "model": model,
                    "category": category,
                    "nudge_type": nudge_type,
                    "results_dir": results_base_dir,
                    "reasoning_condition": reasoning_condition,
                    **result,
                }
            )

    if not data_points:
        print(
            "\nNo data points found! Check that results exist for the specified combinations."
        )
        return None

    print(f"\nCollected {len(data_points)} data points")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color by model, marker by nudge type
    for point in data_points:
        model = point["model"]
        nudge_type = point["nudge_type"]

        color = get_model_color(model)
        marker = get_nudge_marker(nudge_type)

        ax.scatter(
            point["baseline_bias"],
            point["steerability_bias"],
            c=color,
            marker=marker,
            s=150,
            alpha=0.8,
            edgecolors="white",
            linewidths=1.5,
        )

    # Add reference lines
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Get factor levels for axis labels
    level_A, level_B = get_factor_levels(category)

    # Labels and title
    ax.set_xlabel(f"Baseline Bias (+ → {level_B})", fontsize=14)
    ax.set_ylabel(f"Steerability Bias (+ → {level_B})", fontsize=14)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        ax.set_title(
            f"Baseline vs Steerability Bias\n(Category: {category.replace('_', ' ').title()})",
            fontsize=16,
            fontweight="bold",
        )

    # Legend - create separate sections for models (colors) and nudge types (markers)
    if show_legend:
        from matplotlib.lines import Line2D

        legend_handles = []
        legend_labels = []

        # Add model entries (colors) - include reasoning condition
        # Group by model to get unique (model, reasoning) pairs
        unique_model_reasoning = list(
            dict.fromkeys((p["model"], p["reasoning_condition"]) for p in data_points)
        )
        for model, reasoning in unique_model_reasoning:
            color = get_model_color(model)
            display_name = get_model_display_name(model)
            label = f"{display_name} ({reasoning})"
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=label,
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        # Add separator
        legend_handles.append(Line2D([0], [0], color="none"))
        legend_labels.append("")

        # Add nudge type entries (markers)
        unique_nudges = list(dict.fromkeys(p["nudge_type"] for p in data_points))
        for nudge_type in unique_nudges:
            marker = get_nudge_marker(nudge_type)
            label = nudge_type.replace("_", " ").title()
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markeredgecolor="gray",
                markersize=10,
                label=label,
            )
            legend_handles.append(handle)
            legend_labels.append(label)

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            framealpha=0.9,
        )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved plot to: {output_path}")

    return fig, data_points


def main():
    parser = argparse.ArgumentParser(
        description="Create scatter plot of baseline bias vs steerability bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all models and nudge types from results directories
    python plot_baseline_vs_steerability.py --category gender

    # Specify results directories
    python plot_baseline_vs_steerability.py --category gender \\
        --results-dirs results results2

    # Filter by models and/or nudge types
    python plot_baseline_vs_steerability.py --category gender \\
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \\
        --nudge-types survey_preference weak_evidence
        """,
    )

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category/factor to plot (e.g., gender, wealth, age_group)",
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
        "--results-dirs",
        nargs="+",
        default=["results"],
        help="List of results directories to search (default: results)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: baseline_vs_steerability_{category}.pdf)",
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
        default=[10, 8],
        help="Figure size (width height)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"baseline_vs_steerability_{args.category}.pdf"

    # Discover experiments
    print("=" * 70)
    print("Baseline Bias vs Steerability Bias Scatter Plot")
    print("=" * 70)
    print(f"Category: {args.category}")
    print(f"Results directories: {args.results_dirs}")

    # Discover available models and nudge types
    available_models, available_nudges = get_available_models_and_nudges(
        args.results_dirs, args.category
    )

    if not available_models:
        print(f"\nNo experiments found for category '{args.category}'")
        print(f"Searched in: {args.results_dirs}")
        return

    # Apply filters
    model_filter = args.models
    nudge_filter = args.nudge_types

    if model_filter:
        print(f"Model filter: {model_filter}")
    else:
        print(f"Models (discovered): {sorted(available_models)}")

    if nudge_filter:
        print(f"Nudge type filter: {nudge_filter}")
    else:
        print(f"Nudge types (discovered): {sorted(available_nudges)}")

    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Get experiments matching filters
    experiments = discover_experiments_for_category(
        args.results_dirs,
        args.category,
        model_filter=model_filter,
        nudge_type_filter=nudge_filter,
    )

    if not experiments:
        print("No experiments found matching the filters.")
        return

    print(f"Found {len(experiments)} experiment(s) to process\n")

    # Create plot
    result = create_scatter_plot(
        category=args.category,
        experiments=experiments,
        results_base_dirs=args.results_dirs,
        output_path=output_path,
        title=args.title,
        figsize=tuple(args.figsize),
    )

    if result is None:
        return

    fig, data_points = result

    if data_points:
        print("\n" + "=" * 70)
        print("Data Summary")
        print("=" * 70)
        print(
            f"{'Model':<30} {'Reasoning':<10} {'Nudge Type':<20} {'Base Bias':>12} {'Steer Bias':>12}"
        )
        print("-" * 90)
        for dp in sorted(data_points, key=lambda x: (x["model"], x["nudge_type"])):
            print(
                f"{get_model_display_name(dp['model']):<30} "
                f"{dp['reasoning_condition']:<10} "
                f"{dp['nudge_type']:<20} "
                f"{dp['baseline_bias']:>+12.3f} "
                f"{dp['steerability_bias']:>+12.3f}"
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
