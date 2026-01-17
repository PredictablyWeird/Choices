#!/usr/bin/env python3
"""
Plot baseline bias vs steerability bias scatter plot.

This script creates a scatter plot showing the relationship between
baseline bias (without contextual influence) and steerability bias,
with different models as dots.

Usage:
    # Single nudge type, multiple categories
    python plot_baseline_vs_steerability.py \
        --categories gender wealth age_group social_status \
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \
        --nudge survey_preference

    # Single category, multiple nudge types (if available)
    python plot_baseline_vs_steerability.py \
        --categories gender \
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \
        --nudges survey_preference always_save
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_frequencies,
)


def load_models_config() -> Dict[str, Any]:
    """Load the models configuration from models.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# Color palettes for different groupings
MODEL_COLORS = {
    "grok-41-fast-non-reasoning": "#E63946",  # red
    "deepseek-v3-2-non-reasoning": "#457B9D",  # blue
    "llama-33-70b": "#2A9D8F",  # teal
    "gpt-4o-mini": "#E9C46A",  # yellow/gold
    "gpt-4o": "#F4A261",  # orange
    "claude-3-5-sonnet-latest": "#9B5DE5",  # purple
    "claude-3-opus": "#00BBF9",  # cyan
    "o1-mini": "#F15BB5",  # pink
}

# Extended color palette for models not in MODEL_COLORS
_EXTRA_COLORS = [
    "#264653",  # dark teal
    "#e76f51",  # burnt sienna
    "#8338ec",  # purple
    "#ff006e",  # pink
    "#3a86ff",  # blue
    "#fb5607",  # orange
    "#ffbe0b",  # yellow
    "#06d6a0",  # mint
    "#118ab2",  # ocean blue
    "#ef476f",  # red-pink
]

# Cache for dynamically assigned model colors
_dynamic_model_colors: Dict[str, str] = {}


def get_model_color(model: str) -> str:
    """Get color for a model, auto-assigning from palette if not predefined."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]

    if model not in _dynamic_model_colors:
        # Assign next available color from palette
        idx = len(_dynamic_model_colors) % len(_EXTRA_COLORS)
        _dynamic_model_colors[model] = _EXTRA_COLORS[idx]

    return _dynamic_model_colors[model]


CATEGORY_MARKERS = {
    "gender": "o",  # circle
    "wealth": "s",  # square
    "age_group": "^",  # triangle up
    "social_status": "D",  # diamond
    "ethnicity": "v",  # triangle down
}

NUDGE_MARKERS = {
    "survey_preference": "o",  # circle
    "always_save": "s",  # square
    "utilitarian": "^",  # triangle up
    "deontological": "D",  # diamond
}


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


def compute_factor_frequencies(
    graph_data: Dict[str, Any],
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, float]:
    """Compute win frequencies for each factor level."""
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0.0, "total": 0} for level in target_levels}

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Skip intra-group comparisons
            if level_a == level_b:
                continue

            if level_a not in target_levels or level_b not in target_levels:
                continue

            aux_data = edge_data.get("aux_data", {})
            original_parsed = aux_data.get("original_parsed", [])
            flipped_parsed = aux_data.get("flipped_parsed", [])

            # Process original responses
            for resp in original_parsed:
                if resp == "A" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1
                elif resp == "B" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1

            # Process flipped responses (A in flipped = original B)
            for resp in flipped_parsed:
                if resp == "A" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1
                elif resp == "B" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1

        except Exception:
            continue

    # Compute frequencies
    frequencies = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            frequencies[level] = stats["wins"] / stats["total"]
        else:
            frequencies[level] = 0.5

    return frequencies


def get_factor_levels(category: str) -> Tuple[str, str]:
    """Get the two factor levels for a category."""
    level_mapping = {
        "gender": ("male", "female"),
        "wealth": ("poor", "rich"),
        "age_group": ("young", "old"),
        "social_status": ("low", "high"),
        "ethnicity": ("white", "black"),
    }
    return level_mapping.get(category, (None, None))


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

    # Compute frequencies
    base_freqs = compute_factor_frequencies(base_data, category, target_levels)
    nudge_A_freqs = compute_factor_frequencies(nudge_A_data, category, target_levels)
    nudge_B_freqs = compute_factor_frequencies(nudge_B_data, category, target_levels)

    f_0_A = base_freqs.get(level_A, 0.5)
    f_0_B = base_freqs.get(level_B, 0.5)
    f_A_A = nudge_A_freqs.get(level_A, 0.5)
    f_A_B = nudge_A_freqs.get(level_B, 0.5)
    f_B_A = nudge_B_freqs.get(level_A, 0.5)
    f_B_B = nudge_B_freqs.get(level_B, 0.5)

    # Compute baseline bias: deviation from 50% for level_B
    # Positive means biased towards level_B (consistent with steerability_bias)
    baseline_bias = f_0_B - 0.5

    # Compute steerability bias
    # Positive steerability_bias means more steerable towards B
    steer_A, steer_B, steerability_bias = compute_steerability_bias_from_frequencies(
        f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B
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


# Cache for models config
_models_config: Optional[Dict[str, Any]] = None


def get_model_display_name(model: str) -> str:
    """Get the display name for a model from models.yaml config."""
    global _models_config
    if _models_config is None:
        _models_config = load_models_config()

    model_config = _models_config.get(model, {})
    return model_config.get("display_name", model)


def create_scatter_plot(
    categories: List[str],
    models: List[str],
    nudge_types: List[str],
    results_base_dir: str = "results",
    output_path: str = None,
    title: str = None,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (10, 8),
):
    """
    Create a scatter plot of baseline bias vs steerability bias.

    Either categories or nudge_types should have a single value.
    The dots represent (model, other_aspect) combinations.
    """
    # Determine which dimension varies
    single_category = len(categories) == 1
    single_nudge = len(nudge_types) == 1

    if not single_category and not single_nudge:
        print("Warning: Neither categories nor nudge_types has a single value.")
        print("         Will show (model, category, nudge) combinations.")

    # Collect data points
    data_points = []

    for model in models:
        for category in categories:
            for nudge_type in nudge_types:
                print(f"Processing: {model} / {category} / {nudge_type}")
                result = compute_baseline_and_steerability_bias(
                    category, model, nudge_type, results_base_dir
                )

                if result:
                    data_points.append(
                        {
                            "model": model,
                            "category": category,
                            "nudge_type": nudge_type,
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

    # Set up colors and markers based on grouping
    if single_nudge:
        # Color by model, marker by category
        for point in data_points:
            model = point["model"]
            category = point["category"]

            color = get_model_color(model)
            marker = CATEGORY_MARKERS.get(category, "o")

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

    elif single_category:
        # Color by model, marker by nudge type
        for point in data_points:
            model = point["model"]
            nudge_type = point["nudge_type"]

            color = get_model_color(model)
            marker = NUDGE_MARKERS.get(nudge_type, "o")

            ax.scatter(
                point["baseline_bias"],
                point["steerability_bias"],
                c=color,
                marker=marker,
                s=150,
                alpha=0.8,
                edgecolors="white",
                linewidths=1.5,
                label=f"{get_model_display_name(model)} / {nudge_type}",
            )

    else:
        # Color by model only
        for point in data_points:
            model = point["model"]
            color = get_model_color(model)

            ax.scatter(
                point["baseline_bias"],
                point["steerability_bias"],
                c=color,
                marker="o",
                s=150,
                alpha=0.8,
                edgecolors="white",
                linewidths=1.5,
                label=f"{get_model_display_name(model)} / {point['category']} / {point['nudge_type']}",
            )

    # Add reference lines
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Labels and title
    ax.set_xlabel("Baseline Bias (deviation from 0.5)", fontsize=14)
    ax.set_ylabel("Steerability Bias", fontsize=14)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        if single_nudge:
            ax.set_title(
                f"Baseline vs Steerability Bias\n(Nudge type: {nudge_types[0]})",
                fontsize=16,
                fontweight="bold",
            )
        elif single_category:
            ax.set_title(
                f"Baseline vs Steerability Bias\n(Category: {categories[0]})",
                fontsize=16,
                fontweight="bold",
            )
        else:
            ax.set_title(
                "Baseline vs Steerability Bias", fontsize=16, fontweight="bold"
            )

    # Legend - create separate sections for models (colors) and categories (markers)
    if show_legend and single_nudge:
        from matplotlib.lines import Line2D

        legend_handles = []
        legend_labels = []

        # Add model entries (colors)
        unique_models = list(dict.fromkeys(p["model"] for p in data_points))
        for model in unique_models:
            color = get_model_color(model)
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=get_model_display_name(model),
            )
            legend_handles.append(handle)
            legend_labels.append(get_model_display_name(model))

        # Add separator
        legend_handles.append(Line2D([0], [0], color="none"))
        legend_labels.append("")

        # Add category entries (markers) with level_B info
        unique_categories = list(dict.fromkeys(p["category"] for p in data_points))
        for category in unique_categories:
            marker = CATEGORY_MARKERS.get(category, "o")
            level_A, level_B = get_factor_levels(category)
            # Show category name with level_B in parentheses
            label = f"{category.replace('_', ' ').title()} (+ → {level_B})"
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
    elif show_legend:
        # Fallback for other cases
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
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


def create_model_comparison_plot(
    categories: List[str],
    models: List[str],
    nudge_type: str,
    results_base_dir: str = "results",
    output_path: str = None,
    figsize: Tuple[float, float] = (12, 10),
):
    """
    Create a scatter plot with one subplot per category, comparing models.
    """
    n_categories = len(categories)
    n_cols = min(2, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    all_data_points = []

    for idx, category in enumerate(categories):
        ax = axes[idx]

        level_A, level_B = get_factor_levels(category)

        for model in models:
            result = compute_baseline_and_steerability_bias(
                category, model, nudge_type, results_base_dir
            )

            if result:
                color = get_model_color(model)
                ax.scatter(
                    result["baseline_bias"],
                    result["steerability_bias"],
                    c=color,
                    marker="o",
                    s=200,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=2,
                    label=get_model_display_name(model),
                )
                all_data_points.append(
                    {
                        "model": model,
                        "category": category,
                        "nudge_type": nudge_type,
                        **result,
                    }
                )

        # Reference lines
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Labels - both positive means level_B is favored
        ax.set_xlabel(f"Baseline Bias (+ → {level_B})", fontsize=11)
        ax.set_ylabel(f"Steerability Bias (+ → {level_B})", fontsize=11)
        ax.set_title(
            f"{category.replace('_', ' ').title()}", fontsize=14, fontweight="bold"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n_categories, len(axes)):
        axes[idx].set_visible(False)

    # Create legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(models)),
        fontsize=11,
    )

    fig.suptitle(
        f"Baseline vs Steerability Bias by Category\n(Nudge: {nudge_type})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved plot to: {output_path}")

    return fig, all_data_points


def main():
    parser = argparse.ArgumentParser(
        description="Create scatter plot of baseline bias vs steerability bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single nudge type, multiple categories (one plot per category)
    python plot_baseline_vs_steerability.py \\
        --categories gender wealth age_group social_status \\
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \\
        --nudge survey_preference \\
        --subplot-per-category

    # Single plot with all data
    python plot_baseline_vs_steerability.py \\
        --categories gender wealth \\
        --models grok-41-fast-non-reasoning deepseek-v3-2-non-reasoning \\
        --nudge survey_preference
        """,
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="List of categories/factors (e.g., gender wealth age_group)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names",
    )

    parser.add_argument(
        "--nudge",
        type=str,
        default="survey_preference",
        help="Nudge type (default: survey_preference)",
    )

    parser.add_argument(
        "--nudges",
        nargs="+",
        default=None,
        help="List of nudge types (alternative to --nudge for multiple)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory for results (default: results)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: baseline_vs_steerability.pdf)",
    )

    parser.add_argument(
        "--subplot-per-category",
        action="store_true",
        help="Create one subplot per category",
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

    args = parser.parse_args()

    # Determine nudge types
    if args.nudges:
        nudge_types = args.nudges
    else:
        nudge_types = [args.nudge]

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = "baseline_vs_steerability.pdf"

    print("=" * 60)
    print("Baseline Bias vs Steerability Bias Scatter Plot")
    print("=" * 60)
    print(f"Categories: {args.categories}")
    print(f"Models: {args.models}")
    print(f"Nudge types: {nudge_types}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()

    if args.subplot_per_category and len(nudge_types) == 1:
        fig, data_points = create_model_comparison_plot(
            categories=args.categories,
            models=args.models,
            nudge_type=nudge_types[0],
            results_base_dir=args.results_dir,
            output_path=output_path,
            figsize=tuple(args.figsize),
        )
    else:
        fig, data_points = create_scatter_plot(
            categories=args.categories,
            models=args.models,
            nudge_types=nudge_types,
            results_base_dir=args.results_dir,
            output_path=output_path,
            title=args.title,
            figsize=tuple(args.figsize),
        )

    if data_points:
        print("\n" + "=" * 60)
        print("Data Summary")
        print("=" * 60)
        print(f"{'Model':<30} {'Category':<15} {'Base Bias':>12} {'Steer Bias':>12}")
        print("-" * 75)
        for dp in data_points:
            print(
                f"{get_model_display_name(dp['model']):<30} "
                f"{dp['category']:<15} "
                f"{dp['baseline_bias']:>+12.3f} "
                f"{dp['steerability_bias']:>+12.3f}"
            )

    plt.show()


if __name__ == "__main__":
    main()
