#!/usr/bin/env python3
"""
Plot baseline frequency vs nudged frequency scatter plot.

For each (model, factor, nudge_type, option) combination, this script plots:
- x-axis: f_0(c) = baseline frequency of choosing option c
- y-axis: f_c(c) = frequency of choosing option c when nudged towards c

This shows how nudging affects selection probability as a function of
baseline preference.

Usage:
    # Basic usage
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --results-dirs results

    # Filter by reasoning condition
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --reasoning-conditions before none

    # Group by model, factor, or nudge type
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --groups model
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    get_nudge_target_group,
    load_preference_graph,
)
from choices.analysis.utils import (
    compute_factor_frequencies_with_counts,
    get_model_color,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)


@dataclass
class FrequencyDataPoint:
    """Data point for the scatter plot."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    option: str  # The option being tracked (e.g., "male", "female")
    other_option: str  # The other option
    # Frequencies
    f_0: float  # Baseline frequency of choosing this option
    f_c: float  # Frequency when nudged towards this option
    # Effect size
    effect_size: float  # f_c - f_0
    # Sample sizes
    n_baseline: int
    n_nudged: int


def find_condition_directories(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> List[Dict[str, Path]]:
    """
    Find result directories for each condition (base, and each nudge target).

    Groups directories by both condition AND reasoning_mode to handle cases where
    the same model/factor/nudge_type has results with different reasoning settings.

    Returns:
        List of dictionaries, each mapping condition name -> Path to result directory.
        Each dict represents a complete experiment with consistent reasoning_mode.
        e.g., [
            {'base': Path(...), 'young': Path(...), 'old': Path(...)},  # reasoning_mode="none"
            {'base': Path(...), 'young': Path(...), 'old': Path(...)},  # reasoning_mode="before"
        ]
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return []

    # Group directories by (condition, reasoning_mode)
    # Key: (condition, reasoning_mode), Value: list of directories
    dirs_by_condition_and_reasoning: Dict[Tuple[str, str], List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        # Check if this is a base condition
        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            # Get target group from graph data
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        # Get reasoning_mode from the utility model JSON
        reasoning_mode = get_reasoning_mode_from_results(result_dir)
        if reasoning_mode is None:
            reasoning_mode = "unknown"

        key = (condition, reasoning_mode)
        if key not in dirs_by_condition_and_reasoning:
            dirs_by_condition_and_reasoning[key] = []
        dirs_by_condition_and_reasoning[key].append(result_dir)

    # For each (condition, reasoning_mode), use the most recent directory
    # Then group by reasoning_mode to build complete experiments
    experiments_by_reasoning: Dict[str, Dict[str, Path]] = {}

    for (condition, reasoning_mode), dirs in dirs_by_condition_and_reasoning.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)

        if reasoning_mode not in experiments_by_reasoning:
            experiments_by_reasoning[reasoning_mode] = {}
        experiments_by_reasoning[reasoning_mode][condition] = most_recent

    # Return list of complete experiments (one per reasoning_mode)
    return list(experiments_by_reasoning.values())


def discover_experiments(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Discover all available experiments in the results directories.

    Returns:
        List of (results_dir, factor_name, model, nudge_type) tuples
    """
    experiments = []

    for results_base_dir in results_base_dirs:
        results_path = Path(results_base_dir)
        if not results_path.exists():
            continue

        # Iterate through experiment directories (simple_{factor})
        for exp_dir in results_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
                continue

            factor_name = exp_dir.name[7:]  # Remove 'simple_' prefix

            # Apply factor filter
            if factor_filter and factor_name not in factor_filter:
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

                    experiments.append(
                        (results_base_dir, factor_name, model, nudge_type)
                    )

    return experiments


def _compute_data_points_for_condition_dirs(
    factor_name: str,
    model: str,
    nudge_type: str,
    condition_dirs: Dict[str, Path],
) -> List[FrequencyDataPoint]:
    """
    Compute frequency data points for a single experiment set (one reasoning_mode).

    Returns one data point per option (typically 2 for binary factors).
    """
    if "base" not in condition_dirs:
        return []

    # Load baseline data
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return []

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return []

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        return []

    level_A, level_B = factor_levels[0], factor_levels[1]
    target_levels = [level_A, level_B]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return []

    # Load nudge condition data
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_A_graph or not nudge_B_graph:
        return []

    # Compute frequencies for each condition
    base_stats = compute_factor_frequencies_with_counts(
        base_graph, factor_var_name, target_levels
    )
    nudge_A_stats = compute_factor_frequencies_with_counts(
        nudge_A_graph, factor_var_name, target_levels
    )
    nudge_B_stats = compute_factor_frequencies_with_counts(
        nudge_B_graph, factor_var_name, target_levels
    )

    # Determine reasoning condition
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    data_points = []

    # Data point for option A: f_0(A) vs f_A(A)
    f_0_A = base_stats.get(level_A, {}).get("freq", 0.5)
    n_0_A = base_stats.get(level_A, {}).get("n", 0)
    f_A_A = nudge_A_stats.get(level_A, {}).get("freq", 0.5)
    n_A = nudge_A_stats.get(level_A, {}).get("n", 0)

    data_points.append(
        FrequencyDataPoint(
            model=model,
            reasoning_condition=reasoning_condition,
            factor=factor_name,
            nudge_type=nudge_type,
            option=level_A,
            other_option=level_B,
            f_0=f_0_A,
            f_c=f_A_A,
            effect_size=f_A_A - f_0_A,
            n_baseline=n_0_A,
            n_nudged=n_A,
        )
    )

    # Data point for option B: f_0(B) vs f_B(B)
    f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
    n_0_B = base_stats.get(level_B, {}).get("n", 0)
    f_B_B = nudge_B_stats.get(level_B, {}).get("freq", 0.5)
    n_B = nudge_B_stats.get(level_B, {}).get("n", 0)

    data_points.append(
        FrequencyDataPoint(
            model=model,
            reasoning_condition=reasoning_condition,
            factor=factor_name,
            nudge_type=nudge_type,
            option=level_B,
            other_option=level_A,
            f_0=f_0_B,
            f_c=f_B_B,
            effect_size=f_B_B - f_0_B,
            n_baseline=n_0_B,
            n_nudged=n_B,
        )
    )

    return data_points


def compute_data_points_for_experiment(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> List[FrequencyDataPoint]:
    """
    Compute frequency data points for a single experiment.

    This handles cases where the same model/factor/nudge_type combination has
    multiple experiment runs with different reasoning_mode settings.

    Args:
        factor_name: Name of the factor
        model: Model name
        nudge_type: Type of nudge
        results_base_dir: Base directory for results

    Returns:
        List of FrequencyDataPoint objects (one per option per reasoning_mode)
    """
    # Find all experiment sets (one per reasoning_mode)
    experiment_sets = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    all_data_points = []
    for condition_dirs in experiment_sets:
        points = _compute_data_points_for_condition_dirs(
            factor_name, model, nudge_type, condition_dirs
        )
        all_data_points.extend(points)

    return all_data_points


def compute_all_data_points(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
    reasoning_conditions_filter: Optional[List[str]] = None,
) -> List[FrequencyDataPoint]:
    """
    Compute data points for all available experiments.

    Args:
        results_base_dirs: List of base directories for results
        model_filter: Optional list of models to include
        factor_filter: Optional list of factors to include
        nudge_type_filter: Optional list of nudge types to include
        reasoning_conditions_filter: Optional list of reasoning conditions to include

    Returns:
        List of FrequencyDataPoint objects
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    data_points = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        points = compute_data_points_for_experiment(
            factor_name, model, nudge_type, results_base_dir
        )
        for point in points:
            # Apply reasoning condition filter
            if reasoning_conditions_filter:
                if point.reasoning_condition not in reasoning_conditions_filter:
                    continue
            data_points.append(point)

    return data_points


# Color palette for factors
FACTOR_COLORS = {
    "gender": "#E63946",  # red
    "age_group": "#457B9D",  # blue
    "wealth": "#2A9D8F",  # teal
    "social_status": "#E9C46A",  # yellow/gold
    "nationality": "#9B5DE5",  # purple
    "handedness": "#F4A261",  # orange
    "tech_view": "#00BBF9",  # cyan
    "diet": "#F15BB5",  # pink
    "extraversion": "#264653",  # dark teal
    "hair_color": "#e76f51",  # burnt sienna
}

_EXTRA_FACTOR_COLORS = [
    "#8338ec",
    "#ff006e",
    "#3a86ff",
    "#fb5607",
    "#ffbe0b",
]

_dynamic_factor_colors: Dict[str, str] = {}


def get_factor_color(factor: str) -> str:
    """Get color for a factor, auto-assigning from palette if not predefined."""
    if factor in FACTOR_COLORS:
        return FACTOR_COLORS[factor]

    if factor not in _dynamic_factor_colors:
        idx = len(_dynamic_factor_colors) % len(_EXTRA_FACTOR_COLORS)
        _dynamic_factor_colors[factor] = _EXTRA_FACTOR_COLORS[idx]

    return _dynamic_factor_colors[factor]


# Color palette for nudge types
NUDGE_COLORS = {
    # Information-based nudges
    "survey_preference": "#E63946",  # red
    "weak_evidence": "#457B9D",  # blue
    # Pressure-based nudges
    "emotional": "#2A9D8F",  # teal
    "user_preference": "#E9C46A",  # yellow/gold
    # Other nudges
    "few_shot_3": "#9B5DE5",  # purple
    "few_shot_5": "#F4A261",  # orange
    # Legacy nudges (from older experiments)
    "always_save": "#00BBF9",  # cyan
    "utilitarian": "#F15BB5",  # pink
    "deontological": "#264653",  # dark teal
    "expert_opinion": "#e76f51",  # burnt sienna
    "social_proof": "#8338ec",  # purple
}

_EXTRA_NUDGE_COLORS = [
    "#ff006e",
    "#3a86ff",
    "#fb5607",
    "#ffbe0b",
    "#06d6a0",
]

_dynamic_nudge_colors: Dict[str, str] = {}


def get_nudge_color(nudge_type: str) -> str:
    """Get color for a nudge type, auto-assigning from palette if not predefined."""
    if nudge_type in NUDGE_COLORS:
        return NUDGE_COLORS[nudge_type]

    if nudge_type not in _dynamic_nudge_colors:
        idx = len(_dynamic_nudge_colors) % len(_EXTRA_NUDGE_COLORS)
        _dynamic_nudge_colors[nudge_type] = _EXTRA_NUDGE_COLORS[idx]

    return _dynamic_nudge_colors[nudge_type]


def create_scatter_plot(
    data_points: List[FrequencyDataPoint],
    groups: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    n_bins: int = 10,
    show_diagonal: bool = True,
):
    """
    Create a scatter plot of baseline frequency vs nudged frequency.

    Args:
        data_points: List of FrequencyDataPoint objects
        groups: Grouping variable ('model', 'factor', or 'nudge')
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size as (width, height)
        n_bins: Number of bins for the binned average overlay
        show_diagonal: If True, show the y=x diagonal line

    Returns:
        Tuple of (figure, axes) or None if no data
    """
    if not data_points:
        print("No data points to plot!")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Define helper functions for grouping
    def get_group_model(dp: FrequencyDataPoint):
        return (dp.model, dp.reasoning_condition)

    def get_color_model(g):
        return get_model_color(g[0])

    def get_label_model(g):
        return f"{get_model_display_name(g[0])} ({g[1]})"

    def get_group_factor(dp: FrequencyDataPoint):
        return dp.factor

    def get_label_factor(g):
        return g.replace("_", " ").title()

    def get_group_nudge(dp: FrequencyDataPoint):
        return dp.nudge_type

    def get_label_nudge(g):
        return g.replace("_", " ").title()

    def get_group_none(dp: FrequencyDataPoint):
        return None

    def get_color_none(g):
        return "#457B9D"  # Default blue

    def get_label_none(g):
        return None

    # Get unique groups for coloring
    if groups == "model":
        unique_groups = list(
            dict.fromkeys((dp.model, dp.reasoning_condition) for dp in data_points)
        )
        get_group = get_group_model
        get_color = get_color_model
        get_label = get_label_model
    elif groups == "factor":
        unique_groups = list(dict.fromkeys(dp.factor for dp in data_points))
        get_group = get_group_factor
        get_color = get_factor_color
        get_label = get_label_factor
    elif groups == "nudge":
        unique_groups = list(dict.fromkeys(dp.nudge_type for dp in data_points))
        get_group = get_group_nudge
        get_color = get_nudge_color
        get_label = get_label_nudge
    else:
        unique_groups = None
        get_group = get_group_none
        get_color = get_color_none
        get_label = get_label_none

    # Plot data points
    if unique_groups:
        for group in unique_groups:
            group_points = [dp for dp in data_points if get_group(dp) == group]
            x = [dp.f_0 for dp in group_points]
            y = [dp.f_c for dp in group_points]

            ax.scatter(
                x,
                y,
                c=get_color(group),
                label=get_label(group),
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )
    else:
        x = [dp.f_0 for dp in data_points]
        y = [dp.f_c for dp in data_points]

        ax.scatter(
            x,
            y,
            c=get_color(None),
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Add diagonal line (y = x)
    if show_diagonal:
        ax.plot(
            [0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="y = x (no effect)"
        )

    # Compute and overlay binned averages
    if len(data_points) > 0:
        x_all = np.array([dp.f_0 for dp in data_points])
        y_all = np.array([dp.f_c for dp in data_points])

        # Create bins from 0 to 1
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute average y for each bin
        bin_means = []
        bin_stds = []
        bin_counts = []
        for i in range(n_bins):
            mask = (x_all >= bin_edges[i]) & (x_all < bin_edges[i + 1])
            # Include the right edge for the last bin
            if i == n_bins - 1:
                mask = (x_all >= bin_edges[i]) & (x_all <= bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(y_all[mask]))
                bin_stds.append(np.std(y_all[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
                bin_counts.append(0)

        bin_means = np.array(bin_means)
        bin_counts = np.array(bin_counts)

        # Plot binned averages as a line with error band
        valid_mask = ~np.isnan(bin_means)
        if np.any(valid_mask):
            ax.plot(
                bin_centers[valid_mask],
                bin_means[valid_mask],
                "r-",
                linewidth=2,
                alpha=0.8,
                label=f"Binned avg (n={len(data_points)})",
                zorder=10,
            )
            # Add markers at bin centers
            ax.scatter(
                bin_centers[valid_mask],
                bin_means[valid_mask],
                c="red",
                s=50,
                zorder=11,
                edgecolors="white",
                linewidths=1,
            )

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Labels
    ax.set_xlabel("Baseline Frequency f₀(c)", fontsize=14)
    ax.set_ylabel("Nudged Frequency fₓ(c) (when nudged towards c)", fontsize=14)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        ax.set_title(
            "Baseline vs Nudged Frequency by Option",
            fontsize=16,
            fontweight="bold",
        )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 16:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            framealpha=0.9,
        )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Make plot square
    ax.set_aspect("equal", adjustable="box")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved plot to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Create scatter plot of baseline frequency vs nudged frequency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --results-dirs results

    # Filter by reasoning condition
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --reasoning-conditions before none

    # Group by model, factor, or nudge type
    uv run python -m choices.analysis.plot_baseline_vs_nudged_frequency --groups model
        """,
    )

    parser.add_argument(
        "--groups",
        type=str,
        choices=["model", "factor", "nudge"],
        default=None,
        help="Group data points by this variable (model, factor, or nudge)",
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
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include (e.g., 'before', 'none', 'after')",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: baseline_vs_nudged.pdf)",
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

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for the binned average overlay (default: 10)",
    )

    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Don't show the y=x diagonal line",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.groups}" if args.groups else ""
        output_path = f"baseline_vs_nudged{suffix}.pdf"

    # Print header
    print("=" * 70)
    print("Baseline Frequency vs Nudged Frequency Scatter Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    if args.groups:
        print(f"Grouping by: {args.groups}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning_conditions:
        print(f"Reasoning conditions filter: {args.reasoning_conditions}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Compute data points
    data_points = compute_all_data_points(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        reasoning_conditions_filter=args.reasoning_conditions,
    )

    print(f"Found {len(data_points)} data points\n")

    if not data_points:
        print("No data points found matching the criteria.")
        return

    # Create plot
    result = create_scatter_plot(
        data_points=data_points,
        groups=args.groups,
        output_path=output_path,
        title=args.title,
        figsize=tuple(args.figsize),
        n_bins=args.n_bins,
        show_diagonal=not args.no_diagonal,
    )

    if result is None:
        return

    # Print data summary
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    print(
        f"{'Model':<25} {'Factor':<12} {'Nudge':<18} "
        f"{'Option':<10} {'f_0(c)':>8} {'f_c(c)':>8} {'Effect':>8}"
    )
    print("-" * 99)
    for dp in sorted(
        data_points, key=lambda x: (x.model, x.factor, x.nudge_type, x.option)
    ):
        print(
            f"{get_model_display_name(dp.model):<25} "
            f"{dp.factor:<12} "
            f"{dp.nudge_type:<18} "
            f"{dp.option:<10} "
            f"{dp.f_0:>8.3f} "
            f"{dp.f_c:>8.3f} "
            f"{dp.effect_size:>+8.3f}"
        )

    # Print statistics by group if grouping is enabled
    if args.groups:
        print("\n" + "=" * 70)
        print(f"Statistics by {args.groups.title()}")
        print("=" * 70)

        if args.groups == "model":
            groups = {}
            for dp in data_points:
                key = (dp.model, dp.reasoning_condition)
                if key not in groups:
                    groups[key] = []
                groups[key].append(dp)

            for (model, reasoning), points in sorted(groups.items()):
                avg_f0 = np.mean([dp.f_0 for dp in points])
                avg_fc = np.mean([dp.f_c for dp in points])
                avg_effect = np.mean([dp.effect_size for dp in points])
                positive_count = sum(1 for dp in points if dp.effect_size > 0)
                print(
                    f"  {get_model_display_name(model)} ({reasoning}): "
                    f"n={len(points)}, "
                    f"avg f₀={avg_f0:.3f}, "
                    f"avg f_c={avg_fc:.3f}, "
                    f"avg effect={avg_effect:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

        elif args.groups == "factor":
            groups = {}
            for dp in data_points:
                if dp.factor not in groups:
                    groups[dp.factor] = []
                groups[dp.factor].append(dp)

            for factor, points in sorted(groups.items()):
                avg_f0 = np.mean([dp.f_0 for dp in points])
                avg_fc = np.mean([dp.f_c for dp in points])
                avg_effect = np.mean([dp.effect_size for dp in points])
                positive_count = sum(1 for dp in points if dp.effect_size > 0)
                print(
                    f"  {factor}: "
                    f"n={len(points)}, "
                    f"avg f₀={avg_f0:.3f}, "
                    f"avg f_c={avg_fc:.3f}, "
                    f"avg effect={avg_effect:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

        elif args.groups == "nudge":
            groups = {}
            for dp in data_points:
                if dp.nudge_type not in groups:
                    groups[dp.nudge_type] = []
                groups[dp.nudge_type].append(dp)

            for nudge_type, points in sorted(groups.items()):
                avg_f0 = np.mean([dp.f_0 for dp in points])
                avg_fc = np.mean([dp.f_c for dp in points])
                avg_effect = np.mean([dp.effect_size for dp in points])
                positive_count = sum(1 for dp in points if dp.effect_size > 0)
                print(
                    f"  {nudge_type}: "
                    f"n={len(points)}, "
                    f"avg f₀={avg_f0:.3f}, "
                    f"avg f_c={avg_fc:.3f}, "
                    f"avg effect={avg_effect:+.3f}, "
                    f"positive={positive_count}/{len(points)}"
                )

    # Overall statistics
    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    x_all = [dp.f_0 for dp in data_points]
    y_all = [dp.f_c for dp in data_points]
    effects = [dp.effect_size for dp in data_points]

    avg_f0 = np.mean(x_all)
    avg_fc = np.mean(y_all)
    avg_effect = np.mean(effects)
    positive_count = sum(1 for e in effects if e > 0)

    # Correlation
    corr, p_value = stats.pearsonr(x_all, y_all)

    # Paired t-test for effect size different from 0
    t_stat, t_pvalue = stats.ttest_1samp(effects, 0)

    print(f"  Total data points: {len(data_points)}")
    print(f"  Average baseline frequency (f₀): {avg_f0:.3f}")
    print(f"  Average nudged frequency (f_c): {avg_fc:.3f}")
    print(f"  Average effect size (f_c - f₀): {avg_effect:+.3f}")
    print(
        f"  Positive effects: {positive_count}/{len(data_points)} "
        f"({100*positive_count/len(data_points):.1f}%)"
    )
    print(f"  Correlation (f₀ vs f_c): r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Effect size t-test: t = {t_stat:.3f} (p = {t_pvalue:.4f})")

    # Additional analysis: effect size vs baseline frequency
    print("\n" + "-" * 70)
    print("Effect Size by Baseline Frequency Quintile")
    print("-" * 70)

    # Split into quintiles based on baseline frequency
    quintiles = np.percentile(x_all, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        lower, upper = quintiles[i], quintiles[i + 1]
        mask = [(lower <= f0 <= upper) for f0 in x_all]
        quintile_effects = [e for e, m in zip(effects, mask) if m]
        if quintile_effects:
            avg_q_effect = np.mean(quintile_effects)
            positive_q = sum(1 for e in quintile_effects if e > 0)
            print(
                f"  [{lower:.2f}, {upper:.2f}]: "
                f"n={len(quintile_effects)}, "
                f"avg effect={avg_q_effect:+.3f}, "
                f"positive={positive_q}/{len(quintile_effects)}"
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
