#!/usr/bin/env python3
"""
Create a frequency-based summary table of nudge experiment results.

This script creates a table showing frequency metrics:
- f_0(B): Baseline frequency of choosing B
- f_A(B): Frequency of choosing B when nudged towards A
- f_B(B): Frequency of choosing B when nudged towards B
- Avg f(B): Average of f_A(B) and f_B(B)
- Steerability bias

Usage:
    # Discover all results from default results directory
    python create_frequency_table.py

    # Specify results directories
    python create_frequency_table.py --results-dirs results results_anthropic

    # Filter by models, factors, nudge types
    python create_frequency_table.py \
        --models claude-haiku-4-5 claude-haiku-4-5-thinking \
        --factors age_group social_status \
        --nudge-types user_preference

    # Output to CSV
    python create_frequency_table.py --output frequencies.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_frequencies,
)
from choices.analysis.utils import (
    compute_factor_frequencies,
    get_base_model_name,
    get_model_display_name,
    get_reasoning_condition,
)


@dataclass
class FrequencyResult:
    """Frequency-based results for a single nudge experiment."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    # Frequency metrics (all for level B)
    f_0_B: float  # P(B) in baseline
    f_A_B: float  # P(B) when nudged towards A
    f_B_B: float  # P(B) when nudged towards B
    avg_f_B: float  # Average of f_A_B and f_B_B
    # Steerability metrics
    steerability_A: Optional[float]  # Steerability towards A
    steerability_B: Optional[float]  # Steerability towards B
    avg_steerability: Optional[float]  # Average of steerability_A and steerability_B
    steerability_bias: Optional[float]  # steerability_B - steerability_A
    # Sample size
    n_comparisons: int


def get_nudge_target_group(result_dir: Path) -> Optional[str]:
    """Get the target group for a nudge condition from the graph data."""
    graph_data = load_preference_graph(result_dir)
    if not graph_data:
        return None

    nudge_config = graph_data.get("nudge_config")
    if nudge_config:
        return nudge_config.get("target_group")
    return None


def find_condition_directories(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Dict[str, Path]:
    """
    Find result directories for each condition (base, and each nudge target).

    Returns:
        Dictionary mapping condition name -> Path to result directory
        e.g., {'base': Path(...), 'young': Path(...), 'old': Path(...)}
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return {}

    result_dirs = {}
    dirs_by_condition: Dict[str, List[Path]] = {}

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

        if condition not in dirs_by_condition:
            dirs_by_condition[condition] = []
        dirs_by_condition[condition].append(result_dir)

    # For each condition, use the most recent directory
    for condition, dirs in dirs_by_condition.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result_dirs[condition] = most_recent

    return result_dirs


def compute_frequency_result(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[FrequencyResult]:
    """
    Compute frequency metrics for a single experiment.

    Args:
        factor_name: Name of the factor (e.g., 'age_group')
        model: Model name
        nudge_type: Type of nudge (e.g., 'user_preference')
        results_base_dir: Base directory for results

    Returns:
        FrequencyResult object or None if data is insufficient
    """
    # Find all condition directories
    condition_dirs = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    if "base" not in condition_dirs:
        return None

    # Load baseline data
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return None

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return None

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        # For now, only support binary factors
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    # Load nudge condition data
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_A_graph or not nudge_B_graph:
        return None

    # Compute frequencies for level B under each condition
    target_levels = [level_A, level_B]

    base_freqs = compute_factor_frequencies(base_graph, factor_var_name, target_levels)
    nudge_A_freqs = compute_factor_frequencies(
        nudge_A_graph, factor_var_name, target_levels
    )
    nudge_B_freqs = compute_factor_frequencies(
        nudge_B_graph, factor_var_name, target_levels
    )

    # Get frequencies for level B
    f_0_A = base_freqs.get(level_A, 0.5)
    f_0_B = base_freqs.get(level_B, 0.5)
    f_A_A = nudge_A_freqs.get(level_A, 0.5)
    f_A_B = nudge_A_freqs.get(level_B, 0.5)
    f_B_A = nudge_B_freqs.get(level_A, 0.5)
    f_B_B = nudge_B_freqs.get(level_B, 0.5)

    # Average of f_A(B) and f_B(B)
    avg_f_B = (f_A_B + f_B_B) / 2

    # Compute steerability metrics
    steerability_A, steerability_B, steerability_bias = (
        compute_steerability_bias_from_frequencies(
            f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B
        )
    )

    # Compute average steerability
    avg_steerability = None
    if steerability_A is not None and steerability_B is not None:
        avg_steerability = (steerability_A + steerability_B) / 2

    # Get sample size from edges
    n_comparisons = len(base_graph.get("edges", {}))

    # Determine reasoning condition
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    return FrequencyResult(
        model=model,
        reasoning_condition=reasoning_condition,
        factor=factor_name,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        f_0_B=f_0_B,
        f_A_B=f_A_B,
        f_B_B=f_B_B,
        avg_f_B=avg_f_B,
        steerability_A=steerability_A,
        steerability_B=steerability_B,
        avg_steerability=avg_steerability,
        steerability_bias=steerability_bias,
        n_comparisons=n_comparisons,
    )


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


def compute_all_results(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[FrequencyResult]:
    """
    Compute frequency results for all available experiments.

    Args:
        results_base_dirs: List of base directories for results
        model_filter: Optional list of models to include
        factor_filter: Optional list of factors to include
        nudge_type_filter: Optional list of nudge types to include

    Returns:
        List of FrequencyResult objects
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    results = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        result = compute_frequency_result(
            factor_name, model, nudge_type, results_base_dir
        )
        if result is not None:
            results.append(result)

    return results


def format_table(
    results: List[FrequencyResult],
    show_display_names: bool = True,
) -> str:
    """Format results as a text table."""
    if not results:
        return "No results found."

    # Sort by base_model, factor, nudge_type, reasoning_condition
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    # Build header
    headers = [
        "Model",
        "Reasoning",
        "Factor (A/B)",
        "Nudge Type",
        "f_0(B)",
        "f_A(B)",
        "f_B(B)",
        "Avg f(B)",
        "Steer(A)",
        "Steer(B)",
        "Avg Steer",
        "Steer Bias",
    ]

    # Build rows
    rows = []
    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        steer_A_str = (
            f"{r.steerability_A:.3f}" if r.steerability_A is not None else "N/A"
        )
        steer_B_str = (
            f"{r.steerability_B:.3f}" if r.steerability_B is not None else "N/A"
        )
        avg_steer_str = (
            f"{r.avg_steerability:.3f}" if r.avg_steerability is not None else "N/A"
        )
        steer_bias_str = (
            f"{r.steerability_bias:+.3f}" if r.steerability_bias is not None else "N/A"
        )
        factor_with_levels = f"{r.factor} ({r.level_A}/{r.level_B})"

        rows.append(
            [
                model_name,
                r.reasoning_condition,
                factor_with_levels,
                r.nudge_type,
                f"{r.f_0_B:.3f}",
                f"{r.f_A_B:.3f}",
                f"{r.f_B_B:.3f}",
                f"{r.avg_f_B:.3f}",
                steer_A_str,
                steer_B_str,
                avg_steer_str,
                steer_bias_str,
            ]
        )

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    # Format rows
    row_lines = []
    for row in rows:
        row_lines.append(
            " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        )

    return "\n".join([header_line, separator] + row_lines)


def write_csv(
    results: List[FrequencyResult],
    output_path: str,
    show_display_names: bool = True,
) -> None:
    """Write results to a CSV file."""
    if not results:
        print("No results to write.")
        return

    # Sort by base_model, factor, nudge_type, reasoning_condition
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    headers = [
        "model",
        "model_display_name",
        "reasoning_condition",
        "factor",
        "level_A",
        "level_B",
        "nudge_type",
        "f_0_B",
        "f_A_B",
        "f_B_B",
        "avg_f_B",
        "steerability_A",
        "steerability_B",
        "avg_steerability",
        "steerability_bias",
        "n_comparisons",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            writer.writerow(
                [
                    r.model,
                    get_model_display_name(r.model) if show_display_names else r.model,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    r.f_0_B,
                    r.f_A_B,
                    r.f_B_B,
                    r.avg_f_B,
                    r.steerability_A if r.steerability_A is not None else "",
                    r.steerability_B if r.steerability_B is not None else "",
                    r.avg_steerability if r.avg_steerability is not None else "",
                    r.steerability_bias if r.steerability_bias is not None else "",
                    r.n_comparisons,
                ]
            )

    print(f"Wrote {len(results)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create frequency-based summary table of nudge experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all results from default results directory
    python create_frequency_table.py

    # Specify results directories
    python create_frequency_table.py --results-dirs results results_anthropic

    # Filter by models, factors, nudge types
    python create_frequency_table.py \\
        --models claude-haiku-4-5 claude-haiku-4-5-thinking \\
        --factors age_group social_status \\
        --nudge-types user_preference

    # Output to CSV
    python create_frequency_table.py --output frequencies.csv
        """,
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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: print to stdout)",
    )

    parser.add_argument(
        "--no-display-names",
        action="store_true",
        help="Use raw model names instead of display names",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Frequency-Based Nudge Experiment Summary Table")
    print("=" * 80)
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    print("=" * 80)
    print()

    # Compute results
    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    print(f"Found {len(results)} complete experiments\n")

    if not results:
        print("No complete experiments found matching the filters.")
        return

    show_display_names = not args.no_display_names

    if args.output:
        write_csv(results, args.output, show_display_names)
    else:
        print(format_table(results, show_display_names))

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    from collections import defaultdict

    # Helper to compute steerability stats
    def get_steer_stats(
        result_list: List[FrequencyResult],
    ) -> Tuple[Optional[float], Optional[float]]:
        steer_results = [r for r in result_list if r.avg_steerability is not None]
        bias_results = [r for r in result_list if r.steerability_bias is not None]
        avg_steer = (
            sum(r.avg_steerability for r in steer_results) / len(steer_results)
            if steer_results
            else None
        )
        avg_bias = (
            sum(abs(r.steerability_bias) for r in bias_results) / len(bias_results)
            if bias_results
            else None
        )
        return avg_steer, avg_bias

    # By model
    model_groups: Dict[str, List[FrequencyResult]] = defaultdict(list)
    for r in results:
        base_model = get_base_model_name(r.model)
        model_groups[base_model].append(r)

    print(f"\nModels ({len(model_groups)}):")
    for base_model in sorted(model_groups.keys()):
        model_results = model_groups[base_model]
        n_factors = len(set(r.factor for r in model_results))
        avg_steer, avg_bias = get_steer_stats(model_results)

        display_name = (
            get_model_display_name(model_results[0].model)
            if show_display_names
            else base_model
        )

        if n_factors > 1:
            # Multiple factors: only show steerability metrics
            steer_str = (
                f"avg_steer={avg_steer:.3f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.3f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(f"  {display_name}: n={len(model_results)}, {steer_str}, {bias_str}")
        else:
            # Single factor: show frequency metrics and steerability
            avg_f_0_B = sum(r.f_0_B for r in model_results) / len(model_results)
            avg_f_A_B = sum(r.f_A_B for r in model_results) / len(model_results)
            avg_f_B_B = sum(r.f_B_B for r in model_results) / len(model_results)
            steer_str = (
                f"avg_steer={avg_steer:.3f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.3f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {display_name}: n={len(model_results)}, "
                f"f_0(B)={avg_f_0_B:.3f}, f_A(B)={avg_f_A_B:.3f}, "
                f"f_B(B)={avg_f_B_B:.3f}, {steer_str}, {bias_str}"
            )

    # By factor (single factor by definition)
    factors = set(r.factor for r in results)
    print(f"\nFactors ({len(factors)}):")
    for factor in sorted(factors):
        factor_results = [r for r in results if r.factor == factor]
        avg_f_0_B = sum(r.f_0_B for r in factor_results) / len(factor_results)
        avg_f_A_B = sum(r.f_A_B for r in factor_results) / len(factor_results)
        avg_f_B_B = sum(r.f_B_B for r in factor_results) / len(factor_results)
        avg_steer, avg_bias = get_steer_stats(factor_results)
        # Get level info
        level_B = factor_results[0].level_B if factor_results else "?"
        steer_str = (
            f"avg_steer={avg_steer:.3f}" if avg_steer is not None else "avg_steer=N/A"
        )
        bias_str = (
            f"|steer_bias|={avg_bias:.3f}"
            if avg_bias is not None
            else "|steer_bias|=N/A"
        )
        print(
            f"  {factor} (B={level_B}): n={len(factor_results)}, "
            f"f_0(B)={avg_f_0_B:.3f}, f_A(B)={avg_f_A_B:.3f}, "
            f"f_B(B)={avg_f_B_B:.3f}, {steer_str}, {bias_str}"
        )

    # By nudge type
    nudge_types = set(r.nudge_type for r in results)
    print(f"\nNudge Types ({len(nudge_types)}):")
    for nudge_type in sorted(nudge_types):
        nudge_results = [r for r in results if r.nudge_type == nudge_type]
        n_factors = len(set(r.factor for r in nudge_results))
        avg_steer, avg_bias = get_steer_stats(nudge_results)

        if n_factors > 1:
            # Multiple factors: only show steerability metrics
            steer_str = (
                f"avg_steer={avg_steer:.3f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.3f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(f"  {nudge_type}: n={len(nudge_results)}, {steer_str}, {bias_str}")
        else:
            # Single factor: show frequency metrics and steerability
            avg_f_0_B = sum(r.f_0_B for r in nudge_results) / len(nudge_results)
            avg_f_A_B = sum(r.f_A_B for r in nudge_results) / len(nudge_results)
            avg_f_B_B = sum(r.f_B_B for r in nudge_results) / len(nudge_results)
            steer_str = (
                f"avg_steer={avg_steer:.3f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.3f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {nudge_type}: n={len(nudge_results)}, "
                f"f_0(B)={avg_f_0_B:.3f}, f_A(B)={avg_f_A_B:.3f}, "
                f"f_B(B)={avg_f_B_B:.3f}, {steer_str}, {bias_str}"
            )


if __name__ == "__main__":
    main()
