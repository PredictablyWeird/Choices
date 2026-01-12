#!/usr/bin/env python3
"""
Analyze simple nudging experiment results.

This script loads results from simple_nudging experiments and computes
preference statistics (AMCE, larger-N preference, factor preference)
for each nudging condition.

Usage:
    python analyze_simple_nudging_results.py --factor gender --model gpt-4o-mini --nudge always_save
    python analyze_simple_nudging_results.py --factor ethnicity --model gpt-4o-mini --nudge survey_preference
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_nudge_config(results_dir: str) -> Optional[Dict]:
    """
    Load nudge configuration from a result directory.

    Args:
        results_dir: Path to result directory

    Returns:
        Dictionary with nudge config or None if not found
    """
    results_path = Path(results_dir)

    # Find preference graph file
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        return None

    with open(graph_files[0], "r") as f:
        graph_data = json.load(f)

    # The nudge_config is at the top level (spread from config dict during save)
    nudge_config = graph_data.get("nudge_config")
    return nudge_config


def find_base_result_directory(
    factor_name: str, model: str, results_base_dir: str = "results"
) -> Optional[Tuple[str, str]]:
    """
    Find the base (no-nudge) result directory.

    Args:
        factor_name: Name of the factor (e.g., "gender")
        model: Model name
        results_base_dir: Base directory for results

    Returns:
        (result_dir_path, "base") tuple or None if not found
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / "base"

    if not base_path.exists():
        return None

    # Find the most recent base result directory
    result_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not result_dirs:
        return None

    # Sort by modification time, get most recent
    most_recent = max(result_dirs, key=lambda d: d.stat().st_mtime)

    # Print note if there are multiple directories
    if len(result_dirs) > 1:
        ignored_dirs = [d.name for d in result_dirs if d != most_recent]
        print(f"Note: Found {len(result_dirs)} result directories for BASE condition.")
        print(f"  Using most recent: {most_recent.name}")
        print(
            f"  Ignoring {len(ignored_dirs)} older director{'y' if len(ignored_dirs) == 1 else 'ies'}: {', '.join(ignored_dirs)}"
        )
        print()

    return (str(most_recent), "base")


def find_nudging_result_directories(
    factor_name: str, model: str, nudge_type: str, results_base_dir: str = "results"
) -> List[Tuple[str, Optional[str]]]:
    """
    Find all result directories for a nudging experiment.

    For each target_group, only the most recent directory is returned.

    Args:
        factor_name: Name of the factor (e.g., "gender")
        model: Model name (e.g., "gpt-4o-mini")
        nudge_type: Type of nudge (e.g., "always_save")
        results_base_dir: Base directory for results (default: "results")

    Returns:
        List of (result_dir_path, target_group) tuples.
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        raise FileNotFoundError(
            f"Results directory not found: {base_path}\n"
            f"Make sure the experiment has been run with --factor {factor_name} --model {model} --nudge {nudge_type}"
        )

    # Group directories by target_group
    dirs_by_target: Dict[str, List[Path]] = {}
    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        # Try to get target_group from nudge config
        target_group = None
        nudge_config = load_nudge_config(str(result_dir))
        if nudge_config:
            target_group = nudge_config.get("target_group")

        if target_group is None:
            print(
                f"Warning: Could not load nudge configuration from {result_dir}, skipping..."
            )
            continue

        if target_group not in dirs_by_target:
            dirs_by_target[target_group] = []
        dirs_by_target[target_group].append(result_dir)

    if not dirs_by_target:
        raise FileNotFoundError(
            f"No result directories found in {base_path}\n"
            f"Make sure the experiment has been run."
        )

    # For each target_group, select the most recent directory
    result_dirs = []
    for target_group, dirs in dirs_by_target.items():
        # Sort by modification time, get most recent
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result_dirs.append((str(most_recent), target_group))

        # Print note if there are multiple directories for this target_group
        if len(dirs) > 1:
            ignored_dirs = [d.name for d in dirs if d != most_recent]
            print(
                f"Note: Found {len(dirs)} result directories for nudge condition '{target_group}'."
            )
            print(f"  Using most recent: {most_recent.name}")
            print(
                f"  Ignoring {len(ignored_dirs)} older director{'y' if len(ignored_dirs) == 1 else 'ies'}: {', '.join(ignored_dirs)}"
            )
            print()

    return sorted(result_dirs, key=lambda x: x[1] or "")


def load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load results from a simple_nudging experiment directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with loaded data
    """
    results_path = Path(results_dir)

    # Find preference graph file
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        raise FileNotFoundError(f"No preference_graph file found in {results_dir}")

    with open(graph_files[0], "r") as f:
        graph_data = json.load(f)

    # Also load utility model if available
    utility_files = list(results_path.glob("utility_model_*.json"))
    utility_data = None
    if utility_files:
        with open(utility_files[0], "r") as f:
            utility_data = json.load(f)

    return {
        "graph": graph_data,
        "utilities": utility_data,
        "results_dir": results_dir,
    }


def compute_preference_stats(
    graph_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute preference statistics from graph data.

    Args:
        graph_data: Loaded preference graph data

    Returns:
        Dictionary with preference statistics
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    variables = graph_data.get("variables", [])

    # Identify factor variable (non-N variable)
    factor_var = None
    for var in variables:
        if var["name"] != "N":
            factor_var = var
            break

    if not factor_var:
        return {"error": "No factor variable found"}

    factor_name = factor_var["name"]
    factor_levels = factor_var["values"]

    # Build option lookup
    options_by_id = {opt["id"]: opt for opt in options}

    # Track statistics per factor level
    level_stats = {level: {"n_presented": 0, "wins": 0.0} for level in factor_levels}

    # Track how often the larger N is chosen
    larger_n_stats = {"n_comparisons": 0, "larger_n_wins": 0.0}

    # Process each edge
    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            prob_a = edge_data.get("probability_A", 0.5)
            num_responses = edge_data.get("aux_data", {}).get("num_responses", 1)

            # Get factor levels
            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Get N values
            n_a = opt_a.get("N")
            n_b = opt_b.get("N")

            # Update factor level stats
            if level_a in level_stats:
                level_stats[level_a]["n_presented"] += num_responses
                level_stats[level_a]["wins"] += prob_a * num_responses

            if level_b in level_stats:
                level_stats[level_b]["n_presented"] += num_responses
                level_stats[level_b]["wins"] += (1 - prob_a) * num_responses

            # Track larger N preference (only for edges with different N values)
            if n_a is not None and n_b is not None and n_a != n_b:
                larger_n_stats["n_comparisons"] += num_responses
                if n_a > n_b:
                    larger_n_stats["larger_n_wins"] += prob_a * num_responses
                else:
                    larger_n_stats["larger_n_wins"] += (1 - prob_a) * num_responses

        except Exception:
            continue

    # Compute probabilities
    factor_probs = {}
    for level, stats in level_stats.items():
        if stats["n_presented"] > 0:
            prob = stats["wins"] / stats["n_presented"]
            se = np.sqrt(prob * (1 - prob) / stats["n_presented"])
        else:
            prob = 0.5
            se = 0.0
        factor_probs[level] = {
            "prob_chosen": prob,
            "se": se,
            "n_presented": stats["n_presented"],
        }

    # Compute larger N preference
    larger_n_prob = None
    larger_n_se = None
    if larger_n_stats["n_comparisons"] > 0:
        larger_n_prob = (
            larger_n_stats["larger_n_wins"] / larger_n_stats["n_comparisons"]
        )
        larger_n_se = np.sqrt(
            larger_n_prob * (1 - larger_n_prob) / larger_n_stats["n_comparisons"]
        )

    return {
        "factor_name": factor_name,
        "factor_levels": factor_levels,
        "factor_probs": factor_probs,
        "larger_n_prob": larger_n_prob,
        "larger_n_se": larger_n_se,
        "larger_n_comparisons": larger_n_stats["n_comparisons"],
    }


def format_probability(prob: float, se: float = None) -> str:
    """Format a probability for display."""
    if prob is None:
        return "N/A"
    if se is not None:
        return f"{prob:.1%} (±{se:.1%})"
    return f"{prob:.1%}"


def analyze_simple_nudging_experiment(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
):
    """
    Analyze a complete simple nudging experiment.

    Args:
        factor_name: Name of the factor (e.g., "gender")
        model: Model name
        nudge_type: Type of nudge
        results_base_dir: Base directory for results
    """
    print("=" * 80)
    print("Simple Nudging Experiment Analysis")
    print("=" * 80)
    print(f"Factor: {factor_name}")
    print(f"Model: {model}")
    print(f"Nudge Type: {nudge_type}")
    print("=" * 80)
    print()

    # Find base condition (no nudge)
    base_result_dir = find_base_result_directory(factor_name, model, results_base_dir)

    # Find all result directories for nudging conditions
    result_dirs = find_nudging_result_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    # Combine base and nudging conditions
    all_result_dirs = []
    if base_result_dir:
        all_result_dirs.append(base_result_dir)
        print("Found BASE condition (no nudge)")
    all_result_dirs.extend(result_dirs)

    print(f"Found {len(result_dirs)} nudging conditions:")
    for result_dir, target_group in result_dirs:
        print(f"  - {target_group}")
    print()

    # Collect results for all conditions
    all_results = []

    for result_dir, target_group in all_result_dirs:
        condition_label = (
            "BASE (no nudge)" if target_group == "base" else f"nudge → {target_group}"
        )
        print(f"Analyzing condition: {condition_label}")
        print("-" * 80)

        # Load results
        try:
            results_data = load_results(result_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        # Load nudge config
        nudge_config = load_nudge_config(result_dir)
        if nudge_config:
            print(f"Nudge Text: {nudge_config.get('nudge_text', 'N/A')}")

        # Compute preference stats
        stats = compute_preference_stats(results_data["graph"])

        if "error" in stats:
            print(f"Error: {stats['error']}")
            continue

        # Print factor preferences
        print(f"\nFactor Preferences ({stats['factor_name']}):")
        sorted_probs = sorted(
            stats["factor_probs"].items(),
            key=lambda x: x[1]["prob_chosen"],
            reverse=True,
        )
        for level, data in sorted_probs:
            print(
                f"  {level:20s}: {format_probability(data['prob_chosen'], data['se'])} "
                f"(n={data['n_presented']})"
            )

        # Print larger N preference
        if stats["larger_n_prob"] is not None:
            print("\nLarger N Preference:")
            print(
                f"  Larger N chosen: {format_probability(stats['larger_n_prob'], stats['larger_n_se'])} "
                f"(n={stats['larger_n_comparisons']})"
            )

        all_results.append(
            {
                "target_group": target_group,
                "nudge_config": nudge_config,
                "stats": stats,
            }
        )

        print()

    # Summary comparison table
    if len(all_results) >= 2:
        print("=" * 80)
        print("Summary: Preference by Nudging Condition")
        print("=" * 80)

        # Get factor levels from first result
        factor_levels = all_results[0]["stats"]["factor_levels"]
        display_factor_name = all_results[0]["stats"]["factor_name"]

        # Sort results: base first, then others alphabetically
        def sort_key(result):
            target = result["target_group"]
            return (0 if target == "base" else 1, target)

        sorted_results = sorted(all_results, key=sort_key)

        # Print header
        print(f"\n{display_factor_name.upper()} PREFERENCE:")
        print(f"\n{'Level':<20s}", end="")
        for result in sorted_results:
            label = (
                "base" if result["target_group"] == "base" else result["target_group"]
            )
            print(f"  {label:>15s}", end="")
        print()

        print("-" * (20 + 18 * len(sorted_results)))

        # Print factor level preferences
        for level in factor_levels:
            print(f"{level:<20s}", end="")
            for result in sorted_results:
                prob = result["stats"]["factor_probs"].get(level, {}).get("prob_chosen")
                if prob is not None:
                    print(f"  {prob:>14.1%}", end="")
                else:
                    print(f"  {'N/A':>14s}", end="")
            print()

        print()

        # Print larger N preference comparison
        print("LARGER N PREFERENCE:")
        print(f"\n{'':20s}", end="")
        for result in sorted_results:
            label = (
                "base" if result["target_group"] == "base" else result["target_group"]
            )
            print(f"  {label:>15s}", end="")
        print()

        print("-" * (20 + 18 * len(sorted_results)))

        print(f"{'Larger N chosen':<20s}", end="")
        for result in sorted_results:
            prob = result["stats"]["larger_n_prob"]
            if prob is not None:
                print(f"  {prob:>14.1%}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()

        print()

        # Compute and display nudge effects
        print("=" * 80)
        print("NUDGE EFFECTS (change from base)")
        print("=" * 80)
        print()

        # Find base results
        base_result = next(
            (r for r in sorted_results if r["target_group"] == "base"), None
        )
        if base_result:
            base_stats = base_result["stats"]

            for result in sorted_results:
                if result["target_group"] == "base":
                    continue

                target = result["target_group"]
                stats = result["stats"]

                print(f"Nudge towards '{target}':")

                # Factor level changes
                for level in factor_levels:
                    base_prob = (
                        base_stats["factor_probs"]
                        .get(level, {})
                        .get("prob_chosen", 0.5)
                    )
                    nudge_prob = (
                        stats["factor_probs"].get(level, {}).get("prob_chosen", 0.5)
                    )
                    change = nudge_prob - base_prob

                    marker = "↑" if change > 0 else "↓" if change < 0 else "→"
                    print(
                        f"  {level}: {base_prob:.1%} → {nudge_prob:.1%} ({marker} {abs(change):.1%})"
                    )

                # Larger N preference change
                if (
                    base_stats["larger_n_prob"] is not None
                    and stats["larger_n_prob"] is not None
                ):
                    base_n_prob = base_stats["larger_n_prob"]
                    nudge_n_prob = stats["larger_n_prob"]
                    n_change = nudge_n_prob - base_n_prob

                    marker = "↑" if n_change > 0 else "↓" if n_change < 0 else "→"
                    print(
                        f"  Larger N: {base_n_prob:.1%} → {nudge_n_prob:.1%} ({marker} {abs(n_change):.1%})"
                    )

                print()

        # Effectiveness summary
        print("=" * 80)
        print("NUDGE EFFECTIVENESS SUMMARY")
        print("=" * 80)
        print()

        if base_result:
            base_stats = base_result["stats"]

            for result in sorted_results:
                if result["target_group"] == "base":
                    continue

                target = result["target_group"]
                stats = result["stats"]

                base_prob = (
                    base_stats["factor_probs"].get(target, {}).get("prob_chosen", 0.5)
                )
                nudge_prob = (
                    stats["factor_probs"].get(target, {}).get("prob_chosen", 0.5)
                )
                change = nudge_prob - base_prob

                if change > 0.05:
                    effectiveness = "EFFECTIVE"
                elif change > 0:
                    effectiveness = "SLIGHTLY EFFECTIVE"
                elif change < -0.05:
                    effectiveness = "BACKFIRED"
                else:
                    effectiveness = "NO EFFECT"

                print(f"Nudge towards '{target}': {effectiveness}")
                print(
                    f"  Preference for {target}: {base_prob:.1%} → {nudge_prob:.1%} (Δ = {change:+.1%})"
                )
                print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze simple nudging experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_simple_nudging_results.py --factor gender --model gpt-4o-mini --nudge always_save
  python analyze_simple_nudging_results.py --factor ethnicity --model gpt-4o-mini --nudge survey_preference
        """,
    )

    parser.add_argument(
        "--factor",
        type=str,
        required=True,
        help="Factor name (e.g., gender, ethnicity)",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o-mini)",
    )

    parser.add_argument(
        "--nudge",
        type=str,
        required=True,
        help="Type of nudge (e.g., always_save, survey_preference)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory for results (default: results)",
    )

    args = parser.parse_args()

    analyze_simple_nudging_experiment(
        factor_name=args.factor,
        model=args.model,
        nudge_type=args.nudge,
        results_base_dir=args.results_dir,
    )
