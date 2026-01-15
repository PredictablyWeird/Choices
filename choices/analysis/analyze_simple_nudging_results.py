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

from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_frequencies,
)


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
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[Tuple[str, str]]:
    """
    Find the base (no-nudge) result directory.

    First looks for base results in the nudge directory (with "_base" suffix),
    then falls back to the legacy "base" directory.

    Args:
        factor_name: Name of the factor (e.g., "gender")
        model: Model name
        nudge_type: Type of nudge (to look for base in nudge directory)
        results_base_dir: Base directory for results

    Returns:
        (result_dir_path, "base") tuple or None if not found
    """
    experiment_name = f"simple_{factor_name}"

    # First, try to find base in the nudge directory (new location)
    nudge_path = Path(results_base_dir) / experiment_name / model / nudge_type
    if nudge_path.exists():
        # Look for directories ending with "_base"
        base_dirs = [
            d for d in nudge_path.iterdir() if d.is_dir() and d.name.endswith("_base")
        ]
        if base_dirs:
            most_recent = max(base_dirs, key=lambda d: d.stat().st_mtime)

            # Print note if there are multiple directories
            if len(base_dirs) > 1:
                ignored_dirs = [d.name for d in base_dirs if d != most_recent]
                print(
                    f"Note: Found {len(base_dirs)} result directories for BASE condition."
                )
                print(f"  Using most recent: {most_recent.name}")
                print(
                    f"  Ignoring {len(ignored_dirs)} older director{'y' if len(ignored_dirs) == 1 else 'ies'}: {', '.join(ignored_dirs)}"
                )
                print()

            return (str(most_recent), "base")

    # Fall back to legacy "base" directory
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

        # Skip _base directories - these are handled by find_base_result_directory
        if result_dir.name.endswith("_base"):
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
            num_responses = edge_data.get("aux_data", {}).get("total_responses", 1)

            # Get factor levels
            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Get N values
            n_a = opt_a.get("N")
            n_b = opt_b.get("N")

            # Update factor level stats (only for edges with different factor levels)
            # Intra-group comparisons (e.g., poor vs poor) don't tell us about
            # factor preference, so we exclude them - similar to larger_n_stats below
            if level_a != level_b:
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

    # Find base condition (no nudge) - looks in nudge dir first, then falls back to base dir
    base_result_dir = find_base_result_directory(
        factor_name, model, nudge_type, results_base_dir
    )

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

        # Compute and display steerability bias
        _display_steerability_bias(sorted_results, factor_levels)

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def _display_steerability_bias(
    sorted_results: List[Dict[str, Any]],
    factor_levels: List[str],
) -> None:
    """
    Compute and display steerability bias for pairwise factor level comparisons.

    Args:
        sorted_results: List of result dictionaries sorted with base first
        factor_levels: List of factor level names
    """
    # Need base condition and at least 2 factor levels
    if len(factor_levels) < 2:
        return

    base_result = next((r for r in sorted_results if r["target_group"] == "base"), None)
    if not base_result:
        print("Note: Cannot compute steerability bias without base condition")
        return

    base_stats = base_result["stats"]

    # Build lookup: {target_group: stats}
    stats_by_target = {}
    for result in sorted_results:
        stats_by_target[result["target_group"]] = result["stats"]

    print("=" * 80)
    print("STEERABILITY BIAS ANALYSIS")
    print("=" * 80)
    print()
    print(
        "Steerability measures how much nudging changes the odds ratio for each option."
    )
    print(
        "Bias measures differential steerability (positive = more steerable toward B)."
    )
    print()

    # Compute pairwise steerability biases
    pairwise_results = []

    for i, level_A in enumerate(factor_levels):
        for level_B in factor_levels[i + 1 :]:
            # Check if we have nudge conditions for both levels
            if level_A not in stats_by_target or level_B not in stats_by_target:
                continue

            # Get frequencies for base condition
            f_0_A = base_stats["factor_probs"].get(level_A, {}).get("prob_chosen")
            f_0_B = base_stats["factor_probs"].get(level_B, {}).get("prob_chosen")

            if f_0_A is None or f_0_B is None:
                continue

            # Get frequencies for nudge towards A
            nudge_A_stats = stats_by_target.get(level_A, {})
            if not nudge_A_stats:
                continue
            f_A_A = nudge_A_stats["factor_probs"].get(level_A, {}).get("prob_chosen")
            f_A_B = nudge_A_stats["factor_probs"].get(level_B, {}).get("prob_chosen")

            if f_A_A is None or f_A_B is None:
                continue

            # Get frequencies for nudge towards B
            nudge_B_stats = stats_by_target.get(level_B, {})
            if not nudge_B_stats:
                continue
            f_B_A = nudge_B_stats["factor_probs"].get(level_A, {}).get("prob_chosen")
            f_B_B = nudge_B_stats["factor_probs"].get(level_B, {}).get("prob_chosen")

            if f_B_A is None or f_B_B is None:
                continue

            # Compute steerability bias
            steer_A, steer_B, bias = compute_steerability_bias_from_frequencies(
                f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B
            )

            if steer_A is not None:
                pairwise_results.append(
                    {
                        "level_A": level_A,
                        "level_B": level_B,
                        "f_0_A": f_0_A,
                        "f_0_B": f_0_B,
                        "f_A_A": f_A_A,
                        "f_A_B": f_A_B,
                        "f_B_A": f_B_A,
                        "f_B_B": f_B_B,
                        "steerability_A": steer_A,
                        "steerability_B": steer_B,
                        "bias": bias,
                    }
                )

    if not pairwise_results:
        print(
            "Could not compute steerability bias (missing nudge conditions or near-zero frequencies)"
        )
        print()
        return

    # Display detailed results for each pair
    for result in pairwise_results:
        level_A = result["level_A"]
        level_B = result["level_B"]

        print(f"{level_A} (A) vs {level_B} (B):")
        print("-" * 40)

        # Show frequency table
        print(
            f"  {'Condition':<20s} {'P(' + level_A + ')':>12s} {'P(' + level_B + ')':>12s} {'Odds(A/B)':>12s}"
        )
        print(f"  {'-'*56}")

        # Base condition
        r_0 = result["f_0_A"] / result["f_0_B"] if result["f_0_B"] > 0 else float("inf")
        print(
            f"  {'Base (no nudge)':<20s} {result['f_0_A']:>11.1%} {result['f_0_B']:>12.1%} {r_0:>12.2f}"
        )

        # Nudge towards A
        r_A = result["f_A_A"] / result["f_A_B"] if result["f_A_B"] > 0 else float("inf")
        print(
            f"  {'Nudge → ' + level_A:<20s} {result['f_A_A']:>11.1%} {result['f_A_B']:>12.1%} {r_A:>12.2f}"
        )

        # Nudge towards B
        r_B = result["f_B_A"] / result["f_B_B"] if result["f_B_B"] > 0 else float("inf")
        print(
            f"  {'Nudge → ' + level_B:<20s} {result['f_B_A']:>11.1%} {result['f_B_B']:>12.1%} {r_B:>12.2f}"
        )

        print()

        # Steerability metrics
        print(
            f"  Steerability towards {level_A}: s(A) = {result['steerability_A']:+.3f}"
        )
        print(
            f"  Steerability towards {level_B}: s(B) = {result['steerability_B']:+.3f}"
        )
        print()

        bias = result["bias"]
        if abs(bias) < 0.05:
            interpretation = "roughly equal steerability"
        elif bias > 0:
            interpretation = f"more steerable towards {level_B}"
        else:
            interpretation = f"more steerable towards {level_A}"

        print(f"  Steerability Bias: {bias:+.3f} ({interpretation})")
        print()

    # Summary table if multiple pairs
    if len(pairwise_results) > 1:
        print("STEERABILITY BIAS SUMMARY:")
        print("-" * 60)
        print(f"  {'Pair':<30s} {'s(A)':>10s} {'s(B)':>10s} {'Bias':>10s}")
        print(f"  {'-'*60}")
        for result in pairwise_results:
            pair_name = f"{result['level_A']} vs {result['level_B']}"
            print(
                f"  {pair_name:<30s} "
                f"{result['steerability_A']:>+10.3f} "
                f"{result['steerability_B']:>+10.3f} "
                f"{result['bias']:>+10.3f}"
            )
        print()

    # Bias matrix for >2 factor levels
    if len(factor_levels) > 2:
        print("STEERABILITY BIAS MATRIX:")
        print("(positive value in row A, column B means more steerable towards B)")
        print()

        # Build lookup for quick access
        bias_lookup = {}
        for result in pairwise_results:
            bias_lookup[(result["level_A"], result["level_B"])] = result["bias"]

        col_width = max(len(level) for level in factor_levels) + 2
        header = " " * col_width + "".join(
            f"{level:>{col_width}}" for level in factor_levels
        )
        print(header)
        print("-" * len(header))

        for level_A in factor_levels:
            row = f"{level_A:<{col_width}}"
            for level_B in factor_levels:
                if level_A == level_B:
                    row += f"{'—':>{col_width}}"
                elif (level_A, level_B) in bias_lookup:
                    bias = bias_lookup[(level_A, level_B)]
                    row += f"{bias:>+{col_width}.2f}"
                elif (level_B, level_A) in bias_lookup:
                    # Bias is antisymmetric
                    bias = -bias_lookup[(level_B, level_A)]
                    row += f"{bias:>+{col_width}.2f}"
                else:
                    row += f"{'N/A':>{col_width}}"
            print(row)
        print()


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
