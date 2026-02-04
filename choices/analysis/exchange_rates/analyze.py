#!/usr/bin/env python3
"""
Analyze nudging experiment results.

This script loads results from nudging experiments and computes exchange rates
for each nudging condition (target group).

Usage:
    python analyze_nudging_results.py --config age_group_deaths --model gpt-4o-mini --nudge always_save
    python analyze_nudging_results.py --config gender_illness_hospital --model gpt-4o-mini --nudge survey_preference
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from choices.analysis.exchange_rates.plots import (
    fit_utility_curves,
    get_factor_values_and_N_values,
    load_exchange_rates_data,
    two_way_geometric_exchange_rate,
)
from choices.analysis.metrics import compute_steerability_asym
from choices.results import ExperimentResults
from choices.utils import find_result_files


def load_nudge_config(results_dir: str) -> Optional[Dict]:
    """
    Load nudge configuration from a result directory.

    Args:
        results_dir: Path to result directory

    Returns:
        Dictionary with nudge config or None if not found
    """
    graph_path, model_path, suffix = find_result_files(results_dir)
    if graph_path is None:
        return None

    # Load the preference graph JSON to get the config
    with open(graph_path, "r") as f:
        graph_data = json.load(f)

    # The nudge_config is at the top level (spread from config dict)
    nudge_config = graph_data.get("nudge_config")
    return nudge_config


def find_base_result_directory(
    config_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[Tuple[str, str]]:
    """
    Find the base (no-nudge) result directory.

    First looks for base results in the nudge directory (with "_base" suffix),
    then falls back to the legacy "base" directory.

    Args:
        config_name: Name of the base config
        model: Model name
        nudge_type: Type of nudge (to look for base in nudge directory)
        results_base_dir: Base directory for results

    Returns:
        (result_dir_path, "base") tuple or None if not found
    """
    # First, try to find base in the nudge directory (new location)
    nudge_path = Path(results_base_dir) / config_name / model / nudge_type
    if nudge_path.exists():
        # Look for directories ending with "_base"
        base_dirs = [
            d for d in nudge_path.iterdir() if d.is_dir() and d.name.endswith("_base")
        ]
        if base_dirs:
            most_recent = max(base_dirs, key=lambda d: d.stat().st_mtime)
            return (str(most_recent), "base")

    # Fall back to legacy "base" directory
    base_path = Path(results_base_dir) / config_name / model / "base"

    if not base_path.exists():
        return None

    # Find the most recent base result directory
    result_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not result_dirs:
        return None

    # Sort by modification time, get most recent
    most_recent = max(result_dirs, key=lambda d: d.stat().st_mtime)
    return (str(most_recent), "base")


def find_nudging_result_directories(
    config_name: str, model: str, nudge_type: str, results_base_dir: str = "results"
) -> List[Tuple[str, Optional[str]]]:
    """
    Find all result directories for a nudging experiment.

    Args:
        config_name: Name of the base config (e.g., "age_group_deaths")
        model: Model name (e.g., "gpt-4o-mini")
        nudge_type: Type of nudge (e.g., "always_save")
        results_base_dir: Base directory for results (default: "results")

    Returns:
        List of (result_dir_path, target_group) tuples. target_group is None if not found in config.
    """
    base_path = Path(results_base_dir) / config_name / model / nudge_type

    if not base_path.exists():
        raise FileNotFoundError(
            f"Results directory not found: {base_path}\n"
            f"Make sure the experiment has been run with --config {config_name} --model {model} --nudge {nudge_type}"
        )

    result_dirs = []
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

        result_dirs.append((str(result_dir), target_group))

    if not result_dirs:
        raise FileNotFoundError(
            f"No result directories found in {base_path}\n"
            f"Make sure the experiment has been run."
        )

    return sorted(result_dirs, key=lambda x: x[1] or "")


def compute_exchange_rates_for_condition(
    results_dir: str, factor_name: str, canonical_group: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute exchange rates for a single nudging condition.

    Args:
        results_dir: Path to result directory
        factor_name: Name of the categorical factor
        canonical_group: Group to use as reference (if None, uses first group alphabetically)

    Returns:
        Dictionary mapping group names to exchange rates (relative to canonical_group)
    """
    # Load the data
    df, numerical_var = load_exchange_rates_data(results_dir, factor_name)

    if df.empty:
        return {}

    # Get factor values and N values
    X_values, N_values_list = get_factor_values_and_N_values(df, numerical_var)

    if not X_values:
        return {}

    # Fit utility curves
    slopes, intercepts = fit_utility_curves(df, return_mse=False)

    if not slopes:
        return {}

    # Determine canonical group (reference group)
    if canonical_group is None:
        # Use first group alphabetically that has a valid slope
        valid_groups = [x for x in X_values if x in slopes and slopes[x] > 0]
        if not valid_groups:
            # If no positive slopes, use any group
            valid_groups = [x for x in X_values if x in slopes]
        if not valid_groups:
            return {}
        canonical_group = sorted(valid_groups)[0]

    if canonical_group not in slopes:
        # Fallback to first available group
        canonical_group = sorted(slopes.keys())[0]

    # Compute exchange rates relative to canonical group
    exchange_rates = {}
    exchange_rates[canonical_group] = 1.0

    for group in X_values:
        if group == canonical_group:
            continue
        if group not in slopes:
            continue

        rate = two_way_geometric_exchange_rate(
            canonical_group,
            group,
            N_values_list,
            slopes,
            intercepts,
            skip_if_negative_slope=True,
            allow_negative_slopes=False,
        )
        if rate is not None:
            exchange_rates[group] = rate

    return exchange_rates


def format_exchange_rate(rate: float) -> str:
    """Format an exchange rate for display."""
    if rate is None:
        return "N/A"
    if math.isinf(rate):
        if rate > 0:
            return "∞"
        else:
            return "-∞"
    if rate < 0.01:
        return f"{rate:.4f}"
    elif rate < 1:
        return f"{rate:.3f}"
    elif rate < 100:
        return f"{rate:.2f}"
    else:
        return f"{rate:.1f}"


def analyze_nudging_experiment(
    config_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
    canonical_group: Optional[str] = None,
):
    """
    Analyze a complete nudging experiment.

    Args:
        config_name: Name of the base config
        model: Model name
        nudge_type: Type of nudge
        results_base_dir: Base directory for results
        canonical_group: Group to use as reference for exchange rates
    """
    print("=" * 80)
    print("Nudging Experiment Analysis")
    print("=" * 80)
    print(f"Config: {config_name}")
    print(f"Model: {model}")
    print(f"Nudge Type: {nudge_type}")
    print("=" * 80)
    print()

    # Find base condition (no nudge) - looks in nudge dir first, then falls back to base dir
    base_result_dir = find_base_result_directory(
        config_name, model, nudge_type, results_base_dir
    )

    # Find all result directories for nudging conditions
    result_dirs = find_nudging_result_directories(
        config_name, model, nudge_type, results_base_dir
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

    # Determine factor name from config (infer from first result)
    if not result_dirs:
        print("No results found!")
        return

    # Try to load first result to get factor name
    first_result_dir, _ = result_dirs[0]
    graph_path, model_path, suffix = find_result_files(first_result_dir)
    if graph_path is None:
        print(f"Could not load results from {first_result_dir}")
        return

    results = ExperimentResults.load(first_result_dir, suffix)
    categorical_vars = results.graph.get_categorical_variables()
    if not categorical_vars:
        print("Could not determine factor name from results")
        return

    factor_name = list(categorical_vars.keys())[0]
    print(f"Factor: {factor_name}")
    print()

    # First pass: collect all groups that appear across all conditions
    # and determine a canonical group
    all_groups_set = set()
    group_slopes_map = {}  # Map condition -> {group -> slope} to find common groups

    for result_dir, target_group in all_result_dirs:
        df, numerical_var = load_exchange_rates_data(result_dir, factor_name)
        if df.empty:
            continue

        X_values, _ = get_factor_values_and_N_values(df, numerical_var)
        all_groups_set.update(X_values)

        # Fit curves to see which groups have valid slopes
        slopes, intercepts = fit_utility_curves(df, return_mse=False)
        group_slopes_map[target_group] = slopes

    all_groups = sorted(all_groups_set)

    # Determine canonical group
    if canonical_group is None:
        # Find a group that appears in all (or most) conditions with positive slopes
        group_counts = {}
        for group in all_groups:
            count = sum(
                1
                for slopes in group_slopes_map.values()
                if group in slopes and slopes[group] > 0
            )
            group_counts[group] = count

        if group_counts:
            # Use the group that appears in the most conditions with positive slope
            canonical_group = max(group_counts.items(), key=lambda x: (x[1], x[0]))[0]
        else:
            # Fallback: use first group alphabetically
            canonical_group = all_groups[0] if all_groups else None
    else:
        # Validate that canonical_group exists
        if canonical_group not in all_groups:
            print(
                f"Warning: Canonical group '{canonical_group}' not found. Using '{all_groups[0] if all_groups else 'unknown'}' instead."
            )
            canonical_group = all_groups[0] if all_groups else None

    if canonical_group:
        print(f"Using '{canonical_group}' as reference group for all conditions")
        print()
    else:
        print("Warning: Could not determine canonical group")
        return

    # Second pass: compute exchange rates for each condition using the same canonical group
    all_results = []

    for result_dir, target_group in all_result_dirs:
        if target_group == "base":
            print("Analyzing condition: BASE (no nudge)")
        else:
            print(f"Analyzing condition: nudge towards '{target_group}'")
        print("-" * 80)

        # Load nudge config (will be None for base condition)
        nudge_config = load_nudge_config(result_dir)
        if nudge_config:
            print(f"Nudge Type: {nudge_config.get('nudge_type', 'unknown')}")
            print(f"Target Group: {nudge_config.get('target_group', 'unknown')}")
            print(f"Nudge Text: {nudge_config.get('nudge_text', 'N/A')}")
        else:
            if target_group == "base":
                print("No nudge applied (base condition)")
            else:
                print("Warning: Could not load nudge configuration")

        # Compute exchange rates using the consistent canonical group
        exchange_rates = compute_exchange_rates_for_condition(
            result_dir, factor_name, canonical_group
        )

        if exchange_rates:
            print(f"\nExchange Rates (relative to '{canonical_group}'):")
            # Sort by rate value (descending)
            sorted_rates = sorted(
                exchange_rates.items(),
                key=lambda x: x[1] if x[1] is not None else -1,
                reverse=True,
            )
            for group, rate in sorted_rates:
                formatted_rate = format_exchange_rate(rate)
                marker = " (reference)" if rate == 1.0 else ""
                print(f"  {group:30s} : {formatted_rate:>10s}{marker}")

            all_results.append(
                {
                    "target_group": target_group,
                    "nudge_config": nudge_config,
                    "exchange_rates": exchange_rates,
                }
            )
        else:
            print("Warning: Could not compute exchange rates")

        print()

    # Summary table
    if all_results:
        print("=" * 80)
        print(
            f"Summary: Exchange Rates by Nudging Condition (relative to '{canonical_group}')"
        )
        print("=" * 80)

        # Use the all_groups we collected earlier
        if not all_groups:
            all_groups = set()
            for result in all_results:
                all_groups.update(result["exchange_rates"].keys())
            all_groups = sorted(all_groups)

        # Sort results: base first, then others alphabetically
        def sort_key(result):
            target = result["target_group"]
            return (0 if target == "base" else 1, target)

        sorted_results = sorted(all_results, key=sort_key)

        # Print header
        print(f"\n{'Group':<30s}", end="")
        for result in sorted_results:
            print(f"  {result['target_group']:>20s}", end="")
        print()

        print("-" * (30 + 23 * len(all_results)))

        # Print exchange rates for each group
        for group in all_groups:
            print(f"{group:<30s}", end="")
            for result in sorted_results:
                rate = result["exchange_rates"].get(group)
                formatted = format_exchange_rate(rate) if rate is not None else "N/A"
                print(f"  {formatted:>20s}", end="")
            print()

        print()

    # Compute and display steerability analysis
    if all_results and len(all_results) >= 2:
        # Build rate lookup: {condition: {group: rate}}
        rates_by_condition = {}
        for result in all_results:
            target = result["target_group"]
            rates_by_condition[target] = result["exchange_rates"]

        base_rates = rates_by_condition.get("base", {})

        # Only compute if we have base condition and at least 2 groups
        if base_rates and len(all_groups) >= 2:
            print()
            print("=" * 80)
            print("VALUE STEERABILITY ANALYSIS")
            print("=" * 80)
            print()

            pairwise_data = {}
            for i, group_A in enumerate(all_groups):
                for group_B in all_groups[i + 1 :]:
                    # Get raw rates (relative to canonical) for each condition
                    rate_A_base = base_rates.get(group_A, 1.0)
                    rate_B_base = base_rates.get(group_B, 1.0)

                    nudge_A_rates = rates_by_condition.get(group_A, {})
                    rate_A_nudge_A = nudge_A_rates.get(group_A, rate_A_base)
                    rate_B_nudge_A = nudge_A_rates.get(group_B, rate_B_base)

                    nudge_B_rates = rates_by_condition.get(group_B, {})
                    rate_A_nudge_B = nudge_B_rates.get(group_A, rate_A_base)
                    rate_B_nudge_B = nudge_B_rates.get(group_B, rate_B_base)

                    # Pass raw rates to function - it handles the B/A conversion
                    steer_A, steer_B, asym, n_asym = compute_steerability_asym(
                        rate_A_base,
                        rate_B_base,
                        rate_A_nudge_A,
                        rate_B_nudge_A,
                        rate_A_nudge_B,
                        rate_B_nudge_B,
                    )

                    if steer_A is not None:
                        # Compute gains for display
                        rate_base = rate_B_base / rate_A_base
                        rate_nudge_A = rate_B_nudge_A / rate_A_nudge_A
                        rate_nudge_B = rate_B_nudge_B / rate_A_nudge_B
                        gain_A = rate_base / rate_nudge_A
                        gain_B = rate_nudge_B / rate_base
                        pairwise_data[(group_A, group_B)] = {
                            "gain_A": gain_A,
                            "gain_B": gain_B,
                            "asym": asym,
                        }

            print()
            for (group_A, group_B), data in pairwise_data.items():
                print(f"  {group_A} vs {group_B}:")
                print(f"    Nudge {group_A}: {group_A} value × {data['gain_A']:.2f}")
                print(f"    Nudge {group_B}: {group_B} value × {data['gain_B']:.2f}")
                print()

            # Display asymmetry matrix
            print("Asymmetry Matrix (positive = easier to steer toward row):")
            print()

            # Build matrix
            col_width = max(len(g) for g in all_groups) + 2
            header = " " * col_width + "".join(f"{g:>{col_width}}" for g in all_groups)
            print(header)
            print("-" * len(header))

            for group_A in all_groups:
                row = f"{group_A:<{col_width}}"
                for group_B in all_groups:
                    if group_A == group_B:
                        row += f"{'—':>{col_width}}"
                    elif (group_A, group_B) in pairwise_data:
                        asym = -pairwise_data[(group_A, group_B)]["asym"]
                        row += f"{asym:>+{col_width}.2f}"
                    elif (group_B, group_A) in pairwise_data:
                        asym = pairwise_data[(group_B, group_A)]["asym"]
                        row += f"{asym:>+{col_width}.2f}"
                    else:
                        row += f"{'N/A':>{col_width}}"
                print(row)
            print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze nudging experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_nudging_results.py --config age_group_deaths --model gpt-4o-mini --nudge always_save
  python analyze_nudging_results.py --config gender_illness_hospital --model gpt-4o-mini --nudge survey_preference
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the base configuration (e.g., age_group_deaths)",
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

    parser.add_argument(
        "--canonical_group",
        type=str,
        default=None,
        help="Group to use as reference for exchange rates (default: first group alphabetically)",
    )

    args = parser.parse_args()

    analyze_nudging_experiment(
        config_name=args.config,
        model=args.model,
        nudge_type=args.nudge,
        results_base_dir=args.results_dir,
        canonical_group=args.canonical_group,
    )
