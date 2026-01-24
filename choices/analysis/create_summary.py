#!/usr/bin/env python3
"""
Create a summary table of nudge experiment results with frequency-based metrics.

This script creates a table showing frequency metrics:
- f_0(B): Baseline frequency of choosing B
- f_A(B): Frequency of choosing B when nudged towards A
- f_B(B): Frequency of choosing B when nudged towards B
- Avg f(B): Average of f_A(B) and f_B(B)
- Steerability metrics and bias
- Statistical significance markers

Usage:
    # Discover all results from default results directory
    uv run python -m choices.analysis.create_summary

    # Specify results directories
    uv run python -m choices.analysis.create_summary --results-dirs results results_anthropic

    # Filter by models, factors, nudge types
    uv run python -m choices.analysis.create_summary \
        --models claude-haiku-4-5 claude-haiku-4-5-thinking \
        --factors age_group social_status \
        --nudge-types user_preference

    # Filter by reasoning condition (as displayed in the Reasoning column)
    uv run python -m choices.analysis.create_summary --reasoning low medium high
    uv run python -m choices.analysis.create_summary --reasoning none before after

    # Filter by baseline significance (whether f_0(B) differs from 0.5)
    uv run python -m choices.analysis.create_summary --baseline-sig sig      # only biased baselines
    uv run python -m choices.analysis.create_summary --baseline-sig not-sig  # only unbiased baselines

    # Sort by a specific column (ascending by default)
    uv run python -m choices.analysis.create_summary --sort steer_bias
    uv run python -m choices.analysis.create_summary --sort abs_effect --reverse

    # Sort by absolute value (prefix column with "abs-")
    uv run python -m choices.analysis.create_summary --sort abs-steer_bias --reverse

    # Output to CSV
    uv run python -m choices.analysis.create_summary --output summary.csv

    # Set decimal places for displayed values
    uv run python -m choices.analysis.create_summary --decimals 3
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from choices.analysis.analyze_simple_nudging_results import (
    binomial_test_vs_half,
    two_proportion_z_test,
)
from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.steerability_metric import (
    compute_steerability_bias_from_counts,
)
from choices.analysis.utils import (
    get_base_model_name,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)

# Default significance level (95% confidence)
DEFAULT_ALPHA = 0.05

# Mapping from display column names to FrequencyResult attributes
# Keys are lowercase for case-insensitive matching
COLUMN_TO_ATTR = {
    "model": "model",
    "reasoning": "reasoning_condition",
    "factor": "factor",
    "nudge_type": "nudge_type",
    "nudge type": "nudge_type",
    "invalid%": "invalid_pct",
    "invalid_pct": "invalid_pct",
    "f_0(b)": "f_0_B",
    "f_0_b": "f_0_B",
    "f_a(b)": "f_A_B",
    "f_a_b": "f_A_B",
    "f_b(b)": "f_B_B",
    "f_b_b": "f_B_B",
    "avg f(b)": "avg_f_B",
    "avg_f_b": "avg_f_B",
    "|effect|": "abs_effect",
    "abs_effect": "abs_effect",
    "effect": "abs_effect",
    "steer(a)": "steerability_A",
    "steerability_a": "steerability_A",
    "steer_a": "steerability_A",
    "steer(b)": "steerability_B",
    "steerability_b": "steerability_B",
    "steer_b": "steerability_B",
    "avg steer": "avg_steerability",
    "avg_steer": "avg_steerability",
    "avg_steerability": "avg_steerability",
    "|steer|": "abs_steerability",
    "abs_steer": "abs_steerability",
    "abs_steerability": "abs_steerability",
    "steer bias": "steerability_bias",
    "steer_bias": "steerability_bias",
    "steerability_bias": "steerability_bias",
    "n_comparisons": "n_comparisons",
}


def sort_results(
    results: List["FrequencyResult"],
    sort_column: Optional[str] = None,
    reverse: bool = False,
) -> List["FrequencyResult"]:
    """
    Sort results by the specified column.

    Args:
        results: List of FrequencyResult objects
        sort_column: Column name to sort by. Prefix with "abs-" to sort by absolute value.
        reverse: If True, sort in descending order

    Returns:
        Sorted list of results
    """
    if not sort_column:
        # Default sort: by base_model, factor, nudge_type, reasoning_condition
        return sorted(
            results,
            key=lambda r: (
                get_base_model_name(r.model),
                r.factor,
                r.nudge_type,
                r.reasoning_condition,
            ),
        )

    # Check for abs- prefix
    use_abs = sort_column.lower().startswith("abs-")
    if use_abs:
        sort_column = sort_column[4:]  # Remove "abs-" prefix

    # Look up the attribute name
    attr_name = COLUMN_TO_ATTR.get(sort_column.lower())
    if not attr_name:
        print(f"Warning: Unknown sort column '{sort_column}'. Using default sort.")
        print(f"Valid columns: {', '.join(sorted(set(COLUMN_TO_ATTR.values())))}")
        return sort_results(results, None, reverse)

    def sort_key(r: "FrequencyResult"):
        val = getattr(r, attr_name)
        # Handle None values - put them at the end
        if val is None:
            return (1, 0)  # (is_none, value) - None values sort last
        if use_abs:
            return (0, abs(val))
        return (0, val)

    return sorted(results, key=sort_key, reverse=reverse)


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
    # Effect size metric
    abs_effect: float  # (|f_A(A) - f_0(A)| + |f_B(B) - f_0(B)|) / 2
    # Steerability metrics
    steerability_A: Optional[float]  # Steerability towards A
    steerability_B: Optional[float]  # Steerability towards B
    avg_steerability: Optional[float]  # Average of steerability_A and steerability_B
    abs_steerability: Optional[float]  # (|Steer(A)| + |Steer(B)|) / 2
    steerability_bias: Optional[float]  # steerability_B - steerability_A
    # Backfire metrics (nudge decreases frequency of target option)
    backfire_A: bool  # True if f_A(A) < f_0(A) (nudging towards A decreased A)
    backfire_B: bool  # True if f_B(B) < f_0(B) (nudging towards B decreased B)
    # Significance metrics (z-test comparing nudge to baseline)
    sig_A: bool  # True if f_A(A) differs significantly from f_0(A)
    sig_B: bool  # True if f_B(B) differs significantly from f_0(B)
    # Baseline significance (binomial test vs 0.5)
    sig_baseline_B: bool  # True if f_0(B) differs significantly from 0.5
    # Sample info
    n_comparisons: int  # Number of pairwise comparisons
    invalid_pct: float  # Percentage of invalid responses


def compute_factor_frequencies_with_counts(
    graph_data: Dict,
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute win frequencies and sample counts for each factor level.

    Returns:
        Dictionary mapping level -> {"freq": float, "n": int, "wins": int}
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0, "total": 0} for level in target_levels}

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

    # Compute frequencies with counts
    result = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            result[level] = {
                "freq": stats["wins"] / stats["total"],
                "n": stats["total"],
                "wins": stats["wins"],
            }
        else:
            result[level] = {"freq": 0.5, "n": 0, "wins": 0}

    return result


def get_nudge_target_group(result_dir: Path) -> Optional[str]:
    """Get the target group for a nudge condition from the graph data."""
    graph_data = load_preference_graph(result_dir)
    if not graph_data:
        return None

    nudge_config = graph_data.get("nudge_config")
    if nudge_config:
        return nudge_config.get("target_group")
    return None


def count_responses(graph_data: Dict) -> tuple[int, int]:
    """
    Count valid and total responses in a preference graph.

    Returns:
        Tuple of (valid_count, total_count)
    """
    edges = graph_data.get("edges", {})
    valid_count = 0
    total_count = 0

    for edge_data in edges.values():
        aux_data = edge_data.get("aux_data", {})
        original_parsed = aux_data.get("original_parsed", [])
        flipped_parsed = aux_data.get("flipped_parsed", [])

        # Count all responses
        total_count += len(original_parsed) + len(flipped_parsed)

        # Count valid responses (A or B, not None or other)
        for resp in original_parsed:
            if resp in ("A", "B"):
                valid_count += 1
        for resp in flipped_parsed:
            if resp in ("A", "B"):
                valid_count += 1

    return valid_count, total_count


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


def _compute_single_frequency_result(
    factor_name: str,
    model: str,
    nudge_type: str,
    condition_dirs: Dict[str, Path],
) -> Optional[FrequencyResult]:
    """
    Compute frequency metrics for a single experiment given its condition directories.

    Args:
        factor_name: Name of the factor (e.g., 'age_group')
        model: Model name
        nudge_type: Type of nudge (e.g., 'user_preference')
        condition_dirs: Dictionary mapping condition name -> Path to result directory

    Returns:
        FrequencyResult object or None if data is insufficient
    """
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

    # Compute frequencies for level B under each condition (with sample counts)
    target_levels = [level_A, level_B]

    base_stats = compute_factor_frequencies_with_counts(
        base_graph, factor_var_name, target_levels
    )
    nudge_A_stats = compute_factor_frequencies_with_counts(
        nudge_A_graph, factor_var_name, target_levels
    )
    nudge_B_stats = compute_factor_frequencies_with_counts(
        nudge_B_graph, factor_var_name, target_levels
    )

    # Get frequencies, sample sizes, and win counts for both levels
    f_0_A = base_stats.get(level_A, {}).get("freq", 0.5)
    f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
    n_0_A = base_stats.get(level_A, {}).get("n", 0)
    n_0_B = base_stats.get(level_B, {}).get("n", 0)
    c_0_A = base_stats.get(level_A, {}).get("wins", 0)
    c_0_B = base_stats.get(level_B, {}).get("wins", 0)

    f_A_A = nudge_A_stats.get(level_A, {}).get("freq", 0.5)
    f_A_B = nudge_A_stats.get(level_B, {}).get("freq", 0.5)
    n_A_A = nudge_A_stats.get(level_A, {}).get("n", 0)
    c_A_A = nudge_A_stats.get(level_A, {}).get("wins", 0)
    c_A_B = nudge_A_stats.get(level_B, {}).get("wins", 0)

    f_B_B = nudge_B_stats.get(level_B, {}).get("freq", 0.5)
    n_B_B = nudge_B_stats.get(level_B, {}).get("n", 0)
    c_B_A = nudge_B_stats.get(level_A, {}).get("wins", 0)
    c_B_B = nudge_B_stats.get(level_B, {}).get("wins", 0)

    # Average of f_A(B) and f_B(B)
    avg_f_B = (f_A_B + f_B_B) / 2

    # Compute absolute effect size: (|f_A(A) - f_0(A)| + |f_B(B) - f_0(B)|) / 2
    abs_effect = (abs(f_A_A - f_0_A) + abs(f_B_B - f_0_B)) / 2

    # Compute backfire metrics (nudge decreases frequency of target option)
    backfire_A = f_A_A < f_0_A  # Nudging towards A decreased frequency of A
    backfire_B = f_B_B < f_0_B  # Nudging towards B decreased frequency of B

    # Compute significance using two-proportion z-test
    # Test if nudge towards A significantly changed frequency of A
    test_A = two_proportion_z_test(f_0_A, n_0_A, f_A_A, n_A_A, DEFAULT_ALPHA)
    sig_A = test_A["is_significant"]

    # Test if nudge towards B significantly changed frequency of B
    test_B = two_proportion_z_test(f_0_B, n_0_B, f_B_B, n_B_B, DEFAULT_ALPHA)
    sig_B = test_B["is_significant"]

    # Test if baseline f_0(B) differs significantly from 0.5 (binomial test)
    test_baseline_B = binomial_test_vs_half(c_0_B, n_0_B, DEFAULT_ALPHA)
    sig_baseline_B = test_baseline_B["is_significant"]

    # Compute steerability metrics using counts (with Haldane-Anscombe correction)
    steerability_A, steerability_B, steerability_bias = (
        compute_steerability_bias_from_counts(c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B)
    )

    # Compute average steerability (signed)
    avg_steerability = None
    if steerability_A is not None and steerability_B is not None:
        avg_steerability = (steerability_A + steerability_B) / 2

    # Compute absolute steerability: (|Steer(A)| + |Steer(B)|) / 2
    abs_steerability = None
    if steerability_A is not None and steerability_B is not None:
        abs_steerability = (abs(steerability_A) + abs(steerability_B)) / 2

    # Get sample info
    n_comparisons = len(base_graph.get("edges", {}))

    # Count valid and total responses across all conditions
    valid_base, total_base = count_responses(base_graph)
    valid_A, total_A = count_responses(nudge_A_graph)
    valid_B, total_B = count_responses(nudge_B_graph)

    total_valid = valid_base + valid_A + valid_B
    total_responses = total_base + total_A + total_B

    # Compute invalid percentage
    invalid_pct = (
        ((total_responses - total_valid) / total_responses * 100)
        if total_responses > 0
        else 0.0
    )

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
        abs_effect=abs_effect,
        steerability_A=steerability_A,
        steerability_B=steerability_B,
        avg_steerability=avg_steerability,
        abs_steerability=abs_steerability,
        steerability_bias=steerability_bias,
        backfire_A=backfire_A,
        backfire_B=backfire_B,
        sig_A=sig_A,
        sig_B=sig_B,
        sig_baseline_B=sig_baseline_B,
        n_comparisons=n_comparisons,
        invalid_pct=invalid_pct,
    )


def compute_frequency_results(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> List[FrequencyResult]:
    """
    Compute frequency metrics for all experiments matching the given parameters.

    This handles cases where the same model/factor/nudge_type combination has
    multiple experiment runs with different reasoning_mode settings.

    Args:
        factor_name: Name of the factor (e.g., 'age_group')
        model: Model name
        nudge_type: Type of nudge (e.g., 'user_preference')
        results_base_dir: Base directory for results

    Returns:
        List of FrequencyResult objects (one per unique reasoning_mode)
    """
    # Find all experiment sets (one per reasoning_mode)
    experiment_sets = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    results = []
    for condition_dirs in experiment_sets:
        result = _compute_single_frequency_result(
            factor_name, model, nudge_type, condition_dirs
        )
        if result is not None:
            results.append(result)

    return results


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
        # compute_frequency_results returns a list (one per reasoning_mode)
        experiment_results = compute_frequency_results(
            factor_name, model, nudge_type, results_base_dir
        )
        results.extend(experiment_results)

    return results


# Ordered list of (display_header, key) tuples - single source of truth for columns
TABLE_COLUMNS = [
    ("Model", "model"),
    ("Reasoning", "reasoning"),
    ("Factor", "factor"),
    ("Nudge Type", "nudge_type"),
    ("Invalid%", "invalid_pct"),
    ("f_0(B)", "f_0_b"),
    ("f_A(B)", "f_a_b"),
    ("f_B(B)", "f_b_b"),
    ("Avg f(B)", "avg_f_b"),
    ("|Effect|", "effect"),
    ("Steer(A)", "steer_a"),
    ("Steer(B)", "steer_b"),
    ("Avg Steer", "avg_steer"),
    ("|Steer|", "abs_steer"),
    ("Steer Bias", "steer_bias"),
    ("Backfire", "backfire"),
]

# Derived mappings for lookups
HEADER_TO_KEY = {header.lower(): key for header, key in TABLE_COLUMNS}
KEY_TO_HEADER = {key: header for header, key in TABLE_COLUMNS}


def format_table(
    results: List[FrequencyResult],
    show_display_names: bool = True,
    decimals: int = 2,
    sort_column: Optional[str] = None,
    reverse: bool = False,
    hide_columns: Optional[List[str]] = None,
) -> str:
    """Format results as a text table."""
    if not results:
        return "No results found."

    # Sort results
    results = sort_results(results, sort_column, reverse)

    # Normalize hidden columns to lowercase
    hidden_set = set()
    if hide_columns:
        for col in hide_columns:
            col_lower = col.lower().replace("-", "_").replace(" ", "_")
            hidden_set.add(col_lower)

    # Filter out hidden columns
    visible_columns = [
        (header, key) for header, key in TABLE_COLUMNS if key not in hidden_set
    ]
    headers = [header for header, _ in visible_columns]
    visible_keys = [key for _, key in visible_columns]

    # Build rows
    rows = []
    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        steer_A_str = (
            f"{r.steerability_A:.{decimals}f}"
            if r.steerability_A is not None
            else "N/A"
        )
        steer_B_str = (
            f"{r.steerability_B:.{decimals}f}"
            if r.steerability_B is not None
            else "N/A"
        )
        avg_steer_str = (
            f"{r.avg_steerability:.{decimals}f}"
            if r.avg_steerability is not None
            else "N/A"
        )
        abs_steer_str = (
            f"{r.abs_steerability:.{decimals}f}"
            if r.abs_steerability is not None
            else "N/A"
        )
        steer_bias_str = (
            f"{r.steerability_bias:+.{decimals}f}"
            if r.steerability_bias is not None
            else "N/A"
        )
        factor_with_levels = f"{r.level_A}/{r.level_B}"

        # Format frequency columns with asterisks for significant changes
        f_A_B_str = f"{r.f_A_B:.{decimals}f}{'*' if r.sig_A else ''}"
        f_B_B_str = f"{r.f_B_B:.{decimals}f}{'*' if r.sig_B else ''}"

        # Backfire column: show which nudges backfired (no significance markers here)
        backfire_parts = []
        if r.backfire_A:
            backfire_parts.append("A")
        if r.backfire_B:
            backfire_parts.append("B")
        backfire_str = ",".join(backfire_parts) if backfire_parts else "None"

        # Map keys to values
        all_values = {
            "model": model_name,
            "reasoning": r.reasoning_condition,
            "factor": factor_with_levels,
            "nudge_type": r.nudge_type,
            "invalid_pct": f"{r.invalid_pct:.1f}%",
            "f_0_b": f"{r.f_0_B:.{decimals}f}",
            "f_a_b": f_A_B_str,
            "f_b_b": f_B_B_str,
            "avg_f_b": f"{r.avg_f_B:.{decimals}f}",
            "effect": f"{r.abs_effect:.{decimals}f}",
            "steer_a": steer_A_str,
            "steer_b": steer_B_str,
            "avg_steer": avg_steer_str,
            "abs_steer": abs_steer_str,
            "steer_bias": steer_bias_str,
            "backfire": backfire_str,
        }

        rows.append([all_values[key] for key in visible_keys])

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
    sort_column: Optional[str] = None,
    reverse: bool = False,
) -> None:
    """Write results to a CSV file."""
    if not results:
        print("No results to write.")
        return

    # Sort results
    results = sort_results(results, sort_column, reverse)

    headers = [
        "model",
        "model_display_name",
        "reasoning_condition",
        "factor",
        "level_A",
        "level_B",
        "nudge_type",
        "invalid_pct",
        "n_comparisons",
        "f_0_B",
        "f_A_B",
        "f_B_B",
        "avg_f_B",
        "abs_effect",
        "steerability_A",
        "steerability_B",
        "avg_steerability",
        "abs_steerability",
        "steerability_bias",
        "backfire_A",
        "backfire_B",
        "sig_A",
        "sig_B",
        "sig_baseline_B",
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
                    r.invalid_pct,
                    r.n_comparisons,
                    r.f_0_B,
                    r.f_A_B,
                    r.f_B_B,
                    r.avg_f_B,
                    r.abs_effect,
                    r.steerability_A if r.steerability_A is not None else "",
                    r.steerability_B if r.steerability_B is not None else "",
                    r.avg_steerability if r.avg_steerability is not None else "",
                    r.abs_steerability if r.abs_steerability is not None else "",
                    r.steerability_bias if r.steerability_bias is not None else "",
                    r.backfire_A,
                    r.backfire_B,
                    r.sig_A,
                    r.sig_B,
                    r.sig_baseline_B,
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

    # Filter by reasoning condition (as shown in the table)
    python create_frequency_table.py --reasoning low medium high
    python create_frequency_table.py --reasoning none before after

    # Sort by a column (use abs- prefix for absolute value sorting)
    python create_frequency_table.py --sort steer_bias
    python create_frequency_table.py --sort abs-steer_bias --reverse

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
        "--reasoning",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include, as they appear in the table "
        "(e.g., 'low', 'medium', 'high', 'off', 'before', 'after', 'none', '10000')",
    )

    parser.add_argument(
        "--baseline-sig",
        type=str,
        choices=["any", "sig", "not-sig"],
        default="any",
        help="Filter by significance of f_0(B) vs 0.5: "
        "'any' = no filter (default), "
        "'sig' = only include cases where f_0(B) differs significantly from 0.5, "
        "'not-sig' = only include cases where f_0(B) does NOT differ significantly from 0.5",
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

    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default=None,
        help="Column to sort by. Prefix with 'abs-' to sort by absolute value "
        "(e.g., 'steer_bias', 'abs-steer_bias', 'abs_effect'). "
        "Valid columns: model, reasoning, factor, nudge_type, invalid_pct, "
        "f_0_B, f_A_B, f_B_B, avg_f_B, abs_effect, steerability_A, steerability_B, "
        "avg_steerability, abs_steerability, steerability_bias",
    )

    parser.add_argument(
        "--reverse",
        "-r",
        action="store_true",
        help="Sort in descending order (default: ascending)",
    )

    parser.add_argument(
        "--decimals",
        "-d",
        type=int,
        default=2,
        help="Number of decimal places for frequency values (default: 2)",
    )

    parser.add_argument(
        "--hide-columns",
        nargs="+",
        default=None,
        help="List of columns to hide. Column names (case-insensitive): "
        + ", ".join(key for _, key in TABLE_COLUMNS),
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
    if args.reasoning:
        print(f"Reasoning condition filter: {args.reasoning}")
    if args.baseline_sig != "any":
        sig_desc = (
            "significantly different from 0.5"
            if args.baseline_sig == "sig"
            else "NOT significantly different from 0.5"
        )
        print(f"Baseline significance filter: f_0(B) {sig_desc}")
    if args.sort:
        sort_desc = f"Sort by: {args.sort}"
        if args.reverse:
            sort_desc += " (descending)"
        print(sort_desc)
    if args.hide_columns:
        print(f"Hidden columns: {args.hide_columns}")
    print("=" * 80)
    print()

    # Compute results
    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    # Apply reasoning condition filter (post-computation since it's derived from results)
    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]

    # Apply baseline significance filter
    if args.baseline_sig == "sig":
        results = [r for r in results if r.sig_baseline_B]
    elif args.baseline_sig == "not-sig":
        results = [r for r in results if not r.sig_baseline_B]

    print(f"Found {len(results)} complete experiments\n")

    if not results:
        print("No complete experiments found matching the filters.")
        return

    show_display_names = not args.no_display_names
    decimals = args.decimals
    sort_column = args.sort
    reverse = args.reverse

    if args.output:
        write_csv(results, args.output, show_display_names, sort_column, reverse)
    else:
        print(
            format_table(
                results,
                show_display_names,
                decimals,
                sort_column,
                reverse,
                args.hide_columns,
            )
        )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    from collections import defaultdict

    # Helper to compute steerability stats
    def get_steer_stats(
        result_list: List[FrequencyResult],
    ) -> Tuple[
        Optional[float],
        Optional[float],
        float,
        Optional[float],
        float,
        float,
        float,
    ]:
        """
        Returns (avg_steer, avg_bias, avg_effect, avg_abs_steer,
                 sig_rate, sig_backfire_rate, backfire_rate).
        - sig_rate: fraction of nudges with significant change
        - sig_backfire_rate: fraction of nudges that backfired significantly
        - backfire_rate: fraction of nudges that backfired (regardless of significance)
        """
        steer_results = [r for r in result_list if r.avg_steerability is not None]
        bias_results = [r for r in result_list if r.steerability_bias is not None]
        abs_steer_results = [r for r in result_list if r.abs_steerability is not None]
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
        avg_effect = (
            sum(r.abs_effect for r in result_list) / len(result_list)
            if result_list
            else 0.0
        )
        avg_abs_steer = (
            sum(r.abs_steerability for r in abs_steer_results) / len(abs_steer_results)
            if abs_steer_results
            else None
        )
        # Each result has 2 nudges (towards A and towards B), so total nudges = 2 * n
        total_nudges = 2 * len(result_list)

        # Compute significance rate: fraction of nudges with significant change
        # Note: sig_A and sig_B may be numpy bool_ which doesn't add correctly,
        # so we convert to int explicitly
        sig_count = sum(int(r.sig_A) + int(r.sig_B) for r in result_list)
        sig_rate = sig_count / total_nudges if total_nudges > 0 else 0.0

        # Compute significant backfire rate: fraction of nudges that backfired significantly
        # Only count backfires that are also statistically significant
        sig_backfire_count = sum(
            int(r.backfire_A and r.sig_A) + int(r.backfire_B and r.sig_B)
            for r in result_list
        )
        sig_backfire_rate = (
            sig_backfire_count / total_nudges if total_nudges > 0 else 0.0
        )

        # Compute backfire rate: fraction of nudges that backfired (regardless of significance)
        backfire_count = sum(int(r.backfire_A) + int(r.backfire_B) for r in result_list)
        backfire_rate = backfire_count / total_nudges if total_nudges > 0 else 0.0

        return (
            avg_steer,
            avg_bias,
            avg_effect,
            avg_abs_steer,
            sig_rate,
            sig_backfire_rate,
            backfire_rate,
        )

    # By model and reasoning condition
    model_groups: Dict[Tuple[str, str], List[FrequencyResult]] = defaultdict(list)
    for r in results:
        base_model = get_base_model_name(r.model)
        model_groups[(base_model, r.reasoning_condition)].append(r)

    print(f"\nModels ({len(model_groups)}):")
    for base_model, reasoning_condition in sorted(model_groups.keys()):
        model_results = model_groups[(base_model, reasoning_condition)]
        n_factors = len(set(r.factor for r in model_results))
        (
            avg_steer,
            avg_bias,
            avg_effect,
            avg_abs_steer,
            sig_rate,
            sig_backfire_rate,
            backfire_rate,
        ) = get_steer_stats(model_results)

        display_name = (
            get_model_display_name(model_results[0].model)
            if show_display_names
            else base_model
        )

        effect_str = f"|effect|={avg_effect:.{decimals}f}"
        abs_steer_str = (
            f"|steer|={avg_abs_steer:.{decimals}f}"
            if avg_abs_steer is not None
            else "|steer|=N/A"
        )
        sig_str = f"sig={sig_rate:.1%}"
        backfire_str = f"sig_backfire={sig_backfire_rate:.1%}"

        if n_factors > 1:
            # Multiple factors: only show steerability metrics
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {display_name} ({reasoning_condition}): n={len(model_results)}, {effect_str}, "
                f"{abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )
        else:
            # Single factor: show frequency metrics and steerability
            avg_f_0_B = sum(r.f_0_B for r in model_results) / len(model_results)
            avg_f_A_B = sum(r.f_A_B for r in model_results) / len(model_results)
            avg_f_B_B = sum(r.f_B_B for r in model_results) / len(model_results)
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {display_name} ({reasoning_condition}): n={len(model_results)}, "
                f"f_0(B)={avg_f_0_B:.{decimals}f}, f_A(B)={avg_f_A_B:.{decimals}f}, "
                f"f_B(B)={avg_f_B_B:.{decimals}f}, {effect_str}, {abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )

    # By reasoning condition
    reasoning_groups: Dict[str, List[FrequencyResult]] = defaultdict(list)
    for r in results:
        reasoning_groups[r.reasoning_condition].append(r)

    print(f"\nReasoning Conditions ({len(reasoning_groups)}):")
    for reasoning_condition in sorted(reasoning_groups.keys()):
        reasoning_results = reasoning_groups[reasoning_condition]
        n_factors = len(set(r.factor for r in reasoning_results))
        (
            avg_steer,
            avg_bias,
            avg_effect,
            avg_abs_steer,
            sig_rate,
            sig_backfire_rate,
            backfire_rate,
        ) = get_steer_stats(reasoning_results)

        effect_str = f"|effect|={avg_effect:.{decimals}f}"
        abs_steer_str = (
            f"|steer|={avg_abs_steer:.{decimals}f}"
            if avg_abs_steer is not None
            else "|steer|=N/A"
        )
        sig_str = f"sig={sig_rate:.1%}"
        backfire_str = f"sig_backfire={sig_backfire_rate:.1%}"

        if n_factors > 1:
            # Multiple factors: only show steerability metrics
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {reasoning_condition}: n={len(reasoning_results)}, {effect_str}, "
                f"{abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )
        else:
            # Single factor: show frequency metrics and steerability
            avg_f_0_B = sum(r.f_0_B for r in reasoning_results) / len(reasoning_results)
            avg_f_A_B = sum(r.f_A_B for r in reasoning_results) / len(reasoning_results)
            avg_f_B_B = sum(r.f_B_B for r in reasoning_results) / len(reasoning_results)
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {reasoning_condition}: n={len(reasoning_results)}, "
                f"f_0(B)={avg_f_0_B:.{decimals}f}, f_A(B)={avg_f_A_B:.{decimals}f}, "
                f"f_B(B)={avg_f_B_B:.{decimals}f}, {effect_str}, {abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )

    # By factor (single factor by definition)
    factors = set(r.factor for r in results)
    print(f"\nFactors ({len(factors)}):")
    for factor in sorted(factors):
        factor_results = [r for r in results if r.factor == factor]
        avg_f_0_B = sum(r.f_0_B for r in factor_results) / len(factor_results)
        avg_f_A_B = sum(r.f_A_B for r in factor_results) / len(factor_results)
        avg_f_B_B = sum(r.f_B_B for r in factor_results) / len(factor_results)
        (
            avg_steer,
            avg_bias,
            avg_effect,
            avg_abs_steer,
            sig_rate,
            sig_backfire_rate,
            backfire_rate,
        ) = get_steer_stats(factor_results)
        # Get level info
        level_A = factor_results[0].level_A if factor_results else "?"
        level_B = factor_results[0].level_B if factor_results else "?"
        effect_str = f"|effect|={avg_effect:.{decimals}f}"
        abs_steer_str = (
            f"|steer|={avg_abs_steer:.{decimals}f}"
            if avg_abs_steer is not None
            else "|steer|=N/A"
        )
        steer_str = (
            f"avg_steer={avg_steer:.{decimals}f}"
            if avg_steer is not None
            else "avg_steer=N/A"
        )
        # Compute raw (non-absolute) steerability bias for factors
        bias_results = [r for r in factor_results if r.steerability_bias is not None]
        raw_avg_bias = (
            sum(r.steerability_bias for r in bias_results) / len(bias_results)
            if bias_results
            else None
        )
        raw_bias_str = (
            f"steer_bias={raw_avg_bias:.{decimals}f}"
            if raw_avg_bias is not None
            else "steer_bias=N/A"
        )
        abs_bias_str = (
            f"|steer_bias|={avg_bias:.{decimals}f}"
            if avg_bias is not None
            else "|steer_bias|=N/A"
        )
        sig_str = f"sig={sig_rate:.1%}"
        backfire_str = f"sig_backfire={sig_backfire_rate:.1%}"
        # Compute sig rates towards each level (considering all nudges)
        # sig_towards_A = nudge towards A worked + nudge towards B backfired
        # sig_towards_B = nudge towards B worked + nudge towards A backfired
        total_nudges_factor = 2 * len(factor_results)
        sig_towards_A = 0
        sig_towards_B = 0
        for r in factor_results:
            # Nudge towards A
            if r.sig_A:
                if r.backfire_A:
                    sig_towards_B += 1  # Backfired: shifted towards B
                else:
                    sig_towards_A += 1  # Worked: shifted towards A
            # Nudge towards B
            if r.sig_B:
                if r.backfire_B:
                    sig_towards_A += 1  # Backfired: shifted towards A
                else:
                    sig_towards_B += 1  # Worked: shifted towards B
        sig_A_rate = sig_towards_A / total_nudges_factor
        sig_B_rate = sig_towards_B / total_nudges_factor
        sig_A_str = f"sig({level_A})={sig_A_rate:.1%}"
        sig_B_str = f"sig({level_B})={sig_B_rate:.1%}"
        print(
            f"  {factor} (A={level_A}, B={level_B}): n={len(factor_results)}, "
            f"f_0(B)={avg_f_0_B:.{decimals}f}, f_A(B)={avg_f_A_B:.{decimals}f}, "
            f"f_B(B)={avg_f_B_B:.{decimals}f}, {effect_str}, {abs_steer_str}, {steer_str}, {raw_bias_str}, {abs_bias_str}, {sig_str}, {sig_A_str}, {sig_B_str}, {backfire_str}"
        )

    # By nudge type
    nudge_types = set(r.nudge_type for r in results)
    print(f"\nNudge Types ({len(nudge_types)}):")
    for nudge_type in sorted(nudge_types):
        nudge_results = [r for r in results if r.nudge_type == nudge_type]
        n_factors = len(set(r.factor for r in nudge_results))
        (
            avg_steer,
            avg_bias,
            avg_effect,
            avg_abs_steer,
            sig_rate,
            sig_backfire_rate,
            backfire_rate,
        ) = get_steer_stats(nudge_results)

        effect_str = f"|effect|={avg_effect:.{decimals}f}"
        abs_steer_str = (
            f"|steer|={avg_abs_steer:.{decimals}f}"
            if avg_abs_steer is not None
            else "|steer|=N/A"
        )
        sig_str = f"sig={sig_rate:.1%}"
        backfire_str = f"sig_backfire={sig_backfire_rate:.1%}"

        if n_factors > 1:
            # Multiple factors: only show steerability metrics
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {nudge_type}: n={len(nudge_results)}, {effect_str}, "
                f"{abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )
        else:
            # Single factor: show frequency metrics and steerability
            avg_f_0_B = sum(r.f_0_B for r in nudge_results) / len(nudge_results)
            avg_f_A_B = sum(r.f_A_B for r in nudge_results) / len(nudge_results)
            avg_f_B_B = sum(r.f_B_B for r in nudge_results) / len(nudge_results)
            steer_str = (
                f"avg_steer={avg_steer:.{decimals}f}"
                if avg_steer is not None
                else "avg_steer=N/A"
            )
            bias_str = (
                f"|steer_bias|={avg_bias:.{decimals}f}"
                if avg_bias is not None
                else "|steer_bias|=N/A"
            )
            print(
                f"  {nudge_type}: n={len(nudge_results)}, "
                f"f_0(B)={avg_f_0_B:.{decimals}f}, f_A(B)={avg_f_A_B:.{decimals}f}, "
                f"f_B(B)={avg_f_B_B:.{decimals}f}, {effect_str}, {abs_steer_str}, {steer_str}, {bias_str}, {sig_str}, {backfire_str}"
            )

    # Overall statistics
    (
        _,
        _,
        _,
        _,
        overall_sig_rate,
        overall_sig_backfire_rate,
        overall_backfire_rate,
    ) = get_steer_stats(results)
    total_nudges = 2 * len(results)

    # Significance statistics
    total_sig = sum(int(r.sig_A) + int(r.sig_B) for r in results)
    print(
        f"\nOverall Significant Change Rate: {overall_sig_rate:.1%} ({total_sig}/{total_nudges})"
    )

    # Significant backfire statistics (only counting statistically significant backfires)
    total_sig_backfire = sum(
        int(r.backfire_A and r.sig_A) + int(r.backfire_B and r.sig_B) for r in results
    )
    print(
        f"Overall Significant Backfire Rate: {overall_sig_backfire_rate:.1%} ({total_sig_backfire}/{total_nudges})"
    )


if __name__ == "__main__":
    main()
