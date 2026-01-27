#!/usr/bin/env python3
"""
Analyze invalid response rates across experiments.

This script analyzes invalid response rates broken down by:
- Condition (base, nudge towards A, nudge towards B)
- Comparison type (equal-n, larger-A, larger-B)

For each condition, shows invalid response rates for:
- Overall: all comparisons
- equal-n: comparisons where both options have same N
- larger-A: comparisons where A has larger N (based on factor level A/B naming)
- larger-B: comparisons where B has larger N

When aggregating across factors, larger-A and larger-B are combined as "different-n".

Usage:
    # Analyze invalid response rates from default results directory
    uv run python -m choices.analysis.analyze_invalid_responses

    # Specify results directories
    uv run python -m choices.analysis.analyze_invalid_responses --results-dirs results results_reasoning

    # Filter by models, factors, nudge types
    uv run python -m choices.analysis.analyze_invalid_responses \\
        --models llama-33-70b gpt-4o-mini \\
        --factors age_group social_status \\
        --nudge-types user_preference identity

    # Filter by reasoning condition
    uv run python -m choices.analysis.analyze_invalid_responses --reasoning none before after

    # Output to CSV
    uv run python -m choices.analysis.analyze_invalid_responses --output invalid_rates.csv
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.utils import (
    get_base_model_name,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)


@dataclass
class InvalidRateCounts:
    """Counts for computing invalid response rates."""

    valid: int = 0
    invalid: int = 0

    @property
    def total(self) -> int:
        return self.valid + self.invalid

    @property
    def invalid_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.invalid / self.total

    def __add__(self, other: "InvalidRateCounts") -> "InvalidRateCounts":
        return InvalidRateCounts(
            valid=self.valid + other.valid,
            invalid=self.invalid + other.invalid,
        )


@dataclass
class ConditionInvalidRates:
    """Invalid rates broken down by comparison type for a single condition."""

    overall: InvalidRateCounts = field(default_factory=InvalidRateCounts)
    equal_n: InvalidRateCounts = field(default_factory=InvalidRateCounts)
    larger_A: InvalidRateCounts = field(default_factory=InvalidRateCounts)
    larger_B: InvalidRateCounts = field(default_factory=InvalidRateCounts)

    def __add__(self, other: "ConditionInvalidRates") -> "ConditionInvalidRates":
        return ConditionInvalidRates(
            overall=self.overall + other.overall,
            equal_n=self.equal_n + other.equal_n,
            larger_A=self.larger_A + other.larger_A,
            larger_B=self.larger_B + other.larger_B,
        )


@dataclass
class ExperimentInvalidRates:
    """Invalid rates for a complete experiment (base + both nudge conditions)."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    # Invalid rates by condition
    base: ConditionInvalidRates = field(default_factory=ConditionInvalidRates)
    nudge_A: ConditionInvalidRates = field(default_factory=ConditionInvalidRates)
    nudge_B: ConditionInvalidRates = field(default_factory=ConditionInvalidRates)

    @property
    def all_conditions(self) -> ConditionInvalidRates:
        """Combined invalid rates across all conditions."""
        return self.base + self.nudge_A + self.nudge_B


def classify_comparison(
    opt_a: Dict[str, Any],
    opt_b: Dict[str, Any],
    factor_name: str,
    level_A: str,
    level_B: str,
) -> Optional[str]:
    """
    Classify a comparison based on factor levels and N values.

    Returns:
        - "equal_n" if N_a == N_b
        - "larger_A" if the option with factor level_A has larger N
        - "larger_B" if the option with factor level_B has larger N
        - None if this is an intra-group comparison (same factor level)
    """
    factor_a = opt_a.get(factor_name)
    factor_b = opt_b.get(factor_name)

    # Skip intra-group comparisons
    if factor_a == factor_b:
        return None

    n_a = opt_a.get("N", 1)
    n_b = opt_b.get("N", 1)

    if n_a == n_b:
        return "equal_n"

    # Determine which option corresponds to level_A and level_B
    if factor_a == level_A and factor_b == level_B:
        # opt_a is level_A, opt_b is level_B
        n_level_A = n_a
        n_level_B = n_b
    elif factor_a == level_B and factor_b == level_A:
        # opt_a is level_B, opt_b is level_A
        n_level_A = n_b
        n_level_B = n_a
    else:
        # Neither matches - should not happen
        return None

    if n_level_A > n_level_B:
        return "larger_A"
    else:
        return "larger_B"


def count_invalid_responses_by_type(
    graph_data: Dict[str, Any],
    factor_name: str,
    level_A: str,
    level_B: str,
) -> ConditionInvalidRates:
    """
    Count valid and invalid responses by comparison type.

    Returns:
        ConditionInvalidRates with counts for overall, equal_n, larger_A, larger_B
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    rates = ConditionInvalidRates()

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            # Classify this comparison
            comp_type = classify_comparison(opt_a, opt_b, factor_name, level_A, level_B)

            if comp_type is None:
                # Intra-group comparison, skip
                continue

            # Get response data
            aux_data = edge_data.get("aux_data", {})
            original_parsed = aux_data.get("original_parsed", [])
            flipped_parsed = aux_data.get("flipped_parsed", [])

            # Count valid and invalid responses
            for resp in original_parsed:
                if resp in ("A", "B"):
                    rates.overall.valid += 1
                    getattr(rates, comp_type).valid += 1
                else:
                    rates.overall.invalid += 1
                    getattr(rates, comp_type).invalid += 1

            for resp in flipped_parsed:
                if resp in ("A", "B"):
                    rates.overall.valid += 1
                    getattr(rates, comp_type).valid += 1
                else:
                    rates.overall.invalid += 1
                    getattr(rates, comp_type).invalid += 1

        except Exception:
            continue

    return rates


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
) -> List[Dict[str, Path]]:
    """
    Find result directories for each condition (base, and each nudge target).

    Groups directories by both condition AND reasoning_mode.

    Returns:
        List of dictionaries, each mapping condition name -> Path to result directory.
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return []

    dirs_by_condition_and_reasoning: Dict[Tuple[str, str], List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        reasoning_mode = get_reasoning_mode_from_results(result_dir)
        if reasoning_mode is None:
            reasoning_mode = "unknown"

        key = (condition, reasoning_mode)
        if key not in dirs_by_condition_and_reasoning:
            dirs_by_condition_and_reasoning[key] = []
        dirs_by_condition_and_reasoning[key].append(result_dir)

    experiments_by_reasoning: Dict[str, Dict[str, Path]] = {}

    for (condition, reasoning_mode), dirs in dirs_by_condition_and_reasoning.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)

        if reasoning_mode not in experiments_by_reasoning:
            experiments_by_reasoning[reasoning_mode] = {}
        experiments_by_reasoning[reasoning_mode][condition] = most_recent

    return list(experiments_by_reasoning.values())


def compute_experiment_invalid_rates(
    factor_name: str,
    model: str,
    nudge_type: str,
    condition_dirs: Dict[str, Path],
) -> Optional[ExperimentInvalidRates]:
    """
    Compute invalid rates for a single experiment given its condition directories.
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

    # Compute invalid rates for each condition
    base_rates = count_invalid_responses_by_type(
        base_graph, factor_var_name, level_A, level_B
    )
    nudge_A_rates = count_invalid_responses_by_type(
        nudge_A_graph, factor_var_name, level_A, level_B
    )
    nudge_B_rates = count_invalid_responses_by_type(
        nudge_B_graph, factor_var_name, level_A, level_B
    )

    # Determine reasoning condition
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    return ExperimentInvalidRates(
        model=model,
        reasoning_condition=reasoning_condition,
        factor=factor_name,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        base=base_rates,
        nudge_A=nudge_A_rates,
        nudge_B=nudge_B_rates,
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

        for exp_dir in results_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
                continue

            factor_name = exp_dir.name[7:]  # Remove 'simple_' prefix

            if factor_filter and factor_name not in factor_filter:
                continue

            for model_dir in exp_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name

                if model_filter and model not in model_filter:
                    continue

                for nudge_dir in model_dir.iterdir():
                    if not nudge_dir.is_dir():
                        continue

                    nudge_type = nudge_dir.name

                    if nudge_type_filter and nudge_type not in nudge_type_filter:
                        continue

                    experiments.append(
                        (results_base_dir, factor_name, model, nudge_type)
                    )

    return experiments


def compute_all_invalid_rates(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[ExperimentInvalidRates]:
    """
    Compute invalid rates for all available experiments.
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    results = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        experiment_sets = find_condition_directories(
            factor_name, model, nudge_type, results_base_dir
        )

        for condition_dirs in experiment_sets:
            result = compute_experiment_invalid_rates(
                factor_name, model, nudge_type, condition_dirs
            )
            if result is not None:
                results.append(result)

    return results


def exceeds_threshold(result: ExperimentInvalidRates, threshold: float) -> bool:
    """
    Check if any invalid rate in the experiment exceeds the threshold.

    Args:
        result: ExperimentInvalidRates object
        threshold: Threshold in percent (e.g., 1.0 for 1%)

    Returns:
        True if any invalid rate >= threshold
    """
    threshold_fraction = threshold / 100.0

    # Check all conditions and all comparison types
    for condition in [result.base, result.nudge_A, result.nudge_B]:
        if condition.overall.invalid_rate >= threshold_fraction:
            return True
        if condition.equal_n.invalid_rate >= threshold_fraction:
            return True
        if condition.larger_A.invalid_rate >= threshold_fraction:
            return True
        if condition.larger_B.invalid_rate >= threshold_fraction:
            return True

    return False


def format_rate(rate: float, total: int, decimals: int = 1) -> str:
    """Format invalid rate as percentage with sample size."""
    return f"{rate*100:.{decimals}f}% ({total})"


def format_rate_short(rate: float, decimals: int = 1) -> str:
    """Format invalid rate as percentage without sample size."""
    return f"{rate*100:.{decimals}f}%"


def format_detailed_table(
    results: List[ExperimentInvalidRates],
    show_display_names: bool = True,
    decimals: int = 1,
) -> str:
    """Format results as a detailed text table."""
    if not results:
        return "No results found."

    # Sort results
    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    lines = []

    # Header
    headers = [
        "Model",
        "Reas",
        "Factor",
        "Nudge Type",
        "Cond",
        "Overall",
        "Equal-N",
        "Larger-A",
        "Larger-B",
    ]

    col_widths = [20, 8, 12, 18, 8, 14, 14, 14, 14]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        factor_str = f"{r.level_A}/{r.level_B}"

        # Base condition row
        base_row = [
            model_name[:20],
            r.reasoning_condition[:8],
            factor_str[:12],
            r.nudge_type[:18],
            "base",
            format_rate(r.base.overall.invalid_rate, r.base.overall.total, decimals),
            format_rate(r.base.equal_n.invalid_rate, r.base.equal_n.total, decimals),
            format_rate(r.base.larger_A.invalid_rate, r.base.larger_A.total, decimals),
            format_rate(r.base.larger_B.invalid_rate, r.base.larger_B.total, decimals),
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(base_row, col_widths)))

        # Nudge A condition row
        nudge_A_row = [
            "",
            "",
            "",
            "",
            f"nudge_{r.level_A[:3]}",
            format_rate(
                r.nudge_A.overall.invalid_rate, r.nudge_A.overall.total, decimals
            ),
            format_rate(
                r.nudge_A.equal_n.invalid_rate, r.nudge_A.equal_n.total, decimals
            ),
            format_rate(
                r.nudge_A.larger_A.invalid_rate, r.nudge_A.larger_A.total, decimals
            ),
            format_rate(
                r.nudge_A.larger_B.invalid_rate, r.nudge_A.larger_B.total, decimals
            ),
        ]
        lines.append(
            " | ".join(str(v).ljust(w) for v, w in zip(nudge_A_row, col_widths))
        )

        # Nudge B condition row
        nudge_B_row = [
            "",
            "",
            "",
            "",
            f"nudge_{r.level_B[:3]}",
            format_rate(
                r.nudge_B.overall.invalid_rate, r.nudge_B.overall.total, decimals
            ),
            format_rate(
                r.nudge_B.equal_n.invalid_rate, r.nudge_B.equal_n.total, decimals
            ),
            format_rate(
                r.nudge_B.larger_A.invalid_rate, r.nudge_B.larger_A.total, decimals
            ),
            format_rate(
                r.nudge_B.larger_B.invalid_rate, r.nudge_B.larger_B.total, decimals
            ),
        ]
        lines.append(
            " | ".join(str(v).ljust(w) for v, w in zip(nudge_B_row, col_widths))
        )

        # Separator between experiments
        lines.append("-" * len(header_line))

    return "\n".join(lines)


def format_summary_table(
    results: List[ExperimentInvalidRates],
    show_display_names: bool = True,
    decimals: int = 1,
) -> str:
    """Format results as a condensed summary table (one row per experiment)."""
    if not results:
        return "No results found."

    results = sorted(
        results,
        key=lambda r: (
            get_base_model_name(r.model),
            r.factor,
            r.nudge_type,
            r.reasoning_condition,
        ),
    )

    lines = []

    # Header - more compact format showing all conditions on one row
    headers = [
        "Model",
        "Reas",
        "Factor",
        "Nudge Type",
        "Base",
        "Nudge A",
        "Nudge B",
        "Overall",
    ]

    col_widths = [20, 8, 12, 18, 10, 10, 10, 10]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        factor_str = f"{r.level_A}/{r.level_B}"

        all_cond = r.all_conditions

        row = [
            model_name[:20],
            r.reasoning_condition[:8],
            factor_str[:12],
            r.nudge_type[:18],
            format_rate_short(r.base.overall.invalid_rate, decimals),
            format_rate_short(r.nudge_A.overall.invalid_rate, decimals),
            format_rate_short(r.nudge_B.overall.invalid_rate, decimals),
            format_rate_short(all_cond.overall.invalid_rate, decimals),
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    return "\n".join(lines)


def print_aggregate_stats(
    results: List[ExperimentInvalidRates],
    show_display_names: bool = True,
    decimals: int = 1,
) -> None:
    """Print aggregate statistics."""
    if not results:
        print("No results to aggregate.")
        return

    def format_aggregate_rates(
        rates: ConditionInvalidRates, show_diff_n: bool = False
    ) -> str:
        """Format aggregate rates for display."""
        overall_str = f"overall={rates.overall.invalid_rate*100:.{decimals}f}%"
        equal_n_str = f"equal-n={rates.equal_n.invalid_rate*100:.{decimals}f}%"

        if show_diff_n:
            # Combine larger-A and larger-B as different-n
            diff_n = rates.larger_A + rates.larger_B
            diff_n_str = f"diff-n={diff_n.invalid_rate*100:.{decimals}f}%"
            return f"{overall_str}, {equal_n_str}, {diff_n_str}"
        else:
            larger_A_str = f"larger-A={rates.larger_A.invalid_rate*100:.{decimals}f}%"
            larger_B_str = f"larger-B={rates.larger_B.invalid_rate*100:.{decimals}f}%"
            return f"{overall_str}, {equal_n_str}, {larger_A_str}, {larger_B_str}"

    print("\n" + "=" * 100)
    print("Aggregate Invalid Response Rates")
    print("=" * 100)

    # Overall
    total_base = ConditionInvalidRates()
    total_nudge_A = ConditionInvalidRates()
    total_nudge_B = ConditionInvalidRates()

    for r in results:
        total_base = total_base + r.base
        total_nudge_A = total_nudge_A + r.nudge_A
        total_nudge_B = total_nudge_B + r.nudge_B

    total_all = total_base + total_nudge_A + total_nudge_B

    print(f"\nOverall (n={len(results)} experiments):")
    print(f"  All conditions: {format_aggregate_rates(total_all, show_diff_n=True)}")
    print(f"  Base:           {format_aggregate_rates(total_base, show_diff_n=True)}")
    print(
        f"  Nudge (avg):    {format_aggregate_rates(total_nudge_A + total_nudge_B, show_diff_n=True)}"
    )

    # By model
    model_groups: Dict[Tuple[str, str], List[ExperimentInvalidRates]] = defaultdict(
        list
    )
    for r in results:
        base_model = get_base_model_name(r.model)
        model_groups[(base_model, r.reasoning_condition)].append(r)

    print(f"\nBy Model ({len(model_groups)} groups):")
    for base_model, reasoning_condition in sorted(model_groups.keys()):
        model_results = model_groups[(base_model, reasoning_condition)]

        # Aggregate across experiments
        agg_base = ConditionInvalidRates()
        agg_nudge_A = ConditionInvalidRates()
        agg_nudge_B = ConditionInvalidRates()

        for r in model_results:
            agg_base = agg_base + r.base
            agg_nudge_A = agg_nudge_A + r.nudge_A
            agg_nudge_B = agg_nudge_B + r.nudge_B

        agg_all = agg_base + agg_nudge_A + agg_nudge_B

        display_name = (
            get_model_display_name(model_results[0].model)
            if show_display_names
            else base_model
        )

        # Show diff-n for multi-factor aggregation
        show_diff = len(set(r.factor for r in model_results)) > 1

        print(f"\n  {display_name} ({reasoning_condition}): n={len(model_results)}")
        print(f"    All:    {format_aggregate_rates(agg_all, show_diff_n=show_diff)}")
        print(f"    Base:   {format_aggregate_rates(agg_base, show_diff_n=show_diff)}")
        print(
            f"    Nudge:  {format_aggregate_rates(agg_nudge_A + agg_nudge_B, show_diff_n=show_diff)}"
        )

    # By factor
    factor_groups: Dict[str, List[ExperimentInvalidRates]] = defaultdict(list)
    for r in results:
        factor_groups[r.factor].append(r)

    print(f"\nBy Factor ({len(factor_groups)} groups):")
    for factor in sorted(factor_groups.keys()):
        factor_results = factor_groups[factor]

        agg_base = ConditionInvalidRates()
        agg_nudge_A = ConditionInvalidRates()
        agg_nudge_B = ConditionInvalidRates()

        for r in factor_results:
            agg_base = agg_base + r.base
            agg_nudge_A = agg_nudge_A + r.nudge_A
            agg_nudge_B = agg_nudge_B + r.nudge_B

        agg_all = agg_base + agg_nudge_A + agg_nudge_B

        level_A = factor_results[0].level_A
        level_B = factor_results[0].level_B

        # For single factor, show larger-A and larger-B separately
        print(f"\n  {factor} (A={level_A}, B={level_B}): n={len(factor_results)}")
        print(f"    All:    {format_aggregate_rates(agg_all, show_diff_n=False)}")
        print(f"    Base:   {format_aggregate_rates(agg_base, show_diff_n=False)}")
        print(
            f"    Nudge→{level_A[:3]}: {format_aggregate_rates(agg_nudge_A, show_diff_n=False)}"
        )
        print(
            f"    Nudge→{level_B[:3]}: {format_aggregate_rates(agg_nudge_B, show_diff_n=False)}"
        )

    # By nudge type
    nudge_groups: Dict[str, List[ExperimentInvalidRates]] = defaultdict(list)
    for r in results:
        nudge_groups[r.nudge_type].append(r)

    print(f"\nBy Nudge Type ({len(nudge_groups)} groups):")
    for nudge_type in sorted(nudge_groups.keys()):
        nudge_results = nudge_groups[nudge_type]

        agg_base = ConditionInvalidRates()
        agg_nudge_A = ConditionInvalidRates()
        agg_nudge_B = ConditionInvalidRates()

        for r in nudge_results:
            agg_base = agg_base + r.base
            agg_nudge_A = agg_nudge_A + r.nudge_A
            agg_nudge_B = agg_nudge_B + r.nudge_B

        agg_all = agg_base + agg_nudge_A + agg_nudge_B

        show_diff = len(set(r.factor for r in nudge_results)) > 1

        print(f"\n  {nudge_type}: n={len(nudge_results)}")
        print(f"    All:    {format_aggregate_rates(agg_all, show_diff_n=show_diff)}")
        print(f"    Base:   {format_aggregate_rates(agg_base, show_diff_n=show_diff)}")
        print(
            f"    Nudge:  {format_aggregate_rates(agg_nudge_A + agg_nudge_B, show_diff_n=show_diff)}"
        )

    # By reasoning condition
    reasoning_groups: Dict[str, List[ExperimentInvalidRates]] = defaultdict(list)
    for r in results:
        reasoning_groups[r.reasoning_condition].append(r)

    print(f"\nBy Reasoning Condition ({len(reasoning_groups)} groups):")
    for reasoning in sorted(reasoning_groups.keys()):
        reasoning_results = reasoning_groups[reasoning]

        agg_base = ConditionInvalidRates()
        agg_nudge_A = ConditionInvalidRates()
        agg_nudge_B = ConditionInvalidRates()

        for r in reasoning_results:
            agg_base = agg_base + r.base
            agg_nudge_A = agg_nudge_A + r.nudge_A
            agg_nudge_B = agg_nudge_B + r.nudge_B

        agg_all = agg_base + agg_nudge_A + agg_nudge_B

        show_diff = len(set(r.factor for r in reasoning_results)) > 1

        print(f"\n  {reasoning}: n={len(reasoning_results)}")
        print(f"    All:    {format_aggregate_rates(agg_all, show_diff_n=show_diff)}")
        print(f"    Base:   {format_aggregate_rates(agg_base, show_diff_n=show_diff)}")
        print(
            f"    Nudge:  {format_aggregate_rates(agg_nudge_A + agg_nudge_B, show_diff_n=show_diff)}"
        )


def write_csv(
    results: List[ExperimentInvalidRates],
    output_path: str,
    show_display_names: bool = True,
) -> None:
    """Write results to a CSV file."""
    if not results:
        print("No results to write.")
        return

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
        "condition",
        "invalid_rate_overall",
        "n_overall",
        "invalid_rate_equal_n",
        "n_equal_n",
        "invalid_rate_larger_A",
        "n_larger_A",
        "invalid_rate_larger_B",
        "n_larger_B",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            display_name = (
                get_model_display_name(r.model) if show_display_names else r.model
            )

            # Write base condition
            writer.writerow(
                [
                    r.model,
                    display_name,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    "base",
                    r.base.overall.invalid_rate,
                    r.base.overall.total,
                    r.base.equal_n.invalid_rate,
                    r.base.equal_n.total,
                    r.base.larger_A.invalid_rate,
                    r.base.larger_A.total,
                    r.base.larger_B.invalid_rate,
                    r.base.larger_B.total,
                ]
            )

            # Write nudge A condition
            writer.writerow(
                [
                    r.model,
                    display_name,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    f"nudge_{r.level_A}",
                    r.nudge_A.overall.invalid_rate,
                    r.nudge_A.overall.total,
                    r.nudge_A.equal_n.invalid_rate,
                    r.nudge_A.equal_n.total,
                    r.nudge_A.larger_A.invalid_rate,
                    r.nudge_A.larger_A.total,
                    r.nudge_A.larger_B.invalid_rate,
                    r.nudge_A.larger_B.total,
                ]
            )

            # Write nudge B condition
            writer.writerow(
                [
                    r.model,
                    display_name,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    f"nudge_{r.level_B}",
                    r.nudge_B.overall.invalid_rate,
                    r.nudge_B.overall.total,
                    r.nudge_B.equal_n.invalid_rate,
                    r.nudge_B.equal_n.total,
                    r.nudge_B.larger_A.invalid_rate,
                    r.nudge_B.larger_A.total,
                    r.nudge_B.larger_B.invalid_rate,
                    r.nudge_B.larger_B.total,
                ]
            )

    print(f"Wrote {len(results) * 3} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze invalid response rates across experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze invalid response rates from default results directory
    python -m choices.analysis.analyze_invalid_responses

    # Specify results directories
    python -m choices.analysis.analyze_invalid_responses --results-dirs results results_reasoning

    # Filter by models, factors, nudge types
    python -m choices.analysis.analyze_invalid_responses \\
        --models llama-33-70b gpt-4o-mini \\
        --factors age_group social_status

    # Output to CSV
    python -m choices.analysis.analyze_invalid_responses --output invalid_rates.csv
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
        help="List of reasoning conditions to include (e.g., 'none', 'before', 'after')",
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
        "--decimals",
        "-d",
        type=int,
        default=1,
        help="Number of decimal places for percentages (default: 1)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show condensed summary table instead of detailed table",
    )

    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Don't show aggregate statistics",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Only show experiments where any invalid rate >= threshold (in percent). "
        "E.g., --threshold 1.0 shows only experiments with at least 1%% invalid rate somewhere.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("Invalid Response Rate Analysis")
    print("=" * 100)
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning:
        print(f"Reasoning condition filter: {args.reasoning}")
    if args.threshold is not None:
        print(f"Threshold filter: >= {args.threshold}%")
    print("=" * 100)
    print()

    # Compute results
    results = compute_all_invalid_rates(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    # Apply reasoning condition filter
    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]

    print(f"Found {len(results)} complete experiments")

    # Apply threshold filter for table display
    display_results = results
    if args.threshold is not None:
        display_results = [r for r in results if exceeds_threshold(r, args.threshold)]
        print(
            f"Showing {len(display_results)} experiments with invalid rate >= {args.threshold}%"
        )

    print()

    if not results:
        print("No complete experiments found matching the filters.")
        return

    show_display_names = not args.no_display_names
    decimals = args.decimals

    if display_results:
        if args.output:
            write_csv(display_results, args.output, show_display_names)
        else:
            if args.summary:
                print(
                    format_summary_table(display_results, show_display_names, decimals)
                )
            else:
                print(
                    format_detailed_table(display_results, show_display_names, decimals)
                )
    else:
        print("No experiments exceed the threshold.")

    if not args.no_aggregate:
        print_aggregate_stats(results, show_display_names, decimals)


if __name__ == "__main__":
    main()
