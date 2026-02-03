#!/usr/bin/env python3
"""
Analyze backfiring rates in nudging experiments.

This script computes backfiring rates (cases where nudging moves preferences
in the opposite direction to intended) based on baseline preference significance.

The analysis is stratified by:
1. Cases without significant baseline bias (f_0 ~ 0.5)
2. Cases with significant baseline bias, further split by:
   - Overall backfiring rate
   - Backfiring when nudging towards the already preferred option
   - Backfiring when nudging away from the baseline preferred option

For each category, the script reports:
- Raw count and total samples
- Percentage of total samples
- Percentage of significant nudges that backfired
- Rate of significant nudging effects (as reference)

Usage:
    # Basic usage with default results directory
    uv run python -m choices.analysis.get_backfiring_rates

    # Specify results directories
    uv run python -m choices.analysis.get_backfiring_rates --results-dirs results results_anthropic

    # Filter by model, factor, nudge type, or reasoning condition
    uv run python -m choices.analysis.get_backfiring_rates \
        --models gpt-4o-mini claude-haiku-4-5 \
        --factors age_group social_status \
        --nudge-types user_preference \
        --reasoning none before after
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from choices.analysis.metrics import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.utils import (
    binomial_test_vs_half,
    two_proportion_z_test,
)
from choices.analysis.utils import (
    compute_factor_frequencies_with_counts,
    get_base_model_name,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)

# Default significance level (95% confidence)
DEFAULT_ALPHA = 0.05


@dataclass
class NudgeSample:
    """A single nudge sample (one direction of nudging for one experiment)."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    # Which direction is this nudge? ('A' or 'B')
    nudge_direction: str
    # Baseline preference for level B (f_0_B)
    f_0_B: float
    # Frequency after nudging (f_A_A or f_B_B depending on direction)
    f_nudged: float
    # Baseline frequency for the target (f_0_A or f_0_B)
    f_0_target: float
    # Sample sizes
    n_baseline: int
    n_nudged: int
    wins_baseline: int
    # Is baseline preference significantly different from 0.5?
    sig_baseline: bool
    # Is the nudge effect significant?
    sig_nudge: bool
    # Did the nudge backfire (decrease target frequency)?
    backfired: bool
    # Is nudging towards or away from baseline preference?
    # "towards_preferred" = nudging toward the option with f_0 > 0.5
    # "away_from_preferred" = nudging toward the option with f_0 < 0.5
    nudge_alignment: Optional[str]


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

    # Group directories by (condition, reasoning_mode)
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
    experiments_by_reasoning: Dict[str, Dict[str, Path]] = {}

    for (condition, reasoning_mode), dirs in dirs_by_condition_and_reasoning.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)

        if reasoning_mode not in experiments_by_reasoning:
            experiments_by_reasoning[reasoning_mode] = {}
        experiments_by_reasoning[reasoning_mode][condition] = most_recent

    return list(experiments_by_reasoning.values())


def compute_nudge_samples(
    factor_name: str,
    model: str,
    nudge_type: str,
    condition_dirs: Dict[str, Path],
    alpha: float = DEFAULT_ALPHA,
) -> List[NudgeSample]:
    """
    Compute nudge samples for a single experiment.

    Each experiment produces 2 samples: nudge towards A and nudge towards B.
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

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return []

    # Load nudge condition data
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_A_graph or not nudge_B_graph:
        return []

    # Compute frequencies with counts
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

    # Get frequencies and counts
    f_0_A = base_stats.get(level_A, {}).get("freq", 0.5)
    f_0_B = base_stats.get(level_B, {}).get("freq", 0.5)
    n_0_A = base_stats.get(level_A, {}).get("n", 0)
    n_0_B = base_stats.get(level_B, {}).get("n", 0)
    c_0_A = base_stats.get(level_A, {}).get("wins", 0)
    c_0_B = base_stats.get(level_B, {}).get("wins", 0)

    f_A_A = nudge_A_stats.get(level_A, {}).get("freq", 0.5)
    n_A_A = nudge_A_stats.get(level_A, {}).get("n", 0)

    f_B_B = nudge_B_stats.get(level_B, {}).get("freq", 0.5)
    n_B_B = nudge_B_stats.get(level_B, {}).get("n", 0)

    # Test if baseline f_0(B) differs significantly from 0.5
    test_baseline_B = binomial_test_vs_half(c_0_B, n_0_B, alpha)
    sig_baseline_B = test_baseline_B["is_significant"]

    # Also test f_0(A) for completeness (should be complementary)
    test_baseline_A = binomial_test_vs_half(c_0_A, n_0_A, alpha)
    sig_baseline_A = test_baseline_A["is_significant"]

    # Use either baseline test (they test the same thing)
    sig_baseline = sig_baseline_B or sig_baseline_A

    # Test if nudge towards A significantly changed frequency of A
    test_A = two_proportion_z_test(f_0_A, n_0_A, f_A_A, n_A_A, alpha)
    sig_A = test_A["is_significant"]

    # Test if nudge towards B significantly changed frequency of B
    test_B = two_proportion_z_test(f_0_B, n_0_B, f_B_B, n_B_B, alpha)
    sig_B = test_B["is_significant"]

    # Determine backfiring
    backfire_A = f_A_A < f_0_A  # Nudging towards A decreased frequency of A
    backfire_B = f_B_B < f_0_B  # Nudging towards B decreased frequency of B

    # Determine nudge alignment (only meaningful when baseline is significantly biased)
    # f_0_B > 0.5 means B is preferred at baseline
    if sig_baseline:
        if f_0_B > 0.5:
            # B is preferred at baseline
            alignment_A = (
                "away_from_preferred"  # Nudging towards A goes against baseline
            )
            alignment_B = "towards_preferred"  # Nudging towards B reinforces baseline
        else:
            # A is preferred at baseline (f_0_A > 0.5)
            alignment_A = "towards_preferred"  # Nudging towards A reinforces baseline
            alignment_B = (
                "away_from_preferred"  # Nudging towards B goes against baseline
            )
    else:
        alignment_A = None
        alignment_B = None

    # Determine reasoning condition
    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    samples = []

    # Sample for nudge towards A
    samples.append(
        NudgeSample(
            model=model,
            reasoning_condition=reasoning_condition,
            factor=factor_name,
            nudge_type=nudge_type,
            level_A=level_A,
            level_B=level_B,
            nudge_direction="A",
            f_0_B=f_0_B,
            f_nudged=f_A_A,
            f_0_target=f_0_A,
            n_baseline=n_0_A,
            n_nudged=n_A_A,
            wins_baseline=c_0_A,
            sig_baseline=sig_baseline,
            sig_nudge=sig_A,
            backfired=backfire_A,
            nudge_alignment=alignment_A,
        )
    )

    # Sample for nudge towards B
    samples.append(
        NudgeSample(
            model=model,
            reasoning_condition=reasoning_condition,
            factor=factor_name,
            nudge_type=nudge_type,
            level_A=level_A,
            level_B=level_B,
            nudge_direction="B",
            f_0_B=f_0_B,
            f_nudged=f_B_B,
            f_0_target=f_0_B,
            n_baseline=n_0_B,
            n_nudged=n_B_B,
            wins_baseline=c_0_B,
            sig_baseline=sig_baseline,
            sig_nudge=sig_B,
            backfired=backfire_B,
            nudge_alignment=alignment_B,
        )
    )

    return samples


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


def collect_all_samples(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
    reasoning_filter: Optional[List[str]] = None,
    alpha: float = DEFAULT_ALPHA,
) -> List[NudgeSample]:
    """
    Collect all nudge samples from the specified results directories.
    """
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    all_samples = []

    for results_base_dir, factor_name, model, nudge_type in experiments:
        # Find all experiment sets (one per reasoning_mode)
        experiment_sets = find_condition_directories(
            factor_name, model, nudge_type, results_base_dir
        )

        for condition_dirs in experiment_sets:
            samples = compute_nudge_samples(
                factor_name, model, nudge_type, condition_dirs, alpha
            )
            all_samples.extend(samples)

    # Apply reasoning filter (post-computation since it's derived from results)
    if reasoning_filter:
        all_samples = [
            s for s in all_samples if s.reasoning_condition in reasoning_filter
        ]

    return all_samples


@dataclass
class BackfireStats:
    """Statistics for a group of nudge samples.

    Note: Backfiring is only counted when the nudge effect is statistically significant.
    A "backfire" means the nudge significantly moved preferences in the opposite direction.
    """

    total_samples: int
    sig_count: int  # Number of samples with significant nudge effect
    backfire_count: int  # Number of significant backfires (sig AND backfired)

    @property
    def sig_rate(self) -> float:
        """Percentage of samples with significant nudge effect."""
        return self.sig_count / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def backfire_rate(self) -> float:
        """Percentage of all samples that had significant backfiring."""
        return (
            self.backfire_count / self.total_samples if self.total_samples > 0 else 0.0
        )

    @property
    def sig_backfire_rate(self) -> float:
        """Percentage of significant effects that backfired."""
        return self.backfire_count / self.sig_count if self.sig_count > 0 else 0.0


def compute_backfire_stats(samples: List[NudgeSample]) -> BackfireStats:
    """Compute backfire statistics for a list of samples.

    Backfiring is only counted when the nudge effect is statistically significant.
    """
    total = len(samples)
    sig_count = sum(1 for s in samples if s.sig_nudge)
    # Backfire only counts when effect is significant
    backfire_count = sum(1 for s in samples if s.backfired and s.sig_nudge)

    return BackfireStats(
        total_samples=total,
        sig_count=sig_count,
        backfire_count=backfire_count,
    )


def format_stats(stats: BackfireStats, label: str, indent: int = 0) -> str:
    """Format backfire statistics for display."""
    ind = " " * indent
    lines = [
        f"{ind}{label}:",
        f"{ind}  Total samples: {stats.total_samples}",
        f"{ind}  Significant nudge effects: {stats.sig_count}/{stats.total_samples} ({stats.sig_rate:.1%})",
        f"{ind}  Backfired (significant): {stats.backfire_count}/{stats.total_samples} ({stats.backfire_rate:.1%} of total)",
        f"{ind}  Backfire rate among significant: {stats.backfire_count}/{stats.sig_count} ({stats.sig_backfire_rate:.1%})",
    ]
    return "\n".join(lines)


def analyze_backfiring_rates(
    samples: List[NudgeSample],
    show_display_names: bool = True,
) -> None:
    """
    Analyze and display backfiring rates, stratified by baseline significance.
    """
    if not samples:
        print("No samples found matching the filters.")
        return

    print("=" * 80)
    print("BACKFIRING RATE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total nudge samples: {len(samples)}")
    print()

    # Split samples by baseline significance
    unbiased_samples = [s for s in samples if not s.sig_baseline]
    biased_samples = [s for s in samples if s.sig_baseline]

    print(f"Samples without significant baseline bias: {len(unbiased_samples)}")
    print(f"Samples with significant baseline bias: {len(biased_samples)}")
    print()

    # =========================================================================
    # Section 1: Overall statistics
    # =========================================================================
    print("=" * 80)
    print("OVERALL BACKFIRING RATES")
    print("=" * 80)
    print()

    overall_stats = compute_backfire_stats(samples)
    print(format_stats(overall_stats, "All samples"))
    print()

    # =========================================================================
    # Section 2: Cases without significant baseline bias
    # =========================================================================
    print("=" * 80)
    print("CASES WITHOUT SIGNIFICANT BASELINE BIAS (f_0 ~ 0.5)")
    print("=" * 80)
    print()

    if unbiased_samples:
        unbiased_stats = compute_backfire_stats(unbiased_samples)
        print(format_stats(unbiased_stats, "Unbiased baseline samples"))
    else:
        print("No samples without significant baseline bias.")
    print()

    # =========================================================================
    # Section 3: Cases with significant baseline bias
    # =========================================================================
    print("=" * 80)
    print("CASES WITH SIGNIFICANT BASELINE BIAS")
    print("=" * 80)
    print()

    if biased_samples:
        # Overall for biased samples
        biased_stats = compute_backfire_stats(biased_samples)
        print(format_stats(biased_stats, "All biased baseline samples"))
        print()

        # Split by nudge alignment
        towards_preferred = [
            s for s in biased_samples if s.nudge_alignment == "towards_preferred"
        ]
        away_from_preferred = [
            s for s in biased_samples if s.nudge_alignment == "away_from_preferred"
        ]

        print("-" * 60)
        print("Stratified by nudge direction relative to baseline preference:")
        print("-" * 60)
        print()

        if towards_preferred:
            towards_stats = compute_backfire_stats(towards_preferred)
            print(
                format_stats(
                    towards_stats, "Nudging TOWARDS already preferred option", indent=2
                )
            )
            print()

        if away_from_preferred:
            away_stats = compute_backfire_stats(away_from_preferred)
            print(
                format_stats(
                    away_stats, "Nudging AWAY FROM baseline preferred option", indent=2
                )
            )
            print()
    else:
        print("No samples with significant baseline bias.")
    print()

    # =========================================================================
    # Section 4: Breakdown by model
    # =========================================================================
    print("=" * 80)
    print("BREAKDOWN BY MODEL")
    print("=" * 80)
    print()

    # Group by (base_model, reasoning_condition)
    by_model: Dict[Tuple[str, str], List[NudgeSample]] = defaultdict(list)
    for s in samples:
        base_model = get_base_model_name(s.model)
        by_model[(base_model, s.reasoning_condition)].append(s)

    for (base_model, reasoning), model_samples in sorted(by_model.items()):
        display_name = (
            get_model_display_name(model_samples[0].model)
            if show_display_names
            else base_model
        )
        model_label = f"{display_name} ({reasoning})"

        model_stats = compute_backfire_stats(model_samples)
        print(
            f"{model_label}: n={model_stats.total_samples}, "
            f"backfire={model_stats.backfire_rate:.1%}, "
            f"sig={model_stats.sig_rate:.1%}, "
            f"sig_backfire={model_stats.sig_backfire_rate:.1%}"
        )

        # Split by baseline significance
        unbiased = [s for s in model_samples if not s.sig_baseline]
        biased = [s for s in model_samples if s.sig_baseline]

        if unbiased:
            unbiased_stats = compute_backfire_stats(unbiased)
            print(
                f"  Unbiased baseline (n={unbiased_stats.total_samples}): "
                f"backfire={unbiased_stats.backfire_rate:.1%}, "
                f"sig_backfire={unbiased_stats.sig_backfire_rate:.1%}"
            )

        if biased:
            biased_stats = compute_backfire_stats(biased)
            print(
                f"  Biased baseline (n={biased_stats.total_samples}): "
                f"backfire={biased_stats.backfire_rate:.1%}, "
                f"sig_backfire={biased_stats.sig_backfire_rate:.1%}"
            )

            # Further split by nudge alignment
            towards = [s for s in biased if s.nudge_alignment == "towards_preferred"]
            away = [s for s in biased if s.nudge_alignment == "away_from_preferred"]

            if towards:
                towards_stats = compute_backfire_stats(towards)
                print(
                    f"    Towards preferred (n={towards_stats.total_samples}): "
                    f"backfire={towards_stats.backfire_rate:.1%}, "
                    f"sig_backfire={towards_stats.sig_backfire_rate:.1%}"
                )

            if away:
                away_stats = compute_backfire_stats(away)
                print(
                    f"    Away from preferred (n={away_stats.total_samples}): "
                    f"backfire={away_stats.backfire_rate:.1%}, "
                    f"sig_backfire={away_stats.sig_backfire_rate:.1%}"
                )
        print()

    # =========================================================================
    # Section 5: Breakdown by factor
    # =========================================================================
    print("=" * 80)
    print("BREAKDOWN BY FACTOR")
    print("=" * 80)
    print()

    by_factor: Dict[str, List[NudgeSample]] = defaultdict(list)
    for s in samples:
        by_factor[s.factor].append(s)

    for factor, factor_samples in sorted(by_factor.items()):
        # Get level names
        level_A = factor_samples[0].level_A if factor_samples else "?"
        level_B = factor_samples[0].level_B if factor_samples else "?"

        factor_stats = compute_backfire_stats(factor_samples)
        print(
            f"{factor} (A={level_A}, B={level_B}): n={factor_stats.total_samples}, "
            f"backfire={factor_stats.backfire_rate:.1%}, "
            f"sig={factor_stats.sig_rate:.1%}, "
            f"sig_backfire={factor_stats.sig_backfire_rate:.1%}"
        )

        # Split by baseline significance
        unbiased = [s for s in factor_samples if not s.sig_baseline]
        biased = [s for s in factor_samples if s.sig_baseline]

        if unbiased:
            unbiased_stats = compute_backfire_stats(unbiased)
            print(
                f"  Unbiased baseline (n={unbiased_stats.total_samples}): "
                f"backfire={unbiased_stats.backfire_rate:.1%}, "
                f"sig_backfire={unbiased_stats.sig_backfire_rate:.1%}"
            )

        if biased:
            biased_stats = compute_backfire_stats(biased)
            print(
                f"  Biased baseline (n={biased_stats.total_samples}): "
                f"backfire={biased_stats.backfire_rate:.1%}, "
                f"sig_backfire={biased_stats.sig_backfire_rate:.1%}"
            )

            # Further split by nudge alignment
            towards = [s for s in biased if s.nudge_alignment == "towards_preferred"]
            away = [s for s in biased if s.nudge_alignment == "away_from_preferred"]

            if towards:
                towards_stats = compute_backfire_stats(towards)
                print(
                    f"    Towards preferred (n={towards_stats.total_samples}): "
                    f"backfire={towards_stats.backfire_rate:.1%}, "
                    f"sig_backfire={towards_stats.sig_backfire_rate:.1%}"
                )

            if away:
                away_stats = compute_backfire_stats(away)
                print(
                    f"    Away from preferred (n={away_stats.total_samples}): "
                    f"backfire={away_stats.backfire_rate:.1%}, "
                    f"sig_backfire={away_stats.sig_backfire_rate:.1%}"
                )
        print()

    # =========================================================================
    # Section 6: Breakdown by nudge type
    # =========================================================================
    print("=" * 80)
    print("BREAKDOWN BY NUDGE TYPE")
    print("=" * 80)
    print()

    by_nudge: Dict[str, List[NudgeSample]] = defaultdict(list)
    for s in samples:
        by_nudge[s.nudge_type].append(s)

    for nudge_type, nudge_samples in sorted(by_nudge.items()):
        nudge_stats = compute_backfire_stats(nudge_samples)
        print(
            f"{nudge_type}: n={nudge_stats.total_samples}, "
            f"backfire={nudge_stats.backfire_rate:.1%}, "
            f"sig={nudge_stats.sig_rate:.1%}, "
            f"sig_backfire={nudge_stats.sig_backfire_rate:.1%}"
        )

        # Split by baseline significance
        unbiased = [s for s in nudge_samples if not s.sig_baseline]
        biased = [s for s in nudge_samples if s.sig_baseline]

        if unbiased:
            unbiased_stats = compute_backfire_stats(unbiased)
            print(
                f"  Unbiased baseline (n={unbiased_stats.total_samples}): "
                f"backfire={unbiased_stats.backfire_rate:.1%}, "
                f"sig_backfire={unbiased_stats.sig_backfire_rate:.1%}"
            )

        if biased:
            biased_stats = compute_backfire_stats(biased)
            print(
                f"  Biased baseline (n={biased_stats.total_samples}): "
                f"backfire={biased_stats.backfire_rate:.1%}, "
                f"sig_backfire={biased_stats.sig_backfire_rate:.1%}"
            )

            # Further split by nudge alignment
            towards = [s for s in biased if s.nudge_alignment == "towards_preferred"]
            away = [s for s in biased if s.nudge_alignment == "away_from_preferred"]

            if towards:
                towards_stats = compute_backfire_stats(towards)
                print(
                    f"    Towards preferred (n={towards_stats.total_samples}): "
                    f"backfire={towards_stats.backfire_rate:.1%}, "
                    f"sig_backfire={towards_stats.sig_backfire_rate:.1%}"
                )

            if away:
                away_stats = compute_backfire_stats(away)
                print(
                    f"    Away from preferred (n={away_stats.total_samples}): "
                    f"backfire={away_stats.backfire_rate:.1%}, "
                    f"sig_backfire={away_stats.sig_backfire_rate:.1%}"
                )
        print()

    # =========================================================================
    # Section 7: Summary Table
    # =========================================================================
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    # Header
    headers = [
        "Category",
        "N",
        "Sig",
        "Backfire",
        "Backfire %",
        "Sig %",
        "Sig Backfire %",
    ]
    col_widths = [35, 6, 6, 10, 12, 10, 15]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(separator)

    def print_row(label: str, stats: BackfireStats) -> None:
        row = [
            label,
            str(stats.total_samples),
            str(stats.sig_count),
            f"{stats.backfire_count}/{stats.total_samples}",
            f"{stats.backfire_rate:.1%}",
            f"{stats.sig_rate:.1%}",
            f"{stats.sig_backfire_rate:.1%}",
        ]
        print(" | ".join(val.ljust(w) for val, w in zip(row, col_widths)))

    # Overall
    print_row("All samples", overall_stats)

    # Unbiased
    if unbiased_samples:
        print_row(
            "  Unbiased baseline (f_0 ~ 0.5)", compute_backfire_stats(unbiased_samples)
        )

    # Biased
    if biased_samples:
        print_row("  Biased baseline", compute_backfire_stats(biased_samples))
        towards_preferred = [
            s for s in biased_samples if s.nudge_alignment == "towards_preferred"
        ]
        away_from_preferred = [
            s for s in biased_samples if s.nudge_alignment == "away_from_preferred"
        ]

        if towards_preferred:
            print_row(
                "    -> Towards preferred", compute_backfire_stats(towards_preferred)
            )
        if away_from_preferred:
            print_row(
                "    -> Away from preferred",
                compute_backfire_stats(away_from_preferred),
            )

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze backfiring rates in nudging experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default results directory
    uv run python -m choices.analysis.get_backfiring_rates

    # Specify results directories
    uv run python -m choices.analysis.get_backfiring_rates --results-dirs results results_anthropic

    # Filter by model, factor, nudge type, or reasoning condition
    uv run python -m choices.analysis.get_backfiring_rates \\
        --models gpt-4o-mini claude-haiku-4-5 \\
        --factors age_group social_status \\
        --nudge-types user_preference \\
        --reasoning none before after
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
        help="List of reasoning conditions to include "
        "(e.g., 'low', 'medium', 'high', 'off', 'before', 'after', 'none')",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Significance level for statistical tests (default: {DEFAULT_ALPHA})",
    )

    parser.add_argument(
        "--no-display-names",
        action="store_true",
        help="Use raw model names instead of display names",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Backfiring Rate Analysis")
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
    print(f"Significance level (alpha): {args.alpha}")
    print("=" * 80)
    print()

    # Collect all samples
    samples = collect_all_samples(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
        reasoning_filter=args.reasoning,
        alpha=args.alpha,
    )

    # Analyze and display results
    show_display_names = not args.no_display_names
    analyze_backfiring_rates(samples, show_display_names)


if __name__ == "__main__":
    main()
