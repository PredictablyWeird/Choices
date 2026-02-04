#!/usr/bin/env python3
"""
Case study: Backfiring in nudging experiments.

Research question:
"Cases where backfiring leads to more extreme preferences: Consider cases of
unequal n where the model chooses egalitarian or even favors one group without
context, but strongly favors the other when nudged to the first group
(e.g. for 3-young-7-old, old might be 70% in baseline and 30% in nudged)"

This script:
1. Finds edges where nudging towards group X causes preference to shift AWAY from X
2. Checks both directions (A→B backfire and B→A backfire)
3. Extracts paired traces (baseline vs nudged) for each edge
4. Shows aggregate statistics by factor and n-difference

Usage:
    uv run python -m choices.analysis.reasoning_traces.case_study_backfire \
        --results-dirs results/main_results/results_main0 \
        --model claude-3-5-sonnet-20241022 \
        --min-effect 0.2 \
        --min-n-diff 2
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from choices.analysis.reasoning_traces.edge_filtering import (
    EdgeComparison,
    EdgeFilteringPipeline,
    TraceWithContext,
    extract_traces_for_edge,
    n_difference,
)


# =============================================================================
# Case Study Data Structures
# =============================================================================


@dataclass
class BackfireCase:
    """
    A case where nudging backfired.

    Contains the edge data plus which direction the backfire occurred.
    """

    edge: EdgeComparison

    # Which option was nudged (the one that backfired)
    nudged_option: str  # "A" or "B"

    # The option that ended up being preferred after backfire
    preferred_option: str  # "B" or "A" (opposite of nudged)

    # Key metrics
    baseline_freq: float  # f_0 for the nudged option
    nudged_freq: float  # f_X(X) for the nudged option (should be lower)
    effect_magnitude: float  # |baseline_freq - nudged_freq|

    # N values context (convenience mapping based on nudged_option)
    nudged_n: int  # N for the nudged option
    other_n: int  # N for the other option

    @property
    def nudged_has_smaller_n(self) -> bool:
        """True if the nudged option has fewer people."""
        return self.nudged_n < self.other_n

    @property
    def nudged_has_larger_n(self) -> bool:
        """True if the nudged option has more people."""
        return self.nudged_n > self.other_n


@dataclass
class PairedTraces:
    """
    Paired traces for a single edge: baseline vs nudged condition.

    This allows comparing reasoning between conditions.
    """

    case: BackfireCase
    baseline_traces: List[TraceWithContext]
    nudged_traces: List[TraceWithContext]

    @property
    def n_baseline(self) -> int:
        return len(self.baseline_traces)

    @property
    def n_nudged(self) -> int:
        return len(self.nudged_traces)

    def get_baseline_choices(self) -> Counter:
        """Count of choices in baseline condition."""
        return Counter(t.choice for t in self.baseline_traces)

    def get_nudged_choices(self) -> Counter:
        """Count of choices in nudged condition."""
        return Counter(t.choice for t in self.nudged_traces)


# =============================================================================
# Filtering Logic
# =============================================================================


def find_backfire_cases(
    edges: List[EdgeComparison],
    min_effect: float = 0.0,
) -> List[BackfireCase]:
    """
    Find edges where nudging backfired.

    Checks both directions (A and B) and returns all matching cases.

    Args:
        edges: List of EdgeComparison objects
        min_effect: Minimum effect magnitude |f_0(X) - f_X(X)| to include

    Returns:
        List of BackfireCase objects
    """
    cases = []

    for edge in edges:
        # Check direction A: nudging towards A backfires
        # Backfire means f_A(A) < f_0(A)
        if edge.backfire_A:
            effect_a = edge.backfire_A_magnitude  # f_0(A) - f_A(A)
            if effect_a >= min_effect:
                cases.append(
                    BackfireCase(
                        edge=edge,
                        nudged_option="A",
                        preferred_option="B",
                        baseline_freq=edge.f_0_A,
                        nudged_freq=edge.f_A_A,
                        effect_magnitude=effect_a,
                        nudged_n=edge.option_a_n,
                        other_n=edge.option_b_n,
                    )
                )

        # Check direction B: nudging towards B backfires
        # Backfire means f_B(B) < f_0(B)
        if edge.backfire_B:
            effect_b = edge.backfire_B_magnitude  # f_0(B) - f_B(B)
            if effect_b >= min_effect:
                cases.append(
                    BackfireCase(
                        edge=edge,
                        nudged_option="B",
                        preferred_option="A",
                        baseline_freq=edge.f_0_B,
                        nudged_freq=edge.f_B_B,
                        effect_magnitude=effect_b,
                        nudged_n=edge.option_b_n,
                        other_n=edge.option_a_n,
                    )
                )

    return cases


def extract_paired_traces(
    case: BackfireCase,
) -> PairedTraces:
    """
    Extract paired traces for a backfire case.

    Gets traces from:
    - Baseline condition
    - The nudged condition (where backfire occurred)

    Args:
        case: BackfireCase to extract traces for

    Returns:
        PairedTraces object with baseline and nudged traces
    """
    edge = case.edge

    # Determine which condition to pull nudged traces from
    if case.nudged_option == "A":
        nudge_condition = edge.level_A
    else:
        nudge_condition = edge.level_B

    # Extract baseline traces
    baseline_traces = extract_traces_for_edge(edge, conditions=["base"])

    # Extract nudged condition traces
    nudged_traces = extract_traces_for_edge(edge, conditions=[nudge_condition])

    return PairedTraces(
        case=case,
        baseline_traces=baseline_traces,
        nudged_traces=nudged_traces,
    )


# =============================================================================
# Statistics and Reporting
# =============================================================================


def compute_statistics(
    cases: List[BackfireCase],
    paired_traces: List[PairedTraces],
) -> dict:
    """
    Compute aggregate statistics for the case study.

    Returns:
        Dictionary with various statistics
    """
    stats = {
        "total_cases": len(cases),
        "by_factor": Counter(c.edge.factor for c in cases),
        "by_nudge_type": Counter(c.edge.nudge_type for c in cases),
        "by_nudged_option": Counter(c.nudged_option for c in cases),
        "by_n_difference": Counter(c.edge.n_difference for c in cases),
        "nudged_smaller_n": sum(1 for c in cases if c.nudged_has_smaller_n),
        "nudged_larger_n": sum(1 for c in cases if c.nudged_has_larger_n),
    }

    # Effect by n-difference
    effect_by_n_diff = defaultdict(list)
    for c in cases:
        effect_by_n_diff[c.edge.n_difference].append(c.effect_magnitude)

    stats["effect_by_n_diff"] = {
        n: {
            "count": len(effects),
            "mean": sum(effects) / len(effects),
            "max": max(effects),
        }
        for n, effects in sorted(effect_by_n_diff.items())
    }

    # Trace statistics
    total_baseline = sum(p.n_baseline for p in paired_traces)
    total_nudged = sum(p.n_nudged for p in paired_traces)
    stats["total_baseline_traces"] = total_baseline
    stats["total_nudged_traces"] = total_nudged

    return stats


def print_statistics(stats: dict):
    """Print formatted statistics."""
    print("=" * 70)
    print("BACKFIRE CASE STUDY - STATISTICS")
    print("=" * 70)

    print(f"\nTotal cases found: {stats['total_cases']}")

    print("\n--- By Factor ---")
    for factor, count in stats["by_factor"].most_common():
        print(f"  {factor}: {count}")

    print("\n--- By Nudge Type ---")
    for nudge_type, count in stats["by_nudge_type"].most_common():
        print(f"  {nudge_type}: {count}")

    print("\n--- By Nudged Option ---")
    for option, count in stats["by_nudged_option"].most_common():
        print(f"  {option}: {count}")

    print("\n--- N-Difference Distribution ---")
    print(f"  Nudged option had SMALLER N: {stats['nudged_smaller_n']}")
    print(f"  Nudged option had LARGER N: {stats['nudged_larger_n']}")

    print("\n--- Effect by N-Difference ---")
    for n_diff, effect_stats in stats["effect_by_n_diff"].items():
        print(
            f"  n_diff={n_diff}: count={effect_stats['count']}, "
            f"mean_effect={effect_stats['mean']:.3f}, max={effect_stats['max']:.3f}"
        )

    print("\n--- Traces ---")
    print(f"  Total baseline traces: {stats['total_baseline_traces']}")
    print(f"  Total nudged traces: {stats['total_nudged_traces']}")


def print_sample_cases(
    paired_traces: List[PairedTraces],
    n_samples: int = 5,
):
    """Print sample cases for inspection."""
    print("\n" + "=" * 70)
    print("SAMPLE CASES")
    print("=" * 70)

    # Sort by effect magnitude to show most interesting first
    sorted_pairs = sorted(
        paired_traces,
        key=lambda p: p.case.effect_magnitude,
        reverse=True,
    )

    for i, pair in enumerate(sorted_pairs[:n_samples], 1):
        case = pair.case
        edge = case.edge

        print(f"\n--- Case {i} ---")
        print(f"Model: {edge.model}")
        print(f"Factor: {edge.factor} ({edge.level_A} vs {edge.level_B})")
        print(f"Nudge type: {edge.nudge_type}")
        print(f"Options: {edge.option_a_label} vs {edge.option_b_label}")
        print(
            f"N values: A={edge.option_a_n}, B={edge.option_b_n} (diff={edge.n_difference})"
        )
        print(
            f"\nBackfire direction: Nudged towards {case.nudged_option}, preferred {case.preferred_option}"
        )
        print(f"Baseline f({case.nudged_option}): {case.baseline_freq:.3f}")
        print(f"Nudged f({case.nudged_option}): {case.nudged_freq:.3f}")
        print(f"Effect magnitude: {case.effect_magnitude:.3f}")

        print(f"\nBaseline traces: {pair.n_baseline}")
        baseline_choices = pair.get_baseline_choices()
        print(
            f"  Choices: A={baseline_choices.get('A', 0)}, B={baseline_choices.get('B', 0)}"
        )

        print(f"Nudged traces: {pair.n_nudged}")
        nudged_choices = pair.get_nudged_choices()
        print(
            f"  Choices: A={nudged_choices.get('A', 0)}, B={nudged_choices.get('B', 0)}"
        )

        # Show one sample trace from each condition
        if pair.baseline_traces:
            t = pair.baseline_traces[0]
            print(f"\nSample baseline reasoning (chose {t.choice}):")
            print(f"  {t.reasoning[:300]}...")

        if pair.nudged_traces:
            t = pair.nudged_traces[0]
            print(f"\nSample nudged reasoning (chose {t.choice}):")
            print(f"  {t.reasoning[:300]}...")


# =============================================================================
# Main Pipeline
# =============================================================================


def run_case_study(
    results_dirs: List[str],
    model: str,
    min_effect: float = 0.0,
    min_n_diff: int = 1,
    factors: Optional[List[str]] = None,
    nudge_types: Optional[List[str]] = None,
    show_samples: int = 5,
) -> Tuple[List[BackfireCase], List[PairedTraces], dict]:
    """
    Run the backfire case study.

    Args:
        results_dirs: List of results directory paths
        model: Model to analyze (required)
        min_effect: Minimum effect magnitude for backfire
        min_n_diff: Minimum N difference (1 = any unequal, 0 = include equal)
        factors: Optional list of factors to include
        nudge_types: Optional list of nudge types to include
        show_samples: Number of sample cases to print (0 to skip)

    Returns:
        Tuple of (cases, paired_traces, statistics)
    """
    print(f"Running case study for model: {model}")
    print(f"Parameters: min_effect={min_effect}, min_n_diff={min_n_diff}")

    # Step 1: Extract edges
    edge_filter = n_difference(min_n_diff) if min_n_diff > 0 else None
    pipeline = EdgeFilteringPipeline(
        results_dirs=results_dirs,
        edge_filter=edge_filter,
        models=[model],
        factors=factors,
        nudge_types=nudge_types,
    )

    edges = pipeline.get_filtered_edges()
    print(
        f"\nExtracted {len(edges)} edges"
        + (f" with n_diff >= {min_n_diff}" if min_n_diff > 0 else "")
    )

    # Step 2: Find backfire cases
    cases = find_backfire_cases(edges, min_effect=min_effect)
    print(f"Found {len(cases)} backfire cases")

    # Step 3: Extract paired traces
    paired_traces = [extract_paired_traces(case) for case in cases]

    # Step 4: Compute and print statistics
    stats = compute_statistics(cases, paired_traces)
    print_statistics(stats)

    # Step 5: Print sample cases
    if show_samples > 0 and paired_traces:
        print_sample_cases(paired_traces, n_samples=show_samples)

    return cases, paired_traces, stats


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Case study: Backfiring in nudging experiments"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=[
            "results/",
        ],
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to analyze (required)",
    )
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.0,
        help="Minimum effect magnitude for backfire (default: 0.0)",
    )
    parser.add_argument(
        "--min-n-diff",
        type=int,
        default=1,
        help="Minimum N difference (default: 1, use 0 to include equal N)",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Filter to specific factors",
    )
    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="Filter to specific nudge types",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample cases to show (default: 5, 0 to skip)",
    )

    args = parser.parse_args()

    run_case_study(
        results_dirs=args.results_dirs,
        model=args.model,
        min_effect=args.min_effect,
        min_n_diff=args.min_n_diff,
        factors=args.factors,
        nudge_types=args.nudge_types,
        show_samples=args.show_samples,
    )


if __name__ == "__main__":
    main()
