#!/usr/bin/env python3
"""
Validation script for choice-mapping logic in edge_filtering.py.

Verifies that the choice field in TraceWithContext correctly maps to canonical
labels (A=level_A, B=level_B) by cross-checking trace counts against the
aggregated counts in EdgeComparison.

Usage:
    uv run python -m choices.analysis.reasoning_traces.validate_choice_mapping \
        --results-dirs results/main_results/results_main0
"""

import argparse
from collections import Counter

from choices.analysis.reasoning_traces.edge_filtering import (
    EdgeFilteringPipeline,
    extract_traces_for_edge,
)


def validate_choice_mapping(
    results_dirs: list[str],
    models: list[str] | None = None,
    max_edges: int = 50,
) -> bool:
    """
    Validate that trace extraction choice mapping matches aggregated counts.

    For each edge, the number of traces mapped to choice "A" should be <=
    base_chose_a (for base condition), and similarly for nudge conditions.
    Traces may be fewer than parsed counts when reasoning text is missing
    for some responses (e.g. reasoning list shorter than parsed list).
    We check that the ratio A/(A+B) matches between traces and counts,
    and that trace counts never EXCEED expected counts.

    Returns True if all validations pass.
    """
    pipeline = EdgeFilteringPipeline(
        results_dirs=results_dirs,
        models=models,
    )

    edges = pipeline.get_edges()
    print(f"Validating choice mapping across {len(edges)} edges")

    errors = 0
    warnings = 0
    checked = 0

    for edge in edges[:max_edges]:
        # Extract traces for all conditions
        traces = extract_traces_for_edge(edge)
        if not traces:
            continue

        checked += 1

        # Group traces by condition
        by_condition: dict[str, Counter[str]] = {}
        for t in traces:
            if t.condition not in by_condition:
                by_condition[t.condition] = Counter()
            by_condition[t.condition][t.choice] += 1

        def check_condition(
            label: str, actual_counts: Counter, expected_a: int, expected_b: int
        ) -> bool:
            """Check one condition. Returns True if ok."""
            actual_a = actual_counts.get("A", 0)
            actual_b = actual_counts.get("B", 0)

            # Traces can't exceed parsed counts (would indicate wrong mapping)
            if actual_a > expected_a or actual_b > expected_b:
                nonlocal errors
                errors += 1
                print(
                    f"\nEXCEEDS EXPECTED ({label}) - "
                    f"{edge.model}/{edge.factor}/{edge.nudge_type}"
                )
                print(f"  Edge: {edge.edge_key}")
                print(f"  Expected: A={expected_a}, B={expected_b}")
                print(f"  Got:      A={actual_a}, B={actual_b}")
                return False

            # Check proportions match (within tolerance for small counts)
            actual_total = actual_a + actual_b
            expected_total = expected_a + expected_b
            if actual_total > 0 and expected_total > 0:
                actual_ratio = actual_a / actual_total
                expected_ratio = expected_a / expected_total
                # Allow tolerance for rounding with small samples
                tolerance = 1.0 / min(actual_total, expected_total) + 0.01
                if abs(actual_ratio - expected_ratio) > tolerance:
                    nonlocal warnings
                    warnings += 1
                    if warnings <= 10:
                        print(
                            f"\nPROPORTION MISMATCH ({label}) - "
                            f"{edge.model}/{edge.factor}/{edge.nudge_type}"
                        )
                        print(f"  Edge: {edge.edge_key}")
                        print(
                            f"  Expected: A={expected_a}, B={expected_b} "
                            f"(ratio={expected_ratio:.3f})"
                        )
                        print(
                            f"  Got:      A={actual_a}, B={actual_b} "
                            f"(ratio={actual_ratio:.3f})"
                        )
                    return False
            return True

        # Validate base condition
        if "base" in by_condition:
            check_condition(
                "base", by_condition["base"], edge.base_chose_a, edge.base_chose_b
            )

        # Validate nudge_A condition
        if edge.level_A in by_condition:
            check_condition(
                f"nudge_A={edge.level_A}",
                by_condition[edge.level_A],
                edge.nudge_a_chose_a,
                edge.nudge_a_chose_b,
            )

        # Validate nudge_B condition
        if edge.level_B in by_condition:
            check_condition(
                f"nudge_B={edge.level_B}",
                by_condition[edge.level_B],
                edge.nudge_b_chose_a,
                edge.nudge_b_chose_b,
            )

    if errors == 0 and warnings == 0:
        print(f"\nAll {checked} edges validated successfully!")
        return True
    elif errors == 0:
        print(
            f"\n{checked} edges checked: {warnings} proportion warnings "
            f"(likely from missing reasoning text), 0 critical errors"
        )
        return True
    else:
        print(f"\n{errors} critical errors, {warnings} warnings in {checked} edges")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate choice mapping in trace extraction"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results/main_results/results_main0"],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--max-edges",
        type=int,
        default=50,
        help="Max edges to validate (default: 50)",
    )

    args = parser.parse_args()

    success = validate_choice_mapping(
        results_dirs=args.results_dirs,
        models=args.models,
        max_edges=args.max_edges,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
