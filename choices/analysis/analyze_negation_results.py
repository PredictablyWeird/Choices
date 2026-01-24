#!/usr/bin/env python3
"""
Analyze negation nudge results.

This script analyzes how negation nudges compare to their normal counterparts.
Negation nudges (ending with "_negation") should theoretically have the opposite
effect of their normal counterparts.

Creates a contingency matrix:
- Y-axis: Normal nudge effect (significant vs insignificant)
- X-axis: Negation effect direction (opposite, insignificant, same direction)

Usage:
    # Analyze negation results from multiple directories
    uv run python -m choices.analysis.analyze_negation_results \
        --results-dirs results_main1 results_negation

    # Filter by specific models or factors
    uv run python -m choices.analysis.analyze_negation_results \
        --results-dirs results_main1 results_negation \
        --models gpt-5-2-reasoning gpt-5-2-non-reasoning

    # Output to CSV
    uv run python -m choices.analysis.analyze_negation_results \
        --results-dirs results_main1 results_negation \
        --output negation_analysis.csv
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import (
    get_model_display_name,
)


@dataclass
class NegationPair:
    """A pair of normal and negation nudge results."""

    model: str
    reasoning_condition: str
    factor: str
    level_A: str
    level_B: str
    normal_nudge_type: str
    # Normal nudge results
    normal_sig_A: bool  # Significant effect for nudge towards A
    normal_sig_B: bool  # Significant effect for nudge towards B
    normal_effect_A: float  # f_A(A) - f_0(A)
    normal_effect_B: float  # f_B(B) - f_0(B)
    normal_backfire_A: bool
    normal_backfire_B: bool
    # Negation nudge results
    negation_sig_A: bool
    negation_sig_B: bool
    negation_effect_A: float
    negation_effect_B: float
    negation_backfire_A: bool
    negation_backfire_B: bool


def get_base_nudge_name(nudge_type: str) -> Optional[str]:
    """
    Extract the base nudge name from a negation nudge type.

    Returns None if not a negation nudge.
    """
    if nudge_type.endswith("_negation"):
        return nudge_type[: -len("_negation")]
    return None


def is_negation_nudge(nudge_type: str) -> bool:
    """Check if a nudge type is a negation nudge."""
    return nudge_type.endswith("_negation")


def compute_effect_from_result(result: FrequencyResult) -> Tuple[float, float]:
    """
    Compute the signed effect for each direction.

    Returns:
        (effect_A, effect_B) where:
        - effect_A = f_A(A) - f_0(A) (change in A when nudged towards A)
        - effect_B = f_B(B) - f_0(B) (change in B when nudged towards B)

    Note: f_A(A) = 1 - f_A(B), f_0(A) = 1 - f_0(B)
    """
    # The result stores frequencies for level B
    # f_0_B = baseline freq of B
    # f_A_B = freq of B when nudged towards A
    # f_B_B = freq of B when nudged towards B

    f_0_B = result.f_0_B
    f_A_B = result.f_A_B
    f_B_B = result.f_B_B

    # Convert to level A frequencies
    f_0_A = 1 - f_0_B
    f_A_A = 1 - f_A_B

    # Effect A: change in A frequency when nudged towards A
    effect_A = f_A_A - f_0_A

    # Effect B: change in B frequency when nudged towards B
    effect_B = f_B_B - f_0_B

    return effect_A, effect_B


def find_matching_pairs(
    results: List[FrequencyResult],
) -> List[NegationPair]:
    """
    Find matching pairs of normal and negation nudge results.

    Args:
        results: All frequency results

    Returns:
        List of NegationPair objects for matched experiments
    """
    # Index results by (model, reasoning_condition, factor, nudge_type)
    results_index: Dict[Tuple[str, str, str, str], FrequencyResult] = {}

    for result in results:
        key = (
            result.model,
            result.reasoning_condition,
            result.factor,
            result.nudge_type,
        )
        results_index[key] = result

    # Find pairs
    pairs = []

    for result in results:
        # Only process negation nudges
        if not is_negation_nudge(result.nudge_type):
            continue

        base_nudge = get_base_nudge_name(result.nudge_type)
        if not base_nudge:
            continue

        # Look for the matching normal nudge
        normal_key = (
            result.model,
            result.reasoning_condition,
            result.factor,
            base_nudge,
        )
        normal_result = results_index.get(normal_key)

        if normal_result is None:
            # No matching normal nudge found
            continue

        # Compute effects
        normal_effect_A, normal_effect_B = compute_effect_from_result(normal_result)
        negation_effect_A, negation_effect_B = compute_effect_from_result(result)

        pair = NegationPair(
            model=result.model,
            reasoning_condition=result.reasoning_condition,
            factor=result.factor,
            level_A=result.level_A,
            level_B=result.level_B,
            normal_nudge_type=base_nudge,
            # Normal nudge
            normal_sig_A=normal_result.sig_A,
            normal_sig_B=normal_result.sig_B,
            normal_effect_A=normal_effect_A,
            normal_effect_B=normal_effect_B,
            normal_backfire_A=normal_result.backfire_A,
            normal_backfire_B=normal_result.backfire_B,
            # Negation nudge
            negation_sig_A=result.sig_A,
            negation_sig_B=result.sig_B,
            negation_effect_A=negation_effect_A,
            negation_effect_B=negation_effect_B,
            negation_backfire_A=result.backfire_A,
            negation_backfire_B=result.backfire_B,
        )
        pairs.append(pair)

    return pairs


def classify_negation_effect(
    normal_effect: float,
    normal_sig: bool,
    negation_effect: float,
    negation_sig: bool,
) -> str:
    """
    Classify the negation effect relative to the normal nudge.

    The negation nudge says "do NOT prefer X" while targeting X.
    This should ideally decrease preference for X (opposite direction).

    Returns:
        "opposite" - Negation significantly moved in opposite direction
        "insignificant" - Negation effect was not significant
        "same" - Negation significantly moved in same direction (unexpected)
    """
    if not negation_sig:
        return "insignificant"

    # Both are significant - check direction
    # Normal nudge towards A should increase A (positive effect_A)
    # Negation towards A with "not prefer A" should decrease A (negative effect_A)

    # Opposite direction means signs are different
    # (or comparing against expected: negation should go negative if normal went positive)
    same_direction = (normal_effect >= 0 and negation_effect >= 0) or (
        normal_effect < 0 and negation_effect < 0
    )

    if same_direction:
        return "same"
    else:
        return "opposite"


def create_contingency_matrix(
    pairs: List[NegationPair],
) -> Dict[str, Dict[str, int]]:
    """
    Create a contingency matrix for negation analysis.

    Y-axis: Normal nudge effect (significant vs insignificant)
    X-axis: Negation effect (opposite, insignificant, same direction)

    Returns:
        Nested dict: matrix[normal_effect][negation_effect] = count
    """
    matrix = {
        "significant": {"opposite": 0, "insignificant": 0, "same": 0},
        "insignificant": {"opposite": 0, "insignificant": 0, "same": 0},
    }

    for pair in pairs:
        # Analyze for direction A
        normal_cat_A = "significant" if pair.normal_sig_A else "insignificant"
        negation_cat_A = classify_negation_effect(
            pair.normal_effect_A,
            pair.normal_sig_A,
            pair.negation_effect_A,
            pair.negation_sig_A,
        )
        matrix[normal_cat_A][negation_cat_A] += 1

        # Analyze for direction B
        normal_cat_B = "significant" if pair.normal_sig_B else "insignificant"
        negation_cat_B = classify_negation_effect(
            pair.normal_effect_B,
            pair.normal_sig_B,
            pair.negation_effect_B,
            pair.negation_sig_B,
        )
        matrix[normal_cat_B][negation_cat_B] += 1

    return matrix


def format_contingency_matrix(
    matrix: Dict[str, Dict[str, int]],
    show_percentages: bool = True,
    title: Optional[str] = None,
) -> str:
    """Format the contingency matrix as a text table."""
    lines = []

    # Header
    lines.append("")
    if title:
        lines.append(title)
    else:
        lines.append(
            "Contingency Matrix: Normal Nudge Effect vs Negation Effect Direction"
        )
    lines.append("=" * 75)
    lines.append("")

    # Column headers
    col_headers = ["Opposite", "Insignificant", "Same Dir."]
    header_line = (
        f"{'Normal Nudge':>20} | "
        + " | ".join(f"{h:^14}" for h in col_headers)
        + " | Total"
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    row_labels = [("Significant", "significant"), ("Insignificant", "insignificant")]

    for label, key in row_labels:
        row_data = matrix[key]
        row_total = sum(row_data.values())

        if show_percentages and row_total > 0:
            cells = []
            for col_key in ["opposite", "insignificant", "same"]:
                count = row_data[col_key]
                pct = count / row_total * 100
                cells.append(f"{count:>4} ({pct:>5.1f}%)")
            row_line = (
                f"{label:>20} | "
                + " | ".join(f"{c:^14}" for c in cells)
                + f" | {row_total:>5}"
            )
        else:
            cells = [
                str(row_data[col_key])
                for col_key in ["opposite", "insignificant", "same"]
            ]
            row_line = (
                f"{label:>20} | "
                + " | ".join(f"{c:^14}" for c in cells)
                + f" | {row_total:>5}"
            )

        lines.append(row_line)

    # Column totals
    lines.append("-" * len(header_line))
    col_totals = {}
    for col_key in ["opposite", "insignificant", "same"]:
        col_totals[col_key] = sum(
            matrix[row_key][col_key] for row_key in ["significant", "insignificant"]
        )

    grand_total = sum(col_totals.values())

    if show_percentages and grand_total > 0:
        total_cells = []
        for col_key in ["opposite", "insignificant", "same"]:
            count = col_totals[col_key]
            pct = count / grand_total * 100
            total_cells.append(f"{count:>4} ({pct:>5.1f}%)")
        total_line = (
            f"{'Total':>20} | "
            + " | ".join(f"{c:^14}" for c in total_cells)
            + f" | {grand_total:>5}"
        )
    else:
        total_cells = [
            str(col_totals[col_key])
            for col_key in ["opposite", "insignificant", "same"]
        ]
        total_line = (
            f"{'Total':>20} | "
            + " | ".join(f"{c:^14}" for c in total_cells)
            + f" | {grand_total:>5}"
        )

    lines.append(total_line)
    lines.append("")

    return "\n".join(lines)


def create_per_model_matrices(
    pairs: List[NegationPair],
) -> Dict[Tuple[str, str], Dict[str, Dict[str, int]]]:
    """
    Create contingency matrices for each (model, reasoning_condition) combination.

    Returns:
        Dict mapping (model, reasoning_condition) -> contingency matrix
    """
    # Group pairs by (model, reasoning_condition)
    pairs_by_model: Dict[Tuple[str, str], List[NegationPair]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in pairs_by_model:
            pairs_by_model[key] = []
        pairs_by_model[key].append(pair)

    # Create matrix for each group
    matrices = {}
    for key, model_pairs in pairs_by_model.items():
        matrices[key] = create_contingency_matrix(model_pairs)

    return matrices


def format_per_model_matrices(
    matrices: Dict[Tuple[str, str], Dict[str, Dict[str, int]]],
    show_display_names: bool = True,
    show_percentages: bool = True,
) -> str:
    """Format per-model contingency matrices as text tables."""
    lines = []

    lines.append("")
    lines.append("=" * 75)
    lines.append("Per-Model Contingency Matrices")
    lines.append("=" * 75)

    # Sort by model name, then reasoning condition
    sorted_keys = sorted(matrices.keys(), key=lambda x: (x[0], x[1]))

    for model, reasoning_condition in sorted_keys:
        matrix = matrices[(model, reasoning_condition)]

        model_name = get_model_display_name(model) if show_display_names else model
        title = f"{model_name} ({reasoning_condition})"

        lines.append(format_contingency_matrix(matrix, show_percentages, title))

    return "\n".join(lines)


def format_pairs_table(
    pairs: List[NegationPair],
    show_display_names: bool = True,
    decimals: int = 3,
) -> str:
    """Format detailed pair results as a text table."""
    if not pairs:
        return "No matching pairs found."

    lines = []
    lines.append("")
    lines.append("Detailed Pair Results")
    lines.append("=" * 120)
    lines.append("")

    headers = [
        "Model",
        "Reasoning",
        "Factor",
        "Nudge",
        "Dir",
        "Normal Eff",
        "Neg Eff",
        "Normal Sig",
        "Neg Cat",
    ]
    col_widths = [25, 10, 12, 20, 3, 12, 12, 10, 12]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for pair in pairs:
        model_name = (
            get_model_display_name(pair.model) if show_display_names else pair.model
        )
        factor_str = f"{pair.level_A}/{pair.level_B}"

        # Row for direction A
        normal_cat_A = "Sig" if pair.normal_sig_A else "Not Sig"
        neg_cat_A = classify_negation_effect(
            pair.normal_effect_A,
            pair.normal_sig_A,
            pair.negation_effect_A,
            pair.negation_sig_A,
        )

        row_A = [
            model_name[:25],
            pair.reasoning_condition[:10],
            factor_str[:12],
            pair.normal_nudge_type[:20],
            "A",
            f"{pair.normal_effect_A:+.{decimals}f}",
            f"{pair.negation_effect_A:+.{decimals}f}",
            normal_cat_A,
            neg_cat_A,
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row_A, col_widths)))

        # Row for direction B
        normal_cat_B = "Sig" if pair.normal_sig_B else "Not Sig"
        neg_cat_B = classify_negation_effect(
            pair.normal_effect_B,
            pair.normal_sig_B,
            pair.negation_effect_B,
            pair.negation_sig_B,
        )

        row_B = [
            "",  # Don't repeat model
            "",  # Don't repeat reasoning
            "",  # Don't repeat factor
            "",  # Don't repeat nudge
            "B",
            f"{pair.normal_effect_B:+.{decimals}f}",
            f"{pair.negation_effect_B:+.{decimals}f}",
            normal_cat_B,
            neg_cat_B,
        ]
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row_B, col_widths)))

    return "\n".join(lines)


def write_pairs_csv(
    pairs: List[NegationPair],
    output_path: str,
) -> None:
    """Write detailed pair results to CSV."""
    headers = [
        "model",
        "reasoning_condition",
        "factor",
        "level_A",
        "level_B",
        "nudge_type",
        "direction",
        "normal_effect",
        "negation_effect",
        "normal_significant",
        "negation_significant",
        "negation_category",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for pair in pairs:
            # Direction A
            neg_cat_A = classify_negation_effect(
                pair.normal_effect_A,
                pair.normal_sig_A,
                pair.negation_effect_A,
                pair.negation_sig_A,
            )
            writer.writerow(
                [
                    pair.model,
                    pair.reasoning_condition,
                    pair.factor,
                    pair.level_A,
                    pair.level_B,
                    pair.normal_nudge_type,
                    "A",
                    pair.normal_effect_A,
                    pair.negation_effect_A,
                    pair.normal_sig_A,
                    pair.negation_sig_A,
                    neg_cat_A,
                ]
            )

            # Direction B
            neg_cat_B = classify_negation_effect(
                pair.normal_effect_B,
                pair.normal_sig_B,
                pair.negation_effect_B,
                pair.negation_sig_B,
            )
            writer.writerow(
                [
                    pair.model,
                    pair.reasoning_condition,
                    pair.factor,
                    pair.level_A,
                    pair.level_B,
                    pair.normal_nudge_type,
                    "B",
                    pair.normal_effect_B,
                    pair.negation_effect_B,
                    pair.normal_sig_B,
                    pair.negation_sig_B,
                    neg_cat_B,
                ]
            )

    print(f"Wrote detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze negation nudge results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results"],
        help="Directories containing results (default: results)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter by specific models",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Filter by specific factors",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for detailed results",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Don't show detailed pair results",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for effect sizes (default: 3)",
    )

    args = parser.parse_args()

    # Compute all frequency results
    print(f"Loading results from: {', '.join(args.results_dirs)}")

    all_results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=None,  # We need both normal and negation nudges
    )

    print(f"Found {len(all_results)} total experiment results")

    # Count normal and negation nudges
    normal_count = sum(1 for r in all_results if not is_negation_nudge(r.nudge_type))
    negation_count = sum(1 for r in all_results if is_negation_nudge(r.nudge_type))
    print(f"  - {normal_count} normal nudge results")
    print(f"  - {negation_count} negation nudge results")

    # Find matching pairs
    pairs = find_matching_pairs(all_results)
    print(f"\nFound {len(pairs)} matching normal-negation pairs")

    if not pairs:
        print(
            "\nNo matching pairs found. Make sure both normal and negation nudge results exist."
        )
        print("Normal nudges should be in one results directory (e.g., results_main1)")
        print("Negation nudges should be in another (e.g., results_negation)")
        return

    # Create and display overall contingency matrix
    matrix = create_contingency_matrix(pairs)
    print(format_contingency_matrix(matrix, title="Overall Contingency Matrix"))

    # Create and display per-model contingency matrices
    per_model_matrices = create_per_model_matrices(pairs)
    print(format_per_model_matrices(per_model_matrices))

    # Display detailed results if requested
    if not args.no_details:
        print(format_pairs_table(pairs, decimals=args.decimals))

    # Write CSV if requested
    if args.output:
        write_pairs_csv(pairs, args.output)


if __name__ == "__main__":
    main()
