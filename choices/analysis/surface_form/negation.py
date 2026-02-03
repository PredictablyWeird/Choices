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

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import (
    PLOTS_OUTPUT_DIR,
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


def classify_negation_vs_opposite(
    negation_effect: float,
    negation_sig: bool,
    opposite_normal_effect: float,
    opposite_normal_sig: bool,
    ignore_significance: bool = False,
) -> str:
    """
    Classify the negation effect relative to the OPPOSITE normal nudge.

    "NOT prefer A" should behave like "prefer B".
    We compare negation_effect_A vs normal_effect_B.

    KEY INSIGHT: The signs will be OPPOSITE if they produce the same outcome!
    - negation_effect_A < 0 means A decreased (shifted toward B)
    - normal_effect_B > 0 means B increased (shifted toward B)
    Both represent shifting toward B, but have opposite signs.

    So "matches" = signs are OPPOSITE (same real-world outcome)
    And "mismatches" = signs are SAME (different real-world outcome)

    Args:
        ignore_significance: If True, skip significance check and just compare directions

    Returns:
        "matches" - Negation behaves like the opposite nudge (same outcome)
        "insignificant" - Negation effect was not significant (only if ignore_significance=False)
        "mismatches" - Negation does NOT behave like opposite nudge
    """
    if not ignore_significance and not negation_sig:
        return "insignificant"

    # "NOT prefer A" decreasing A (negative) = "prefer B" increasing B (positive)
    # So OPPOSITE signs mean matching behavior!
    opposite_signs = (opposite_normal_effect >= 0 and negation_effect < 0) or (
        opposite_normal_effect < 0 and negation_effect >= 0
    )

    if opposite_signs:
        return "matches"
    else:
        return "mismatches"


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


def create_contingency_matrix_vs_opposite(
    pairs: List[NegationPair],
) -> Dict[str, Dict[str, int]]:
    """
    Create a contingency matrix comparing negation to OPPOSITE normal nudge.

    This tests: does "NOT prefer A" behave like "prefer B"?

    Y-axis: Opposite normal nudge effect (significant vs insignificant)
           i.e., for "NOT prefer A", we look at "prefer B"'s significance
    X-axis: Does negation match the opposite nudge direction?

    Returns:
        Nested dict: matrix[opposite_normal_effect][negation_match] = count
    """
    matrix = {
        "significant": {"matches": 0, "insignificant": 0, "mismatches": 0},
        "insignificant": {"matches": 0, "insignificant": 0, "mismatches": 0},
    }

    for pair in pairs:
        # For "NOT prefer A" (negation_A), compare to "prefer B" (normal_B)
        opposite_cat_A = "significant" if pair.normal_sig_B else "insignificant"
        negation_cat_A = classify_negation_vs_opposite(
            pair.negation_effect_A,
            pair.negation_sig_A,
            pair.normal_effect_B,
            pair.normal_sig_B,
        )
        matrix[opposite_cat_A][negation_cat_A] += 1

        # For "NOT prefer B" (negation_B), compare to "prefer A" (normal_A)
        opposite_cat_B = "significant" if pair.normal_sig_A else "insignificant"
        negation_cat_B = classify_negation_vs_opposite(
            pair.negation_effect_B,
            pair.negation_sig_B,
            pair.normal_effect_A,
            pair.normal_sig_A,
        )
        matrix[opposite_cat_B][negation_cat_B] += 1

    return matrix


def format_contingency_matrix_vs_opposite(
    matrix: Dict[str, Dict[str, int]],
    show_percentages: bool = True,
    title: Optional[str] = None,
) -> str:
    """Format the vs-opposite contingency matrix as a text table."""
    lines = []

    # Header
    lines.append("")
    if title:
        lines.append(title)
    else:
        lines.append("Does 'NOT prefer A' behave like 'prefer B'?")
    lines.append("=" * 75)
    lines.append("")

    # Column headers
    col_headers = ["Matches", "Insignificant", "Mismatches"]
    header_line = (
        f"{'Opposite Nudge':>20} | "
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
            for col_key in ["matches", "insignificant", "mismatches"]:
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
                for col_key in ["matches", "insignificant", "mismatches"]
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
    for col_key in ["matches", "insignificant", "mismatches"]:
        col_totals[col_key] = sum(
            matrix[row_key][col_key] for row_key in ["significant", "insignificant"]
        )

    grand_total = sum(col_totals.values())

    if show_percentages and grand_total > 0:
        total_cells = []
        for col_key in ["matches", "insignificant", "mismatches"]:
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
            for col_key in ["matches", "insignificant", "mismatches"]
        ]
        total_line = (
            f"{'Total':>20} | "
            + " | ".join(f"{c:^14}" for c in total_cells)
            + f" | {grand_total:>5}"
        )

    lines.append(total_line)
    lines.append("")

    return "\n".join(lines)


def create_per_model_matrices_vs_opposite(
    pairs: List[NegationPair],
) -> Dict[Tuple[str, str], Dict[str, Dict[str, int]]]:
    """
    Create vs-opposite contingency matrices for each (model, reasoning_condition).
    """
    pairs_by_model: Dict[Tuple[str, str], List[NegationPair]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in pairs_by_model:
            pairs_by_model[key] = []
        pairs_by_model[key].append(pair)

    matrices = {}
    for key, model_pairs in pairs_by_model.items():
        matrices[key] = create_contingency_matrix_vs_opposite(model_pairs)

    return matrices


def format_per_model_matrices_vs_opposite(
    matrices: Dict[Tuple[str, str], Dict[str, Dict[str, int]]],
    show_display_names: bool = True,
    show_percentages: bool = True,
) -> str:
    """Format per-model vs-opposite contingency matrices."""
    lines = []

    lines.append("")
    lines.append("=" * 75)
    lines.append("Per-Model: Does 'NOT prefer A' behave like 'prefer B'?")
    lines.append("=" * 75)

    sorted_keys = sorted(matrices.keys(), key=lambda x: (x[0], x[1]))

    for model, reasoning_condition in sorted_keys:
        matrix = matrices[(model, reasoning_condition)]
        model_name = get_model_display_name(model) if show_display_names else model
        title = f"{model_name} ({reasoning_condition})"
        lines.append(
            format_contingency_matrix_vs_opposite(matrix, show_percentages, title)
        )

    return "\n".join(lines)


def print_detailed_vs_opposite_analysis(
    pairs: List[NegationPair],
) -> None:
    """
    Print detailed analysis comparing negation to opposite nudge.
    Shows individual cases where negation does NOT match opposite nudge.
    """
    print("\n" + "=" * 100)
    print("DETAILED: Cases where 'NOT prefer A' does NOT behave like 'prefer B'")
    print("=" * 100)

    # Group by model
    model_data: Dict[Tuple[str, str], List[dict]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in model_data:
            model_data[key] = []

        # For "NOT prefer A", compare to "prefer B"
        cat_A = classify_negation_vs_opposite(
            pair.negation_effect_A,
            pair.negation_sig_A,
            pair.normal_effect_B,
            pair.normal_sig_B,
        )
        model_data[key].append(
            {
                "factor": pair.factor,
                "level_A": pair.level_A,
                "level_B": pair.level_B,
                "nudge": pair.normal_nudge_type,
                "negation_target": pair.level_A,
                "opposite_target": pair.level_B,
                "negation_effect": pair.negation_effect_A,
                "opposite_effect": pair.normal_effect_B,
                "negation_sig": pair.negation_sig_A,
                "opposite_sig": pair.normal_sig_B,
                "category": cat_A,
            }
        )

        # For "NOT prefer B", compare to "prefer A"
        cat_B = classify_negation_vs_opposite(
            pair.negation_effect_B,
            pair.negation_sig_B,
            pair.normal_effect_A,
            pair.normal_sig_A,
        )
        model_data[key].append(
            {
                "factor": pair.factor,
                "level_A": pair.level_A,
                "level_B": pair.level_B,
                "nudge": pair.normal_nudge_type,
                "negation_target": pair.level_B,
                "opposite_target": pair.level_A,
                "negation_effect": pair.negation_effect_B,
                "opposite_effect": pair.normal_effect_A,
                "negation_sig": pair.negation_sig_B,
                "opposite_sig": pair.normal_sig_A,
                "category": cat_B,
            }
        )

    def sort_key(item):
        model, reasoning = item[0]
        reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(reasoning, 2)
        return (model, reasoning_order)

    for (model, reasoning), data in sorted(model_data.items(), key=sort_key):
        model_name = get_model_display_name(model)

        # Only show mismatches where opposite nudge was significant
        mismatches = [
            d for d in data if d["category"] == "mismatches" and d["opposite_sig"]
        ]

        if not mismatches:
            continue

        print(f"\n{'─' * 100}")
        print(f"{model_name} ({reasoning}) - {len(mismatches)} mismatches")
        print(f"{'─' * 100}")

        for d in mismatches:
            neg_dir = (
                "→ " + d["level_A"] if d["negation_effect"] > 0 else "→ " + d["level_B"]
            )
            opp_dir = (
                "→ " + d["level_B"] if d["opposite_effect"] > 0 else "→ " + d["level_A"]
            )
            print(
                f"  {d['nudge']:20} | NOT prefer {d['negation_target']:6} {neg_dir:12} ({d['negation_effect']:+.3f})"
            )
            print(
                f"  {' ':20} | prefer {d['opposite_target']:10} {opp_dir:12} ({d['opposite_effect']:+.3f})"
            )
            print()


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


def print_detailed_effect_analysis(
    pairs: List[NegationPair],
    effect_threshold: float = 0.05,
) -> None:
    """
    Print detailed analysis of effect sizes to understand if 'same direction'
    findings are meaningful or just noise.

    This addresses the question: Is GPT-5.2's 'same direction' pattern real,
    or is it just that GPT-5.2 doesn't respond to nudges and effects are noise?
    """
    print("\n" + "=" * 100)
    print("DETAILED EFFECT SIZE ANALYSIS")
    print("=" * 100)
    print(
        f"(Helps distinguish real patterns from noise. Threshold for 'substantial': |effect| > {effect_threshold})"
    )

    # Group by model
    model_data: Dict[Tuple[str, str], List[dict]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in model_data:
            model_data[key] = []

        # Process direction A
        neg_cat_A = classify_negation_effect(
            pair.normal_effect_A,
            pair.normal_sig_A,
            pair.negation_effect_A,
            pair.negation_sig_A,
        )
        model_data[key].append(
            {
                "factor": pair.factor,
                "nudge": pair.normal_nudge_type,
                "direction": "A",
                "normal_effect": pair.normal_effect_A,
                "negation_effect": pair.negation_effect_A,
                "normal_sig": pair.normal_sig_A,
                "negation_sig": pair.negation_sig_A,
                "category": neg_cat_A,
                "normal_backfire": pair.normal_backfire_A,
                "negation_backfire": pair.negation_backfire_A,
            }
        )

        # Process direction B
        neg_cat_B = classify_negation_effect(
            pair.normal_effect_B,
            pair.normal_sig_B,
            pair.negation_effect_B,
            pair.negation_sig_B,
        )
        model_data[key].append(
            {
                "factor": pair.factor,
                "nudge": pair.normal_nudge_type,
                "direction": "B",
                "normal_effect": pair.normal_effect_B,
                "negation_effect": pair.negation_effect_B,
                "normal_sig": pair.normal_sig_B,
                "negation_sig": pair.negation_sig_B,
                "category": neg_cat_B,
                "normal_backfire": pair.normal_backfire_B,
                "negation_backfire": pair.negation_backfire_B,
            }
        )

    # Sort by model name
    def sort_key(item):
        model, reasoning = item[0]
        reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(reasoning, 2)
        return (model, reasoning_order)

    for (model, reasoning), data in sorted(model_data.items(), key=sort_key):
        model_name = get_model_display_name(model)
        print(f"\n{'─' * 100}")
        print(f"{model_name} ({reasoning})")
        print(f"{'─' * 100}")

        # Separate by normal nudge significance
        sig_cases = [d for d in data if d["normal_sig"]]
        _insig_cases = [d for d in data if not d["normal_sig"]]

        # For significant normal nudges, break down by negation category
        opposite = [d for d in sig_cases if d["category"] == "opposite"]
        same = [d for d in sig_cases if d["category"] == "same"]
        insig_neg = [d for d in sig_cases if d["category"] == "insignificant"]

        print(f"\n  When normal nudge is SIGNIFICANT ({len(sig_cases)} cases):")

        # Overall effect size stats
        if sig_cases:
            avg_normal = np.mean([abs(d["normal_effect"]) for d in sig_cases])
            print(f"    Average |normal effect|: {avg_normal:.3f}")

        # Opposite direction
        if opposite:
            avg_normal_opp = np.mean([abs(d["normal_effect"]) for d in opposite])
            avg_neg_opp = np.mean([abs(d["negation_effect"]) for d in opposite])
            print(
                f"\n    OPPOSITE direction ({len(opposite)} cases): avg |normal|={avg_normal_opp:.3f}, avg |negation|={avg_neg_opp:.3f}"
            )

        # Same direction - THE KEY QUESTION
        if same:
            avg_normal_same = np.mean([abs(d["normal_effect"]) for d in same])
            avg_neg_same = np.mean([abs(d["negation_effect"]) for d in same])
            substantial_normal = sum(
                1 for d in same if abs(d["normal_effect"]) > effect_threshold
            )
            substantial_neg = sum(
                1 for d in same if abs(d["negation_effect"]) > effect_threshold
            )
            backfire_normal = sum(1 for d in same if d["normal_backfire"])
            backfire_neg = sum(1 for d in same if d["negation_backfire"])

            print(
                f"\n    SAME direction ({len(same)} cases): avg |normal|={avg_normal_same:.3f}, avg |negation|={avg_neg_same:.3f}"
            )
            print(
                f"      - Substantial normal effects (>{effect_threshold}): {substantial_normal}/{len(same)}"
            )
            print(
                f"      - Substantial negation effects (>{effect_threshold}): {substantial_neg}/{len(same)}"
            )
            print(f"      - Normal backfiring: {backfire_normal}/{len(same)}")
            print(f"      - Negation backfiring: {backfire_neg}/{len(same)}")

            # Show each same-direction case
            print("\n      Individual same-direction cases:")
            for d in same:
                backfire_marker = (
                    " [BACKFIRE]"
                    if d["normal_backfire"] or d["negation_backfire"]
                    else ""
                )
                print(
                    f"        {d['factor']:12} {d['nudge']:20} {d['direction']}: "
                    f"normal={d['normal_effect']:+.3f}, neg={d['negation_effect']:+.3f}{backfire_marker}"
                )

        # Insignificant negation
        if insig_neg:
            avg_normal_insig = np.mean([abs(d["normal_effect"]) for d in insig_neg])
            avg_neg_insig = np.mean([abs(d["negation_effect"]) for d in insig_neg])
            print(
                f"\n    INSIGNIFICANT negation ({len(insig_neg)} cases): avg |normal|={avg_normal_insig:.3f}, avg |negation|={avg_neg_insig:.3f}"
            )

        # Summary interpretation
        if sig_cases:
            print("\n  INTERPRETATION:")
            total_substantial = sum(
                1 for d in sig_cases if abs(d["normal_effect"]) > effect_threshold
            )
            print(
                f"    - Cases with substantial normal effect (>{effect_threshold}): {total_substantial}/{len(sig_cases)}"
            )

            if same:
                same_substantial = sum(
                    1 for d in same if abs(d["normal_effect"]) > effect_threshold
                )
                if same_substantial == 0:
                    print(
                        "    - ALL 'same direction' cases have small normal effects - likely noise, not real pattern"
                    )
                elif same_substantial < len(same) / 2:
                    print(
                        "    - Most 'same direction' cases have small normal effects - pattern may be overstated"
                    )
                else:
                    print(
                        "    - 'Same direction' cases have substantial effects - this appears to be a real pattern"
                    )

    # Overall comparison across models
    print("\n" + "=" * 100)
    print("CROSS-MODEL COMPARISON: Are 'same direction' findings meaningful?")
    print("=" * 100)

    print(
        f"\n{'Model':<30} {'Sig Cases':<12} {'Same Dir':<12} {'Avg |Norm| (same)':<20} {'Substantial':<15}"
    )
    print("-" * 90)

    for (model, reasoning), data in sorted(model_data.items(), key=sort_key):
        model_name = f"{get_model_display_name(model)} ({reasoning})"
        sig_cases = [d for d in data if d["normal_sig"]]
        same = [d for d in sig_cases if d["category"] == "same"]

        if same:
            avg_norm = np.mean([abs(d["normal_effect"]) for d in same])
            substantial = sum(
                1 for d in same if abs(d["normal_effect"]) > effect_threshold
            )
            print(
                f"{model_name:<30} {len(sig_cases):<12} {len(same):<12} {avg_norm:<20.3f} {substantial}/{len(same)}"
            )
        else:
            print(
                f"{model_name:<30} {len(sig_cases):<12} {'0':<12} {'N/A':<20} {'N/A'}"
            )


def print_appendix_summary_table(pairs: List[NegationPair]) -> None:
    """
    Print a summary table for the paper appendix.
    Shows % matches by model and factor (e.g., wealth vs gender).
    """
    # Collect data by (model, reasoning, factor)
    data: Dict[Tuple[str, str, str], Dict[str, int]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition, pair.factor)
        if key not in data:
            data[key] = {"matches": 0, "mismatches": 0, "insignificant": 0, "total": 0}

        # For "NOT prefer A", compare to "prefer B"
        if pair.normal_sig_B:  # Only when opposite nudge was significant
            data[key]["total"] += 1
            cat_A = classify_negation_vs_opposite(
                pair.negation_effect_A,
                pair.negation_sig_A,
                pair.normal_effect_B,
                pair.normal_sig_B,
            )
            data[key][cat_A] += 1

        # For "NOT prefer B", compare to "prefer A"
        if pair.normal_sig_A:  # Only when opposite nudge was significant
            data[key]["total"] += 1
            cat_B = classify_negation_vs_opposite(
                pair.negation_effect_B,
                pair.negation_sig_B,
                pair.normal_effect_A,
                pair.normal_sig_A,
            )
            data[key][cat_B] += 1

    print("\n" + "=" * 90)
    print("APPENDIX TABLE: Negation Understanding by Model and Factor")
    print("(% of cases where 'NOT prefer A' behaves like 'prefer B')")
    print("=" * 90)

    # Get unique factors
    factors = sorted(set(k[2] for k in data.keys()))

    # Print header
    header = f"{'Model':<20} {'Reas.':<8}"
    for factor in factors:
        header += f" {factor:>12}"
    print(header)
    print("-" * len(header))

    # Group by (model, reasoning)
    model_reasoning_pairs = sorted(
        set((k[0], k[1]) for k in data.keys()),
        key=lambda x: (x[0], {"off": 0, "none": 0, "low": 1, "before": 1}.get(x[1], 2)),
    )

    for model, reasoning in model_reasoning_pairs:
        model_name = get_model_display_name(model)
        row = f"{model_name:<20} {reasoning:<8}"

        for factor in factors:
            key = (model, reasoning, factor)
            if key in data and data[key]["total"] > 0:
                # Show --- if all cases are insignificant (no matches AND no mismatches)
                if data[key]["matches"] == 0 and data[key]["mismatches"] == 0:
                    row += f" {'---':>12}"
                else:
                    pct_matches = 100 * data[key]["matches"] / data[key]["total"]
                    row += f" {pct_matches:>11.0f}%"
            else:
                row += f" {'---':>12}"
        print(row)

    # Also print a summary by model only (aggregating across factors)
    print("\n" + "-" * 90)
    print("Summary by Model (aggregated across all factors):")
    print("-" * 90)
    print(
        f"{'Model':<20} {'Reas.':<8} {'% Match':>10} {'% Mismatch':>12} {'% Insig':>10} {'n':>6}"
    )
    print("-" * 70)

    model_agg: Dict[Tuple[str, str], Dict[str, int]] = {}
    for (model, reasoning, _factor), counts in data.items():
        key = (model, reasoning)
        if key not in model_agg:
            model_agg[key] = {
                "matches": 0,
                "mismatches": 0,
                "insignificant": 0,
                "total": 0,
            }
        for k in ["matches", "mismatches", "insignificant", "total"]:
            model_agg[key][k] += counts[k]

    for (model, reasoning), counts in sorted(
        model_agg.items(),
        key=lambda x: (
            x[0][0],
            {"off": 0, "none": 0, "low": 1, "before": 1}.get(x[0][1], 2),
        ),
    ):
        model_name = get_model_display_name(model)
        if counts["total"] > 0:
            pct_m = 100 * counts["matches"] / counts["total"]
            pct_mm = 100 * counts["mismatches"] / counts["total"]
            pct_i = 100 * counts["insignificant"] / counts["total"]
            print(
                f"{model_name:<20} {reasoning:<8} {pct_m:>9.1f}% {pct_mm:>11.1f}% {pct_i:>9.1f}% {counts['total']:>6}"
            )
        else:
            print(
                f"{model_name:<20} {reasoning:<8} {'---':>10} {'---':>12} {'---':>10} {'0':>6}"
            )

    print()


def create_stacked_bar_chart(
    pairs: List[NegationPair],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a stacked bar chart showing negation effect categories by model.

    Only includes cases where the normal nudge was significant.
    Shows % Opposite / % Insignificant / % Same Direction for each model.
    """
    # Group by model and compute percentages
    model_data: Dict[Tuple[str, str], Dict[str, int]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in model_data:
            model_data[key] = {"opposite": 0, "insignificant": 0, "same": 0, "total": 0}

        # Process direction A (only if normal nudge was significant)
        if pair.normal_sig_A:
            model_data[key]["total"] += 1
            cat_A = classify_negation_effect(
                pair.normal_effect_A,
                pair.normal_sig_A,
                pair.negation_effect_A,
                pair.negation_sig_A,
            )
            model_data[key][cat_A] += 1

        # Process direction B (only if normal nudge was significant)
        if pair.normal_sig_B:
            model_data[key]["total"] += 1
            cat_B = classify_negation_effect(
                pair.normal_effect_B,
                pair.normal_sig_B,
                pair.negation_effect_B,
                pair.negation_sig_B,
            )
            model_data[key][cat_B] += 1

    if not model_data:
        print("No data for stacked bar chart!")
        return None

    # Sort: group by model, non-reasoning before reasoning
    def sort_key(item):
        model, reasoning = item[0]
        reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(reasoning, 2)
        return (model, reasoning_order)

    sorted_models = sorted(model_data.items(), key=sort_key)

    # Prepare data for plotting
    labels = []
    opposite_pcts = []
    insig_pcts = []
    same_pcts = []

    for (model, reasoning), counts in sorted_models:
        if counts["total"] == 0:
            continue
        labels.append(f"{get_model_display_name(model)}\n({reasoning})")
        total = counts["total"]
        opposite_pcts.append(100 * counts["opposite"] / total)
        insig_pcts.append(100 * counts["insignificant"] / total)
        same_pcts.append(100 * counts["same"] / total)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.6

    # Colors: green for opposite (good), gray for insignificant, red for same (unexpected)
    colors = {
        "opposite": "#2E7D32",  # green
        "insignificant": "#9E9E9E",  # gray
        "same": "#C62828",  # red
    }

    # Stack the bars
    bars1 = ax.bar(
        x,
        opposite_pcts,
        width,
        label="Opposite (expected)",
        color=colors["opposite"],
        alpha=0.85,
    )
    bars2 = ax.bar(
        x,
        insig_pcts,
        width,
        bottom=opposite_pcts,
        label="Insignificant",
        color=colors["insignificant"],
        alpha=0.85,
    )
    bars3 = ax.bar(
        x,
        same_pcts,
        width,
        bottom=[o + i for o, i in zip(opposite_pcts, insig_pcts)],
        label="Same direction",
        color=colors["same"],
        alpha=0.85,
    )

    # Styling
    ax.set_ylabel("Percentage of Cases", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add percentage labels on bars (only for segments > 15%)
    def add_labels(bars, values, bottoms):
        for bar, val, bottom in zip(bars, values, bottoms):
            if val > 15:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + val / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

    add_labels(bars1, opposite_pcts, [0] * len(opposite_pcts))
    add_labels(bars2, insig_pcts, opposite_pcts)
    add_labels(bars3, same_pcts, [o + i for o, i in zip(opposite_pcts, insig_pcts)])

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Effect of Negated Nudges", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved stacked bar chart to: {output_path}")

    return fig, ax


def create_stacked_bar_chart_vs_opposite(
    pairs: List[NegationPair],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    include_insignificant: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a stacked bar chart comparing negation to OPPOSITE nudge.

    Tests: does "NOT prefer A" behave like "prefer B"?
    By default, only includes cases where the opposite nudge was significant.
    If include_insignificant=True, includes all cases.
    Shows % Matches / % Insignificant / % Mismatches for each model.
    """
    model_data: Dict[Tuple[str, str], Dict[str, int]] = {}

    for pair in pairs:
        key = (pair.model, pair.reasoning_condition)
        if key not in model_data:
            model_data[key] = {
                "matches": 0,
                "insignificant": 0,
                "mismatches": 0,
                "total": 0,
            }

        # For "NOT prefer A", compare to "prefer B"
        if include_insignificant or pair.normal_sig_B:
            model_data[key]["total"] += 1
            cat_A = classify_negation_vs_opposite(
                pair.negation_effect_A,
                pair.negation_sig_A,
                pair.normal_effect_B,
                pair.normal_sig_B,
                ignore_significance=include_insignificant,
            )
            model_data[key][cat_A] += 1

        # For "NOT prefer B", compare to "prefer A"
        if include_insignificant or pair.normal_sig_A:
            model_data[key]["total"] += 1
            cat_B = classify_negation_vs_opposite(
                pair.negation_effect_B,
                pair.negation_sig_B,
                pair.normal_effect_A,
                pair.normal_sig_A,
                ignore_significance=include_insignificant,
            )
            model_data[key][cat_B] += 1

    if not model_data:
        print("No data for stacked bar chart!")
        return None

    def sort_key(item):
        model, reasoning = item[0]
        reasoning_order = {"off": 0, "none": 0, "low": 1, "before": 1}.get(reasoning, 2)
        return (model, reasoning_order)

    sorted_models = sorted(model_data.items(), key=sort_key)

    labels = []
    matches_pcts = []
    insig_pcts = []
    mismatches_pcts = []

    for (model, reasoning), counts in sorted_models:
        if counts["total"] == 0:
            continue
        labels.append(f"{get_model_display_name(model)}\n({reasoning})")
        total = counts["total"]
        matches_pcts.append(100 * counts["matches"] / total)
        insig_pcts.append(100 * counts["insignificant"] / total)
        mismatches_pcts.append(100 * counts["mismatches"] / total)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.6

    # Colors: green for matches (good), gray for insignificant, red for mismatches (bad)
    colors = {
        "matches": "#2E7D32",  # green
        "insignificant": "#9E9E9E",  # gray
        "mismatches": "#C62828",  # red
    }

    bars1 = ax.bar(
        x,
        matches_pcts,
        width,
        label="Matches opposite nudge",
        color=colors["matches"],
        alpha=0.85,
    )
    bars2 = ax.bar(
        x,
        insig_pcts,
        width,
        bottom=matches_pcts,
        label="Insignificant",
        color=colors["insignificant"],
        alpha=0.85,
    )
    bars3 = ax.bar(
        x,
        mismatches_pcts,
        width,
        bottom=[m + i for m, i in zip(matches_pcts, insig_pcts)],
        label="Mismatches",
        color=colors["mismatches"],
        alpha=0.85,
    )

    ax.set_ylabel("Percentage of Cases", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def add_labels(bars, values, bottoms):
        for bar, val, bottom in zip(bars, values, bottoms):
            if val > 15:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + val / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

    add_labels(bars1, matches_pcts, [0] * len(matches_pcts))
    add_labels(bars2, insig_pcts, matches_pcts)
    add_labels(
        bars3, mismatches_pcts, [m + i for m, i in zip(matches_pcts, insig_pcts)]
    )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        if include_insignificant:
            ax.set_title(
                "Does 'NOT prefer A' behave like 'prefer B'?\n(all cases)",
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.set_title(
                "Does 'NOT prefer A' behave like 'prefer B'?",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"\nSaved vs-opposite bar chart to: {output_path}")

    return fig, ax


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
    parser.add_argument(
        "--deep-analysis",
        action="store_true",
        help="Show detailed effect size analysis to distinguish real patterns from noise",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=0.05,
        help="Threshold for 'substantial' effects in deep analysis (default: 0.05)",
    )
    parser.add_argument(
        "--bar-chart",
        action="store_true",
        help="Generate stacked bar chart showing negation effect categories by model",
    )
    parser.add_argument(
        "--figure-output",
        type=str,
        default=None,
        help="Output path for the figure (default: negation_analysis.pdf)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )
    parser.add_argument(
        "--vs-opposite",
        action="store_true",
        help="Compare 'NOT prefer A' to 'prefer B' instead of to 'prefer A'",
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

    # VS-OPPOSITE analysis: does "NOT prefer A" behave like "prefer B"?
    if args.vs_opposite:
        print("\n" + "=" * 100)
        print("VS-OPPOSITE ANALYSIS: Does 'NOT prefer A' behave like 'prefer B'?")
        print("=" * 100)

        matrix_vs_opp = create_contingency_matrix_vs_opposite(pairs)
        print(
            format_contingency_matrix_vs_opposite(
                matrix_vs_opp, title="Overall: NOT-A vs B"
            )
        )

        per_model_vs_opp = create_per_model_matrices_vs_opposite(pairs)
        print(format_per_model_matrices_vs_opposite(per_model_vs_opp))

        print_appendix_summary_table(pairs)

        print_detailed_vs_opposite_analysis(pairs)

    # Display detailed results if requested
    if not args.no_details:
        print(format_pairs_table(pairs, decimals=args.decimals))

    # Deep analysis of effect sizes
    if args.deep_analysis:
        print_detailed_effect_analysis(pairs, effect_threshold=args.effect_threshold)

    # Generate bar chart if requested
    if args.bar_chart:
        figure_path = args.figure_output or f"{PLOTS_OUTPUT_DIR}/negation_analysis.pdf"
        if args.vs_opposite:
            result = create_stacked_bar_chart_vs_opposite(
                pairs, output_path=figure_path
            )
            # Also generate version including insignificant cases
            base, ext = (
                figure_path.rsplit(".", 1)
                if "." in figure_path
                else (figure_path, "pdf")
            )
            insig_path = f"{base}_include_insignificant.{ext}"
            create_stacked_bar_chart_vs_opposite(
                pairs, output_path=insig_path, include_insignificant=True
            )
        else:
            result = create_stacked_bar_chart(pairs, output_path=figure_path)
        if result and not args.no_show:
            plt.show()

    # Write CSV if requested
    if args.output:
        write_pairs_csv(pairs, args.output)


if __name__ == "__main__":
    main()
