#!/usr/bin/env python3
"""
Sample random edges across models, factors and nudge types, and export traces.

For each sampled edge, both nudge directions (towards A and towards B) are
exported as separate cases, each containing:
- All baseline traces (condition_a)
- All nudged traces for that direction (condition_b)
- The signed effect: nudged_freq − baseline_freq

The output JSON is compatible with downstream scripts:
``compliance_classification.py``, ``rationale_detection.py``, and the
corresponding plot scripts.

Usage:
    uv run python -m choices.analysis.reasoning_traces.sample_edges \
        --results-dirs results_main0 results_main1 \
        --models gpt-5-2-reasoning \
        --max-samples 100 --seed 42 \
        --output sampled_edges.json

    # Include equal-N edges too
    uv run python -m choices.analysis.reasoning_traces.sample_edges \
        --results-dirs results_main0 results_main1 \
        --models gpt-5-2-reasoning \
        --min-n-diff 0 --max-samples 200 \
        --output sampled_edges.json
"""

import argparse
import json
import random
from datetime import datetime

from choices.analysis.reasoning_traces.edge_filtering import (
    EdgeComparison,
    EdgeFilteringPipeline,
    TraceWithContext,
    extract_traces_for_edge,
    n_difference,
)


# =============================================================================
# Generic edge → case conversion
# =============================================================================


def extract_traces_for_direction(
    edge: EdgeComparison,
    nudged_option: str,
) -> tuple[list[TraceWithContext], list[TraceWithContext]]:
    """
    Extract all baseline and nudged traces for a given nudge direction.

    Args:
        edge: The edge to extract traces from.
        nudged_option: ``"A"`` or ``"B"`` – which option the nudge targets.

    Returns:
        (baseline_traces, nudged_traces)
    """
    baseline_traces = extract_traces_for_edge(edge, conditions=["base"])
    nudge_condition = edge.level_A if nudged_option == "A" else edge.level_B
    nudged_traces = extract_traces_for_edge(edge, conditions=[nudge_condition])
    return baseline_traces, nudged_traces


def edge_to_case_dict(
    edge: EdgeComparison,
    nudged_option: str,
    baseline_traces: list[TraceWithContext],
    nudged_traces: list[TraceWithContext],
) -> dict:
    """
    Build a standard case dict from an edge and its traces.

    The resulting dict is compatible with ``compliance_classification.py``,
    ``rationale_detection.py`` and the plotting scripts.

    Args:
        edge: The EdgeComparison.
        nudged_option: ``"A"`` or ``"B"``.
        baseline_traces: Traces from the baseline (no nudge) condition.
        nudged_traces: Traces from the nudged condition.

    Returns:
        A case dict ready for JSON serialization.
    """
    if nudged_option == "A":
        baseline_freq = edge.f_0_A
        nudged_freq = edge.f_A_A
        nudged_n = edge.option_a_n
        other_n = edge.option_b_n
    else:
        baseline_freq = edge.f_0_B
        nudged_freq = edge.f_B_B
        nudged_n = edge.option_b_n
        other_n = edge.option_a_n

    return {
        # Edge identification
        "edge_key": edge.edge_key,
        # Experiment metadata
        "model": edge.model,
        "factor": edge.factor,
        "nudge_type": edge.nudge_type,
        "level_A": edge.level_A,
        "level_B": edge.level_B,
        # Option info
        "option_a_label": edge.option_a_label,
        "option_b_label": edge.option_b_label,
        "option_a_n": edge.option_a_n,
        "option_b_n": edge.option_b_n,
        # Cross-condition frequencies
        "f_0_A": edge.f_0_A,
        "f_A_A": edge.f_A_A,
        "f_B_A": edge.f_B_A,
        # Direction info
        "nudged_option": nudged_option,
        "baseline_freq": baseline_freq,
        "nudged_freq": nudged_freq,
        "nudged_n": nudged_n,
        "other_n": other_n,
        # Traces
        "condition_a_name": "baseline",
        "condition_b_name": f"nudged_towards_{nudged_option}",
        "condition_a_traces": [
            {
                "choice": t.choice,
                "reasoning": t.reasoning,
                "is_flipped": t.is_flipped,
            }
            for t in baseline_traces
        ],
        "condition_b_traces": [
            {
                "choice": t.choice,
                "reasoning": t.reasoning,
                "is_flipped": t.is_flipped,
            }
            for t in nudged_traces
        ],
    }


def save_cases_json(
    cases: list[dict],
    metadata: dict,
    output_path: str,
):
    """Save a list of case dicts to JSON with metadata."""
    data = {
        "metadata": {
            **metadata,
            "saved_at": datetime.now().isoformat(),
            "n_cases": len(cases),
        },
        "cases": cases,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(cases)} cases to {output_path}")


# =============================================================================
# Edge → case expansion (both directions)
# =============================================================================


def _expand_direction_list(directions: str) -> list[str]:
    """Return the list of nudge directions implied by *directions*."""
    out: list[str] = []
    if directions in ("both", "A"):
        out.append("A")
    if directions in ("both", "B"):
        out.append("B")
    return out


def edges_to_cases(
    edges: list[EdgeComparison],
    directions: str = "both",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Convert edges to case dicts, extracting traces for each nudge direction.

    When *max_samples* is given the edges are sampled **before** trace
    extraction so that we only do expensive I/O for the edges we keep.

    Args:
        edges: List of EdgeComparison objects.
        directions: Which nudge directions to include:
            ``"both"`` – one case per direction (A and B) per edge,
            ``"A"`` – only nudge-towards-A cases,
            ``"B"`` – only nudge-towards-B cases.
        max_samples: If set, randomly sample at most this many (edge, direction)
            pairs *before* extracting traces.
        seed: Random seed for sampling.

    Returns:
        List of case dicts.
    """
    target_directions = _expand_direction_list(directions)

    # Build lightweight (edge, direction) pairs first – no I/O yet
    pairs: list[tuple[EdgeComparison, str]] = [
        (edge, d) for edge in edges for d in target_directions
    ]

    # Sample early to avoid extracting traces we'll throw away
    if max_samples is not None and max_samples < len(pairs):
        random.seed(seed)
        pairs = random.sample(pairs, max_samples)
        print(f"Sampled {max_samples} (edge, direction) pairs (seed={seed})")

    cases: list[dict] = []
    for edge, nudged_option in pairs:
        baseline_traces, nudged_traces = extract_traces_for_direction(
            edge, nudged_option
        )
        if not nudged_traces:
            continue
        cases.append(
            edge_to_case_dict(edge, nudged_option, baseline_traces, nudged_traces)
        )

    return cases


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sample random edges and export traces for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --results-dirs results_main0 results_main1 \\\n"
            "    --models gpt-5-2-reasoning --max-samples 100 -o sampled.json\n"
            "\n"
            "  %(prog)s --results-dirs results/ --models gpt-5-2-reasoning \\\n"
            "    --factors age_group gender --max-samples 50 -o sampled.json"
        ),
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results/"],
        help="Directories to search for result files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific models",
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
        "--min-n-diff",
        type=int,
        default=0,
        help="Minimum |n_A − n_B| (default: 0, include all edges)",
    )
    parser.add_argument(
        "--directions",
        choices=["both", "A", "B"],
        default="both",
        help="Which nudge directions to include (default: both)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of cases to keep (sampled randomly)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON file",
    )

    args = parser.parse_args()

    # Build edge pipeline
    edge_filter = n_difference(args.min_n_diff) if args.min_n_diff > 0 else None
    pipeline = EdgeFilteringPipeline(
        results_dirs=args.results_dirs,
        edge_filter=edge_filter,
        models=args.models,
        factors=args.factors,
        nudge_types=args.nudge_types,
    )

    edges = pipeline.get_filtered_edges()
    print(f"Extracted {len(edges)} edges")

    # Convert to cases – sampling happens *before* trace extraction
    print("Extracting traces...")
    cases = edges_to_cases(
        edges,
        directions=args.directions,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"Built {len(cases)} cases")

    # Save
    metadata = {
        "results_dirs": args.results_dirs,
        "models": args.models,
        "factors": args.factors,
        "nudge_types": args.nudge_types,
        "min_n_diff": args.min_n_diff,
        "directions": args.directions,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }
    save_cases_json(cases, metadata, args.output)


if __name__ == "__main__":
    main()
