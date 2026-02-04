"""
Reasoning trace classification and analysis.

This module contains scripts for classifying and analyzing model reasoning traces.
"""

from choices.analysis.reasoning_traces.edge_filtering import (
    EdgeComparison,
    EdgeFilter,
    EdgeFilteringPipeline,
    TraceWithContext,
    a_has_larger_n,
    b_has_larger_n,
    backfire_A,
    backfire_B,
    baseline_prefers_a,
    baseline_prefers_b,
    effect_A,
    effect_B,
    equal_n,
    extract_edge_comparisons,
    extract_traces_for_edge,
    filter_traces,
    larger_n,
    n_difference,
    sort_by,
    threshold,
    top_k,
)

__all__ = [
    # Data structures
    "EdgeComparison",
    "EdgeFilter",
    "TraceWithContext",
    # Pipeline
    "EdgeFilteringPipeline",
    # Filter factory functions
    "backfire_A",
    "backfire_B",
    "effect_A",
    "effect_B",
    "larger_n",
    "a_has_larger_n",
    "b_has_larger_n",
    "equal_n",
    "n_difference",
    "baseline_prefers_a",
    "baseline_prefers_b",
    "threshold",
    # Extraction functions
    "extract_edge_comparisons",
    "extract_traces_for_edge",
    "filter_traces",
    # Sorting helpers
    "top_k",
    "sort_by",
]
