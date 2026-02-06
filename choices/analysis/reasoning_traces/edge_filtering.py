#!/usr/bin/env python3
"""
Edge filtering pipeline for reasoning trace analysis.

This module provides tools for filtering edges (pairwise comparisons) based on
cross-condition statistics like backfire rates, effect sizes, and N-value differences.

Key components:
- EdgeComparison: Dataclass holding edge-level data across conditions
- EdgeFilter: Composable filter with predicate and metadata
- TraceWithContext: Trace data with associated edge context
- EdgeFilteringPipeline: Orchestrates extraction and filtering

Example usage:
    from choices.analysis.reasoning_traces.edge_filtering import (
        EdgeFilteringPipeline,
        backfire_A, a_has_larger_n, effect_A, n_difference, top_k
    )

    # Find backfiring cases where A has larger N
    pipeline = EdgeFilteringPipeline(
        results_dirs=["results/main_results/results_main0"],
        edge_filter=backfire_A(0.1) & a_has_larger_n(2),
        factors=["age_group"],
    )
    edges = pipeline.get_filtered_edges()

    # Get traces where model chose B (the backfire choice)
    traces = pipeline.get_traces(conditions=["young"], choices=["B"])
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from choices.analysis.create_summary import (
    discover_experiments,
    find_condition_directories,
)
from choices.analysis.metrics import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EdgeComparison:
    """
    Edge-level data including cross-condition statistics and experiment metadata.

    An edge represents a pairwise comparison between two options (A and B).
    This dataclass holds the frequencies and counts across all three conditions:
    - base: No nudge
    - nudge_A: Nudging towards option A (level_A)
    - nudge_B: Nudging towards option B (level_B)

    Frequencies are stored for option A; option B frequencies are 1 - f_A.
    """

    # Experiment metadata
    results_dir: str
    model: str
    factor: str
    nudge_type: str
    level_A: str  # Factor level for "A" (e.g., "young")
    level_B: str  # Factor level for "B" (e.g., "old")

    # Edge identification
    edge_key: str

    # Option info
    option_a_label: str
    option_b_label: str
    option_a_n: int
    option_b_n: int

    # Cross-condition frequencies (all for option A)
    f_0_A: float  # P(A) in baseline
    f_A_A: float  # P(A) when nudged towards A (level_A)
    f_B_A: float  # P(A) when nudged towards B (level_B)

    # Raw counts for statistical tests
    base_chose_a: int
    base_chose_b: int
    nudge_a_chose_a: int
    nudge_a_chose_b: int
    nudge_b_chose_a: int
    nudge_b_chose_b: int

    # Condition directories (for trace extraction)
    condition_dirs: Dict[str, Path] = field(default_factory=dict, repr=False)

    @property
    def n_difference(self) -> int:
        """Absolute difference in N values between options."""
        return abs(self.option_a_n - self.option_b_n)

    @property
    def n_difference_signed(self) -> int:
        """Signed difference: option_a_n - option_b_n."""
        return self.option_a_n - self.option_b_n

    @property
    def f_0_B(self) -> float:
        """P(B) in baseline."""
        return 1 - self.f_0_A

    @property
    def f_A_B(self) -> float:
        """P(B) when nudged towards A."""
        return 1 - self.f_A_A

    @property
    def f_B_B(self) -> float:
        """P(B) when nudged towards B."""
        return 1 - self.f_B_A

    @property
    def effect_A(self) -> float:
        """Effect of nudging towards A: f_A(A) - f_0(A). Positive = compliance."""
        return self.f_A_A - self.f_0_A

    @property
    def effect_B(self) -> float:
        """Effect of nudging towards B: f_B(B) - f_0(B). Positive = compliance."""
        return self.f_B_B - self.f_0_B

    @property
    def effect_A_magnitude(self) -> float:
        """Absolute effect of nudging towards A."""
        return abs(self.effect_A)

    @property
    def effect_B_magnitude(self) -> float:
        """Absolute effect of nudging towards B."""
        return abs(self.effect_B)

    @property
    def backfire_A(self) -> bool:
        """True if nudging towards A decreased A's frequency."""
        return self.f_A_A < self.f_0_A

    @property
    def backfire_B(self) -> bool:
        """True if nudging towards B decreased B's frequency."""
        return self.f_B_B < self.f_0_B

    @property
    def backfire_A_magnitude(self) -> float:
        """Magnitude of backfire towards A (positive if backfired)."""
        return self.f_0_A - self.f_A_A

    @property
    def backfire_B_magnitude(self) -> float:
        """Magnitude of backfire towards B (positive if backfired)."""
        return self.f_0_B - self.f_B_B

    @property
    def base_total(self) -> int:
        """Total responses in base condition."""
        return self.base_chose_a + self.base_chose_b

    @property
    def nudge_a_total(self) -> int:
        """Total responses in nudge_A condition."""
        return self.nudge_a_chose_a + self.nudge_a_chose_b

    @property
    def nudge_b_total(self) -> int:
        """Total responses in nudge_B condition."""
        return self.nudge_b_chose_a + self.nudge_b_chose_b


@dataclass
class TraceWithContext:
    """
    A reasoning trace with associated edge context.

    This combines the reasoning trace with the edge-level statistics,
    allowing analysis of traces in the context of cross-condition behavior.
    """

    # Edge context
    edge: EdgeComparison

    # Condition this trace is from
    condition: str  # "base", level_A, or level_B

    # Trace data
    reasoning: str
    choice: str  # "A" or "B"
    is_flipped: bool

    @property
    def chose_a(self) -> bool:
        """True if the model chose option A."""
        return self.choice == "A"

    @property
    def chose_b(self) -> bool:
        """True if the model chose option B."""
        return self.choice == "B"

    @property
    def chose_larger_n(self) -> bool:
        """True if model chose the option with more people."""
        if self.edge.option_a_n > self.edge.option_b_n:
            return self.choice == "A"
        elif self.edge.option_b_n > self.edge.option_a_n:
            return self.choice == "B"
        return True  # Equal N

    @property
    def chose_smaller_n(self) -> bool:
        """True if model chose the option with fewer people."""
        if self.edge.option_a_n < self.edge.option_b_n:
            return self.choice == "A"
        elif self.edge.option_b_n < self.edge.option_a_n:
            return self.choice == "B"
        return True  # Equal N


# =============================================================================
# Edge Filter
# =============================================================================


@dataclass
class EdgeFilter:
    """
    A composable filter for EdgeComparison objects.

    Wraps a predicate function with metadata for introspection.
    Supports composition via &, |, and ~ operators.

    Example:
        f = backfire_A(0.1) & a_has_larger_n(2)
        matching_edges = [e for e in edges if f(e)]
    """

    predicate: Callable[[EdgeComparison], bool]
    name: str

    def __call__(self, edge: EdgeComparison) -> bool:
        """Apply the filter to an edge."""
        return self.predicate(edge)

    def __and__(self, other: "EdgeFilter") -> "EdgeFilter":
        """Combine filters with AND logic."""
        return EdgeFilter(
            lambda e: self(e) and other(e),
            f"({self.name} & {other.name})",
        )

    def __or__(self, other: "EdgeFilter") -> "EdgeFilter":
        """Combine filters with OR logic."""
        return EdgeFilter(
            lambda e: self(e) or other(e),
            f"({self.name} | {other.name})",
        )

    def __invert__(self) -> "EdgeFilter":
        """Negate the filter."""
        return EdgeFilter(lambda e: not self(e), f"~{self.name}")

    def __repr__(self) -> str:
        return f"EdgeFilter({self.name})"


# =============================================================================
# Filter Factory Functions
# =============================================================================


def backfire_A() -> EdgeFilter:
    """Filter for edges where nudging towards A decreased A's frequency (f_A(A) < f_0(A))."""
    return EdgeFilter(lambda e: e.backfire_A, "backfire_A")


def backfire_B() -> EdgeFilter:
    """Filter for edges where nudging towards B decreased B's frequency (f_B(B) < f_0(B))."""
    return EdgeFilter(lambda e: e.backfire_B, "backfire_B")


def effect_A(min_effect: float = 0.0) -> EdgeFilter:
    """
    Filter for edges with significant effect when nudging towards A.

    Args:
        min_effect: Minimum absolute effect |f_A(A) - f_0(A)|
    """
    return EdgeFilter(
        lambda e: e.effect_A_magnitude >= min_effect,
        f"effect_A(>={min_effect})",
    )


def effect_B(min_effect: float = 0.0) -> EdgeFilter:
    """
    Filter for edges with significant effect when nudging towards B.

    Args:
        min_effect: Minimum absolute effect |f_B(B) - f_0(B)|
    """
    return EdgeFilter(
        lambda e: e.effect_B_magnitude >= min_effect,
        f"effect_B(>={min_effect})",
    )


def larger_n(option: str | None = None, min_diff: int = 1) -> EdgeFilter:
    """
    Filter for edges where specified option has more people.

    Args:
        option: "A" (A has larger N), "B" (B has larger N), or None (no filter)
        min_diff: Minimum difference required

    Returns:
        EdgeFilter that passes all edges if option is None
    """
    if option is None:
        return EdgeFilter(lambda e: True, "larger_n(any)")
    elif option.upper() == "A":
        return EdgeFilter(
            lambda e: e.n_difference_signed >= min_diff,
            f"larger_n(A, >={min_diff})",
        )
    elif option.upper() == "B":
        return EdgeFilter(
            lambda e: -e.n_difference_signed >= min_diff,
            f"larger_n(B, >={min_diff})",
        )
    else:
        raise ValueError(f"option must be 'A', 'B', or None, got {option}")


def a_has_larger_n(min_diff: int = 1) -> EdgeFilter:
    """Filter for edges where option A has more people than B."""
    return larger_n("A", min_diff)


def b_has_larger_n(min_diff: int = 1) -> EdgeFilter:
    """Filter for edges where option B has more people than A."""
    return larger_n("B", min_diff)


def equal_n() -> EdgeFilter:
    """Filter for edges where both options have equal N values."""
    return EdgeFilter(lambda e: e.option_a_n == e.option_b_n, "equal_n")


def n_difference(min_diff: int = 1) -> EdgeFilter:
    """
    Filter for edges with at least min_diff difference in N values.

    Args:
        min_diff: Minimum absolute difference |option_a_n - option_b_n|
    """
    return EdgeFilter(
        lambda e: e.n_difference >= min_diff,
        f"n_diff(>={min_diff})",
    )


def baseline_prefers_a(min_pref: float = 0.5) -> EdgeFilter:
    """
    Filter for edges where baseline prefers A.

    Args:
        min_pref: Minimum f_0(A) threshold
    """
    return EdgeFilter(
        lambda e: e.f_0_A > min_pref,
        f"baseline_A(>{min_pref})",
    )


def baseline_prefers_b(min_pref: float = 0.5) -> EdgeFilter:
    """
    Filter for edges where baseline prefers B.

    Args:
        min_pref: Minimum f_0(B) threshold
    """
    return EdgeFilter(
        lambda e: e.f_0_B > min_pref,
        f"baseline_B(>{min_pref})",
    )


def threshold(attr: str, op: str, value: float) -> EdgeFilter:
    """
    Generic threshold filter on any numeric EdgeComparison attribute.

    Args:
        attr: Attribute name (e.g., "f_0_A", "option_a_n", "effect_A")
        op: Operator ("gt", "lt", "gte", "lte", "eq")
        value: Threshold value

    Example:
        threshold("f_0_A", "gt", 0.6)  # f_0(A) > 0.6
    """
    ops = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
    }
    if op not in ops:
        raise ValueError(f"Unknown operator: {op}. Use one of {list(ops.keys())}")

    op_symbols = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "=="}

    return EdgeFilter(
        lambda e: ops[op](getattr(e, attr), value),
        f"{attr}{op_symbols[op]}{value}",
    )


# =============================================================================
# Sorting Helpers
# =============================================================================


def top_k(
    edges: List[EdgeComparison],
    key: str,
    k: int,
    reverse: bool = True,
) -> List[EdgeComparison]:
    """
    Get top k edges sorted by attribute.

    Args:
        edges: List of EdgeComparison objects
        key: Attribute name to sort by (e.g., "effect_A", "n_difference")
        k: Number of edges to return
        reverse: If True (default), sort descending (largest first)
    """
    return sorted(edges, key=lambda e: getattr(e, key), reverse=reverse)[:k]


def sort_by(
    edges: List[EdgeComparison],
    key: str,
    reverse: bool = True,
) -> List[EdgeComparison]:
    """
    Sort edges by attribute.

    Args:
        edges: List of EdgeComparison objects
        key: Attribute name to sort by
        reverse: If True (default), sort descending
    """
    return sorted(edges, key=lambda e: getattr(e, key), reverse=reverse)


# =============================================================================
# Edge Extraction
# =============================================================================


def extract_edge_comparisons(
    condition_dirs: Dict[str, Path],
    factor_name: str,
    results_dir: str,
    model: str,
    nudge_type: str,
) -> List[EdgeComparison]:
    """
    Extract edge-level data across all conditions.

    Args:
        condition_dirs: Dict mapping condition -> Path
            e.g., {"base": Path, "young": Path, "old": Path}
        factor_name: The factor variable name in the graph
        results_dir: Results directory path (for metadata)
        model: Model name (for metadata)
        nudge_type: Nudge type (for metadata)

    Returns:
        List of EdgeComparison objects with matched edge data
    """
    if "base" not in condition_dirs:
        return []

    # Load base graph
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return []

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return []

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        return []  # Only support binary factors for now

    level_A, level_B = factor_levels[0], factor_levels[1]

    if level_A not in condition_dirs or level_B not in condition_dirs:
        return []

    # Load nudge condition graphs
    nudge_a_graph = load_preference_graph(condition_dirs[level_A])
    nudge_b_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_a_graph or not nudge_b_graph:
        return []

    # Build option lookup
    options = base_graph.get("options", [])
    options_by_id = {opt["id"]: opt for opt in options}

    # Extract edge data
    comparisons = []
    base_edges = base_graph.get("edges", {})
    nudge_a_edges = nudge_a_graph.get("edges", {})
    nudge_b_edges = nudge_b_graph.get("edges", {})

    for edge_key, base_edge in base_edges.items():
        # Skip if edge not in all conditions
        if edge_key not in nudge_a_edges or edge_key not in nudge_b_edges:
            continue

        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            opt_a_group = opt_a.get(factor_var_name)
            opt_b_group = opt_b.get(factor_var_name)

            # Skip intra-group comparisons
            if opt_a_group == opt_b_group:
                continue

            # Ensure opt_a corresponds to level_A
            # If not, swap them
            if opt_a_group != level_A:
                opt_a, opt_b = opt_b, opt_a
                opt_a_group, opt_b_group = opt_b_group, opt_a_group

            # Get counts from each condition
            def get_counts(edge_data) -> Tuple[int, int]:
                aux = edge_data.get("aux_data", {})
                orig = aux.get("original_parsed", [])
                flip = aux.get("flipped_parsed", [])

                # Determine which position opt_a is in for this edge
                edge_ids = eval(edge_key)
                opt_a_is_first = edge_ids[0] == opt_a["id"]

                if opt_a_is_first:
                    # Original: A=opt_a, B=opt_b
                    # Flipped: A=opt_b, B=opt_a
                    chose_a = sum(1 for r in orig if r == "A") + sum(
                        1 for r in flip if r == "B"
                    )
                    chose_b = sum(1 for r in orig if r == "B") + sum(
                        1 for r in flip if r == "A"
                    )
                else:
                    # Original: A=opt_b, B=opt_a
                    # Flipped: A=opt_a, B=opt_b
                    chose_a = sum(1 for r in orig if r == "B") + sum(
                        1 for r in flip if r == "A"
                    )
                    chose_b = sum(1 for r in orig if r == "A") + sum(
                        1 for r in flip if r == "B"
                    )

                return chose_a, chose_b

            base_a, base_b = get_counts(base_edge)
            nudge_a_a, nudge_a_b = get_counts(nudge_a_edges[edge_key])
            nudge_b_a, nudge_b_b = get_counts(nudge_b_edges[edge_key])

            base_total = base_a + base_b
            nudge_a_total = nudge_a_a + nudge_a_b
            nudge_b_total = nudge_b_a + nudge_b_b

            # Compute frequencies
            f_0_A = base_a / base_total if base_total > 0 else 0.5
            f_A_A = nudge_a_a / nudge_a_total if nudge_a_total > 0 else 0.5
            f_B_A = nudge_b_a / nudge_b_total if nudge_b_total > 0 else 0.5

            comparisons.append(
                EdgeComparison(
                    results_dir=results_dir,
                    model=model,
                    factor=factor_name,
                    nudge_type=nudge_type,
                    level_A=level_A,
                    level_B=level_B,
                    edge_key=edge_key,
                    option_a_label=opt_a.get("label", ""),
                    option_b_label=opt_b.get("label", ""),
                    option_a_n=opt_a.get("N", 0),
                    option_b_n=opt_b.get("N", 0),
                    f_0_A=f_0_A,
                    f_A_A=f_A_A,
                    f_B_A=f_B_A,
                    base_chose_a=base_a,
                    base_chose_b=base_b,
                    nudge_a_chose_a=nudge_a_a,
                    nudge_a_chose_b=nudge_a_b,
                    nudge_b_chose_a=nudge_b_a,
                    nudge_b_chose_b=nudge_b_b,
                    condition_dirs=condition_dirs,
                )
            )
        except Exception:
            continue

    return comparisons


# =============================================================================
# Trace Extraction
# =============================================================================


def extract_traces_for_edge(
    edge: EdgeComparison,
    conditions: Optional[List[str]] = None,
) -> List[TraceWithContext]:
    """
    Extract reasoning traces for a specific edge.

    Args:
        edge: EdgeComparison to extract traces for
        conditions: List of conditions to extract from. If None, extracts from all.
            Valid values: "base", edge.level_A, edge.level_B

    Returns:
        List of TraceWithContext objects
    """
    traces = []

    # Determine which conditions to extract
    if conditions is None:
        target_conditions = ["base", edge.level_A, edge.level_B]
    else:
        target_conditions = conditions

    # Map condition names to directory paths
    condition_to_path = {
        "base": edge.condition_dirs.get("base"),
        edge.level_A: edge.condition_dirs.get(edge.level_A),
        edge.level_B: edge.condition_dirs.get(edge.level_B),
    }

    for condition in target_conditions:
        path = condition_to_path.get(condition)
        if not path:
            continue

        graph = load_preference_graph(path)
        if not graph:
            continue

        edges_data = graph.get("edges", {})
        if edge.edge_key not in edges_data:
            continue

        edge_data = edges_data[edge.edge_key]
        aux = edge_data.get("aux_data", {})

        # Get factor info to determine option positions
        options = graph.get("options", [])
        options_by_id = {opt["id"]: opt for opt in options}
        factor_var_name = get_factor_name_from_graph(graph)

        edge_ids = eval(edge.edge_key)
        opt_first = options_by_id.get(edge_ids[0])
        opt_second = options_by_id.get(edge_ids[1])

        if not opt_first or not opt_second:
            continue

        # Determine which option is level_A
        first_is_level_a = opt_first.get(factor_var_name) == edge.level_A

        # Extract original traces (summaries preferred, fall back to full reasoning)
        orig_reasoning = aux.get("original_reasoning_summaries") or aux.get(
            "original_reasoning", []
        )
        orig_parsed = aux.get("original_parsed", [])

        for reasoning, choice in zip(orig_reasoning, orig_parsed):
            if not reasoning or choice not in ("A", "B"):
                continue

            # Map choice to level_A/level_B
            if first_is_level_a:
                actual_choice = choice  # A means level_A, B means level_B
            else:
                actual_choice = "B" if choice == "A" else "A"  # Swap

            traces.append(
                TraceWithContext(
                    edge=edge,
                    condition=condition,
                    reasoning=reasoning,
                    choice=actual_choice,
                    is_flipped=False,
                )
            )

        # Extract flipped traces (summaries preferred, fall back to full reasoning)
        flip_reasoning = aux.get("flipped_reasoning_summaries") or aux.get(
            "flipped_reasoning", []
        )
        flip_parsed = aux.get("flipped_parsed", [])

        for reasoning, choice in zip(flip_reasoning, flip_parsed):
            if not reasoning or choice not in ("A", "B"):
                continue

            # In flipped, positions are swapped
            if first_is_level_a:
                # Flipped: A means level_B, B means level_A
                actual_choice = "B" if choice == "A" else "A"
            else:
                actual_choice = choice

            traces.append(
                TraceWithContext(
                    edge=edge,
                    condition=condition,
                    reasoning=reasoning,
                    choice=actual_choice,
                    is_flipped=True,
                )
            )

    return traces


def filter_traces(
    traces: List[TraceWithContext],
    conditions: Optional[List[str]] = None,
    choices: Optional[List[str]] = None,
) -> List[TraceWithContext]:
    """
    Filter traces by condition and/or choice.

    Args:
        traces: List of TraceWithContext objects
        conditions: Filter to these conditions (e.g., ["base", "young"])
        choices: Filter to these choices (e.g., ["A"], ["B"], ["A", "B"])

    Returns:
        Filtered list of traces
    """
    result = traces

    if conditions is not None:
        result = [t for t in result if t.condition in conditions]

    if choices is not None:
        result = [t for t in result if t.choice in choices]

    return result


# =============================================================================
# Pipeline
# =============================================================================


class EdgeFilteringPipeline:
    """
    Pipeline for extracting and filtering edges across conditions.

    This class orchestrates the full extraction and filtering workflow:
    1. Discover experiments from results directories
    2. Extract edge-level data with cross-condition statistics
    3. Apply edge filters
    4. Extract reasoning traces for filtered edges

    Example:
        pipeline = EdgeFilteringPipeline(
            results_dirs=["results/main_results/results_main0"],
            edge_filter=backfire_A(0.1) & a_has_larger_n(2),
            factors=["age_group"],
        )
        edges = pipeline.get_filtered_edges()
        traces = pipeline.get_traces(conditions=["young"], choices=["B"])
    """

    def __init__(
        self,
        results_dirs: List[str],
        edge_filter: Optional[EdgeFilter] = None,
        models: Optional[List[str]] = None,
        factors: Optional[List[str]] = None,
        nudge_types: Optional[List[str]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            results_dirs: List of results directory paths
            edge_filter: Optional EdgeFilter to apply
            models: Optional list of models to include
            factors: Optional list of factors to include
            nudge_types: Optional list of nudge types to include
        """
        self.results_dirs = results_dirs
        self.edge_filter = edge_filter
        self.models = models
        self.factors = factors
        self.nudge_types = nudge_types

        # Cache for extracted edges
        self._edges: Optional[List[EdgeComparison]] = None

    def get_edges(self) -> List[EdgeComparison]:
        """
        Extract all edges matching experiment filters.

        Returns:
            List of EdgeComparison objects (unfiltered by edge_filter)
        """
        if self._edges is not None:
            return self._edges

        edges = []

        # Discover experiments
        experiments = discover_experiments(
            self.results_dirs,
            model_filter=self.models,
            factor_filter=self.factors,
            nudge_type_filter=self.nudge_types,
        )

        # Extract edges from each experiment
        for results_dir, factor, model, nudge_type in experiments:
            condition_dirs_list = find_condition_directories(
                factor, model, nudge_type, results_dir
            )

            for condition_dirs in condition_dirs_list:
                experiment_edges = extract_edge_comparisons(
                    condition_dirs=condition_dirs,
                    factor_name=factor,
                    results_dir=results_dir,
                    model=model,
                    nudge_type=nudge_type,
                )
                edges.extend(experiment_edges)

        self._edges = edges
        return edges

    def get_filtered_edges(self) -> List[EdgeComparison]:
        """
        Apply edge_filter to extracted edges.

        Returns:
            List of EdgeComparison objects matching the filter
        """
        edges = self.get_edges()

        if self.edge_filter is None:
            return edges

        return [e for e in edges if self.edge_filter(e)]

    def get_traces(
        self,
        conditions: Optional[List[str]] = None,
        choices: Optional[List[str]] = None,
    ) -> List[TraceWithContext]:
        """
        Get reasoning traces for filtered edges with optional trace filtering.

        Args:
            conditions: Filter traces to these conditions
            choices: Filter traces to these choices ("A" or "B")

        Returns:
            List of TraceWithContext objects
        """
        edges = self.get_filtered_edges()
        all_traces = []

        for edge in edges:
            traces = extract_traces_for_edge(edge, conditions=conditions)
            all_traces.extend(traces)

        if choices is not None:
            all_traces = filter_traces(all_traces, choices=choices)

        return all_traces

    def summary(self) -> dict:
        """
        Return summary statistics about the filtering.

        Returns:
            Dictionary with filtering statistics
        """
        all_edges = self.get_edges()
        filtered_edges = self.get_filtered_edges()

        # Group by model/factor
        models = set(e.model for e in filtered_edges)
        factors = set(e.factor for e in filtered_edges)

        return {
            "filter_name": self.edge_filter.name if self.edge_filter else "none",
            "total_edges": len(all_edges),
            "filtered_edges": len(filtered_edges),
            "filter_rate": (len(filtered_edges) / len(all_edges) if all_edges else 0.0),
            "models": sorted(models),
            "factors": sorted(factors),
        }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI entry point for testing edge filtering."""
    import argparse

    parser = argparse.ArgumentParser(description="Edge filtering pipeline")
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results"],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--factors", nargs="+", default=None)
    parser.add_argument("--nudge-types", nargs="+", default=None)
    parser.add_argument(
        "--backfire-a",
        action="store_true",
        help="Filter for backfire towards A (f_A(A) < f_0(A))",
    )
    parser.add_argument(
        "--backfire-b",
        action="store_true",
        help="Filter for backfire towards B (f_B(B) < f_0(B))",
    )
    parser.add_argument(
        "--effect-a",
        type=float,
        default=None,
        help="Filter for |f_A(A) - f_0(A)| >= value",
    )
    parser.add_argument(
        "--effect-b",
        type=float,
        default=None,
        help="Filter for |f_B(B) - f_0(B)| >= value",
    )
    parser.add_argument(
        "--n-diff",
        type=int,
        default=None,
        help="Filter for |n_A - n_B| >= value",
    )
    parser.add_argument(
        "--larger-n",
        choices=["A", "B", "none"],
        default=None,
        help="Filter for which option has larger N (A, B, or none for no filter)",
    )
    parser.add_argument(
        "--equal-n",
        action="store_true",
        help="Filter for edges where both options have equal N",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Show top k edges by effect_A_magnitude",
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Show sample traces",
    )

    args = parser.parse_args()

    # Build filter
    filters = []
    if args.backfire_a:
        filters.append(backfire_A())
    if args.backfire_b:
        filters.append(backfire_B())
    if args.effect_a is not None:
        filters.append(effect_A(args.effect_a))
    if args.effect_b is not None:
        filters.append(effect_B(args.effect_b))
    if args.n_diff is not None:
        filters.append(n_difference(args.n_diff))
    if args.larger_n is not None and args.larger_n != "none":
        filters.append(larger_n(args.larger_n))
    if args.equal_n:
        filters.append(equal_n())

    edge_filter = None
    if filters:
        edge_filter = filters[0]
        for f in filters[1:]:
            edge_filter = edge_filter & f

    # Run pipeline
    pipeline = EdgeFilteringPipeline(
        results_dirs=args.results_dirs,
        edge_filter=edge_filter,
        models=args.models,
        factors=args.factors,
        nudge_types=args.nudge_types,
    )

    summary = pipeline.summary()
    print(f"Filter: {summary['filter_name']}")
    print(f"Total edges: {summary['total_edges']}")
    print(f"Filtered edges: {summary['filtered_edges']}")
    print(f"Filter rate: {summary['filter_rate']:.1%}")
    print(f"Models: {summary['models']}")
    print(f"Factors: {summary['factors']}")

    edges = pipeline.get_filtered_edges()

    if args.top_k and edges:
        print(f"\nTop {args.top_k} edges by effect_A_magnitude:")
        top_edges = top_k(edges, "effect_A_magnitude", args.top_k)
        for i, e in enumerate(top_edges, 1):
            print(
                f"  {i}. {e.model}/{e.factor}/{e.nudge_type} ({e.option_a_n} {e.level_A} vs {e.option_b_n} {e.level_B}): effect_A={e.effect_A:.3f}, "
                f"n_diff={e.n_difference}, f_0_A={e.f_0_A:.3f}, f_A_A={e.f_A_A:.3f}, f_B_A={e.f_B_A:.3f}"
            )

    if args.show_traces and edges:
        print("\nSample traces from first edge:")
        traces = extract_traces_for_edge(edges[0])[:3]
        print(
            f"  Edge: {edges[0].model}/{edges[0].factor}/{edges[0].nudge_type} ({edges[0].option_a_n} {edges[0].level_A} vs {edges[0].option_b_n} {edges[0].level_B})"
        )
        for t in traces:
            print(f"  Condition: {t.condition}, Choice: {t.choice}")
            print(f"  Reasoning: {t.reasoning[:200]}...")
            print()


if __name__ == "__main__":
    main()
