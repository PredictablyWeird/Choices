"""
Analysis of coded reasoning traces.

Provides statistical analysis of coded traces by various dimensions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .argument_coder import CodedTrace
from .codebook import Codebook


@dataclass
class GroupStats:
    """Statistics for a group of coded traces."""

    name: str
    count: int
    argument_counts: dict[str, int]
    argument_rates: dict[str, float]
    consistency_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "argument_counts": self.argument_counts,
            "argument_rates": self.argument_rates,
            "consistency_rate": self.consistency_rate,
        }


@dataclass
class TraceAnalysis:
    """Complete analysis results."""

    total_traces: int
    codebook: Codebook
    overall_stats: GroupStats
    group_analyses: dict[str, list[GroupStats]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "codebook": self.codebook.to_dict(),
            "overall_stats": self.overall_stats.to_dict(),
            "group_analyses": {
                k: [g.to_dict() for g in v] for k, v in self.group_analyses.items()
            },
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the analysis."""
        print(f"\nTotal traces: {self.total_traces}")
        print("\nOverall argument frequencies:")
        for arg_id, rate in self.overall_stats.argument_rates.items():
            count = self.overall_stats.argument_counts[arg_id]
            print(f"  {arg_id}: {count}/{self.total_traces} ({rate:.1f}%)")

        if self.overall_stats.consistency_rate is not None:
            print(f"\nOverall consistency: {self.overall_stats.consistency_rate:.1f}%")

        for analysis_name, groups in self.group_analyses.items():
            print(f"\n{'=' * 60}")
            print(f"Analysis: {analysis_name}")
            print("=" * 60)

            for group in groups:
                print(f"\n{group.name}: {group.count} traces")
                for arg_id, rate in group.argument_rates.items():
                    count = group.argument_counts[arg_id]
                    print(f"  {arg_id}: {count} ({rate:.1f}%)")
                if group.consistency_rate is not None:
                    print(f"  consistency: {group.consistency_rate:.1f}%")


def _compute_group_stats(
    traces: list[CodedTrace],
    name: str,
    codebook: Codebook,
) -> GroupStats:
    """Compute statistics for a group of traces."""
    if not traces:
        return GroupStats(
            name=name,
            count=0,
            argument_counts={arg.id: 0 for arg in codebook.arguments},
            argument_rates={arg.id: 0.0 for arg in codebook.arguments},
            consistency_rate=None,
        )

    argument_counts = {arg.id: 0 for arg in codebook.arguments}
    consistent_count = 0
    has_consistency = False

    for trace in traces:
        for arg in codebook.arguments:
            if trace.argument_codes.get(arg.id, 0) == 1:
                argument_counts[arg.id] += 1

        if trace.trace.is_consistent is not None:
            has_consistency = True
            if trace.trace.is_consistent:
                consistent_count += 1

    count = len(traces)
    argument_rates = {arg_id: 100 * c / count for arg_id, c in argument_counts.items()}

    consistency_rate = None
    if has_consistency:
        consistency_rate = 100 * consistent_count / count

    return GroupStats(
        name=name,
        count=count,
        argument_counts=argument_counts,
        argument_rates=argument_rates,
        consistency_rate=consistency_rate,
    )


def analyze_by_groups(
    traces: list[CodedTrace],
    codebook: Codebook,
    group_fn: Callable[[CodedTrace], str | None],
    analysis_name: str,
) -> list[GroupStats]:
    """
    Analyze traces grouped by a custom function.

    Args:
        traces: List of coded traces
        codebook: Codebook used for coding
        group_fn: Function that returns group name for each trace (None to exclude)
        analysis_name: Name for this analysis

    Returns:
        List of GroupStats for each group
    """
    groups: dict[str, list[CodedTrace]] = defaultdict(list)

    for trace in traces:
        group_name = group_fn(trace)
        if group_name is not None:
            groups[group_name].append(trace)

    return [
        _compute_group_stats(group_traces, group_name, codebook)
        for group_name, group_traces in sorted(groups.items())
    ]


def analyze_by_argument_pattern(
    traces: list[CodedTrace],
    codebook: Codebook,
) -> list[GroupStats]:
    """
    Analyze traces grouped by which arguments are present.

    Creates groups like "numerical_only", "vulnerability_only", "both", "neither".
    """

    def get_pattern(trace: CodedTrace) -> str:
        present = [
            arg.id
            for arg in codebook.arguments
            if trace.argument_codes.get(arg.id) == 1
        ]
        if len(present) == 0:
            return "neither"
        elif len(present) == len(codebook.arguments):
            return "both" if len(present) == 2 else "all"
        elif len(present) == 1:
            return f"{present[0]}_only"
        else:
            return "+".join(sorted(present))

    return analyze_by_groups(traces, codebook, get_pattern, "argument_pattern")


def analyze_by_decision(
    traces: list[CodedTrace],
    codebook: Codebook,
) -> list[GroupStats]:
    """Analyze traces grouped by the decision made (A or B)."""

    def get_decision(trace: CodedTrace) -> str | None:
        answer = trace.trace.parsed_answer
        return answer if answer in ["A", "B"] else None

    return analyze_by_groups(traces, codebook, get_decision, "decision")


def analyze_by_consistency(
    traces: list[CodedTrace],
    codebook: Codebook,
) -> list[GroupStats]:
    """Analyze traces grouped by consistency status."""

    def get_consistency(trace: CodedTrace) -> str | None:
        if trace.trace.is_consistent is None:
            return None
        return "consistent" if trace.trace.is_consistent else "inconsistent"

    return analyze_by_groups(traces, codebook, get_consistency, "consistency")


def analyze_by_metadata_field(
    traces: list[CodedTrace],
    codebook: Codebook,
    field_name: str,
    option: str = "a",  # "a", "b", or "pair"
) -> list[GroupStats]:
    """
    Analyze traces grouped by a metadata field value.

    Args:
        traces: List of coded traces
        codebook: Codebook used for coding
        field_name: Name of the metadata field to group by
        option: Which option's metadata to use ("a", "b", or "pair" for both)
    """

    def get_field_value(trace: CodedTrace) -> str | None:
        metadata = trace.trace.metadata
        if option == "a":
            return metadata.get("option_a_metadata", {}).get(field_name)
        elif option == "b":
            return metadata.get("option_b_metadata", {}).get(field_name)
        else:  # pair
            val_a = metadata.get("option_a_metadata", {}).get(field_name)
            val_b = metadata.get("option_b_metadata", {}).get(field_name)
            if val_a is not None and val_b is not None:
                return f"{val_a} vs {val_b}"
            return None

    return analyze_by_groups(
        traces, codebook, get_field_value, f"{field_name}_{option}"
    )


def analyze_coded_traces(
    coded_traces: list[CodedTrace],
    codebook: Codebook,
    custom_analyses: dict[str, Callable[[CodedTrace], str | None]] | None = None,
) -> TraceAnalysis:
    """
    Run complete analysis on coded traces.

    Args:
        coded_traces: List of coded traces
        codebook: Codebook used for coding
        custom_analyses: Optional dict of {analysis_name: group_fn} for custom groupings

    Returns:
        TraceAnalysis with all results
    """
    overall_stats = _compute_group_stats(coded_traces, "overall", codebook)

    group_analyses = {
        "by_argument_pattern": analyze_by_argument_pattern(coded_traces, codebook),
        "by_decision": analyze_by_decision(coded_traces, codebook),
        "by_consistency": analyze_by_consistency(coded_traces, codebook),
    }

    if custom_analyses:
        for name, group_fn in custom_analyses.items():
            group_analyses[name] = analyze_by_groups(
                coded_traces, codebook, group_fn, name
            )

    return TraceAnalysis(
        total_traces=len(coded_traces),
        codebook=codebook,
        overall_stats=overall_stats,
        group_analyses=group_analyses,
    )


# Convenience functions for common exchange-rate style analyses


def analyze_exchange_rate_traces(
    coded_traces: list[CodedTrace],
    codebook: Codebook,
    country_field: str = "nationality",
    quantity_field: str = "N",
    developing_countries: list[str] | None = None,
) -> TraceAnalysis:
    """
    Specialized analysis for exchange rate experiments.

    Groups by:
    - Country pair type (same country, developed vs developed, developing involved)
    - Quantity ratio buckets
    - Whether developing country was chosen despite fewer people
    """
    if developing_countries is None:
        developing_countries = ["Nigeria"]

    def get_country_pair_type(trace: CodedTrace) -> str | None:
        metadata = trace.trace.metadata
        country_a = metadata.get("option_a_metadata", {}).get(country_field)
        country_b = metadata.get("option_b_metadata", {}).get(country_field)

        if country_a is None or country_b is None:
            return None

        if country_a == country_b:
            return "Same Country"
        elif country_a in developing_countries or country_b in developing_countries:
            return "Developing Country Involved"
        else:
            return "Developed vs Developed"

    def get_quantity_ratio_bucket(trace: CodedTrace) -> str | None:
        metadata = trace.trace.metadata
        qty_a = metadata.get("option_a_metadata", {}).get(quantity_field)
        qty_b = metadata.get("option_b_metadata", {}).get(quantity_field)
        country_a = metadata.get("option_a_metadata", {}).get(country_field)
        country_b = metadata.get("option_b_metadata", {}).get(country_field)

        if qty_a is None or qty_b is None:
            return None

        # Only for developing country comparisons
        if (
            country_a not in developing_countries
            and country_b not in developing_countries
        ):
            return None

        # Determine which is developing country and compute ratio
        if country_a in developing_countries:
            dev_qty = qty_a
            other_qty = qty_b
        else:
            dev_qty = qty_b
            other_qty = qty_a

        ratio = other_qty / dev_qty if dev_qty > 0 else float("inf")

        if ratio < 2:
            return "< 2x"
        elif ratio < 5:
            return "2-5x"
        elif ratio < 10:
            return "5-10x"
        elif ratio < 50:
            return "10-50x"
        else:
            return ">= 50x"

    custom_analyses = {
        "by_country_pair_type": get_country_pair_type,
        "by_quantity_ratio": get_quantity_ratio_bucket,
    }

    return analyze_coded_traces(coded_traces, codebook, custom_analyses)
