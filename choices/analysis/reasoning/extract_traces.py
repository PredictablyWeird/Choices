"""
Extract reasoning traces from experiment results.

Works with any Choices experiment that has with_reasoning enabled.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from choices.results import ExperimentResults


@dataclass
class ReasoningTrace:
    """
    A single reasoning trace from an experiment.

    Attributes:
        edge_key: Key identifying the comparison edge
        response_index: Index of this response within the edge
        trace_type: "original" or "flipped"
        option_a: Dictionary with option A details
        option_b: Dictionary with option B details
        reasoning: The reasoning text from the model
        parsed_answer: The parsed choice ("A" or "B")
        is_consistent: Whether this response is consistent with its flipped counterpart
        metadata: Additional metadata extracted from options
    """

    edge_key: str
    response_index: int
    trace_type: str  # "original" or "flipped"
    option_a: dict[str, Any]
    option_b: dict[str, Any]
    reasoning: str
    parsed_answer: str
    is_consistent: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_key": self.edge_key,
            "response_index": self.response_index,
            "trace_type": self.trace_type,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "reasoning": self.reasoning,
            "parsed_answer": self.parsed_answer,
            "is_consistent": self.is_consistent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningTrace":
        """Create from dictionary."""
        return cls(
            edge_key=data["edge_key"],
            response_index=data["response_index"],
            trace_type=data["trace_type"],
            option_a=data["option_a"],
            option_b=data["option_b"],
            reasoning=data["reasoning"],
            parsed_answer=data["parsed_answer"],
            is_consistent=data.get("is_consistent"),
            metadata=data.get("metadata", {}),
        )


def _get_option_dict(option: Any, results: ExperimentResults) -> dict[str, Any]:
    """
    Convert an option reference to a full option dictionary.

    Handles both direct option dicts and string references (id or label).
    """
    if isinstance(option, dict):
        return option
    if isinstance(option, str):
        # Look up by id or label
        for opt in results.graph.options:
            if str(opt.id) == option or opt.label == option:
                return opt.to_dict()
        raise ValueError(f"Could not find option with id/label: {option}")
    if hasattr(option, "to_dict"):
        return option.to_dict()
    raise ValueError(f"Unknown option type: {type(option)}")


def _extract_metadata_from_option(
    option: dict[str, Any], fields: list[str] | None = None
) -> dict[str, Any]:
    """
    Extract metadata fields from an option dictionary.

    Args:
        option: Option dictionary
        fields: List of fields to extract. If None, extracts all non-id/label fields.
    """
    metadata = {}
    skip_fields = {"id", "label", "text", "description"}

    for key, value in option.items():
        if key in skip_fields:
            continue
        if fields is None or key in fields:
            metadata[key] = value

    return metadata


def extract_reasoning_traces(
    results: ExperimentResults,
    metadata_fields: list[str] | None = None,
) -> list[ReasoningTrace]:
    """
    Extract all reasoning traces from experiment results.

    Args:
        results: ExperimentResults object (loaded from experiment output)
        metadata_fields: Optional list of fields to extract from options as metadata.
                        If None, extracts all non-standard fields.

    Returns:
        List of ReasoningTrace objects
    """
    traces: list[ReasoningTrace] = []

    edges = results.graph.edges
    if not edges:
        return traces

    # First pass: collect all traces
    edge_response_map: dict[tuple[str, int], dict[str, ReasoningTrace]] = {}

    for edge_key, edge_data in edges.items():
        option_a = _get_option_dict(edge_data["option_A"], results)
        option_b = _get_option_dict(edge_data["option_B"], results)
        aux_data = edge_data.get("aux_data", {})

        # Skip edges without reasoning data
        original_reasoning = aux_data.get("original_reasoning", [])
        flipped_reasoning = aux_data.get("flipped_reasoning", [])

        if not original_reasoning and not flipped_reasoning:
            continue

        original_parsed = aux_data.get("original_parsed", [])
        flipped_parsed = aux_data.get("flipped_parsed", [])

        # Extract metadata from options
        metadata_a = _extract_metadata_from_option(option_a, metadata_fields)
        metadata_b = _extract_metadata_from_option(option_b, metadata_fields)

        # Process original responses
        for idx, (reasoning, parsed) in enumerate(
            zip(original_reasoning, original_parsed)
        ):
            trace = ReasoningTrace(
                edge_key=edge_key,
                response_index=idx,
                trace_type="original",
                option_a=option_a,
                option_b=option_b,
                reasoning=reasoning,
                parsed_answer=parsed,
                metadata={
                    "option_a_metadata": metadata_a,
                    "option_b_metadata": metadata_b,
                },
            )
            traces.append(trace)

            # Track for consistency check
            key = (edge_key, idx)
            if key not in edge_response_map:
                edge_response_map[key] = {}
            edge_response_map[key]["original"] = trace

        # Process flipped responses
        # Note: In flipped, the options were presented in reverse order
        for idx, (reasoning, parsed) in enumerate(
            zip(flipped_reasoning, flipped_parsed)
        ):
            trace = ReasoningTrace(
                edge_key=edge_key,
                response_index=idx,
                trace_type="flipped",
                # Swap options to reflect the actual presentation order
                option_a=option_b,
                option_b=option_a,
                reasoning=reasoning,
                parsed_answer=parsed,
                metadata={
                    "option_a_metadata": metadata_b,
                    "option_b_metadata": metadata_a,
                },
            )
            traces.append(trace)

            key = (edge_key, idx)
            if key not in edge_response_map:
                edge_response_map[key] = {}
            edge_response_map[key]["flipped"] = trace

    # Second pass: check consistency
    # Consistent if original chose A and flipped chose B (or vice versa)
    # This means the model chose the same underlying option regardless of position
    for (edge_key, idx), type_map in edge_response_map.items():
        if "original" in type_map and "flipped" in type_map:
            original_answer = type_map["original"].parsed_answer
            flipped_answer = type_map["flipped"].parsed_answer

            is_consistent = (original_answer == "A" and flipped_answer == "B") or (
                original_answer == "B" and flipped_answer == "A"
            )

            type_map["original"].is_consistent = is_consistent
            type_map["flipped"].is_consistent = is_consistent

    return traces


def load_traces_from_results_dir(
    results_dir: str | Path,
    suffix: str | None = None,
    metadata_fields: list[str] | None = None,
) -> list[ReasoningTrace]:
    """
    Load reasoning traces from a results directory.

    Args:
        results_dir: Path to results directory containing preference_graph and utility_model files
        suffix: Optional suffix for result files (e.g., "" for preference_graph.json)
        metadata_fields: Optional list of fields to extract from options as metadata

    Returns:
        List of ReasoningTrace objects
    """
    results = ExperimentResults.load(str(results_dir), suffix)
    return extract_reasoning_traces(results, metadata_fields)


def save_traces(traces: list[ReasoningTrace], output_path: str | Path) -> None:
    """Save reasoning traces to JSON file."""
    import json

    with open(output_path, "w") as f:
        json.dump([t.to_dict() for t in traces], f, indent=2)


def load_traces(input_path: str | Path) -> list[ReasoningTrace]:
    """Load reasoning traces from JSON file."""
    import json

    with open(input_path) as f:
        data = json.load(f)
    return [ReasoningTrace.from_dict(d) for d in data]
