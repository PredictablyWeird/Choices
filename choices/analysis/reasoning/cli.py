#!/usr/bin/env python3
"""
CLI for reasoning trace analysis.

Usage:
    # Extract traces from experiment results
    python -m choices.analysis.reasoning.cli extract results/exp/model/timestamp -o traces.json

    # Code traces for arguments
    python -m choices.analysis.reasoning.cli code traces.json -o coded_traces.json --codebook exchange_rate

    # Analyze coded traces
    python -m choices.analysis.reasoning.cli analyze coded_traces.json

    # Full pipeline
    python -m choices.analysis.reasoning.cli pipeline results/exp/model/timestamp --codebook exchange_rate
"""

import argparse
import asyncio
import json
from pathlib import Path

from .codebook import (
    Codebook,
    create_exchange_rate_codebook,
    create_age_codebook,
    create_occupation_codebook,
)
from .extract_traces import (
    load_traces_from_results_dir,
    save_traces,
    load_traces,
)
from .argument_coder import (
    ArgumentCoder,
    save_coded_traces,
    load_coded_traces,
)
from .analyze_traces import (
    analyze_coded_traces,
    analyze_exchange_rate_traces,
)


BUILTIN_CODEBOOKS = {
    "exchange_rate": create_exchange_rate_codebook,
    "age": create_age_codebook,
    "occupation": create_occupation_codebook,
}


def get_codebook(codebook_arg: str) -> Codebook:
    """Get codebook from name or file path."""
    if codebook_arg in BUILTIN_CODEBOOKS:
        return BUILTIN_CODEBOOKS[codebook_arg]()
    else:
        return Codebook.load(codebook_arg)


def cmd_extract(args):
    """Extract reasoning traces from experiment results."""
    print(f"Extracting traces from: {args.results_dir}")

    traces = load_traces_from_results_dir(args.results_dir)

    print(f"Extracted {len(traces)} traces")

    if args.output:
        save_traces(traces, args.output)
        print(f"Saved to {args.output}")
    else:
        # Print sample
        for trace in traces[:3]:
            print(f"\n--- {trace.edge_key} ({trace.trace_type}) ---")
            print(f"Answer: {trace.parsed_answer}")
            print(f"Reasoning: {trace.reasoning[:200]}...")


def cmd_code(args):
    """Code traces for arguments."""
    print(f"Loading traces from: {args.traces_file}")
    traces = load_traces(args.traces_file)

    print(f"Loaded {len(traces)} traces")

    codebook = get_codebook(args.codebook)
    print(f"Using codebook with {len(codebook.arguments)} arguments:")
    for arg in codebook.arguments:
        print(f"  - {arg.name}")

    coder = ArgumentCoder(
        codebook=codebook,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    coded_traces = asyncio.run(coder.code_traces(traces))

    if args.output:
        save_coded_traces(coded_traces, codebook, coder.model, args.output)
    else:
        print("\nUse --output to save coded traces")


def cmd_analyze(args):
    """Analyze coded traces."""
    print(f"Loading coded traces from: {args.coded_traces_file}")
    coded_traces, codebook, model_used = load_coded_traces(args.coded_traces_file)

    print(f"Loaded {len(coded_traces)} coded traces")
    print(f"Model used: {model_used}")

    if args.exchange_rate:
        analysis = analyze_exchange_rate_traces(
            coded_traces,
            codebook,
            country_field=args.country_field,
            quantity_field=args.quantity_field,
        )
    else:
        analysis = analyze_coded_traces(coded_traces, codebook)

    analysis.print_summary()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
        print(f"\nSaved analysis to {args.output}")


def cmd_pipeline(args):
    """Run full pipeline: extract -> code -> analyze."""
    print(f"Running full pipeline on: {args.results_dir}")

    # Extract
    print("\n=== EXTRACTING TRACES ===")
    traces = load_traces_from_results_dir(args.results_dir)
    print(f"Extracted {len(traces)} traces")

    # Code
    print("\n=== CODING ARGUMENTS ===")
    codebook = get_codebook(args.codebook)
    print(f"Using codebook with {len(codebook.arguments)} arguments")

    coder = ArgumentCoder(
        codebook=codebook,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    coded_traces = asyncio.run(coder.code_traces(traces))

    # Analyze
    print("\n=== ANALYZING TRACES ===")
    if args.exchange_rate:
        analysis = analyze_exchange_rate_traces(coded_traces, codebook)
    else:
        analysis = analyze_coded_traces(coded_traces, codebook)

    analysis.print_summary()

    # Save if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_traces(traces, output_dir / "reasoning_traces.json")
        save_coded_traces(
            coded_traces, codebook, coder.model, str(output_dir / "coded_traces.json")
        )
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)

        print(f"\nSaved all outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning trace analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract reasoning traces from experiment results"
    )
    extract_parser.add_argument(
        "results_dir", help="Directory containing experiment results"
    )
    extract_parser.add_argument("-o", "--output", help="Output file for traces (JSON)")

    # Code command
    code_parser = subparsers.add_parser("code", help="Code traces for arguments")
    code_parser.add_argument("traces_file", help="JSON file with reasoning traces")
    code_parser.add_argument(
        "--codebook",
        default="exchange_rate",
        help="Codebook name (exchange_rate, age, occupation) or path to JSON file",
    )
    code_parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="Model for coding",
    )
    code_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent API requests",
    )
    code_parser.add_argument(
        "-o", "--output", help="Output file for coded traces (JSON)"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze coded traces")
    analyze_parser.add_argument("coded_traces_file", help="JSON file with coded traces")
    analyze_parser.add_argument(
        "--exchange-rate",
        action="store_true",
        help="Use exchange rate specific analysis",
    )
    analyze_parser.add_argument(
        "--country-field",
        default="nationality",
        help="Field name for country (for exchange rate analysis)",
    )
    analyze_parser.add_argument(
        "--quantity-field",
        default="N",
        help="Field name for quantity (for exchange rate analysis)",
    )
    analyze_parser.add_argument(
        "-o", "--output", help="Output file for analysis results (JSON)"
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full pipeline: extract -> code -> analyze"
    )
    pipeline_parser.add_argument(
        "results_dir", help="Directory containing experiment results"
    )
    pipeline_parser.add_argument(
        "--codebook",
        default="exchange_rate",
        help="Codebook name or path",
    )
    pipeline_parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="Model for coding",
    )
    pipeline_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent API requests",
    )
    pipeline_parser.add_argument(
        "--exchange-rate",
        action="store_true",
        help="Use exchange rate specific analysis",
    )
    pipeline_parser.add_argument(
        "-o", "--output", help="Output directory for all results"
    )

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "code":
        cmd_code(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
