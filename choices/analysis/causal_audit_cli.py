#!/usr/bin/env python3
"""
CLI for causal audit analysis.

Usage:
    # List available categories
    python -m choices.analysis.causal_audit_cli list

    # Run a causal audit experiment
    python -m choices.analysis.causal_audit_cli run countries --max-comparisons 50

    # Run with resampling (for high contamination rate categories)
    python -m choices.analysis.causal_audit_cli resample age --target-clean 100

    # Analyze existing results
    python -m choices.analysis.causal_audit_cli analyze results_countries.json
"""

import argparse
import asyncio

from .causal_audit import (
    CAUSAL_AUDIT_CATEGORIES,
    run_causal_audit,
    run_causal_audit_with_resampling,
    load_causal_audit_results,
    CausalAuditResults,
)


def cmd_list(args):
    """List available causal audit categories."""
    print("\nAvailable causal audit categories:")
    print("=" * 60)

    for name, config in CAUSAL_AUDIT_CATEGORIES.items():
        print(f"\n{name}:")
        print(f"  Description: {config.description}")
        print(f"  Groups: {[g[0] for g in config.groups]}")
        print(f"  Quantities: {config.quantities}")
        print(f"  Forbidden reason: {config.forbidden_reason_name}")


def cmd_run(args):
    """Run a causal audit experiment."""
    asyncio.run(
        run_causal_audit(
            category=args.category,
            samples_per_comparison=args.samples,
            max_comparisons=args.max_comparisons,
            output_file=args.output,
            decision_model=args.decision_model,
            coding_model=args.coding_model,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
        )
    )


def cmd_resample(args):
    """Run causal audit with resampling."""
    asyncio.run(
        run_causal_audit_with_resampling(
            category=args.category,
            target_clean_samples=args.target_clean,
            max_samples_per_comparison=args.max_samples_per,
            max_comparisons=args.max_comparisons,
            output_file=args.output,
            decision_model=args.decision_model,
            coding_model=args.coding_model,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
        )
    )


def cmd_analyze(args):
    """Analyze existing causal audit results."""
    data = load_causal_audit_results(args.results_file)

    results_data = data["results"]
    results = CausalAuditResults(
        config_name=results_data["config"],
        target_group=results_data["target_group"],
        total_responses=results_data["total_responses"],
        valid_responses=results_data["valid_responses"],
        clean_samples=results_data["clean_samples"],
        contaminated_samples=results_data["contaminated_samples"],
        summary=results_data["summary"],
        by_ratio=results_data["by_ratio"],
    )

    results.print_summary()

    # Additional analysis
    if args.verbose:
        print("\n--- Sampling Statistics ---")
        if "sampling_stats" in data:
            ss = data["sampling_stats"]
            print(f"Rounds: {ss.get('rounds', 'N/A')}")
            print(f"Total samples: {ss.get('total_samples', 'N/A')}")
            print(f"Target clean: {ss.get('target_clean', 'N/A')}")
            print(f"Actual clean: {ss.get('actual_clean', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Causal audit analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List categories
  python -m choices.analysis.causal_audit_cli list

  # Run experiment on countries
  python -m choices.analysis.causal_audit_cli run countries -o results_countries.json

  # Run with resampling for age category
  python -m choices.analysis.causal_audit_cli resample age --target-clean 100
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available categories")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run causal audit experiment")
    run_parser.add_argument(
        "category",
        choices=list(CAUSAL_AUDIT_CATEGORIES.keys()),
        help="Category to run",
    )
    run_parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Samples per comparison (default: 10)",
    )
    run_parser.add_argument(
        "--max-comparisons",
        type=int,
        default=None,
        help="Maximum comparisons to run",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        help="Output file for results (JSON)",
    )
    run_parser.add_argument(
        "--decision-model",
        default="openai/gpt-4o-mini",
        help="Model for decisions",
    )
    run_parser.add_argument(
        "--coding-model",
        default="google/gemini-2.0-flash-001",
        help="Model for coding",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for decision model",
    )
    run_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent API requests",
    )

    # Resample command
    resample_parser = subparsers.add_parser(
        "resample", help="Run with resampling for clean samples"
    )
    resample_parser.add_argument(
        "category",
        choices=list(CAUSAL_AUDIT_CATEGORIES.keys()),
        help="Category to run",
    )
    resample_parser.add_argument(
        "--target-clean",
        type=int,
        default=100,
        help="Target number of clean samples",
    )
    resample_parser.add_argument(
        "--max-samples-per",
        type=int,
        default=50,
        help="Maximum samples per comparison",
    )
    resample_parser.add_argument(
        "--max-comparisons",
        type=int,
        default=None,
        help="Maximum comparisons to run",
    )
    resample_parser.add_argument(
        "-o",
        "--output",
        help="Output file for results (JSON)",
    )
    resample_parser.add_argument(
        "--decision-model",
        default="openai/gpt-4o-mini",
        help="Model for decisions",
    )
    resample_parser.add_argument(
        "--coding-model",
        default="google/gemini-2.0-flash-001",
        help="Model for coding",
    )
    resample_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for decision model",
    )
    resample_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent API requests",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument(
        "results_file", help="JSON file with causal audit results"
    )
    analyze_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show additional details",
    )

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "resample":
        cmd_resample(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
