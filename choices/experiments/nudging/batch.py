#!/usr/bin/env python3
"""
Batch runner for simplified nudging experiments.

Runs multiple experiments based on a YAML config file or command line arguments,
supporting all combinations of models, nudge types, and factors.

This uses the simplified nudging experiments (random sampling, no active learning)
from nudging.simple. For active learning experiments, see nudging.exchange_rate.

Usage:
    # Using config file
    uv run python -m choices.experiments.nudging.batch run --config experiments_config.yaml

    # Using command line args (runs all combinations)
    uv run python -m choices.experiments.nudging.batch run --models gpt-4o-mini --factors gender --nudges always_save

    # Dry run to see what would be executed
    uv run python -m choices.experiments.nudging.batch run --config experiments_config.yaml --dry-run

    # Low request mode for testing
    uv run python -m choices.experiments.nudging.batch run --models gpt-4o-mini --factors gender --nudges always_save --max-requests 20

    # Generate sample config
    uv run python -m choices.experiments.nudging.batch generate-config > my_experiments.yaml
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from choices.experiments.simple_rates import BINARY_FACTORS, MAX_REQUESTS
from choices.experiments.nudging.templates import NUDGE_TEMPLATES
from choices.experiments.nudging.simple import run_nudging_experiments

app = typer.Typer(help="Batch runner for simplified nudging experiments")


@dataclass
class BatchConfig:
    """Configuration for batch experiment runs."""

    models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    factors: list[str] = field(default_factory=lambda: list(BINARY_FACTORS.keys()))
    nudges: list[str] = field(
        default_factory=lambda: [k for k in NUDGE_TEMPLATES.keys() if k != "custom"]
    )
    max_requests: int = MAX_REQUESTS
    requests_per_edge: int = 4
    n_values: str = "paper"
    reasoning: str = "none"
    setup: str = "preference"
    seed: int = 42
    max_retries: int = 10
    save_dir: str = "results"
    # Global overrides for nudge formatting (None = use nudge type defaults)
    nudge_position: Optional[str] = None
    nudge_brackets: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "BatchConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        settings = data.get("settings", {})
        return cls(
            models=data.get("models", ["gpt-4o-mini"]),
            factors=data.get("factors", list(BINARY_FACTORS.keys())),
            nudges=data.get(
                "nudges", [k for k in NUDGE_TEMPLATES.keys() if k != "custom"]
            ),
            max_requests=settings.get("max_requests", MAX_REQUESTS),
            requests_per_edge=settings.get("requests_per_edge", 4),
            n_values=settings.get("n_values", "paper"),
            reasoning=settings.get("reasoning", "none"),
            setup=settings.get("setup", "preference"),
            seed=settings.get("seed", 42),
            max_retries=settings.get("max_retries", 10),
            save_dir=settings.get("save_dir", "results"),
            nudge_position=settings.get("nudge_position"),  # None = use nudge defaults
            nudge_brackets=settings.get("nudge_brackets"),  # None = use nudge defaults
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        for factor in self.factors:
            if factor not in BINARY_FACTORS:
                errors.append(
                    f"Unknown factor: {factor}. Available: {list(BINARY_FACTORS.keys())}"
                )

        valid_nudges = set(NUDGE_TEMPLATES.keys()) | {
            "few_shot",
            "few_shot_3",
            "few_shot_5",
        }
        for nudge in self.nudges:
            if nudge.startswith("few_shot_"):
                continue
            if nudge not in valid_nudges:
                errors.append(
                    f"Unknown nudge: {nudge}. Available: {list(NUDGE_TEMPLATES.keys())} + few_shot_N"
                )

        if self.n_values not in ["binary", "small", "paper", "original"]:
            errors.append(
                f"Unknown n_values: {self.n_values}. Available: binary, small, paper, original"
            )

        if self.reasoning not in ["none", "before", "after"]:
            errors.append(
                f"Unknown reasoning: {self.reasoning}. Available: none, before, after"
            )

        valid_positions = [
            "system",
            "system_replace",
            "start",
            "after_setup",
            "after_options",
            "end",
        ]
        if (
            self.nudge_position is not None
            and self.nudge_position not in valid_positions
        ):
            errors.append(
                f"Unknown nudge_position: {self.nudge_position}. Available: {valid_positions}"
            )

        valid_brackets = ["parentheses", "quotes", "none", "italic"]
        if (
            self.nudge_brackets is not None
            and self.nudge_brackets not in valid_brackets
        ):
            errors.append(
                f"Unknown nudge_brackets: {self.nudge_brackets}. Available: {valid_brackets}"
            )

        return errors


def generate_experiment_list(config: BatchConfig) -> list[dict]:
    """Generate list of all experiment combinations."""
    experiments = []
    for model in config.models:
        for factor in config.factors:
            for nudge in config.nudges:
                experiments.append({"model": model, "factor": factor, "nudge": nudge})
    return experiments


def print_experiment_plan(config: BatchConfig, experiments: list[dict]) -> None:
    """Print a summary of the planned experiments."""
    print("\n" + "=" * 80)
    print("BATCH EXPERIMENT PLAN")
    print("=" * 80)

    print(f"\nModels ({len(config.models)}):")
    for m in config.models:
        print(f"  - {m}")

    print(f"\nFactors ({len(config.factors)}):")
    for f in config.factors:
        print(f"  - {f}: {BINARY_FACTORS[f].values}")

    print(f"\nNudges ({len(config.nudges)}):")
    for n in config.nudges:
        print(f"  - {n}")

    print("\nSettings:")
    print(f"  max_requests: {config.max_requests}")
    print(f"  requests_per_edge: {config.requests_per_edge}")
    print(f"  n_values: {config.n_values}")
    print(f"  reasoning: {config.reasoning}")
    print(f"  setup: {config.setup}")
    print(f"  seed: {config.seed}")
    print(f"  save_dir: {config.save_dir}")
    position_str = (
        config.nudge_position if config.nudge_position else "(use nudge defaults)"
    )
    brackets_str = (
        config.nudge_brackets if config.nudge_brackets else "(use nudge defaults)"
    )
    print(f"  nudge_position: {position_str}")
    print(f"  nudge_brackets: {brackets_str}")

    print(f"\nTotal experiments: {len(experiments)}")

    runs_per_experiment = 3  # base + 2 groups for binary factors
    total_runs = len(experiments) * runs_per_experiment
    total_requests = total_runs * config.max_requests

    print(f"Total experiment runs: {total_runs} (base + 2 conditions each)")
    print(f"Estimated total API requests: {total_requests:,}")
    print("=" * 80 + "\n")


async def run_batch_async(
    config: BatchConfig,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> dict[str, dict]:
    """Run all experiments in the batch."""
    experiments = generate_experiment_list(config)
    print_experiment_plan(config, experiments)

    if dry_run:
        print("DRY RUN - No experiments will be executed")
        print("\nWould run the following experiments:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp['model']} | {exp['factor']} | {exp['nudge']}")
        return {}

    results = {}
    failed = []
    start_time = datetime.now()

    for i, exp in enumerate(experiments, 1):
        exp_key = f"{exp['model']}_{exp['factor']}_{exp['nudge']}"
        print(f"\n{'#' * 80}")
        print(f"# Experiment {i}/{len(experiments)}: {exp_key}")
        print(f"{'#' * 80}")

        try:
            exp_results = await run_nudging_experiments(
                factor_name=exp["factor"],
                nudge_type=exp["nudge"],
                model=exp["model"],
                max_requests=config.max_requests,
                requests_per_edge=config.requests_per_edge,
                n_values=config.n_values,
                seed=config.seed,
                reasoning=config.reasoning,
                setup=config.setup,
                max_retries=config.max_retries,
                nudge_position=config.nudge_position,
                nudge_brackets=config.nudge_brackets,
                save_dir=config.save_dir,
            )
            results[exp_key] = exp_results
            print(f"\nExperiment {i}/{len(experiments)} completed successfully")

        except Exception as e:
            print(f"\nERROR in experiment {exp_key}: {e}")
            failed.append((exp_key, str(e)))
            if not continue_on_error:
                raise

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Duration: {duration}")

    if failed:
        print("\nFailed experiments:")
        for exp_key, error in failed:
            print(f"  - {exp_key}: {error}")

    print("=" * 80 + "\n")

    return results


@app.command()
def generate_config():
    """Print a sample configuration file to stdout."""
    sample_config = f"""# Batch Nudging Experiments Configuration
# Run with: uv run python -m choices.experiments.nudging.batch run --config this_file.yaml

models:
  # Non-reasoning models:
  - llama-33-70b
  # - qwen3-235b-a22b-2507
  # Reasoning models:
  # - deepseek-v3-2-non-reasoning
  # - grok-41-fast-non-reasoning
  # - gpt-5-2-non-reasoning
  # - deepseek-v3-2-reasoning
  # - grok-41-fast-reasoning
  # - gpt-5-2-reasoning

factors:
  - gender
  - age_group
  - wealth
  - nationality
  - handedness

nudges:
  # Information-based nudges:
  - survey_preference
  - weak_evidence
  # Pressure-based nudges:
  - emotional
  - user_preference
  # Other nudges:
  - few_shot_3

settings:
  max_requests: 20       # API requests per experiment condition (use {MAX_REQUESTS} for real runs)
  requests_per_edge: 4   # Repeats per comparison
  n_values: paper        # Options: binary, small, paper, original
  reasoning: none        # Options: none, before, after
  setup: preference      # Options: original, decision, preference, or custom text
  seed: 42
  max_retries: 10
  save_dir: results      # Base directory for saving results
  # Nudge formatting (omit or set to null to use each nudge type's default)
  # nudge_position: after_setup  # Options: system, system_replace, start, after_setup, after_options, end
  # nudge_brackets: parentheses  # Options: parentheses, quotes, none, italic
"""
    print(sample_config)


@app.command()
def run(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to YAML configuration file")
    ] = None,
    models: Annotated[
        Optional[list[str]], typer.Option(help="Models to test (can specify multiple)")
    ] = None,
    factors: Annotated[
        Optional[list[str]],
        typer.Option(help="Factors to test (can specify multiple)"),
    ] = None,
    all_factors: Annotated[
        bool, typer.Option("--all-factors", help="Test all available factors")
    ] = False,
    nudges: Annotated[
        Optional[list[str]],
        typer.Option(help="Nudge types to test (can specify multiple)"),
    ] = None,
    all_nudges: Annotated[
        bool, typer.Option("--all-nudges", help="Test all available nudge types")
    ] = False,
    max_requests: Annotated[
        Optional[int], typer.Option(help="Max requests per experiment")
    ] = None,
    requests_per_edge: Annotated[
        Optional[int], typer.Option(help="Requests per edge")
    ] = None,
    n_values: Annotated[
        Optional[str], typer.Option(help="N values: binary, small, paper, original")
    ] = None,
    reasoning: Annotated[
        Optional[str], typer.Option(help="Reasoning mode: none, before, after")
    ] = None,
    setup: Annotated[
        Optional[str], typer.Option(help="Setup text key or custom text")
    ] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    nudge_position: Annotated[
        Optional[str],
        typer.Option(
            help="Global nudge position override (default: use nudge type defaults)"
        ),
    ] = None,
    nudge_brackets: Annotated[
        Optional[str],
        typer.Option(
            help="Global nudge brackets override (default: use nudge type defaults)"
        ),
    ] = None,
    save_dir: Annotated[
        Optional[str],
        typer.Option(help="Base directory for saving results (default: results)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would run without executing")
    ] = False,
    stop_on_error: Annotated[
        bool, typer.Option("--stop-on-error", help="Stop batch on first error")
    ] = False,
):
    """Run batch nudging experiments."""
    # Build configuration
    if config:
        batch_config = BatchConfig.from_yaml(str(config))
    else:
        batch_config = BatchConfig()

    # Apply command line overrides
    if models:
        batch_config.models = models
    if factors:
        batch_config.factors = factors
    if all_factors:
        batch_config.factors = list(BINARY_FACTORS.keys())
    if nudges:
        batch_config.nudges = nudges
    if all_nudges:
        batch_config.nudges = [k for k in NUDGE_TEMPLATES.keys() if k != "custom"] + [
            "few_shot_3"
        ]
    if max_requests is not None:
        batch_config.max_requests = max_requests
    if requests_per_edge is not None:
        batch_config.requests_per_edge = requests_per_edge
    if n_values:
        batch_config.n_values = n_values
    if reasoning:
        batch_config.reasoning = reasoning
    if setup:
        batch_config.setup = setup
    if seed is not None:
        batch_config.seed = seed
    if nudge_position is not None:
        batch_config.nudge_position = nudge_position
    if nudge_brackets is not None:
        batch_config.nudge_brackets = nudge_brackets
    if save_dir is not None:
        batch_config.save_dir = save_dir

    # Validate configuration
    errors = batch_config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Run batch
    asyncio.run(
        run_batch_async(
            config=batch_config,
            dry_run=dry_run,
            continue_on_error=not stop_on_error,
        )
    )


@app.command()
def list_options():
    """List all available models, factors, and nudges."""
    print("\nAvailable factors:")
    for name, var in BINARY_FACTORS.items():
        print(f"  {name}: {var.values}")

    print("\nAvailable nudges:")
    for name, nudge in NUDGE_TEMPLATES.items():
        if nudge.template is None:
            continue
        # Truncate long templates for display
        template_display = (
            nudge.template[:50] + "..." if len(nudge.template) > 50 else nudge.template
        )
        print(f'  {name:20} - "{template_display}"')
        print(f"    {'':20}   position: {nudge.position}, brackets: {nudge.brackets}")
    print(f"  {'few_shot_N':20} - Biased examples (e.g., few_shot_3, few_shot_5)")

    print("\nSee models.yaml for available models. Common ones:")
    print("  gpt-4o-mini")
    print("  deepseek-v3-2-non-reasoning")
    print("  deepseek-v3-2-reasoning")
    print("  grok-41-fast-non-reasoning")
    print("  grok-41-fast-reasoning")
    print("  claude-sonnet-4-5")
    print("  claude-opus-4-5")


if __name__ == "__main__":
    app()
