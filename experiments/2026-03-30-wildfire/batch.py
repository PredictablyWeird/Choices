#!/usr/bin/env python3
"""
Batch runner for wildfire evacuation nudging experiments.

Runs multiple experiments based on a YAML config file or command line arguments,
supporting all combinations of models, nudge types, and factors.

Usage:
    # Using config file
    uv run python experiments/2026-03-30-wildfire/batch.py run --config config.yaml

    # Using command line args
    uv run python experiments/2026-03-30-wildfire/batch.py run --models gpt-4o-mini --factors ethnicity_na_asian --nudges emotional

    # Dry run
    uv run python experiments/2026-03-30-wildfire/batch.py run --factors ethnicity_na_asian --nudges emotional --dry-run

    # Generate sample config
    uv run python experiments/2026-03-30-wildfire/batch.py generate-config > my_config.yaml
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import typer
import yaml

from choices.experiments.simple_rates import MAX_REQUESTS
from choices.experiments.nudging.templates import NUDGE_TEMPLATES

# Import from the sibling run module
from run import WILDFIRE_FACTORS, run_wildfire_experiments

app = typer.Typer(help="Batch runner for wildfire evacuation nudging experiments")


@dataclass
class WildfireBatchConfig:
    """Configuration for batch wildfire experiment runs."""

    models: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    factors: list[str] = field(default_factory=lambda: list(WILDFIRE_FACTORS.keys()))
    nudges: list[str] = field(
        default_factory=lambda: [k for k in NUDGE_TEMPLATES.keys() if k != "custom"]
    )
    max_requests: int = MAX_REQUESTS
    requests_per_edge: int = 4
    n_values: str = "paper"
    reasoning: str = "none"
    seed: int = 42
    max_retries: int = 10
    save_dir: str = "results"
    nudge_position: Optional[str] = None
    nudge_brackets: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "WildfireBatchConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        settings = data.get("settings", {})
        return cls(
            models=data.get("models", ["gpt-4o-mini"]),
            factors=data.get("factors", list(WILDFIRE_FACTORS.keys())),
            nudges=data.get(
                "nudges", [k for k in NUDGE_TEMPLATES.keys() if k != "custom"]
            ),
            max_requests=settings.get("max_requests", MAX_REQUESTS),
            requests_per_edge=settings.get("requests_per_edge", 4),
            n_values=settings.get("n_values", "paper"),
            reasoning=settings.get("reasoning", "none"),
            seed=settings.get("seed", 42),
            max_retries=settings.get("max_retries", 10),
            save_dir=settings.get("save_dir", "results"),
            nudge_position=settings.get("nudge_position"),
            nudge_brackets=settings.get("nudge_brackets"),
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        for factor in self.factors:
            if factor not in WILDFIRE_FACTORS:
                errors.append(
                    f"Unknown factor: {factor}. Available: {list(WILDFIRE_FACTORS.keys())}"
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


def generate_experiment_list(config: WildfireBatchConfig) -> list[dict]:
    """Generate list of all experiment combinations."""
    experiments = []
    for model in config.models:
        for factor in config.factors:
            for nudge in config.nudges:
                experiments.append({"model": model, "factor": factor, "nudge": nudge})
    return experiments


def experiment_is_complete(
    save_dir: str, factor_key: str, model: str, nudge: str
) -> bool:
    """Check if an experiment already has results for all conditions."""
    exp_dir = os.path.join(save_dir, f"wildfire_{factor_key}", model, nudge)
    if not os.path.isdir(exp_dir):
        return False

    factor_values = WILDFIRE_FACTORS[factor_key].values
    required_conditions = {"base"} | set(
        v.lower().replace(" ", "_") for v in factor_values
    )
    found_conditions: set[str] = set()

    for entry in os.listdir(exp_dir):
        entry_path = os.path.join(exp_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        parts = entry.split("_", 2)
        if len(parts) < 3:
            continue
        condition = parts[2]
        has_results = any(
            f.startswith("preference_graph_") and f.endswith(".json")
            for f in os.listdir(entry_path)
        )
        if has_results:
            found_conditions.add(condition)

    return required_conditions.issubset(found_conditions)


def print_experiment_plan(config: WildfireBatchConfig, experiments: list[dict]) -> None:
    """Print a summary of the planned experiments."""
    print("\n" + "=" * 80)
    print("WILDFIRE BATCH EXPERIMENT PLAN")
    print("=" * 80)

    print(f"\nModels ({len(config.models)}):")
    for m in config.models:
        print(f"  - {m}")

    print(f"\nFactors ({len(config.factors)}):")
    for f in config.factors:
        print(f"  - {f}: {WILDFIRE_FACTORS[f].values}")

    print(f"\nNudges ({len(config.nudges)}):")
    for n in config.nudges:
        print(f"  - {n}")

    print("\nSettings:")
    print(f"  max_requests: {config.max_requests}")
    print(f"  requests_per_edge: {config.requests_per_edge}")
    print(f"  n_values: {config.n_values}")
    print(f"  reasoning: {config.reasoning}")
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

    runs_per_experiment = 3  # base + 2 groups
    total_runs = len(experiments) * runs_per_experiment
    total_requests = total_runs * config.max_requests

    print(f"Total experiment runs: {total_runs} (base + 2 conditions each)")
    print(f"Estimated total API requests: {total_requests:,}")
    print("=" * 80 + "\n")


async def run_batch_async(
    config: WildfireBatchConfig,
    dry_run: bool = False,
    continue_on_error: bool = True,
    resume: bool = True,
) -> dict[str, dict]:
    """Run all experiments in the batch."""
    experiments = generate_experiment_list(config)
    print_experiment_plan(config, experiments)

    if dry_run:
        print("DRY RUN - No experiments will be executed")
        print("\nWould run the following experiments:")
        for i, exp in enumerate(experiments, 1):
            already_done = resume and experiment_is_complete(
                config.save_dir, exp["factor"], exp["model"], exp["nudge"]
            )
            status = " [SKIP - already complete]" if already_done else ""
            print(f"  {i}. {exp['model']} | {exp['factor']} | {exp['nudge']}{status}")
        return {}

    results = {}
    failed = []
    skipped = 0
    start_time = datetime.now()

    for i, exp in enumerate(experiments, 1):
        exp_key = f"{exp['model']}_{exp['factor']}_{exp['nudge']}"

        if resume and experiment_is_complete(
            config.save_dir, exp["factor"], exp["model"], exp["nudge"]
        ):
            skipped += 1
            print(
                f"\n[SKIP] Experiment {i}/{len(experiments)}: {exp_key} (already complete)"
            )
            continue

        print(f"\n{'#' * 80}")
        print(f"# Experiment {i}/{len(experiments)}: {exp_key}")
        print(f"{'#' * 80}")

        try:
            exp_results = await run_wildfire_experiments(
                factor_key=exp["factor"],
                nudge_type=exp["nudge"],
                model=exp["model"],
                max_requests=config.max_requests,
                requests_per_edge=config.requests_per_edge,
                n_values=config.n_values,
                seed=config.seed,
                reasoning=config.reasoning,
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
    print(f"Skipped (already complete): {skipped}")
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
    sample_config = f"""# Wildfire Evacuation Batch Experiments Configuration
# Run with: uv run python experiments/2026-03-30-wildfire/batch.py run --config this_file.yaml

models:
  - gpt-4o-mini
  # - gpt-5-2-non-reasoning
  # - gpt-5-2-reasoning
  # - deepseek-v3-2-non-reasoning
  # - grok-41-fast-non-reasoning

factors:
  - ethnicity_na_asian

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
  n_values: paper        # Options: binary, small, paper, original (paper = 1-10)
  reasoning: none        # Options: none, before, after
  seed: 42
  max_retries: 10
  save_dir: results
  # nudge_position: after_setup
  # nudge_brackets: parentheses
"""
    print(sample_config)


@app.command()
def run(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to YAML configuration file")
    ] = None,
    models: Annotated[Optional[list[str]], typer.Option(help="Models to test")] = None,
    factors: Annotated[
        Optional[list[str]], typer.Option(help="Factors to test")
    ] = None,
    all_factors: Annotated[
        bool, typer.Option("--all-factors", help="Test all available factors")
    ] = False,
    nudges: Annotated[
        Optional[list[str]], typer.Option(help="Nudge types to test")
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
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    nudge_position: Annotated[
        Optional[str], typer.Option(help="Global nudge position override")
    ] = None,
    nudge_brackets: Annotated[
        Optional[str], typer.Option(help="Global nudge brackets override")
    ] = None,
    save_dir: Annotated[
        Optional[str], typer.Option(help="Base directory for saving results")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print plan without executing")
    ] = False,
    stop_on_error: Annotated[
        bool, typer.Option("--stop-on-error", help="Stop batch on first error")
    ] = False,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Don't skip completed experiments")
    ] = False,
):
    """Run batch wildfire nudging experiments."""
    if config:
        batch_config = WildfireBatchConfig.from_yaml(str(config))
    else:
        batch_config = WildfireBatchConfig()

    if models:
        batch_config.models = models
    if factors:
        batch_config.factors = factors
    if all_factors:
        batch_config.factors = list(WILDFIRE_FACTORS.keys())
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
    if seed is not None:
        batch_config.seed = seed
    if nudge_position is not None:
        batch_config.nudge_position = nudge_position
    if nudge_brackets is not None:
        batch_config.nudge_brackets = nudge_brackets
    if save_dir is not None:
        batch_config.save_dir = save_dir

    errors = batch_config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    asyncio.run(
        run_batch_async(
            config=batch_config,
            dry_run=dry_run,
            continue_on_error=not stop_on_error,
            resume=not no_resume,
        )
    )


@app.command()
def list_options():
    """List all available factors and nudges."""
    print("\nAvailable wildfire factors:")
    for name, var in WILDFIRE_FACTORS.items():
        print(f"  {name}: {var.values}")

    print("\nAvailable nudges:")
    for name, nudge in NUDGE_TEMPLATES.items():
        if nudge.template is None:
            continue
        template_display = (
            nudge.template[:50] + "..." if len(nudge.template) > 50 else nudge.template
        )
        print(f'  {name:20} - "{template_display}"')
    print(f"  {'few_shot_N':20} - Biased examples (e.g., few_shot_3, few_shot_5)")


if __name__ == "__main__":
    app()
