"""
Nudging experiments subpackage.

This package contains tools for running nudging experiments that test
how contextual influences affect LLM preferences.

Modules:
    templates: Shared nudge template definitions
    simple: Simplified nudging experiments (random sampling, fast iteration)
    exchange_rate: Exchange rate nudging experiments (active learning)
    batch: Batch runner for running multiple experiments

Quick usage:
    # Run a single experiment
    python -m choices.experiments.nudging.simple --factor gender --nudge emotional --model gpt-4o-mini

    # Run batch experiments
    python -m choices.experiments.nudging.batch run --models gpt-4o-mini --factors gender --nudges emotional --dry-run
"""

from choices.experiments.nudging.templates import NUDGE_TEMPLATES, get_nudge_names
from choices.experiments.simple_rates import BINARY_FACTORS
from choices.experiments.nudging.simple import run_nudging_experiments

__all__ = [
    "NUDGE_TEMPLATES",
    "get_nudge_names",
    "BINARY_FACTORS",
    "run_nudging_experiments",
]
