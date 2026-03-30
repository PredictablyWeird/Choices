"""
Wildfire evacuation prioritization experiment.

Tests whether LLMs exhibit demographic bias in emergency evacuation decisions
using a simplified zone-based prompt. Supports the same nudging conditions
as the main experiments.

Quick usage:
    # Single experiment
    uv run python experiments/2026-03-30-wildfire/run.py --factor ethnicity_na_asian --nudge emotional

    # Batch experiments
    uv run python experiments/2026-03-30-wildfire/batch.py run --factors ethnicity_na_asian --nudges emotional --models gpt-4o-mini --dry-run
"""
