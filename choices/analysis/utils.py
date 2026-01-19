#!/usr/bin/env python3
"""
Shared utilities for analysis scripts.

This module provides common functions for:
- Loading model configurations
- Determining reasoning conditions
- Computing factor frequencies from preference graphs
- Display name formatting
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# =============================================================================
# Model Configuration
# =============================================================================

# Cache for models config
_models_config: Optional[Dict[str, Any]] = None


def load_models_config() -> Dict[str, Any]:
    """Load the models configuration from models.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _get_models_config() -> Dict[str, Any]:
    """Get cached models config."""
    global _models_config
    if _models_config is None:
        _models_config = load_models_config()
    return _models_config


# =============================================================================
# Reasoning Condition Detection
# =============================================================================


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model is a "reasoning model" (supports extended thinking/reasoning effort).

    Returns True if the model name ends with "-reasoning"/"-non-reasoning" or has
    reasoning_effort explicitly configured in models.yaml.
    """
    # Check model name suffix
    if model.endswith("-reasoning") or model.endswith("-non-reasoning"):
        return True

    # Check config for reasoning_effort - must be explicitly present in config
    model_config = _get_models_config().get(model, {})

    # Only return True if reasoning_effort key exists in the config
    if "reasoning_effort" in model_config:
        return True

    return False


def get_reasoning_effort_from_model(model: str) -> Optional[str]:
    """
    Get the reasoning effort level from model config.

    Returns:
        The effort level ("low", "medium", "high", "off") or None if not configured.
        Note: Returns "off" (not "none") for reasoning models with reasoning disabled,
        to distinguish from chat models with reasoning_mode="none".
    """
    model_config = _get_models_config().get(model, {})

    # Check if reasoning_effort is explicitly in config
    if "reasoning_effort" not in model_config:
        return None

    reasoning_effort = model_config["reasoning_effort"]

    if isinstance(reasoning_effort, dict):
        enabled = reasoning_effort.get("enabled", None)
        effort = reasoning_effort.get("effort", None)

        # If effort is explicitly set, return it (but map "none" to "off")
        if effort is not None:
            return "off" if effort == "none" else effort

        # If enabled is explicitly False, return "off"
        if enabled is False:
            return "off"

        # If enabled is True but no effort specified, assume "low"
        if enabled is True:
            return "low"

    return None


def get_reasoning_mode_from_results(result_dir: Path) -> Optional[str]:
    """
    Get the reasoning mode from the utility model JSON in a result directory.

    Returns:
        The reasoning_mode value (e.g., "before", "none", "after") or None if not found.
    """
    utility_files = list(result_dir.glob("utility_model_*.json"))
    if not utility_files:
        return None

    try:
        with open(utility_files[0], "r") as f:
            utility_data = json.load(f)

        # reasoning_mode is in utility_model_arguments
        args = utility_data.get("utility_model_arguments", {})
        return args.get("reasoning_mode")
    except Exception:
        return None


def get_reasoning_condition(model: str, result_dir: Optional[Path] = None) -> str:
    """
    Determine the reasoning condition for display.

    For reasoning models (extended thinking): shows effort level ("low", "medium", "high", "off")
        - "off" means reasoning is disabled for this reasoning-capable model
    For chat models: shows reasoning_mode from results ("before", "after", "none")
        - "none" means no reasoning instruction was given to this chat model

    Args:
        model: The model identifier
        result_dir: Optional path to result directory to check reasoning_mode

    Returns:
        The reasoning condition string for display
    """
    # Check if this is a reasoning model
    if is_reasoning_model(model):
        effort = get_reasoning_effort_from_model(model)
        if effort is not None:
            return effort
        # Reasoning model but no effort configured - check model name
        if model.endswith("-non-reasoning"):
            return "off"
        return "low"  # Default for reasoning models

    # For chat models, check the reasoning_mode from results
    if result_dir is not None:
        reasoning_mode = get_reasoning_mode_from_results(result_dir)
        if reasoning_mode is not None:
            return reasoning_mode

    return "unknown"


# =============================================================================
# Model Name Utilities
# =============================================================================


def get_base_model_name(model: str) -> str:
    """
    Get the base model name by stripping reasoning-related suffixes.

    E.g., "deepseek-v3-2-reasoning" -> "deepseek-v3-2"
          "deepseek-v3-2-non-reasoning" -> "deepseek-v3-2"
          "llama-33-70b" -> "llama-33-70b"
    """
    # Check -non-reasoning FIRST (it also ends with -reasoning)
    if model.endswith("-non-reasoning"):
        return model[:-14]  # len("-non-reasoning") == 14
    elif model.endswith("-reasoning"):
        return model[:-10]  # len("-reasoning") == 10
    return model


def get_model_display_name(model: str, strip_reasoning_suffix: bool = True) -> str:
    """
    Get the display name for a model from models.yaml config.

    Args:
        model: The model identifier
        strip_reasoning_suffix: If True, remove "(reasoning)" suffix from display name

    Returns:
        The display name for the model
    """
    model_config = _get_models_config().get(model, {})
    display_name = model_config.get("display_name", model)

    # Strip "(reasoning)" suffix if requested
    if strip_reasoning_suffix and display_name.endswith(" (reasoning)"):
        display_name = display_name[:-12]  # len(" (reasoning)") == 12

    return display_name


def get_model_display_with_reasoning(
    model: str, result_dir: Optional[Path] = None
) -> str:
    """
    Get display name with reasoning condition appended.

    E.g., "DeepSeek V3 (off)" or "Llama 3.3 70B (before)"
    """
    display_name = get_model_display_name(model)
    reasoning = get_reasoning_condition(model, result_dir)
    return f"{display_name} ({reasoning})"


# =============================================================================
# Factor Level Mappings
# =============================================================================

# Standard factor level mappings
FACTOR_LEVELS = {
    "gender": ("male", "female"),
    "wealth": ("poor", "rich"),
    "age_group": ("young", "old"),
    "social_status": ("low", "high"),
    "ethnicity": ("white", "black"),
}


def get_factor_levels(category: str) -> Tuple[Optional[str], Optional[str]]:
    """Get the two factor levels for a category."""
    return FACTOR_LEVELS.get(category, (None, None))


# =============================================================================
# Frequency Computation
# =============================================================================


def compute_factor_frequencies(
    graph_data: Dict[str, Any],
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, float]:
    """Compute win frequencies for each factor level."""
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0.0, "total": 0} for level in target_levels}

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Skip intra-group comparisons
            if level_a == level_b:
                continue

            if level_a not in target_levels or level_b not in target_levels:
                continue

            aux_data = edge_data.get("aux_data", {})
            original_parsed = aux_data.get("original_parsed", [])
            flipped_parsed = aux_data.get("flipped_parsed", [])

            # Process original responses
            for resp in original_parsed:
                if resp == "A" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1
                elif resp == "B" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1

            # Process flipped responses (A in flipped = original B)
            for resp in flipped_parsed:
                if resp == "A" and level_b in level_stats:
                    level_stats[level_b]["wins"] += 1
                    level_stats[level_b]["total"] += 1
                    if level_a in level_stats:
                        level_stats[level_a]["total"] += 1
                elif resp == "B" and level_a in level_stats:
                    level_stats[level_a]["wins"] += 1
                    level_stats[level_a]["total"] += 1
                    if level_b in level_stats:
                        level_stats[level_b]["total"] += 1

        except Exception:
            continue

    # Compute frequencies
    frequencies = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            frequencies[level] = stats["wins"] / stats["total"]
        else:
            frequencies[level] = 0.5

    return frequencies


# =============================================================================
# Visualization Palettes
# =============================================================================

# Color palette for models
MODEL_COLORS = {
    "grok-41-fast-non-reasoning": "#E63946",  # red
    "deepseek-v3-2-non-reasoning": "#457B9D",  # blue
    "llama-33-70b": "#2A9D8F",  # teal
    "gpt-4o-mini": "#E9C46A",  # yellow/gold
    "gpt-4o": "#F4A261",  # orange
    "claude-3-5-sonnet-latest": "#9B5DE5",  # purple
    "claude-3-opus": "#00BBF9",  # cyan
    "o1-mini": "#F15BB5",  # pink
}

# Extended color palette for models not in MODEL_COLORS
_EXTRA_COLORS = [
    "#264653",  # dark teal
    "#e76f51",  # burnt sienna
    "#8338ec",  # purple
    "#ff006e",  # pink
    "#3a86ff",  # blue
    "#fb5607",  # orange
    "#ffbe0b",  # yellow
    "#06d6a0",  # mint
    "#118ab2",  # ocean blue
    "#ef476f",  # red-pink
]

# Cache for dynamically assigned model colors
_dynamic_model_colors: Dict[str, str] = {}


def get_model_color(model: str) -> str:
    """Get color for a model, auto-assigning from palette if not predefined."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]

    if model not in _dynamic_model_colors:
        # Assign next available color from palette
        idx = len(_dynamic_model_colors) % len(_EXTRA_COLORS)
        _dynamic_model_colors[model] = _EXTRA_COLORS[idx]

    return _dynamic_model_colors[model]


# Marker palette for categories
CATEGORY_MARKERS = {
    "gender": "o",  # circle
    "wealth": "s",  # square
    "age_group": "^",  # triangle up
    "social_status": "D",  # diamond
    "ethnicity": "v",  # triangle down
}

# Marker palette for nudge types
NUDGE_MARKERS = {
    "survey_preference": "o",  # circle
    "always_save": "s",  # square
    "utilitarian": "^",  # triangle up
    "deontological": "D",  # diamond
    "weak_evidence": "v",  # triangle down
    "emotional": "P",  # plus (filled)
    "expert_opinion": "X",  # x (filled)
    "social_proof": "*",  # star
}

# Extended marker palette for nudge types not in NUDGE_MARKERS
_EXTRA_MARKERS = ["h", "p", "H", "8", "d", "<", ">", "1", "2", "3", "4"]

# Cache for dynamically assigned nudge markers
_dynamic_nudge_markers: Dict[str, str] = {}


def get_nudge_marker(nudge_type: str) -> str:
    """Get marker for a nudge type, auto-assigning from palette if not predefined."""
    if nudge_type in NUDGE_MARKERS:
        return NUDGE_MARKERS[nudge_type]

    if nudge_type not in _dynamic_nudge_markers:
        # Assign next available marker from palette
        idx = len(_dynamic_nudge_markers) % len(_EXTRA_MARKERS)
        _dynamic_nudge_markers[nudge_type] = _EXTRA_MARKERS[idx]

    return _dynamic_nudge_markers[nudge_type]


def get_category_marker(category: str) -> str:
    """Get marker for a category."""
    return CATEGORY_MARKERS.get(category, "o")
