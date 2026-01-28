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

    Returns True if:
    - The model name ends with "-reasoning"/"-non-reasoning"
    - The model has reasoning_effort configured in models.yaml
    - The model has extended_thinking configured (Anthropic extended thinking)
    """
    # Check model name suffix
    if model.endswith("-reasoning") or model.endswith("-non-reasoning"):
        return True

    # Check config for reasoning_effort or extended_thinking
    model_config = _get_models_config().get(model, {})

    # Check for reasoning_effort (non-Anthropic reasoning models)
    if "reasoning_effort" in model_config:
        return True

    # Check for extended_thinking (Anthropic extended thinking)
    if "extended_thinking" in model_config:
        return True

    return False


def get_reasoning_effort_from_model(model: str) -> Optional[str]:
    """
    Get the reasoning effort level from model config.

    Returns:
        For reasoning models: effort level ("low", "medium", "high", "off")
        For Anthropic extended thinking models: budget_tokens as string (e.g., "3000")
        None if not configured.

        Note: Returns "off" (not "none") for reasoning models with reasoning disabled,
        to distinguish from chat models with reasoning_mode="none".
    """
    model_config = _get_models_config().get(model, {})

    # Check for Anthropic extended_thinking configuration (budget_tokens)
    if "extended_thinking" in model_config:
        thinking_config = model_config["extended_thinking"]
        if isinstance(thinking_config, dict):
            budget_tokens = thinking_config.get("budget_tokens")
            if budget_tokens is not None:
                return str(budget_tokens)
        return "thinking"  # Fallback if budget_tokens not specified

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

    For reasoning models (including Anthropic thinking models):
        - Returns effort level ("low", "medium", "high", "off") for reasoning models
        - Returns budget_tokens as string (e.g., "10000") for Anthropic thinking models
        - "off" means reasoning is disabled for this reasoning-capable model
    For chat models: shows reasoning_mode from results ("before", "after", "none")
        - "none" means no reasoning instruction was given to this chat model

    Args:
        model: The model identifier
        result_dir: Optional path to result directory to check reasoning_mode

    Returns:
        The reasoning condition string for display
    """
    # Check if this is a reasoning model (includes Anthropic thinking models)
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
          "claude-haiku-4-5-thinking" -> "claude-haiku-4-5"
          "llama-33-70b" -> "llama-33-70b"
    """
    # Check -non-reasoning FIRST (it also ends with -reasoning)
    if model.endswith("-non-reasoning"):
        return model[:-14]  # len("-non-reasoning") == 14
    elif model.endswith("-thinking"):
        return model[:-9]  # len("-thinking") == 9
    elif model.endswith("-reasoning"):
        return model[:-10]  # len("-reasoning") == 10
    return model


def get_model_display_name(model: str, strip_reasoning_suffix: bool = True) -> str:
    """
    Get the display name for a model from models.yaml config.

    Args:
        model: The model identifier
        strip_reasoning_suffix: If True, remove "(reasoning)" and "(Thinking)" suffixes
            from display name

    Returns:
        The display name for the model
    """
    model_config = _get_models_config().get(model, {})
    display_name = model_config.get("display_name", model)

    # Strip reasoning-related suffixes if requested
    if strip_reasoning_suffix:
        if display_name.endswith(" (reasoning)"):
            display_name = display_name[:-12]  # len(" (reasoning)") == 12
        elif display_name.endswith(" (Thinking)"):
            display_name = display_name[:-11]  # len(" (Thinking)") == 11

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
    stats = compute_factor_frequencies_with_counts(
        graph_data, factor_name, target_levels
    )
    return {level: data["freq"] for level, data in stats.items()}


def compute_factor_frequencies_with_counts(
    graph_data: Dict[str, Any],
    factor_name: str,
    target_levels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute win frequencies and sample counts for each factor level.

    Returns:
        Dictionary mapping level -> {"freq": float, "n": int, "wins": int}
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    level_stats = {level: {"wins": 0, "total": 0} for level in target_levels}

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

    # Compute frequencies with counts
    result = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            result[level] = {
                "freq": stats["wins"] / stats["total"],
                "n": stats["total"],
                "wins": stats["wins"],
            }
        else:
            result[level] = {"freq": 0.5, "n": 0, "wins": 0}

    return result


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
    "user_preference": "s",  # square
    "emotional": "P",  # plus (filled)
    "weak_evidence": "v",  # triangle down
    "few_shot_3": "D",  # diamond
}

# Extended marker palette for nudge types not in NUDGE_MARKERS
_EXTRA_MARKERS = ["X", "*", "^", "h", "p", "H", "8", "d", "<", ">", "1", "2", "3", "4"]

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


# Abbreviations for nudge types (for legend display)
NUDGE_TYPE_ABBREVIATIONS = {
    # Evidence-based
    "survey_preference": "Survey",
    "weak_evidence": "Weak Ev.",
    "strong_evidence": "Strong Ev.",
    "expert_recommendation": "Expert",
    # Pressure-based
    "emotional": "Emotional",
    "identity": "Identity",
    "user_preference": "User Pref.",
    "social_norm": "Social",
    # Direct instruction
    "always_save": "Always",
    "moral_imperative": "Moral",
    # Other
    "few_shot": "Few-shot",
    "few_shot_3": "Few-shot",
    "custom": "Custom",
    # Baselines
    "survey_preference_baseline": "Survey (base)",
    "weak_evidence_baseline": "Weak Ev (base)",
    "emotional_baseline": "Emotional (base)",
    "user_preference_baseline": "User Pref (base)",
    # Negation variants
    "survey_preference_negation": "Survey (neg)",
    "weak_evidence_negation": "Weak Ev (neg)",
    "user_preference_negation": "User Pref (neg)",
    "emotional_negation": "Emotional (neg)",
}


def get_nudge_display_name(nudge_type: str) -> str:
    """
    Get abbreviated display name for a nudge type.

    Args:
        nudge_type: The nudge type identifier (e.g., "survey_preference", "few_shot_3")

    Returns:
        Abbreviated display name for use in legends
    """
    # Check for exact match first
    if nudge_type in NUDGE_TYPE_ABBREVIATIONS:
        return NUDGE_TYPE_ABBREVIATIONS[nudge_type]

    # Handle few_shot_N variants
    if nudge_type.startswith("few_shot"):
        return "Few-shot"

    # Fallback: title case with underscores replaced
    return nudge_type.replace("_", " ").title()


def get_category_marker(category: str) -> str:
    """Get marker for a category."""
    return CATEGORY_MARKERS.get(category, "o")
