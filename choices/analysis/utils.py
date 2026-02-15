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
# Output Configuration
# =============================================================================

# Default directory for saving plot outputs
PLOTS_OUTPUT_DIR = "plots"


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


# Color palettes for factors, nudges, and reasoning conditions
FACTOR_COLORS = {
    "gender": "#E63946",  # red
    "wealth": "#457B9D",  # blue
    "age_group": "#2A9D8F",  # teal
    "social_status": "#E9C46A",  # yellow/gold
    "ethnicity": "#9B5DE5",  # purple
}

NUDGE_COLORS = {
    "survey_preference": "#1f77b4",  # blue
    "user_preference": "#ff7f0e",  # orange
    "emotional": "#2ca02c",  # green
    "weak_evidence": "#d62728",  # red
    "few_shot_3": "#9467bd",  # purple
    "always_save": "#8c564b",  # brown
}

REASONING_COLORS = {
    "none": "#1f77b4",  # blue
    "before": "#ff7f0e",  # orange
    "after": "#2ca02c",  # green
    "off": "#d62728",  # red
    "low": "#9467bd",  # purple
    "medium": "#8c564b",  # brown
    "high": "#e377c2",  # pink
}

# Caches for dynamically assigned colors
_dynamic_factor_colors: Dict[str, str] = {}
_dynamic_nudge_colors: Dict[str, str] = {}
_dynamic_reasoning_colors: Dict[str, str] = {}


def get_factor_color(factor: str) -> str:
    """Get color for a factor, auto-assigning from palette if not predefined."""
    if factor in FACTOR_COLORS:
        return FACTOR_COLORS[factor]

    if factor not in _dynamic_factor_colors:
        idx = len(_dynamic_factor_colors) % len(_EXTRA_COLORS)
        _dynamic_factor_colors[factor] = _EXTRA_COLORS[idx]

    return _dynamic_factor_colors[factor]


def get_nudge_color(nudge_type: str) -> str:
    """Get color for a nudge type, auto-assigning from palette if not predefined."""
    # Strip _baseline or _negation suffixes for color matching
    base_nudge = nudge_type.replace("_baseline", "").replace("_negation", "")

    if base_nudge in NUDGE_COLORS:
        return NUDGE_COLORS[base_nudge]

    if nudge_type not in _dynamic_nudge_colors:
        idx = len(_dynamic_nudge_colors) % len(_EXTRA_COLORS)
        _dynamic_nudge_colors[nudge_type] = _EXTRA_COLORS[idx]

    return _dynamic_nudge_colors[nudge_type]


def get_reasoning_color(reasoning: str) -> str:
    """Get color for a reasoning condition, auto-assigning from palette if not predefined."""
    if reasoning in REASONING_COLORS:
        return REASONING_COLORS[reasoning]

    if reasoning not in _dynamic_reasoning_colors:
        idx = len(_dynamic_reasoning_colors) % len(_EXTRA_COLORS)
        _dynamic_reasoning_colors[reasoning] = _EXTRA_COLORS[idx]

    return _dynamic_reasoning_colors[reasoning]


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
    "virtue_appeal": "Virtue",
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


# =============================================================================
# Result Loading
# =============================================================================


def load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load results from an experiment directory.

    Loads preference graph and optional utility model data.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with keys:
            - "graph": Preference graph data
            - "utilities": Utility model data (or None)
            - "results_dir": The original results directory path
    """
    results_path = Path(results_dir)

    # Find preference graph file
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        raise FileNotFoundError(f"No preference_graph file found in {results_dir}")

    with open(graph_files[0], "r") as f:
        graph_data = json.load(f)

    # Also load utility model if available
    utility_files = list(results_path.glob("utility_model_*.json"))
    utility_data = None
    if utility_files:
        with open(utility_files[0], "r") as f:
            utility_data = json.load(f)

    return {
        "graph": graph_data,
        "utilities": utility_data,
        "results_dir": results_dir,
    }


def find_latest_result_dir(
    experiment_name: str,
    model: str,
    base_dir: str = "results",
) -> Optional[Path]:
    """Find the latest result directory for an experiment."""
    exp_dir = Path(base_dir) / experiment_name / model
    if not exp_dir.exists():
        return None

    # Get all timestamp directories
    dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None

    # Sort by name (timestamp format ensures chronological order)
    return sorted(dirs)[-1]


# =============================================================================
# Experiment Balance Checking
# =============================================================================


def check_balance(
    options: List[Dict],
    edges: Dict[str, Dict],
    variables: List[Dict],
) -> Dict[str, Any]:
    """
    Check if an experiment is balanced.

    For a balanced design:
    - Each option should appear in approximately equal numbers of comparisons
    - For each N value, it should be paired with each factor level equally often
      (e.g., N=1 appears with male and female the same number of times)

    Args:
        options: List of option dictionaries
        edges: Dictionary of edges
        variables: List of variable definitions

    Returns:
        Dictionary with balance statistics:
            - "option_counts": dict of option_id -> presentation count
            - "n_factor_balance": dict of N pairs to their balance info
            - "is_balanced": bool indicating if design is balanced
    """
    from collections import defaultdict

    # Build option lookup
    options_by_id = {opt["id"]: opt for opt in options}

    # Count appearances per option
    option_counts = defaultdict(int)
    for edge_key in edges.keys():
        try:
            ids = eval(edge_key)
            if isinstance(ids, tuple):
                option_counts[ids[0]] += 1
                option_counts[ids[1]] += 1
        except Exception:
            parts = edge_key.strip("()").split(",")
            if len(parts) == 2:
                option_counts[int(parts[0].strip())] += 1
                option_counts[int(parts[1].strip())] += 1

    # Get N variable and factor variable
    n_var = None
    factor_var = None
    for var in variables:
        if var["name"] == "N":
            n_var = var
        else:
            factor_var = var

    if not factor_var or not n_var:
        return {
            "option_counts": dict(option_counts),
            "n_factor_balance": {},
            "is_balanced": True,
        }

    factor_name = factor_var["name"]
    factor_levels = factor_var["values"]
    n_values = sorted(n_var["values"])

    # For each pair of N values, count how often each N is paired with each factor level
    n_factor_balance = {}

    for i, n1 in enumerate(n_values):
        for n2 in n_values[i + 1 :]:
            # For this N pair, count factor level distribution
            factor_with_lower_n = defaultdict(int)
            factor_with_higher_n = defaultdict(int)

            for edge_key in edges.keys():
                try:
                    ids = eval(edge_key)
                    opt_a = options_by_id.get(ids[0])
                    opt_b = options_by_id.get(ids[1])

                    if not opt_a or not opt_b:
                        continue

                    n_a = opt_a.get("N")
                    n_b = opt_b.get("N")
                    factor_a = opt_a.get(factor_name)
                    factor_b = opt_b.get(factor_name)

                    # Check if this edge involves the N pair (n1, n2)
                    if n_a == n1 and n_b == n2:
                        factor_with_lower_n[factor_a] += 1
                        factor_with_higher_n[factor_b] += 1
                    elif n_a == n2 and n_b == n1:
                        factor_with_higher_n[factor_a] += 1
                        factor_with_lower_n[factor_b] += 1

                except Exception:
                    pass

            # Skip N pairs with no samples
            total_samples = sum(factor_with_lower_n.values()) + sum(
                factor_with_higher_n.values()
            )
            if total_samples == 0:
                continue

            # Check if balanced: each factor level should have equal counts
            lower_counts = [factor_with_lower_n[f] for f in factor_levels]
            higher_counts = [factor_with_higher_n[f] for f in factor_levels]

            is_balanced = (
                len(set(lower_counts)) <= 1
                and len(set(higher_counts)) <= 1
                and all(c > 0 for c in lower_counts)
                and all(c > 0 for c in higher_counts)
            )

            n_factor_balance[(n1, n2)] = {
                "factor_with_lower_n": dict(factor_with_lower_n),
                "factor_with_higher_n": dict(factor_with_higher_n),
                "balanced": is_balanced,
            }

    return {
        "option_counts": dict(option_counts),
        "n_factor_balance": n_factor_balance,
        "is_balanced": all(v["balanced"] for v in n_factor_balance.values())
        if n_factor_balance
        else True,
    }


# =============================================================================
# Statistical Tests
# =============================================================================


def binomial_test_vs_half(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if a proportion differs significantly from 0.5 using exact binomial test.

    Args:
        successes: Number of successes (wins)
        n: Total number of trials
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with p_value, ci_low, ci_high, and is_significant
    """
    from scipy import stats

    if n <= 0:
        return {
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 1.0,
            "is_significant": False,
        }

    result = stats.binomtest(successes, n, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=1 - alpha)

    return {
        "p_value": result.pvalue,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "is_significant": result.pvalue < alpha,
    }


def two_proportion_z_test(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if two proportions differ significantly using two-proportion z-test.

    This is the standard test for comparing proportions in academic publications.

    Args:
        p1: First proportion (e.g., base condition)
        n1: Sample size for first proportion
        p2: Second proportion (e.g., nudged condition)
        n2: Sample size for second proportion
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with z_stat, p_value, diff, ci_low, ci_high, and is_significant
    """
    import numpy as np
    from scipy import stats

    if n1 <= 0 or n2 <= 0:
        return {
            "z_stat": 0.0,
            "p_value": 1.0,
            "diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "is_significant": False,
        }

    # Pooled proportion under H0: p1 = p2
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

    # Standard error under H0
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    # Z statistic
    if se_pooled > 0:
        z_stat = (p2 - p1) / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0.0
        p_value = 1.0

    # Confidence interval for the difference (using unpooled SE)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci_low = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "is_significant": p_value < alpha,
    }
