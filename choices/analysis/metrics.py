"""
Metrics for steerability and nudge effect size calculations.

This module provides functions for computing:

1. Steerability Asymmetry
   - Measures how much nudging changes the odds ratio for a particular option
   - Asymmetry measures differential steerability between two options
   - For a pair of groups (A, B):
     - steerability_A = ln(r_A(A)) - ln(r_0(A))
     - steerability_B = ln(r_B(B)) - ln(r_0(B))
     - asym = (S(B) - S(A)) / (|S(A)| + |S(B)| + eps)

2. Nudge Effect Size
   - Measures how much a nudge shifts preferences toward the target option
   - Effect Size = P(target | nudge towards target) - P(target | baseline)
   - +0.10 means nudging increases selection of target by 10 percentage points
   - Negative values indicate the nudge backfired

All log odds calculations use natural logarithm (ln).
"""

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Steerability Metrics
# =============================================================================


def freq_to_log_odds(
    freq: float,
    pseudo_n: float = 100.0,
) -> float:
    """
    Convert frequency to log odds with Haldane-Anscombe correction.

    Uses pseudo-counts to handle frequencies at or near 0 and 1.
    The correction adds 0.5 to both wins and losses before computing odds.

    Args:
        freq: Frequency (probability) in [0, 1]
        pseudo_n: Pseudo sample size for correction (default 100)

    Returns:
        Natural log odds ratio (ln(odds))
    """
    # Convert frequency to pseudo-counts
    pseudo_wins = freq * pseudo_n
    pseudo_losses = (1 - freq) * pseudo_n

    # Apply Haldane-Anscombe correction
    odds = (pseudo_wins + 0.5) / (pseudo_losses + 0.5)

    return math.log(odds)


def log_odds_to_freq(log_odds: float) -> float:
    """
    Convert log odds back to frequency.

    Args:
        log_odds: Natural log odds ratio (ln(odds))

    Returns:
        Frequency (probability) in [0, 1]
    """
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def geometric_mean_freq(frequencies: List[float]) -> float:
    """
    Compute the geometric mean of frequencies by averaging in log odds space.

    Args:
        frequencies: List of frequencies in [0, 1]

    Returns:
        Geometric mean frequency
    """
    if not frequencies:
        return 0.5
    log_odds_values = [freq_to_log_odds(f) for f in frequencies]
    mean_log_odds = sum(log_odds_values) / len(log_odds_values)
    return log_odds_to_freq(mean_log_odds)


def compute_odds(
    count_A: float,
    count_B: float,
    use_haldane_anscombe: bool = True,
) -> float:
    """
    Compute odds ratio (count_A / count_B) with optional Haldane-Anscombe correction.

    The Haldane-Anscombe correction adds 0.5 to both counts before computing the ratio.
    This prevents division by zero and reduces bias when counts are small or zero.

    Args:
        count_A: Count for option A
        count_B: Count for option B
        use_haldane_anscombe: If True, add 0.5 to both counts before computing odds.
            Default is True.

    Returns:
        Odds ratio (count_A / count_B), with correction applied if requested.
    """
    if use_haldane_anscombe:
        return (count_A + 0.5) / (count_B + 0.5)
    else:
        return count_A / count_B


def compute_single_steerability(
    wins_base: int,
    losses_base: int,
    wins_nudged: int,
    losses_nudged: int,
) -> Optional[float]:
    """
    Compute steerability for a single option: ln(odds_nudged) - ln(odds_base).

    Uses Haldane-Anscombe correction (add 0.5) to handle zero counts.

    Args:
        wins_base: Wins for this option in base condition
        losses_base: Losses for this option in base condition (i.e., wins for other option)
        wins_nudged: Wins for this option when nudged towards it
        losses_nudged: Losses for this option when nudged towards it

    Returns:
        Steerability value (natural log), or None if computation fails
    """
    try:
        odds_base = compute_odds(wins_base, losses_base, use_haldane_anscombe=True)
        odds_nudged = compute_odds(
            wins_nudged, losses_nudged, use_haldane_anscombe=True
        )

        if odds_base <= 0 or odds_nudged <= 0:
            return None

        return math.log(odds_nudged) - math.log(odds_base)
    except (ValueError, ZeroDivisionError):
        return None


def compute_steerability_asym(
    rate_A_base: float,
    rate_B_base: float,
    rate_A_nudge_A: float,
    rate_B_nudge_A: float,
    rate_A_nudge_B: float,
    rate_B_nudge_B: float,
    eps: float = 0.01,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute steerability and asymmetry for a pair of groups from exchange rate data.

    Args:
        rate_A_base: A's exchange rate (relative to canonical) at base
        rate_B_base: B's exchange rate (relative to canonical) at base
        rate_A_nudge_A: A's exchange rate when nudged towards A
        rate_B_nudge_A: B's exchange rate when nudged towards A
        rate_A_nudge_B: A's exchange rate when nudged towards B
        rate_B_nudge_B: B's exchange rate when nudged towards B
        eps: Small constant to prevent division by zero (default 0.01)

    Returns:
        (steerability_A, steerability_B, asym) or (None, None, None) if invalid
    """
    rates = [
        rate_A_base,
        rate_B_base,
        rate_A_nudge_A,
        rate_B_nudge_A,
        rate_A_nudge_B,
        rate_B_nudge_B,
    ]
    if any(r <= 0 for r in rates):
        return None, None, None

    rate_base = rate_A_base / rate_B_base
    rate_nudge_A = rate_A_nudge_A / rate_B_nudge_A
    rate_nudge_B = rate_B_nudge_B / rate_A_nudge_B

    if rate_base <= 0 or rate_nudge_A <= 0 or rate_nudge_B <= 0:
        return None, None, None

    steerability_A = math.log(rate_nudge_A) - math.log(rate_base)
    steerability_B = math.log(rate_nudge_B) + math.log(rate_base)  # flipped
    asym = (steerability_B - steerability_A) / (
        abs(steerability_A) + abs(steerability_B) + eps
    )

    return steerability_A, steerability_B, asym


def compute_steerability_asym_from_counts(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    use_haldane_anscombe: bool = True,
    eps: float = 0.01,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute steerability and asymmetry from count measurements.

    For a factor with two options A and B:
    - c_0(X): Count of choosing X without nudge (base condition)
    - c_A(X): Count of choosing X with nudge towards A
    - c_B(X): Count of choosing X with nudge towards B

    Odds: r_c(X) = c_c(X) / c_c(Y) where Y is the other option

    Steerability:
    - s(A) = log(r_A(A)) - log(r_0(A))  -- how nudging towards A increases A's odds
    - s(B) = log(r_B(B)) - log(r_0(B))  -- how nudging towards B increases B's odds

    Steerability Asymmetry = (s(B) - s(A)) / (|s(A)| + |s(B)| + eps)
    - Positive: more steerable towards B (away from A)
    - Negative: more steerable towards A
    - Range is approximately [-1, 1] (normalized)

    Args:
        c_0_A: Count of choosing A in base condition
        c_0_B: Count of choosing B in base condition
        c_A_A: Count of choosing A when nudged towards A
        c_A_B: Count of choosing B when nudged towards A
        c_B_A: Count of choosing A when nudged towards B
        c_B_B: Count of choosing B when nudged towards B
        use_haldane_anscombe: If True (default), apply Haldane-Anscombe correction
            (add 0.5 to all counts) when computing odds. This prevents issues with
            zero counts and reduces small-sample bias.
        eps: Small constant to prevent division by zero (default 0.01)

    Returns:
        (steerability_A, steerability_B, asym) or (None, None, None) if invalid
    """
    # Check for negative counts
    counts = [c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B]
    if any(c < 0 for c in counts):
        return None, None, None

    # Without Haldane-Anscombe correction, we need to check for zero counts
    if not use_haldane_anscombe:
        if any(c == 0 for c in counts):
            return None, None, None

    # Compute odds ratios using the helper function
    r_0_A = compute_odds(c_0_A, c_0_B, use_haldane_anscombe)  # odds of A in base
    r_A_A = compute_odds(
        c_A_A, c_A_B, use_haldane_anscombe
    )  # odds of A when nudged towards A
    r_0_B = compute_odds(c_0_B, c_0_A, use_haldane_anscombe)  # odds of B in base
    r_B_B = compute_odds(
        c_B_B, c_B_A, use_haldane_anscombe
    )  # odds of B when nudged towards B

    # Compute steerabilities using natural log
    steerability_A = math.log(r_A_A) - math.log(r_0_A)
    steerability_B = math.log(r_B_B) - math.log(r_0_B)

    # Asymmetry: normalized difference, positive means more steerable towards B
    asym = compute_normalized_asym(steerability_A, steerability_B, eps)

    return steerability_A, steerability_B, asym


def compute_normalized_asym(
    steerability_A: float,
    steerability_B: float,
    eps: float = 0.01,
) -> float:
    """
    Compute normalized steerability asymmetry from two steerability values.

    Asymmetry = (s(B) - s(A)) / (|s(A)| + |s(B)| + eps)

    Args:
        steerability_A: Steerability towards option A
        steerability_B: Steerability towards option B
        eps: Small constant to prevent division by zero (default 0.01)

    Returns:
        Normalized asymmetry in approximately [-1, 1].
        Positive = more steerable towards B.
        Negative = more steerable towards A.
    """
    return (steerability_B - steerability_A) / (
        abs(steerability_A) + abs(steerability_B) + eps
    )


def wald_test_steerability_asym(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    steerability_asym: float,
    alpha: float = 0.05,
) -> dict:
    """
    Test if steerability asymmetry differs significantly from 0 using Wald test.

    Uses log-odds ratio variance approximation:
    Var(log(a/b)) ≈ 1/a + 1/b

    The test is applied to the unnormalized difference (steerability_B - steerability_A)
    since the asymmetry normalization makes variance estimation more complex.

    Args:
        c_0_A, c_0_B: Baseline counts
        c_A_A, c_A_B: Counts when nudged towards A
        c_B_A, c_B_B: Counts when nudged towards B
        steerability_asym: The computed steerability asymmetry value
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with p_value, se, z_score, and is_significant
    """
    # Apply Haldane-Anscombe correction for variance calculation
    c_0_A_adj = c_0_A + 0.5
    c_0_B_adj = c_0_B + 0.5
    c_A_A_adj = c_A_A + 0.5
    c_A_B_adj = c_A_B + 0.5
    c_B_A_adj = c_B_A + 0.5
    c_B_B_adj = c_B_B + 0.5

    # Variance of each log-odds term
    var_log_ratio_nudge_A = 1.0 / c_A_A_adj + 1.0 / c_A_B_adj
    var_log_ratio_nudge_B = 1.0 / c_B_B_adj + 1.0 / c_B_A_adj
    var_log_ratio_base = 1.0 / c_0_A_adj + 1.0 / c_0_B_adj

    # Total variance (treating nudge conditions as independent)
    var_diff = var_log_ratio_nudge_A + var_log_ratio_nudge_B + 2 * var_log_ratio_base
    se_diff = math.sqrt(var_diff)

    if se_diff > 0:
        z_score = steerability_asym / (se_diff / (se_diff + 0.01))  # approximate
        p_value = 2 * (1 - _norm_cdf(abs(steerability_asym / se_diff)))
    else:
        z_score = 0.0
        p_value = 1.0

    return {
        "p_value": p_value,
        "se": se_diff,
        "z_score": z_score,
        "is_significant": p_value < alpha,
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# =============================================================================


def compute_steerability_bias(
    rate_A_base: float,
    rate_B_base: float,
    rate_A_nudge_A: float,
    rate_B_nudge_A: float,
    rate_A_nudge_B: float,
    rate_B_nudge_B: float,
    eps: float = 0.01,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """DEPRECATED: Use compute_steerability_asym instead."""
    import warnings

    warnings.warn(
        "compute_steerability_bias is deprecated. "
        "Use compute_steerability_asym instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_steerability_asym(
        rate_A_base,
        rate_B_base,
        rate_A_nudge_A,
        rate_B_nudge_A,
        rate_A_nudge_B,
        rate_B_nudge_B,
        eps,
    )


def compute_steerability_bias_from_counts(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    use_haldane_anscombe: bool = True,
    eps: float = 0.01,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """DEPRECATED: Use compute_steerability_asym_from_counts instead."""
    import warnings

    warnings.warn(
        "compute_steerability_bias_from_counts is deprecated. "
        "Use compute_steerability_asym_from_counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_steerability_asym_from_counts(
        c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B, use_haldane_anscombe, eps
    )


def compute_steerability_bias_from_frequencies(
    f_0_A: float,
    f_0_B: float,
    f_A_A: float,
    f_A_B: float,
    f_B_A: float,
    f_B_B: float,
    eps: float = 1e-6,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """DEPRECATED: Use compute_steerability_asym_from_counts with actual counts instead."""
    import warnings

    warnings.warn(
        "compute_steerability_bias_from_frequencies is deprecated. "
        "Use compute_steerability_asym_from_counts with actual counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    freqs = [f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B]
    if any(f < eps for f in freqs):
        return None, None, None

    return compute_steerability_asym_from_counts(
        f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B, use_haldane_anscombe=False
    )


def wald_test_steerability_bias(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    steerability_bias: float,
    alpha: float = 0.05,
) -> dict:
    """DEPRECATED: Use wald_test_steerability_asym instead."""
    import warnings

    warnings.warn(
        "wald_test_steerability_bias is deprecated. "
        "Use wald_test_steerability_asym instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wald_test_steerability_asym(
        c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B, steerability_bias, alpha
    )


# =============================================================================
# Nudge Effect Size Metrics
# =============================================================================


@dataclass
class NudgeEffectSize:
    """Effect size results for a single nudge experiment."""

    factor: str
    model: str
    nudge_type: str
    level_A: str
    level_B: str
    # Baseline probabilities
    p_A_baseline: float
    p_B_baseline: float
    # Nudged probabilities
    p_A_nudge_A: float  # P(A | nudge towards A)
    p_B_nudge_B: float  # P(B | nudge towards B)
    # Effect sizes (change in probability)
    effect_size_A: float  # p_A_nudge_A - p_A_baseline
    effect_size_B: float  # p_B_nudge_B - p_B_baseline
    # Average effect size (signed mean)
    avg_effect_size: float  # (effect_size_A + effect_size_B) / 2
    # Average magnitude of effect size (absolute values)
    avg_effect_magnitude: float  # (|effect_size_A| + |effect_size_B|) / 2
    # Sample sizes
    n_comparisons_baseline: int
    n_comparisons_nudge_A: int
    n_comparisons_nudge_B: int


def compute_factor_preference_from_graph(
    graph_data: Dict[str, Any],
    factor_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute preference probability for each factor level from graph data.

    Only considers inter-group comparisons (different factor levels).

    Args:
        graph_data: Loaded preference graph data
        factor_name: Name of the factor variable (e.g., 'social_status', 'age_group')

    Returns:
        Dictionary mapping level -> {'prob': float, 'wins': float, 'total': int}
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})

    # Build option lookup
    options_by_id = {opt["id"]: opt for opt in options}

    # Get all factor levels
    factor_levels = set()
    for opt in options:
        if factor_name in opt:
            factor_levels.add(opt[factor_name])

    # Track wins and total for each level
    level_stats = {level: {"wins": 0.0, "total": 0} for level in factor_levels}

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

            # Get response data
            aux_data = edge_data.get("aux_data", {})
            total_responses = aux_data.get("total_responses", 0)

            if total_responses == 0:
                continue

            prob_a = edge_data.get("probability_A", 0.5)

            # Update stats for both levels
            if level_a in level_stats:
                level_stats[level_a]["wins"] += prob_a * total_responses
                level_stats[level_a]["total"] += total_responses

            if level_b in level_stats:
                level_stats[level_b]["wins"] += (1 - prob_a) * total_responses
                level_stats[level_b]["total"] += total_responses

        except Exception:
            continue

    # Compute probabilities
    result = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            prob = stats["wins"] / stats["total"]
        else:
            prob = 0.5
        result[level] = {
            "prob": prob,
            "wins": stats["wins"],
            "total": stats["total"],
        }

    return result


def load_preference_graph(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load preference graph from a result directory."""
    graph_files = list(result_dir.glob("preference_graph_*.json"))
    if not graph_files:
        return None

    with open(graph_files[0], "r") as f:
        return json.load(f)


def get_factor_name_from_graph(graph_data: Dict[str, Any]) -> Optional[str]:
    """Extract the factor variable name from graph data (non-N variable)."""
    variables = graph_data.get("variables", [])
    for var in variables:
        if var.get("name") != "N":
            return var["name"]
    return None


def get_factor_levels_from_graph(graph_data: Dict[str, Any]) -> List[str]:
    """Extract factor levels from graph data."""
    factor_name = get_factor_name_from_graph(graph_data)
    if not factor_name:
        return []

    variables = graph_data.get("variables", [])
    for var in variables:
        if var.get("name") == factor_name:
            return var.get("values", [])
    return []


def get_nudge_target_group(result_dir: Path) -> Optional[str]:
    """Get the target group for a nudge condition from the graph data."""
    graph_data = load_preference_graph(result_dir)
    if not graph_data:
        return None

    nudge_config = graph_data.get("nudge_config")
    if nudge_config:
        return nudge_config.get("target_group")
    return None


def find_condition_directories(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Dict[str, Path]:
    """
    Find result directories for each condition (base, and each nudge target).

    Returns:
        Dictionary mapping condition name -> Path to result directory
        e.g., {'base': Path(...), 'low': Path(...), 'high': Path(...)}
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return {}

    result_dirs = {}
    dirs_by_condition: Dict[str, List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        # Check if this is a base condition
        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            # Get target group from graph data
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        if condition not in dirs_by_condition:
            dirs_by_condition[condition] = []
        dirs_by_condition[condition].append(result_dir)

    # For each condition, use the most recent directory
    for condition, dirs in dirs_by_condition.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result_dirs[condition] = most_recent

    return result_dirs


def compute_nudge_effect_size(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[NudgeEffectSize]:
    """
    Compute nudge effect size for a single experiment.

    Args:
        factor_name: Name of the factor (e.g., 'social_status')
        model: Model name
        nudge_type: Type of nudge (e.g., 'weak_evidence')
        results_base_dir: Base directory for results

    Returns:
        NudgeEffectSize object or None if data is insufficient
    """
    # Find all condition directories
    condition_dirs = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    if "base" not in condition_dirs:
        return None

    # Load baseline data
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return None

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return None

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        # For now, only support binary factors
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    # Compute baseline preferences
    base_prefs = compute_factor_preference_from_graph(base_graph, factor_var_name)
    p_A_baseline = base_prefs.get(level_A, {}).get("prob", 0.5)
    p_B_baseline = base_prefs.get(level_B, {}).get("prob", 0.5)
    n_baseline = base_prefs.get(level_A, {}).get("total", 0)

    # Load and compute nudge-A condition
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    if not nudge_A_graph:
        return None

    nudge_A_prefs = compute_factor_preference_from_graph(nudge_A_graph, factor_var_name)
    p_A_nudge_A = nudge_A_prefs.get(level_A, {}).get("prob", 0.5)
    n_nudge_A = nudge_A_prefs.get(level_A, {}).get("total", 0)

    # Load and compute nudge-B condition
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])
    if not nudge_B_graph:
        return None

    nudge_B_prefs = compute_factor_preference_from_graph(nudge_B_graph, factor_var_name)
    p_B_nudge_B = nudge_B_prefs.get(level_B, {}).get("prob", 0.5)
    n_nudge_B = nudge_B_prefs.get(level_B, {}).get("total", 0)

    # Compute effect sizes
    effect_size_A = p_A_nudge_A - p_A_baseline
    effect_size_B = p_B_nudge_B - p_B_baseline
    avg_effect_size = (effect_size_A + effect_size_B) / 2
    avg_effect_magnitude = (abs(effect_size_A) + abs(effect_size_B)) / 2

    return NudgeEffectSize(
        factor=factor_name,
        model=model,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        p_A_baseline=p_A_baseline,
        p_B_baseline=p_B_baseline,
        p_A_nudge_A=p_A_nudge_A,
        p_B_nudge_B=p_B_nudge_B,
        effect_size_A=effect_size_A,
        effect_size_B=effect_size_B,
        avg_effect_size=avg_effect_size,
        avg_effect_magnitude=avg_effect_magnitude,
        n_comparisons_baseline=n_baseline,
        n_comparisons_nudge_A=n_nudge_A,
        n_comparisons_nudge_B=n_nudge_B,
    )


def discover_experiments(
    results_base_dir: str = "results",
) -> List[Tuple[str, str, str]]:
    """
    Discover all available experiments in the results directory.

    Returns:
        List of (factor_name, model, nudge_type) tuples
    """
    results_path = Path(results_base_dir)
    experiments = []

    if not results_path.exists():
        return experiments

    # Iterate through experiment directories (simple_{factor})
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
            continue

        factor_name = exp_dir.name[7:]  # Remove 'simple_' prefix

        # Iterate through model directories
        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name

            # Iterate through nudge type directories
            for nudge_dir in model_dir.iterdir():
                if not nudge_dir.is_dir():
                    continue

                nudge_type = nudge_dir.name
                experiments.append((factor_name, model, nudge_type))

    return experiments


def is_baseline_too_biased(
    effect_size: NudgeEffectSize,
    threshold: float,
) -> bool:
    """
    Check if the baseline is too biased (one option dominates).

    Args:
        effect_size: NudgeEffectSize object with baseline probabilities
        threshold: Maximum allowed deviation from 50%.
                   E.g., threshold=10 means baselines must be in [10%, 90%].
                   threshold=0 means no filtering.

    Returns:
        True if the baseline is too biased and should be filtered out
    """
    if threshold <= 0:
        return False

    lower_bound = threshold / 100.0
    upper_bound = 1.0 - lower_bound

    # Check if either baseline probability is outside the allowed range
    if effect_size.p_A_baseline < lower_bound or effect_size.p_A_baseline > upper_bound:
        return True
    if effect_size.p_B_baseline < lower_bound or effect_size.p_B_baseline > upper_bound:
        return True

    return False


def compute_all_effect_sizes(
    results_base_dir: str = "results",
    baseline_threshold: float = 0,
) -> List[NudgeEffectSize]:
    """
    Compute effect sizes for all available experiments.

    Args:
        results_base_dir: Base directory for results
        baseline_threshold: Filter out experiments where baseline preference
                           is outside [threshold%, (100-threshold)%].
                           E.g., threshold=10 filters baselines >90% or <10%.
                           Default 0 means no filtering.

    Returns:
        List of NudgeEffectSize objects
    """
    experiments = discover_experiments(results_base_dir)
    effect_sizes = []
    filtered_count = 0

    for factor_name, model, nudge_type in experiments:
        effect_size = compute_nudge_effect_size(
            factor_name, model, nudge_type, results_base_dir
        )
        if effect_size is not None:
            if is_baseline_too_biased(effect_size, baseline_threshold):
                filtered_count += 1
            else:
                effect_sizes.append(effect_size)

    if filtered_count > 0 and baseline_threshold > 0:
        print(
            f"Filtered out {filtered_count} experiments with biased baselines (threshold={baseline_threshold}%)"
        )

    return effect_sizes


def aggregate_effect_sizes_by_nudge_type(
    effect_sizes: List[NudgeEffectSize],
    use_magnitude: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate effect sizes by nudge type (averaging across models and factors).

    Args:
        effect_sizes: List of NudgeEffectSize objects
        use_magnitude: If True (default), use absolute value of effect sizes.
                       If False, use signed effect sizes.

    Returns:
        Dictionary mapping nudge_type -> {
            'avg_effect_size': float,
            'std_effect_size': float,
            'n_experiments': int,
            'effect_sizes': list of individual effect sizes
        }
    """
    by_nudge_type: Dict[str, List[float]] = defaultdict(list)

    for es in effect_sizes:
        value = es.avg_effect_magnitude if use_magnitude else es.avg_effect_size
        by_nudge_type[es.nudge_type].append(value)

    result = {}
    for nudge_type, sizes in by_nudge_type.items():
        avg = statistics.mean(sizes) if sizes else 0.0
        std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0

        result[nudge_type] = {
            "avg_effect_size": avg,
            "std_effect_size": std,
            "n_experiments": len(sizes),
            "effect_sizes": sizes,
        }

    return result
