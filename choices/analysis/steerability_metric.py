"""
Value Steerability Asymmetry calculation.

Steerability measures how much nudging changes the odds ratio for a particular option.
Steerability Asymmetry measures differential steerability between two options.

For a pair of groups (A, B):
- steerability_A = ln(r_A(A)) - ln(r_0(A))  [how much nudging towards A increases A's odds]
- steerability_B = ln(r_B(B)) - ln(r_0(B))  [how much nudging towards B increases B's odds]
- asym = (S(B) - S(A)) / (|S(A)| + |S(B)| + eps)  [normalized asymmetry, in [-1, 1]]

Two variants are provided:
- compute_steerability_asym: For exchange rate data (used by analyze_nudging_results.py)
- compute_steerability_asym_from_counts: For count data (used by analyze_simple_nudging_results.py)

All log odds calculations use natural logarithm (ln).
"""

import math
from typing import List, Optional, Tuple


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


# Backward compatibility alias
def compute_steerability_bias(
    rate_A_base: float,
    rate_B_base: float,
    rate_A_nudge_A: float,
    rate_B_nudge_A: float,
    rate_A_nudge_B: float,
    rate_B_nudge_B: float,
    eps: float = 0.01,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    DEPRECATED: Use compute_steerability_asym instead.

    This function is kept for backward compatibility.
    """
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
    asym = (steerability_B - steerability_A) / (
        abs(steerability_A) + abs(steerability_B) + eps
    )

    return steerability_A, steerability_B, asym


# Backward compatibility alias
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
    """
    DEPRECATED: Use compute_steerability_asym_from_counts instead.

    This function is kept for backward compatibility.
    """
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


# Backward compatibility alias - deprecated, use compute_steerability_asym_from_counts
def compute_steerability_bias_from_frequencies(
    f_0_A: float,
    f_0_B: float,
    f_A_A: float,
    f_A_B: float,
    f_B_A: float,
    f_B_B: float,
    eps: float = 1e-6,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    DEPRECATED: Use compute_steerability_asym_from_counts instead.

    This function is kept for backward compatibility but internally converts
    frequencies to pseudo-counts and uses the count-based implementation.

    Note: When using frequencies, Haldane-Anscombe correction is disabled
    since the frequencies are already normalized.
    """
    import warnings

    warnings.warn(
        "compute_steerability_bias_from_frequencies is deprecated. "
        "Use compute_steerability_asym_from_counts with actual counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Check for near-zero frequencies (would cause log(0) issues)
    freqs = [f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B]
    if any(f < eps for f in freqs):
        return None, None, None

    # Use frequencies directly as pseudo-counts without Haldane-Anscombe correction
    # (since frequencies are already normalized, adding 0.5 would distort them)
    return compute_steerability_asym_from_counts(
        f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B, use_haldane_anscombe=False
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
    # steerability_A = log(c_A_A/c_A_B) - log(c_0_A/c_0_B)
    # steerability_B = log(c_B_B/c_B_A) - log(c_0_B/c_0_A)
    # unnormalized diff = steerability_B - steerability_A

    # Var(log(a/b)) ≈ 1/a + 1/b
    var_log_ratio_nudge_A = 1.0 / c_A_A_adj + 1.0 / c_A_B_adj
    var_log_ratio_nudge_B = 1.0 / c_B_B_adj + 1.0 / c_B_A_adj
    var_log_ratio_base = 1.0 / c_0_A_adj + 1.0 / c_0_B_adj

    # Total variance (treating nudge conditions as independent)
    # Baseline terms appear in both steerability_A and steerability_B with opposite signs
    # so they contribute 2 * var_log_ratio_base to total variance
    var_diff = var_log_ratio_nudge_A + var_log_ratio_nudge_B + 2 * var_log_ratio_base
    se_diff = math.sqrt(var_diff)

    # For the test, we use the asymmetry value directly
    # The normalization preserves the sign, so asym != 0 iff unnormalized diff != 0
    # We test if the asymmetry is significantly different from 0
    if se_diff > 0:
        # Use z-test on the unnormalized difference to determine significance
        # But report using the asymmetry value for consistency
        z_score = steerability_asym / (se_diff / (se_diff + 0.01))  # approximate
        # Two-tailed p-value using standard normal CDF
        # p = 2 * (1 - Phi(|z|))
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


# Backward compatibility alias
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
    """
    DEPRECATED: Use wald_test_steerability_asym instead.

    This function is kept for backward compatibility.
    """
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


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
