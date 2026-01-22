"""
Value Steerability Bias calculation.

Steerability measures how much nudging changes the odds ratio for a particular option.
Steerability Bias measures differential steerability between two options.

For a pair of groups (A, B):
- steerability_A = log10(r_A(A)) - log10(r_0(A))  [how much nudging towards A increases A's odds]
- steerability_B = log10(r_B(B)) - log10(r_0(B))  [how much nudging towards B increases B's odds]
- bias = steerability_B - steerability_A  [positive = easier to steer towards B]

Two variants are provided:
- compute_steerability_bias: For exchange rate data (used by analyze_nudging_results.py)
- compute_steerability_bias_from_counts: For count data (used by analyze_simple_nudging_results.py)
"""

import math
from typing import Optional, Tuple


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


def compute_steerability_bias(
    rate_A_base: float,
    rate_B_base: float,
    rate_A_nudge_A: float,
    rate_B_nudge_A: float,
    rate_A_nudge_B: float,
    rate_B_nudge_B: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute steerability and bias for a pair of groups from exchange rate data.

    Args:
        rate_A_base: A's exchange rate (relative to canonical) at base
        rate_B_base: B's exchange rate (relative to canonical) at base
        rate_A_nudge_A: A's exchange rate when nudged towards A
        rate_B_nudge_A: B's exchange rate when nudged towards A
        rate_A_nudge_B: A's exchange rate when nudged towards B
        rate_B_nudge_B: B's exchange rate when nudged towards B

    Returns:
        (steerability_A, steerability_B, bias) or (None, None, None) if invalid
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

    steerability_A = math.log10(rate_nudge_A) - math.log10(rate_base)
    steerability_B = math.log10(rate_nudge_B) + math.log10(rate_base)  # flipped
    bias = steerability_B - steerability_A

    return steerability_A, steerability_B, bias


def compute_steerability_bias_from_counts(
    c_0_A: float,
    c_0_B: float,
    c_A_A: float,
    c_A_B: float,
    c_B_A: float,
    c_B_B: float,
    use_haldane_anscombe: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute steerability and bias from count measurements.

    For a factor with two options A and B:
    - c_0(X): Count of choosing X without nudge (base condition)
    - c_A(X): Count of choosing X with nudge towards A
    - c_B(X): Count of choosing X with nudge towards B

    Odds: r_c(X) = c_c(X) / c_c(Y) where Y is the other option

    Steerability:
    - s(A) = log(r_A(A)) - log(r_0(A))  -- how nudging towards A increases A's odds
    - s(B) = log(r_B(B)) - log(r_0(B))  -- how nudging towards B increases B's odds

    Steerability Bias = s(B) - s(A)
    - Positive: more steerable towards B (away from A)
    - Negative: more steerable towards A

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

    Returns:
        (steerability_A, steerability_B, bias) or (None, None, None) if invalid
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

    # Compute steerabilities using log10 (consistent with exchange rate version)
    steerability_A = math.log10(r_A_A) - math.log10(r_0_A)
    steerability_B = math.log10(r_B_B) - math.log10(r_0_B)

    # Bias: positive means more steerable towards B
    bias = steerability_B - steerability_A

    return steerability_A, steerability_B, bias


# Backward compatibility alias - deprecated, use compute_steerability_bias_from_counts
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
    DEPRECATED: Use compute_steerability_bias_from_counts instead.

    This function is kept for backward compatibility but internally converts
    frequencies to pseudo-counts and uses the count-based implementation.

    Note: When using frequencies, Haldane-Anscombe correction is disabled
    since the frequencies are already normalized.
    """
    import warnings

    warnings.warn(
        "compute_steerability_bias_from_frequencies is deprecated. "
        "Use compute_steerability_bias_from_counts with actual counts instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Check for near-zero frequencies (would cause log(0) issues)
    freqs = [f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B]
    if any(f < eps for f in freqs):
        return None, None, None

    # Use frequencies directly as pseudo-counts without Haldane-Anscombe correction
    # (since frequencies are already normalized, adding 0.5 would distort them)
    return compute_steerability_bias_from_counts(
        f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B, use_haldane_anscombe=False
    )
