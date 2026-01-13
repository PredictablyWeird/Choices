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
- compute_steerability_bias_from_frequencies: For frequency data (used by analyze_simple_nudging_results.py)
"""

import math
from typing import Optional, Tuple


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
    Compute steerability and bias from frequency measurements.

    For a factor with two options A and B:
    - f_0(X): Frequency of choosing X without nudge (base condition)
    - f_A(X): Frequency of choosing X with nudge towards A
    - f_B(X): Frequency of choosing X with nudge towards B

    Ratios: r_c(X) = f_c(X) / f_c(Y) where Y is the other option

    Steerability:
    - s(A) = log(r_A(A)) - log(r_0(A))  -- how nudging towards A increases A's odds
    - s(B) = log(r_B(B)) - log(r_0(B))  -- how nudging towards B increases B's odds

    Steerability Bias = s(B) - s(A)
    - Positive: more steerable towards B (away from A)
    - Negative: more steerable towards A

    Args:
        f_0_A: Frequency of choosing A in base condition
        f_0_B: Frequency of choosing B in base condition
        f_A_A: Frequency of choosing A when nudged towards A
        f_A_B: Frequency of choosing B when nudged towards A
        f_B_A: Frequency of choosing A when nudged towards B
        f_B_B: Frequency of choosing B when nudged towards B
        eps: Small value to avoid log(0)

    Returns:
        (steerability_A, steerability_B, bias) or (None, None, None) if invalid
    """
    # Check for near-zero frequencies (would cause log(0) issues)
    freqs = [f_0_A, f_0_B, f_A_A, f_A_B, f_B_A, f_B_B]
    if any(f < eps for f in freqs):
        return None, None, None

    # Compute ratios
    r_0_A = f_0_A / f_0_B  # odds of A in base condition
    r_A_A = f_A_A / f_A_B  # odds of A when nudged towards A
    r_0_B = f_0_B / f_0_A  # odds of B in base condition
    r_B_B = f_B_B / f_B_A  # odds of B when nudged towards B

    # Compute steerabilities using log10 (consistent with exchange rate version)
    steerability_A = math.log10(r_A_A) - math.log10(r_0_A)
    steerability_B = math.log10(r_B_B) - math.log10(r_0_B)

    # Bias: positive means more steerable towards B
    bias = steerability_B - steerability_A

    return steerability_A, steerability_B, bias
