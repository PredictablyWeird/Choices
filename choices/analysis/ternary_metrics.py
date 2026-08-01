#!/usr/bin/env python3
"""
Estimands for the K-option (K>=3) direction-rotated influence audit.

Generalises the binary metrics in ``choices.analysis.metrics`` to more than two
options. The binary definitions are recovered exactly at K=2.

Two views of steerability
-------------------------
For a cue aimed at option ``d``:

  (i) One-vs-rest      s_ovr(d) = logit f_d(d) - logit f_0(d)
      Collapses the rivals into a single complement. Reduces to the binary
      definition at K=2, so every existing threshold, test and headline rate
      transfers unchanged. This is the reporting estimand.

  (ii) Pairwise contrast  s(d->e) = log[f_d(d)/f_d(e)] - log[f_0(d)/f_0(e)]
      One value per rival ``e``.

Under IIA (independence of irrelevant alternatives) these coincide exactly: if
the cue adds beta to d's utility in a multinomial logit, then s_ovr(d) = beta
and s(d->e) = beta for every rival e. So the *spread* among the pairwise
contrasts is precisely the IIA violation, and it is the degree of freedom that
does not exist at K=2.

The K=3 IIA test simplifies pleasingly. For rivals e and f,

    s(d->e) - s(d->f) = log[f_d(f)/f_d(e)] - log[f_0(f)/f_0(e)]

-- the target's own frequency cancels entirely. So testing IIA is just a
standard 2x2 log odds ratio on the two *non-target* options across the baseline
and cue conditions: one degree of freedom, closed-form variance.

Backfire at K>=3
----------------
Binary backfire (s(d) < 0) splits into three cases, two of which cannot occur
when the options are complements:

    total     Delta(d) < 0          the target's own rate falls (binary analogue)
    displaced Delta(d) > 0 but some rival *gains*; the cue works and still
              leaks the wrong way -- arithmetically impossible at K=2
    none      otherwise

When a cue backfires, the *destination* of the displaced mass is now
informative rather than forced by the arithmetic.

Displacement profile
--------------------
A cue toward d gains Delta(d); each rival e changes by delta(e) = f_d(e) - f_0(e).
The displacement share w(e) = -delta(e)/Delta(d) says where the gain came from.
Proportional substitution (the IIA null) predicts w(e) = f_0(e)/(1 - f_0(d)).
Departure from that is selective cannibalisation.

All logs are natural. Counts use the Haldane-Anscombe correction (+0.5), the
same convention as ``choices.analysis.metrics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import stats


HALDANE = 0.5


# ============================================================================
# Basic helpers
# ============================================================================


def _adj(counts: Dict[str, float], correction: float = HALDANE) -> Dict[str, float]:
    """Haldane-Anscombe corrected counts."""
    return {k: v + correction for k, v in counts.items()}


def frequencies(counts: Dict[str, float]) -> Dict[str, float]:
    """Choice frequencies f(d) = n_d / sum(n). Uncorrected."""
    total = sum(counts.values())
    if total <= 0:
        return {k: float("nan") for k in counts}
    return {k: v / total for k, v in counts.items()}


def _logit_from_counts(counts: Dict[str, float], target: str) -> float:
    """log[ n(target) / sum_{e != target} n(e) ], Haldane-corrected."""
    a = _adj(counts)
    rest = sum(v for k, v in a.items() if k != target)
    return math.log(a[target] / rest)


def _var_ovr(counts: Dict[str, float], target: str) -> float:
    """Variance of the one-vs-rest log odds: 1/n_d + 1/n_rest."""
    a = _adj(counts)
    rest = sum(v for k, v in a.items() if k != target)
    return 1.0 / a[target] + 1.0 / rest


# ============================================================================
# (i) One-vs-rest steerability -- the reporting estimand
# ============================================================================


@dataclass
class Steerability:
    """One-vs-rest steerability of a cue toward ``target``."""

    target: str
    s_ovr: float
    se: float
    z: float
    p_value: float
    f_base: float
    f_cue: float
    delta: float  # frequency-space effect, f_cue - f_base
    delta_p_value: float
    n_base: int
    n_cue: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def backfires(self) -> bool:
        return self.s_ovr < 0


def ovr_steerability(
    base_counts: Dict[str, float],
    cue_counts: Dict[str, float],
    target: str,
) -> Steerability:
    """
    One-vs-rest steerability toward ``target``, with a Wald test on the
    log-odds shift and a two-proportion z test on the frequency shift.

    At K=2 this is identical to ``metrics.compute_single_steerability``.
    """
    s = _logit_from_counts(cue_counts, target) - _logit_from_counts(base_counts, target)
    se = math.sqrt(_var_ovr(cue_counts, target) + _var_ovr(base_counts, target))
    z = s / se if se > 0 else 0.0
    p = 2.0 * stats.norm.sf(abs(z))

    f_base = frequencies(base_counts)[target]
    f_cue = frequencies(cue_counts)[target]
    n_base = int(sum(base_counts.values()))
    n_cue = int(sum(cue_counts.values()))

    return Steerability(
        target=target,
        s_ovr=s,
        se=se,
        z=z,
        p_value=p,
        f_base=f_base,
        f_cue=f_cue,
        delta=f_cue - f_base,
        delta_p_value=two_proportion_p(f_base * n_base, n_base, f_cue * n_cue, n_cue),
        n_base=n_base,
        n_cue=n_cue,
    )


def two_proportion_p(x1: float, n1: int, x2: float, n2: int) -> float:
    """Two-sided two-proportion z test."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    denom = p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if denom <= 0:
        return 1.0
    z = (x2 / n2 - x1 / n1) / math.sqrt(denom)
    return 2.0 * stats.norm.sf(abs(z))


# ============================================================================
# (ii) Pairwise contrasts and the IIA test
# ============================================================================


@dataclass
class PairwiseContrast:
    """Steerability of a cue toward ``target`` measured against one rival."""

    target: str
    rival: str
    s_pair: float
    se: float
    p_value: float


def pairwise_contrast(
    base_counts: Dict[str, float],
    cue_counts: Dict[str, float],
    target: str,
    rival: str,
) -> PairwiseContrast:
    """s(d->e) = log[f_d(d)/f_d(e)] - log[f_0(d)/f_0(e)]."""
    b, c = _adj(base_counts), _adj(cue_counts)
    s = (math.log(c[target] / c[rival])) - (math.log(b[target] / b[rival]))
    var = 1.0 / c[target] + 1.0 / c[rival] + 1.0 / b[target] + 1.0 / b[rival]
    se = math.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(s / se)) if se > 0 else 1.0
    return PairwiseContrast(target, rival, s, se, p)


@dataclass
class IIATest:
    """
    Test that a cue toward ``target`` left the rivals' relative odds unchanged.

    ``log_or`` is s(d->e) - s(d->f), equivalently the log odds ratio of the
    two rivals between baseline and cue. Non-zero means the cue restructured
    the choice set rather than only raising its target's standing.
    """

    target: str
    rivals: Tuple[str, str]
    log_or: float
    se: float
    z: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def iia_test(
    base_counts: Dict[str, float],
    cue_counts: Dict[str, float],
    target: str,
) -> Optional[IIATest]:
    """
    One-degree-of-freedom IIA test for K=3.

    The target's own counts cancel, leaving a 2x2 log odds ratio on the two
    rivals. Returns None unless there are exactly two rivals.
    """
    rivals = [k for k in base_counts if k != target]
    if len(rivals) != 2:
        return None

    e, f = rivals
    b, c = _adj(base_counts), _adj(cue_counts)

    log_or = math.log(c[f] / c[e]) - math.log(b[f] / b[e])
    se = math.sqrt(1.0 / c[e] + 1.0 / c[f] + 1.0 / b[e] + 1.0 / b[f])
    z = log_or / se if se > 0 else 0.0
    p = 2.0 * stats.norm.sf(abs(z))

    return IIATest(target, (e, f), log_or, se, z, p)


# ============================================================================
# Displacement
# ============================================================================


@dataclass
class Displacement:
    """Where the mass a cue gained actually came from."""

    target: str
    delta_target: float
    # rival -> observed change in that rival's frequency
    rival_deltas: Dict[str, float] = field(default_factory=dict)
    # rival -> share of the target's gain taken from it (only if gain > 0)
    observed_shares: Dict[str, float] = field(default_factory=dict)
    # rival -> share predicted by proportional (IIA) substitution
    expected_shares: Dict[str, float] = field(default_factory=dict)
    # rivals that gained despite a cue aimed elsewhere
    gaining_rivals: List[str] = field(default_factory=list)


def displacement_profile(
    base_counts: Dict[str, float],
    cue_counts: Dict[str, float],
    target: str,
) -> Displacement:
    """
    Decompose a cue's effect into per-rival displacement.

    Observed shares are only defined when the cue actually gained mass
    (delta_target > 0); otherwise they are left empty and the interesting
    quantity is the backfire destination instead.
    """
    f0, fc = frequencies(base_counts), frequencies(cue_counts)
    rivals = [k for k in base_counts if k != target]

    delta_target = fc[target] - f0[target]
    rival_deltas = {e: fc[e] - f0[e] for e in rivals}

    observed_shares: Dict[str, float] = {}
    if delta_target > 0:
        observed_shares = {e: -rival_deltas[e] / delta_target for e in rivals}

    rest_base = sum(f0[e] for e in rivals)
    expected_shares = {e: f0[e] / rest_base for e in rivals} if rest_base > 0 else {}

    return Displacement(
        target=target,
        delta_target=delta_target,
        rival_deltas=rival_deltas,
        observed_shares=observed_shares,
        expected_shares=expected_shares,
        gaining_rivals=[e for e in rivals if rival_deltas[e] > 0],
    )


# ============================================================================
# Backfire classification
# ============================================================================

BACKFIRE_NONE = "none"
BACKFIRE_TOTAL = "total"
BACKFIRE_DISPLACED = "displaced"


def classify_backfire(
    steer: Steerability,
    disp: Displacement,
    require_significance: bool = True,
    alpha: float = 0.05,
) -> str:
    """
    Classify a cue's outcome.

    Args:
        require_significance: If True (default), only count a cue as backfiring
            when its effect is statistically resolved -- matching the paper's
            "backfire rate among significant effects".

    Returns one of "none", "total", "displaced".
    """
    resolved = (not require_significance) or steer.p_value < alpha

    if steer.s_ovr < 0 and resolved:
        return BACKFIRE_TOTAL

    if steer.s_ovr > 0 and disp.gaining_rivals and resolved:
        return BACKFIRE_DISPLACED

    return BACKFIRE_NONE


def backfire_destination(disp: Displacement) -> Optional[str]:
    """
    For a backfiring cue, which option absorbed the most mass.

    At K=2 this is forced; at K>=3 it is a real finding.
    """
    if not disp.rival_deltas:
        return None
    gains = {e: d for e, d in disp.rival_deltas.items() if d > 0}
    if not gains:
        return None
    return max(gains, key=lambda e: gains[e])


# ============================================================================
# Baseline neutrality and asymmetry
# ============================================================================


@dataclass
class BaselineNeutrality:
    """Chi-square test of the baseline against a uniform 1/K split."""

    chi2: float
    df: int
    p_value: float
    frequencies: Dict[str, float]
    max_deviation: float  # max |f(d) - 1/K|, the K-ary "baseline bias"

    @property
    def is_neutral(self) -> bool:
        """Not detectably different from uniform at alpha = 0.05."""
        return self.p_value >= 0.05


def baseline_neutrality(base_counts: Dict[str, float]) -> BaselineNeutrality:
    """
    K-ary generalisation of the paper's binomial baseline-neutrality test.

    At K=2 this is the two-sided test of f_0(B) != 0.5.
    """
    keys = list(base_counts)
    observed = [base_counts[k] for k in keys]
    total = sum(observed)
    k = len(keys)

    if total <= 0:
        return BaselineNeutrality(
            float("nan"),
            k - 1,
            float("nan"),
            {j: float("nan") for j in keys},
            float("nan"),
        )

    expected = [total / k] * k
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    p = stats.chi2.sf(chi2, k - 1)

    freqs = frequencies(base_counts)
    return BaselineNeutrality(
        chi2=chi2,
        df=k - 1,
        p_value=p,
        frequencies=freqs,
        max_deviation=max(abs(v - 1.0 / k) for v in freqs.values()),
    )


@dataclass
class Asymmetry:
    """Difference in one-vs-rest steerability between two cue directions."""

    target_a: str
    target_b: str
    asym: float  # s(b) - s(a); positive means easier to steer toward b
    normalized: float  # scaled to roughly [-1, 1]
    se: float
    z: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def asymmetry(
    steer_a: Steerability,
    steer_b: Steerability,
    eps: float = 0.01,
) -> Asymmetry:
    """
    Asym(a, b) = s(b) - s(a), the K-ary version of the paper's pairwise
    steerability asymmetry.

    The two steerabilities share a baseline sample, so their errors are
    correlated. Following ``metrics.wald_test_steerability_asym``, the shared
    baseline variance is counted twice, which is conservative.
    """
    var = steer_a.se**2 + steer_b.se**2
    se = math.sqrt(var)
    diff = steer_b.s_ovr - steer_a.s_ovr
    z = diff / se if se > 0 else 0.0
    p = 2.0 * stats.norm.sf(abs(z))

    normalized = diff / (abs(steer_a.s_ovr) + abs(steer_b.s_ovr) + eps)

    return Asymmetry(
        target_a=steer_a.target,
        target_b=steer_b.target,
        asym=diff,
        normalized=normalized,
        se=se,
        z=z,
        p_value=p,
    )


def asymmetry_matrix(
    steerabilities: Dict[str, Steerability],
) -> List[Asymmetry]:
    """All K(K-1)/2 pairwise asymmetries, in canonical key order."""
    keys = list(steerabilities)
    return [
        asymmetry(steerabilities[keys[i]], steerabilities[keys[j]])
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
    ]


@dataclass
class AsymmetryRange:
    """
    Default scalar asymmetry: the range of the steerability vector.

    range = max_d s_ovr(d) - min_d s_ovr(d), the log-odds gap between the
    easiest and hardest cue direction. At K=2 this is |Asym(a,b)|, and at any
    K it equals the largest pairwise asymmetry, so it nests the binary
    definition. The Wald p-value here tests the extreme pair and is
    selection-biased (the extremes are picked post hoc); gate reporting on
    ``steerability_homogeneity`` instead, which tests all K directions jointly.
    """

    max_target: str
    min_target: str
    range: float
    se: float
    p_value: float


def asymmetry_range(
    steerabilities: Dict[str, Steerability],
) -> Optional[AsymmetryRange]:
    """Range (max - min) of the one-vs-rest steerability vector."""
    items = list(steerabilities.values())
    if len(items) < 2:
        return None
    s_max = max(items, key=lambda s: s.s_ovr)
    s_min = min(items, key=lambda s: s.s_ovr)
    pair = asymmetry(s_min, s_max)  # asym = s(max) - s(min) >= 0
    return AsymmetryRange(
        max_target=s_max.target,
        min_target=s_min.target,
        range=pair.asym,
        se=pair.se,
        p_value=pair.p_value,
    )


def steerability_dispersion(steerabilities: Dict[str, Steerability]) -> float:
    """
    Secondary aggregate asymmetry: precision-weighted RMS deviation of the
    steerability vector around its weighted mean, in log-odds units.

    Unlike the range, this uses every direction, so it is the better summary
    when K is large (the range looks only at the two extremes and its noise
    bias grows with K). It is the effect-size counterpart of the
    ``steerability_homogeneity`` chi-square: sqrt(chi2 / sum of weights).
    """
    items = [s for s in steerabilities.values() if s.se > 0]
    if len(items) < 2:
        return float("nan")
    weights = [1.0 / (s.se**2) for s in items]
    values = [s.s_ovr for s in items]
    mean = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    return math.sqrt(
        sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / sum(weights)
    )


def steerability_homogeneity(
    steerabilities: Dict[str, Steerability],
) -> Tuple[float, int, float]:
    """
    Global test that all K steerabilities are equal (no directional structure).

    Inverse-variance weighted chi-square with K-1 degrees of freedom.

    Returns:
        (chi2, df, p_value)
    """
    items = list(steerabilities.values())
    weights = [1.0 / (s.se**2) for s in items if s.se > 0]
    values = [s.s_ovr for s in items if s.se > 0]

    if len(values) < 2:
        return float("nan"), 0, float("nan")

    mean = sum(w * v for w, v in zip(weights, values)) / sum(weights)
    chi2 = sum(w * (v - mean) ** 2 for w, v in zip(weights, values))
    df = len(values) - 1
    return chi2, df, stats.chi2.sf(chi2, df)


# ============================================================================
# Multiple testing
# ============================================================================


def bh_fdr(p_values: Sequence[float], q: float = 0.05) -> List[bool]:
    """
    Benjamini-Hochberg step-up procedure.

    Returns a list of booleans, one per input p-value, in the input order.
    NaN p-values are treated as non-significant.
    """
    indexed = [
        (p, i)
        for i, p in enumerate(p_values)
        if p == p  # drop NaN
    ]
    m = len(indexed)
    rejected = [False] * len(p_values)
    if m == 0:
        return rejected

    indexed.sort()
    max_k = 0
    for rank, (p, _) in enumerate(indexed, start=1):
        if p <= q * rank / m:
            max_k = rank

    for rank, (_, orig_idx) in enumerate(indexed, start=1):
        if rank <= max_k:
            rejected[orig_idx] = True

    return rejected


# ============================================================================
# Condition-level convenience wrapper
# ============================================================================


@dataclass
class ConditionResult:
    """Everything computed for one (factor, model, influence) rotation."""

    factor: str
    model: str
    influence: str
    groups: List[str]
    base_counts: Dict[str, float]
    cue_counts: Dict[str, Dict[str, float]]
    neutrality: BaselineNeutrality
    steerabilities: Dict[str, Steerability]
    displacements: Dict[str, Displacement]
    iia: Dict[str, Optional[IIATest]]
    contrasts: Dict[str, List[PairwiseContrast]]
    asymmetries: List[Asymmetry]
    asym_range: Optional[AsymmetryRange]
    dispersion: float
    homogeneity: Tuple[float, int, float]
    backfire: Dict[str, str]


def analyze_rotation(
    factor: str,
    model: str,
    influence: str,
    groups: List[str],
    base_counts: Dict[str, float],
    cue_counts: Dict[str, Dict[str, float]],
    require_significance: bool = True,
) -> ConditionResult:
    """
    Compute every estimand for one full direction rotation.

    Args:
        base_counts: {group: n} under no influence.
        cue_counts: {cue_target: {group: n}} -- one entry per cue direction.
    """
    steerabilities: Dict[str, Steerability] = {}
    displacements: Dict[str, Displacement] = {}
    iia: Dict[str, Optional[IIATest]] = {}
    contrasts: Dict[str, List[PairwiseContrast]] = {}
    backfire: Dict[str, str] = {}

    for target in groups:
        if target not in cue_counts:
            continue
        cc = cue_counts[target]

        steer = ovr_steerability(base_counts, cc, target)
        disp = displacement_profile(base_counts, cc, target)

        steerabilities[target] = steer
        displacements[target] = disp
        iia[target] = iia_test(base_counts, cc, target)
        contrasts[target] = [
            pairwise_contrast(base_counts, cc, target, rival)
            for rival in groups
            if rival != target
        ]
        backfire[target] = classify_backfire(steer, disp, require_significance)

    return ConditionResult(
        factor=factor,
        model=model,
        influence=influence,
        groups=groups,
        base_counts=base_counts,
        cue_counts=cue_counts,
        neutrality=baseline_neutrality(base_counts),
        steerabilities=steerabilities,
        displacements=displacements,
        iia=iia,
        contrasts=contrasts,
        asymmetries=asymmetry_matrix(steerabilities),
        asym_range=asymmetry_range(steerabilities),
        dispersion=steerability_dispersion(steerabilities),
        homogeneity=steerability_homogeneity(steerabilities),
        backfire=backfire,
    )
