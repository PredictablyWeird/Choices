#!/usr/bin/env python3
"""
Calibrate group-size assignments to hit a baseline-neutral condition.

The pilot age3 run showed a baseline nowhere near neutral (young 55% / middle
28% / old 17% at generically-sampled sizes), so none of the three cue types
in that run qualify for the paper's asymmetry/backfire analysis, which is
restricted to baseline-neutral conditions (f0 not detectably different from
1/(K) at alpha=0.05).

This module fits a Luce (multinomial logit) model to an already-collected
baseline run:

    utility(g, n) = alpha_g + beta * log(n)
    P(choose g) = softmax_g(utility)

with one group fixed as the reference (alpha = 0). ``alpha_g`` is the group's
intrinsic preference in log-odds units at equal size; ``beta`` is the
per-doubling value of an additional person. Given the fit, it searches the
full {1..K_max}^3 size grid for triples whose *predicted* choice distribution
is close to uniform, i.e. explicit per-group sizes chosen to counteract the
model's intrinsic age preference -- the ternary analogue of the paper's
exchange-rate experiments.

This calibration is exploratory scaffolding, not part of the estimands
themselves: ``ternary_metrics`` reports whatever the model actually does at
whatever sizes are used. Calibration only decides *which* sizes to spend API
budget on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================================
# Fitting
# ============================================================================


@dataclass
class LuceFit:
    """
    A fitted Luce/multinomial-logit model over group + log(size).

    ``beta`` is per-group: utility(g, n) = alpha_g + beta_g * log(n). A shared
    scalar beta (``per_group_beta=False`` in ``fit_luce_model``) is the
    special case beta_g == beta for all g; allow it to vary when a
    likelihood-ratio test shows groups differ in how much an additional
    person is worth (which they may -- that is itself a form of the
    asymmetric-steerability structure this audit is built to find, just
    showing up in the baseline size-response instead of in a cue response).
    """

    groups: List[str]
    reference: str
    alpha: Dict[str, float]  # alpha[reference] == 0.0
    beta: Dict[str, float]  # utility per unit of log(n), per group
    n_obs: int
    log_likelihood: float

    def predict(self, sizes: Dict[str, int]) -> Dict[str, float]:
        """Predicted choice probability for each group at the given sizes."""
        utilities = {
            g: self.alpha[g] + self.beta[g] * math.log(sizes[g])
            for g in self.groups
        }
        m = max(utilities.values())
        exp_u = {g: math.exp(u - m) for g, u in utilities.items()}
        z = sum(exp_u.values())
        return {g: v / z for g, v in exp_u.items()}

    def exchange_rate(self, target: str, reference: str) -> float:
        """
        How many ``target`` are worth one ``reference`` at the fitted model,
        i.e. the size ratio n_target/n_reference that makes the two groups
        equally preferred, holding the *reference*'s count at 1.

        Only exact when beta_target == beta_reference; reported using
        beta_target since that is the group being scaled.
        """
        if self.beta[target] == 0:
            return float("nan")
        return math.exp(
            (self.alpha[reference] - self.alpha[target]) / self.beta[target]
        )


def _parse_sizes(sizes_str: str) -> Dict[str, int]:
    """Parse the ternary runner's "young=1;middle-aged=7;old=3" format."""
    out = {}
    for part in sizes_str.split(";"):
        k, v = part.split("=")
        out[k] = int(v)
    return out


def _load_and_stack(
    trials_dfs: List[pd.DataFrame], groups: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate one or more trials frames into (log_sizes, chosen_idx) arrays."""
    valid = pd.concat(
        [df[df["status"] == "ok"] for df in trials_dfs], ignore_index=True
    )
    if valid.empty:
        raise ValueError("No valid trials to calibrate from.")

    sizes = valid["sizes"].apply(_parse_sizes)
    log_sizes = np.array([[math.log(s[g]) for g in groups] for s in sizes])
    chosen_idx = np.array(
        [groups.index(g) for g in valid["chosen_group"]], dtype=int
    )
    return log_sizes, chosen_idx, len(valid)


def fit_luce_model(
    trials_df,
    groups: List[str],
    reference: str = None,
    per_group_beta: bool = True,
) -> LuceFit:
    """
    Fit the Luce model to base-condition trials.

    Args:
        trials_df: A single DataFrame, or a list of DataFrames to pool
            (e.g. base trials from more than one audit run, which widens the
            size range the fit is calibrated over).
        per_group_beta: If True (default), let each group's log-size
            sensitivity differ; if False, share one beta across all groups.
            Compare via ``log_likelihood`` -- a materially better fit with
            per-group beta means groups don't just differ in intrinsic
            preference but in how much an additional person is worth to
            each, which a shared-beta calibration will under- or
            over-correct for outside the sizes it was fit on.

    Only valid ("ok") trials are used. ``reference`` defaults to the first
    group in ``groups``.
    """
    reference = reference or groups[0]
    others = [g for g in groups if g != reference]

    dfs = [trials_df] if isinstance(trials_df, pd.DataFrame) else list(trials_df)
    log_sizes, chosen_idx, n_obs = _load_and_stack(dfs, groups)

    n_alpha = len(others)
    n_beta = len(groups) if per_group_beta else 1

    def unpack(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = np.zeros(len(groups))
        for i, g in enumerate(others):
            alpha[groups.index(g)] = params[i]
        beta_params = params[n_alpha:]
        beta = beta_params if per_group_beta else np.full(len(groups), beta_params[0])
        return alpha, beta

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha, beta = unpack(params)
        utilities = alpha[None, :] + beta[None, :] * log_sizes
        m = utilities.max(axis=1, keepdims=True)
        log_z = m.squeeze(1) + np.log(np.exp(utilities - m).sum(axis=1))
        chosen_utility = utilities[np.arange(len(chosen_idx)), chosen_idx]
        return -(chosen_utility - log_z).sum()

    x0 = np.zeros(n_alpha + n_beta)
    x0[n_alpha:] = 1.0  # start beta near "more people is better"

    result = minimize(neg_log_likelihood, x0, method="BFGS")
    alpha_arr, beta_arr = unpack(result.x)
    alpha = {g: float(alpha_arr[groups.index(g)]) for g in groups}
    beta = {g: float(beta_arr[groups.index(g)]) for g in groups}

    return LuceFit(
        groups=groups,
        reference=reference,
        alpha=alpha,
        beta=beta,
        n_obs=n_obs,
        log_likelihood=-float(result.fun),
    )


def print_calibration_report(fit: LuceFit) -> None:
    """Human-readable summary of a fit: intrinsic bias and exchange rates."""
    print(f"\nLuce fit ({fit.n_obs} trials, log-likelihood = {fit.log_likelihood:.1f})")
    print(f"  reference group: {fit.reference}  (alpha fixed at 0)")
    print("\n  intrinsic preference at equal size (alpha, log-odds)"
          "  and size sensitivity (beta, per log-unit of size):")
    for g in fit.groups:
        print(f"    {g:<14} alpha {fit.alpha[g]:+.3f}   beta {fit.beta[g]:+.3f}")

    print("\n  implied exchange rates (how many of X equal 1 of Y in preference):")
    for x in fit.groups:
        for y in fit.groups:
            if x == y:
                continue
            rate = fit.exchange_rate(x, y)
            print(f"    1 {y}  ~  {rate:.2f} {x}")


# ============================================================================
# Neutral-triple search
# ============================================================================


def neutral_triples(
    fit: LuceFit,
    n_values: List[int],
    n_triples: int,
    min_n: int = 1,
) -> List[Dict[str, int]]:
    """
    Search the full size grid for explicit per-group assignments whose
    *predicted* choice distribution is closest to uniform.

    Unlike ``ternary.sample_size_triples``, these are not rotated across
    groups afterward -- the whole point is that each group gets the specific
    size the fit says it needs, so rotating would destroy the calibration.

    Returns up to ``n_triples`` distinct assignments, most-neutral first,
    with basic diversity (no duplicate size vectors).
    """
    groups = fit.groups
    grid = [g for g in n_values if g >= min_n]

    candidates = []
    for a in grid:
        for b in grid:
            for c in grid:
                sizes = {groups[0]: a, groups[1]: b, groups[2]: c}
                probs = fit.predict(sizes)
                max_dev = max(abs(p - 1.0 / len(groups)) for p in probs.values())
                candidates.append((max_dev, sizes))

    candidates.sort(key=lambda x: x[0])

    out: List[Dict[str, int]] = []
    seen = set()
    for _, sizes in candidates:
        key = tuple(sorted(sizes.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(sizes)
        if len(out) >= n_triples:
            break

    return out


def summarize_neutral_triples(
    fit: LuceFit, triples: List[Dict[str, int]]
) -> pd.DataFrame:
    """Tidy table of candidate triples with their predicted probabilities."""
    rows = []
    for t in triples:
        probs = fit.predict(t)
        row = {f"n_{g}": t[g] for g in fit.groups}
        row.update({f"p_{g}": probs[g] for g in fit.groups})
        row["max_deviation"] = max(
            abs(p - 1.0 / len(fit.groups)) for p in probs.values()
        )
        rows.append(row)
    return pd.DataFrame(rows)
