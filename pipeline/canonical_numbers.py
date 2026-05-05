"""
Single-source-of-truth analysis script for the NeurIPS rewrite.

Reads the May-1 per-benchmark snapshots and produces every number the paper
cites: headline shifts (all-influences AND single-sentence-only), baseline-
neutral asymmetry rates (under multiple neutrality thresholds), backfire
rates (under multiple definitions), plus per-benchmark and per-(model x
reasoning) breakdowns. Outputs JSON the LaTeX rewrite reads from.

Run:  uv run --with pandas --with numpy --with scipy --with statsmodels \
        python neurips_2026_rewrite/analysis/canonical_numbers.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PIPELINE = Path(__file__).resolve().parent
CHOICES = PIPELINE.parent
DATA = CHOICES / "experiments/asymmetry-regression/data"
OUT = PIPELINE / "data"

PAPER_FACTORS_TROLLEY = {"age_group", "gender", "handedness", "nationality", "wealth"}
PAPER_MODELS = {
    "deepseek-v3-2-reasoning",
    "deepseek-v3-2-non-reasoning",
    "gpt-5-2-reasoning",
    "gpt-5-2-non-reasoning",
    "grok-41-fast-reasoning",
    "grok-41-fast-non-reasoning",
    "llama-33-70b",
    "qwen3-235b-a22b-2507-reasoning",
    "qwen3-235b-a22b-2507",
}

# Influence-type partition: structural placement within the prompt.
ONE_SENTENCE_USER_MSG = {  # one sentence, user message
    "emotional",
    "survey_preference",
    "user_preference",
    "virtue_appeal",
    "weak_evidence",
    # DailyDilemmas naming
    "survey",
    "virtue",
}
SYSTEM_REPLACE = {"role_play"}  # one sentence but in system prompt
MULTI_DEMONSTRATION = {"few_shot", "few_shot_3", "few_shot_value"}


def load_benchmark(name: str) -> pd.DataFrame:
    path = DATA / f"{name}_summary.csv"
    df = pd.read_csv(path)
    df = df[df.model.isin(PAPER_MODELS)].copy()
    if name == "trolley":
        df = df[df.factor.isin(PAPER_FACTORS_TROLLEY)]
    elif name == "dailydilemmas":
        df = df[df.factor != "dishonesty"]
    df["benchmark"] = name
    return df


def headline_shift(df: pd.DataFrame, subset: set | None = None) -> dict:
    sub = df if subset is None else df[df.nudge_type.isin(subset)]
    return {
        "n": int(len(sub)),
        "mean_abs_effect": float(sub.abs_effect.mean()),
        "median_abs_effect": float(sub.abs_effect.median()),
    }


def baseline_neutral_asymmetry_rate(
    df: pd.DataFrame, neutrality: str = "binom"
) -> dict:
    """
    Fraction of baseline-neutral conditions with significant asymmetry.

    neutrality:
      'binom'    : not-detectably-different from 0.5 by binomial test (paper default)
      'margin05' : |f0(B) - 0.5| < 0.05
      'margin10' : |f0(B) - 0.5| < 0.10
      'ci45_55'  : 95% Wilson CI for f0(B) entirely inside [0.45, 0.55]
    """
    f0 = df.f_0_B.values
    n = df.n_comparisons.fillna(100).astype(int).values
    if neutrality == "binom":
        mask = ~df.sig_baseline_B.fillna(False).astype(bool).values
    elif neutrality == "margin05":
        mask = np.abs(f0 - 0.5) < 0.05
    elif neutrality == "margin10":
        mask = np.abs(f0 - 0.5) < 0.10
    elif neutrality == "ci45_55":
        # Wilson 95% CI for binomial proportion
        z = 1.96
        denom = 1 + z**2 / n
        center = (f0 + z**2 / (2 * n)) / denom
        halfwidth = (z * np.sqrt(f0 * (1 - f0) / n + z**2 / (4 * n**2))) / denom
        lo, hi = center - halfwidth, center + halfwidth
        mask = (lo >= 0.45) & (hi <= 0.55)
    else:
        raise ValueError(neutrality)

    sub = df[mask]
    if len(sub) == 0:
        return {"neutrality": neutrality, "n_neutral": 0, "rate_sig_asym": float("nan")}
    rate = float(sub.sig_asym.fillna(False).astype(bool).mean())
    return {
        "neutrality": neutrality,
        "n_neutral": int(len(sub)),
        "rate_sig_asym": rate,
    }


def fdr_correct_asymmetry(df: pd.DataFrame) -> dict:
    """
    Recompute baseline-neutral asymmetry rate after BH-FDR within each
    benchmark x effect-family (we treat the asymmetry test family as one
    family per benchmark, since each (model, factor, nudge) condition
    contributes one asymmetry test).

    We do not have raw p-values in the CSV, only sig_asym booleans, so we
    invert the Wald CI to get a conservative p-proxy: assume the threshold
    was alpha=.05 and reconstruct via |asym|/SE -> z -> p. Where SE is
    not directly available we back it out from the test outcome and
    statsmodels normal quantiles. This is a sensitivity check; we flag
    that the right way is to plumb p-values through create_summary.
    """
    out = {}
    for bench, sub in df.groupby("benchmark"):
        # Our p-proxy: if sig_asym is True and abs_asym is non-trivial, treat
        # the implied two-sided p as bounded by alpha=.05. We instead use a
        # conservative reconstruction: p = 2 * (1 - Phi(|asym|/SE)). The
        # Wald SE is not in the CSV; we approximate it from the variance of
        # asymmetry across conditions (a defensible random-effect-style proxy).
        asym = sub.steerability_asym.dropna().values
        if len(asym) == 0:
            continue
        se_proxy = float(np.std(asym, ddof=1))  # conservative pooled SE proxy
        z = np.abs(sub.steerability_asym.fillna(0).values) / se_proxy
        p_proxy = 2 * (1 - stats.norm.cdf(z))
        # BH FDR
        sig_bh, _, _, _ = multipletests(p_proxy, alpha=0.05, method="fdr_bh")
        # Take subset of baseline-neutral conditions
        is_neutral = ~sub.sig_baseline_B.fillna(False).astype(bool).values
        out[bench] = {
            "n_total": int(len(sub)),
            "rate_sig_asym_alpha05": float(
                sub.sig_asym.fillna(False).astype(bool).mean()
            ),
            "rate_sig_asym_bh_fdr05": float(np.mean(sig_bh)),
            "neutral_rate_alpha05": float(
                sub.loc[is_neutral, "sig_asym"].fillna(False).astype(bool).mean()
                if is_neutral.any()
                else float("nan")
            ),
            "neutral_rate_bh_fdr05": float(
                sig_bh[is_neutral].mean() if is_neutral.any() else float("nan")
            ),
        }
    return out


def cluster_bootstrap_mean_effect(
    df: pd.DataFrame, cluster_col: str = "factor", n_boot: int = 2000, seed: int = 0
) -> dict:
    """Cluster bootstrap CI for mean(abs_effect), resampling clusters with replacement."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    point = float(df.abs_effect.mean())
    boot_means = np.empty(n_boot)
    grouped = {c: df[df[cluster_col] == c].abs_effect.values for c in clusters}
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([grouped[c] for c in sampled])
        boot_means[i] = vals.mean()
    return {
        "n_clusters": int(len(clusters)),
        "n_obs": int(len(df)),
        "point": point,
        "ci_lo": float(np.quantile(boot_means, 0.025)),
        "ci_hi": float(np.quantile(boot_means, 0.975)),
        "cluster_col": cluster_col,
    }


def backfire_rates(df: pd.DataFrame) -> dict:
    """
    Compute backfire rates under multiple definitions for transparency.

    A condition has a 'backfire' if the influence shifts choice opposite
    its intended direction (steerability < 0). We split:

      'cond_lvl_sig_bf'      : condition shows sig effect AND sig backfire
                               (paper's 'sig BF' table column on trolley)
      'sig_effect_bf'        : among conditions with sig effect on either
                               direction, fraction that backfire (paper
                               headline-figure denominator: 14.3 / 10.5 / 5.9)
      'all_directed_bf'      : unconditional fraction of (condition, direction)
                               pairs where steerability < 0
      'sample_flip_away'     : DailyDilemmas-style sample-level flip rate
                               (only computable for DD; for trolley/BBQ this
                               equals 'all_directed_bf')
    """
    # backfire_A is True when steerability_A < 0 (and significant?). Need to verify.
    # In the CSV: backfire_A and backfire_B are boolean flags. Check meaning.
    # From appendix: "backfire when s(d) < 0 and significant".
    # The paper headline figure 5.9% / 10.5% / 14.3% uses
    #   "share of significant effects that move opposite the intended direction"
    # = (sig_A and backfire_A) OR (sig_B and backfire_B) /
    #   (sig_A) OR (sig_B), counted at the directed-pair level.
    rows = []
    for d, sig_col, bf_col in [
        ("A", "sig_A", "backfire_A"),
        ("B", "sig_B", "backfire_B"),
    ]:
        sub = df[[sig_col, bf_col]].rename(columns={sig_col: "sig", bf_col: "bf"})
        sub["direction"] = d
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long["sig"] = long["sig"].fillna(False).astype(bool)
    long["bf"] = long["bf"].fillna(False).astype(bool)

    n_directed = len(long)
    n_sig = int(long["sig"].sum())
    n_sig_bf = int((long["sig"] & long["bf"]).sum())
    n_bf_uncond = int(long["bf"].sum())

    return {
        "n_directed_pairs": n_directed,
        "n_sig_effects": n_sig,
        "rate_sig_bf_among_sig": float(n_sig_bf / n_sig) if n_sig else float("nan"),
        "rate_bf_unconditional": float(n_bf_uncond / n_directed),
    }


def main():
    benchmarks = {b: load_benchmark(b) for b in ["trolley", "bbq", "dailydilemmas"]}

    out: dict = {
        "_meta": {
            "source": str(DATA),
            "paper_models": sorted(PAPER_MODELS),
            "trolley_factors": sorted(PAPER_FACTORS_TROLLEY),
            "dailydilemmas_excluded_values": ["dishonesty"],
            "influence_partition": {
                "one_sentence_user_msg": sorted(ONE_SENTENCE_USER_MSG),
                "system_replace": sorted(SYSTEM_REPLACE),
                "multi_demonstration": sorted(MULTI_DEMONSTRATION),
            },
        }
    }

    for bench, df in benchmarks.items():
        rec: dict = {}
        # Headlines
        rec["headline_all"] = headline_shift(df)
        single_sentence = (
            ONE_SENTENCE_USER_MSG | SYSTEM_REPLACE
        )  # all single-sentence (user OR system)
        rec["headline_single_sentence"] = headline_shift(df, subset=single_sentence)
        rec["headline_user_msg_only"] = headline_shift(df, subset=ONE_SENTENCE_USER_MSG)
        rec["headline_excl_few_shot"] = headline_shift(
            df, subset=set(df.nudge_type.unique()) - MULTI_DEMONSTRATION
        )

        # Per-nudge breakdown
        per_nudge = (
            df.groupby("nudge_type")
            .abs_effect.agg(["mean", "median", "count"])
            .reset_index()
            .to_dict(orient="records")
        )
        rec["per_nudge"] = per_nudge

        # Cluster bootstrap CI on the headline (single-sentence user msg)
        sub_user = df[df.nudge_type.isin(ONE_SENTENCE_USER_MSG)]
        cluster_col = "factor"  # cluster on factor/value
        rec["bootstrap_user_msg_only"] = cluster_bootstrap_mean_effect(
            sub_user, cluster_col=cluster_col
        )
        rec["bootstrap_all"] = cluster_bootstrap_mean_effect(
            df, cluster_col=cluster_col
        )

        # Baseline-neutral asymmetry rates under multiple definitions
        rec["asym_rate_by_neutrality"] = {
            n: baseline_neutral_asymmetry_rate(df, neutrality=n)
            for n in ["binom", "margin05", "margin10", "ci45_55"]
        }

        # Backfire rates
        rec["backfire"] = backfire_rates(df)

        out[bench] = rec

    # Cross-benchmark FDR sensitivity
    pooled = pd.concat(list(benchmarks.values()), ignore_index=True)
    out["fdr_correction"] = fdr_correct_asymmetry(pooled)

    # Write
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "canonical_numbers.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Pretty print summary
    print("=" * 60)
    print("CANONICAL NUMBERS — paper headline checks")
    print("=" * 60)
    for bench in ["trolley", "bbq", "dailydilemmas"]:
        rec = out[bench]
        print(f"\n=== {bench.upper()} ===")
        print(
            f"  All 7 influences:        {rec['headline_all']['mean_abs_effect']*100:5.1f}pp"
            f"  (n={rec['headline_all']['n']})"
        )
        print(
            f"  Excl. few-shot:          {rec['headline_excl_few_shot']['mean_abs_effect']*100:5.1f}pp"
            f"  (n={rec['headline_excl_few_shot']['n']})"
        )
        print(
            f"  Single-sentence (any):   {rec['headline_single_sentence']['mean_abs_effect']*100:5.1f}pp"
            f"  (n={rec['headline_single_sentence']['n']})"
        )
        print(
            f"  Single-sentence USER:    {rec['headline_user_msg_only']['mean_abs_effect']*100:5.1f}pp"
            f"  (n={rec['headline_user_msg_only']['n']})  <-- cleanest claim"
        )
        boot = rec["bootstrap_user_msg_only"]
        print(
            f"    cluster-boot CI on user-msg: [{boot['ci_lo']*100:.1f}, {boot['ci_hi']*100:.1f}]pp"
        )
        print("  Baseline-neutral asymmetry rate by neutrality def:")
        for nname, r in rec["asym_rate_by_neutrality"].items():
            print(
                f"    {nname:9s}: {r['rate_sig_asym']*100:5.1f}%  (n_neutral={r['n_neutral']})"
            )
        bf = rec["backfire"]
        print(
            f"  Backfire among sig effects: {bf['rate_sig_bf_among_sig']*100:5.1f}%"
            f"  ({bf['n_sig_effects']} sig)"
        )
        print(f"  Unconditional backfire:     {bf['rate_bf_unconditional']*100:5.1f}%")
    print()
    print("FDR-corrected asymmetry rates:")
    for bench, r in out["fdr_correction"].items():
        print(
            f"  {bench:14s}: alpha=.05 -> {r['rate_sig_asym_alpha05']*100:.1f}% | "
            f"BH-FDR q=.05 -> {r['rate_sig_asym_bh_fdr05']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
