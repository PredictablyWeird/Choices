"""
Robustness pass:
  (a) BH-FDR correction of asymmetry significance, within each benchmark
  (b) Cluster bootstrap CIs on headline (mean |Effect|), clustered on factor/value
  (c) Equivalence-margin sensitivity for "asymmetry beyond baseline"
  (d) Reconciliation of backfire definitions (esp. DailyDilemmas)
  (e) Cross-benchmark headline summary at multiple subset definitions

Reads:
  - per-benchmark summary CSVs (existing; have abs_effect, sig_*)
  - pvalues_by_condition.csv (computed by extract_pvalues.py for trolley + BBQ)

Writes:
  - robustness.json (full machine-readable output)
  - canonical_numbers_v2.json (paper-citable numbers)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path("/Users/alexmckenzie/code/values")
DATA = REPO / "Choices/experiments/2026-05-01-asymmetry-baseline-regression/data"
ANALYSIS = REPO / "Choices/experiments/2026-05-02-paper-artifacts/data"

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

ONE_SENTENCE_USER_MSG = {
    "emotional",
    "survey_preference",
    "user_preference",
    "virtue_appeal",
    "weak_evidence",
    "survey",
    "virtue",
}
SYSTEM_REPLACE = {"role_play"}
MULTI_DEMONSTRATION = {"few_shot", "few_shot_3", "few_shot_value"}


def load_summary(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}_summary.csv")
    df = df[df.model.isin(PAPER_MODELS)]
    if name == "trolley":
        df = df[df.factor.isin(PAPER_FACTORS_TROLLEY)]
    elif name == "dailydilemmas":
        df = df[df.factor != "dishonesty"]
    df = df.copy()
    df["benchmark"] = name
    return df


def load_pvalues() -> pd.DataFrame:
    """Exact Wald p-values for trolley + BBQ (computed from raw counts)."""
    return pd.read_csv(ANALYSIS / "pvalues_by_condition.csv")


def reconstruct_dd_pvalues(df_dd: pd.DataFrame) -> pd.DataFrame:
    """
    DailyDilemmas: reconstruct p-values from f_*_* and approximate counts.
    Each (model, value, nudge) condition has n_comparisons dilemmas where
    each dilemma had k=3 trials per direction. Total directed trials per
    condition ≈ n_comparisons * 3.
    """
    df = df_dd.copy()
    n = (df.n_comparisons.fillna(20).astype(int) * 3).values
    # Counts: f_0_B = c_0_B / (c_0_A + c_0_B) and similarly for f_A_B, f_B_B.
    # We assume balanced counts: c_0_A + c_0_B ≈ n.
    c0B = (df.f_0_B * n).round().astype(int).values
    c0A = (n - c0B).astype(int)
    cAB = (df.f_A_B * n).round().astype(int).values
    cAA = (n - cAB).astype(int)
    cBB = (df.f_B_B * n).round().astype(int).values
    cBA = (n - cBB).astype(int)

    # Wald p-value for asymmetry, mirrors metrics.wald_test_steerability_asym
    def a(x):
        return x + 0.5

    var = (
        1.0 / a(cAA)
        + 1.0 / a(cAB)
        + 1.0 / a(cBA)
        + 1.0 / a(cBB)
        + 2.0 * (1.0 / a(c0A) + 1.0 / a(c0B))
    )
    se = np.sqrt(var)
    asym = df.steerability_asym.values
    z = np.where(se > 0, np.abs(asym) / se, 0.0)
    p_asym = 2 * (1 - stats.norm.cdf(z))

    # Two-proportion z-test for effect (sig_A direction)
    def two_prop_p(c1, n1, c2, n2):
        with np.errstate(divide="ignore", invalid="ignore"):
            p1, p2 = c1 / np.maximum(n1, 1), c2 / np.maximum(n2, 1)
            ppool = (c1 + c2) / np.maximum(n1 + n2, 1)
            se_ = np.sqrt(
                np.maximum(
                    ppool
                    * (1 - ppool)
                    * (1.0 / np.maximum(n1, 1) + 1.0 / np.maximum(n2, 1)),
                    0,
                )
            )
            z_ = np.where(se_ > 0, (p1 - p2) / se_, 0)
            return 2 * (1 - stats.norm.cdf(np.abs(z_)))

    p_eff_A = two_prop_p(cAA, n, c0A, n)
    p_eff_B = two_prop_p(cBB, n, c0B, n)
    df["p_asym"] = p_asym
    df["p_eff_A"] = p_eff_A
    df["p_eff_B"] = p_eff_B
    return df


def cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Return (point, lo, hi) 95% cluster bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    grouped = {c: values[clusters == c] for c in unique}
    point = float(np.mean(values))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot[i] = np.concatenate([grouped[c] for c in sampled]).mean()
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def equivalence_neutrality_mask(f0: np.ndarray, n: np.ndarray, kind: str) -> np.ndarray:
    if kind == "binom_05":
        # not detectably different from 0.5 at alpha=.05 by binomial test
        c = np.round(f0 * n).astype(int)
        p = stats.binom.cdf(c, n, 0.5)
        # two-sided: p_two = 2*min(p, 1-p)
        p_two = 2 * np.minimum(p, 1 - p)
        return p_two > 0.05
    if kind == "binom_01":
        c = np.round(f0 * n).astype(int)
        p = stats.binom.cdf(c, n, 0.5)
        return 2 * np.minimum(p, 1 - p) > 0.01
    if kind == "margin_05":
        return np.abs(f0 - 0.5) < 0.05
    if kind == "margin_10":
        return np.abs(f0 - 0.5) < 0.10
    if kind == "ci_45_55":
        z = 1.96
        denom = 1 + z**2 / np.maximum(n, 1)
        center = (f0 + z**2 / (2 * np.maximum(n, 1))) / denom
        hw = (
            z
            * np.sqrt(
                np.maximum(
                    f0 * (1 - f0) / np.maximum(n, 1)
                    + z**2 / (4 * np.maximum(n, 1) ** 2),
                    0,
                )
            )
        ) / denom
        lo, hi = center - hw, center + hw
        return (lo >= 0.45) & (hi <= 0.55)
    raise ValueError(kind)


def main():
    summaries = {b: load_summary(b) for b in ["trolley", "bbq", "dailydilemmas"]}
    pvals = load_pvalues()
    out: dict = {"meta": {"source": str(DATA), "models": sorted(PAPER_MODELS)}}

    # ---- 1. Headlines under multiple subset definitions ----
    out["headlines"] = {}
    for bench, df in summaries.items():
        rec = {}
        for name, subset in [
            ("all_7", None),
            ("excl_few_shot", set(df.nudge_type.unique()) - MULTI_DEMONSTRATION),
            ("single_sentence_any", ONE_SENTENCE_USER_MSG | SYSTEM_REPLACE),
            ("single_sentence_user_only", ONE_SENTENCE_USER_MSG),
        ]:
            sub = df if subset is None else df[df.nudge_type.isin(subset)]
            point, lo, hi = cluster_bootstrap(sub.abs_effect.values, sub.factor.values)
            rec[name] = {
                "n": int(len(sub)),
                "mean_abs_effect": point,
                "boot95_lo": lo,
                "boot95_hi": hi,
            }
        out["headlines"][bench] = rec

    # ---- 2. Backfire rate definitions (reconciliation) ----
    out["backfire"] = {}
    for bench, df in summaries.items():
        # Build directed-pair long form
        long_a = df[["sig_A", "backfire_A"]].rename(
            columns={"sig_A": "sig", "backfire_A": "bf"}
        )
        long_b = df[["sig_B", "backfire_B"]].rename(
            columns={"sig_B": "sig", "backfire_B": "bf"}
        )
        long = pd.concat([long_a, long_b], ignore_index=True)
        long["sig"] = long["sig"].fillna(False).astype(bool)
        long["bf"] = long["bf"].fillna(False).astype(bool)

        n_pairs = len(long)
        n_sig = int(long.sig.sum())
        n_sig_bf = int((long.sig & long.bf).sum())
        n_bf_uncond = int(long.bf.sum())
        # Condition-level: condition counts as a sig backfire if either direction has both
        cond_sig_bf = (
            ((df.sig_A & df.backfire_A) | (df.sig_B & df.backfire_B))
            .fillna(False)
            .astype(bool)
        )
        cond_any_sig = (df.sig_A | df.sig_B).fillna(False).astype(bool)
        out["backfire"][bench] = {
            "n_directed_pairs": n_pairs,
            "n_sig_directed": n_sig,
            "rate_sig_bf_among_sig_directed": n_sig_bf / n_sig
            if n_sig
            else float("nan"),
            "rate_bf_unconditional_directed": n_bf_uncond / n_pairs,
            "n_conditions": int(len(df)),
            "n_cond_with_any_sig": int(cond_any_sig.sum()),
            "n_cond_with_sig_bf": int(cond_sig_bf.sum()),
            "rate_cond_sig_bf_total": float(cond_sig_bf.mean()),
            "rate_cond_sig_bf_among_sig": float(
                cond_sig_bf[cond_any_sig].mean() if cond_any_sig.any() else float("nan")
            ),
        }

    # ---- 3. Asymmetry: BH-FDR + equivalence-margin sensitivity ----
    out["asymmetry"] = {}

    # Build a pvalue-equipped frame per benchmark
    pframes: dict[str, pd.DataFrame] = {}
    pframes["trolley"] = pvals[pvals.benchmark == "trolley"].copy()
    pframes["bbq"] = pvals[pvals.benchmark == "bbq"].copy()
    pframes["dailydilemmas"] = reconstruct_dd_pvalues(summaries["dailydilemmas"])

    # Need f_0_B and n for neutrality masks; merge from summary CSVs
    for bench in ["trolley", "bbq"]:
        s = summaries[bench]
        pf = pframes[bench]
        merge_cols = ["model", "factor", "nudge_type"]
        pf = pf.merge(
            s[merge_cols + ["f_0_B", "n_comparisons"]], on=merge_cols, how="left"
        )
        pframes[bench] = pf

    for bench, pf in pframes.items():
        if pf.empty:
            continue

        n_total_trials = pf["n_comparisons"].fillna(100).astype(int).values
        if bench == "trolley" or bench == "bbq":
            n_total_trials = n_total_trials * 8  # ~8 trials per edge
        elif bench == "dailydilemmas":
            n_total_trials = n_total_trials * 3  # k=3 per dilemma direction

        f0 = pf["f_0_B"].fillna(0.5).values
        p_asym = pf["p_asym"].fillna(1.0).values

        # BH-FDR within benchmark
        sig_bh, _, _, _ = multipletests(p_asym, alpha=0.05, method="fdr_bh")
        sig_alpha05 = p_asym < 0.05
        sig_alpha01 = p_asym < 0.01

        rec: dict = {
            "n_conditions": int(len(pf)),
            "rate_sig_alpha05": float(sig_alpha05.mean()),
            "rate_sig_alpha01": float(sig_alpha01.mean()),
            "rate_sig_bh_fdr05": float(sig_bh.mean()),
            "by_neutrality": {},
        }
        for kind in ["binom_05", "binom_01", "margin_05", "margin_10", "ci_45_55"]:
            mask = equivalence_neutrality_mask(f0, n_total_trials, kind)
            n_neutral = int(mask.sum())
            if n_neutral == 0:
                rec["by_neutrality"][kind] = {
                    "n_neutral": 0,
                    "rate_alpha05": float("nan"),
                    "rate_alpha01": float("nan"),
                    "rate_bh_fdr05": float("nan"),
                }
                continue
            rec["by_neutrality"][kind] = {
                "n_neutral": n_neutral,
                "rate_alpha05": float(sig_alpha05[mask].mean()),
                "rate_alpha01": float(sig_alpha01[mask].mean()),
                "rate_bh_fdr05": float(sig_bh[mask].mean()),
            }
        out["asymmetry"][bench] = rec

    # ---- 4. Sanity check sig_asym in our extracted vs CSV ----
    out["sanity"] = {}
    for bench in ["trolley", "bbq"]:
        s = summaries[bench][["model", "factor", "nudge_type", "sig_asym"]].copy()
        s["sig_csv"] = s["sig_asym"].fillna(False).astype(bool)
        pf = pframes[bench][["model", "factor", "nudge_type", "p_asym"]].copy()
        pf["sig_pf"] = (pf["p_asym"] < 0.05).astype(bool)
        merged = pf.merge(
            s[["model", "factor", "nudge_type", "sig_csv"]],
            on=["model", "factor", "nudge_type"],
            how="inner",
        )
        out["sanity"][bench] = {
            "agreement_with_csv_sig_asym": float(
                (merged.sig_pf == merged.sig_csv).mean()
            )
            if len(merged)
            else float("nan"),
            "n_compared": int(len(merged)),
        }

    # Write JSON
    with open(ANALYSIS / "robustness.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Pretty print
    print("=" * 70)
    print("ROBUSTNESS PASS — paper-citable numbers")
    print("=" * 70)
    for bench in ["trolley", "bbq", "dailydilemmas"]:
        print(f"\n=== {bench.upper()} ===")
        h = out["headlines"][bench]
        for name in [
            "all_7",
            "excl_few_shot",
            "single_sentence_any",
            "single_sentence_user_only",
        ]:
            r = h[name]
            print(
                f"  headline {name:28s}: {r['mean_abs_effect']*100:5.1f}pp  "
                f"[{r['boot95_lo']*100:.1f}, {r['boot95_hi']*100:.1f}]  n={r['n']}"
            )
        bf = out["backfire"][bench]
        print(
            f"  backfire (sig-effect denom):   {bf['rate_sig_bf_among_sig_directed']*100:5.1f}%  "
            f"of {bf['n_sig_directed']} sig directed effects"
        )
        print(
            f"  backfire (cond w/ any sig):    {bf['rate_cond_sig_bf_among_sig']*100:5.1f}%  "
            f"of {bf['n_cond_with_any_sig']} conditions w/ any sig effect"
        )
        print(
            f"  backfire (uncond conditions):  {bf['rate_cond_sig_bf_total']*100:5.1f}%  "
            f"of {bf['n_conditions']} conditions"
        )
        if bench in out["asymmetry"]:
            a = out["asymmetry"][bench]
            print(
                f"  asym sig rate:  alpha=.05 {a['rate_sig_alpha05']*100:.1f}% | "
                f"alpha=.01 {a['rate_sig_alpha01']*100:.1f}% | "
                f"BH-FDR q=.05 {a['rate_sig_bh_fdr05']*100:.1f}%"
            )
            for kind, r in a["by_neutrality"].items():
                if r["n_neutral"] == 0:
                    continue
                print(
                    f"    neutral={kind:10s} (n={r['n_neutral']:3d}): "
                    f"alpha=.05 {r['rate_alpha05']*100:5.1f}% | "
                    f"BH-FDR {r['rate_bh_fdr05']*100:5.1f}%"
                )
    if out["sanity"]:
        print("\nSanity (agreement of extracted p-values with CSV's sig_asym):")
        for bench, r in out["sanity"].items():
            print(
                f"  {bench}: {r['agreement_with_csv_sig_asym']*100:.1f}% on {r['n_compared']} conditions"
            )


if __name__ == "__main__":
    main()
