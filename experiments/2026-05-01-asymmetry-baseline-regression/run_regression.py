"""
Exp 1: |asymmetry| on |baseline bias| regression.

Reads the per-condition CSVs produced by `choices.analysis.create_summary`
for trolley (clean+extra arxiv) and BBQ, computes |baseline_bias| and
|asym|, and fits a mixed-effects regression with random intercepts on
model, factor, and influence_type. Runs per-benchmark and pooled.

Outputs:
  data/per_condition.csv             — long-form per-condition table
  data/regression_results.json       — coefficients, R², CIs, sanity-check correlations
  figures/asym_vs_baseline_bias.png  — scatter with regression lines per benchmark
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

EXP_DIR = Path(__file__).parent
DATA_DIR = EXP_DIR / "data"
FIG_DIR = EXP_DIR / "figures"

INPUTS = {
    "trolley": DATA_DIR / "trolley_summary.csv",
    "bbq": DATA_DIR / "bbq_summary.csv",
}


def load_long_table() -> pd.DataFrame:
    frames = []
    for benchmark, path in INPUTS.items():
        df = pd.read_csv(path)
        df["benchmark"] = benchmark
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df = df[df["steerability_asym"].notna()].copy()
    df["abs_baseline_bias"] = (df["f_0_B"] - 0.5).abs()
    df["abs_asym"] = df["steerability_asym"].abs()
    df["abs_n_asym"] = df["normalized_steerability_asym"].abs()

    df["model_reasoning"] = df["model"] + "::" + df["reasoning_condition"].astype(str)

    return df


def fit_mixedlm(df: pd.DataFrame, label: str, include_benchmark_fe: bool) -> dict:
    """
    Mixed-effects: |asym| ~ |baseline_bias| (+ benchmark) + (1|model) + (1|factor) + (1|influence_type).

    statsmodels' MixedLM only supports a single grouping variable for random
    intercepts. We approximate the multi-random-intercept model by stacking
    random effects on a single composite group and using vc_formula for
    crossed random intercepts on factor and influence_type.
    """
    formula = "abs_asym ~ abs_baseline_bias"
    if include_benchmark_fe:
        formula += " + C(benchmark)"

    vc = {
        "factor": "0 + C(factor)",
        "influence": "0 + C(nudge_type)",
    }

    try:
        md = smf.mixedlm(
            formula,
            df,
            groups=df["model_reasoning"],
            vc_formula=vc,
            re_formula="1",
        )
        mdf = md.fit(method="lbfgs", reml=True)
    except Exception as e:
        return {"label": label, "n": int(len(df)), "error": str(e)}

    coef = mdf.params["abs_baseline_bias"]
    se = mdf.bse["abs_baseline_bias"]
    ci_lo, ci_hi = coef - 1.96 * se, coef + 1.96 * se

    # Marginal R²: variance explained by fixed effects / total variance.
    fitted_fixed = mdf.predict(exog=df)
    var_fe = float(np.var(fitted_fixed, ddof=0))
    var_random = (
        sum(float(v) for v in mdf.cov_re.values.diagonal() if np.isfinite(v))
        if mdf.cov_re is not None
        else 0.0
    )
    # Add variance from vc components
    if hasattr(mdf, "vcomp") and mdf.vcomp is not None:
        var_random += float(np.sum(mdf.vcomp))
    var_resid = float(mdf.scale)
    total_var = var_fe + var_random + var_resid
    r2_marginal = var_fe / total_var if total_var > 0 else float("nan")
    r2_conditional = (
        (var_fe + var_random) / total_var if total_var > 0 else float("nan")
    )

    # Also compute simple OLS R² for the fixed effect alone (sanity).
    ols_X = sm.add_constant(df[["abs_baseline_bias"]])
    ols_res = sm.OLS(df["abs_asym"], ols_X).fit()

    return {
        "label": label,
        "n": int(len(df)),
        "formula": formula,
        "coef_abs_baseline_bias": float(coef),
        "se_abs_baseline_bias": float(se),
        "ci95_abs_baseline_bias": [float(ci_lo), float(ci_hi)],
        "p_value": float(mdf.pvalues["abs_baseline_bias"]),
        "r2_marginal": float(r2_marginal),
        "r2_conditional": float(r2_conditional),
        "var_fixed": var_fe,
        "var_random": var_random,
        "var_resid": var_resid,
        "ols_r2": float(ols_res.rsquared),
        "ols_coef": float(ols_res.params["abs_baseline_bias"]),
        "ols_ci95": [
            float(ols_res.conf_int().loc["abs_baseline_bias", 0]),
            float(ols_res.conf_int().loc["abs_baseline_bias", 1]),
        ],
        "converged": bool(mdf.converged),
    }


def correlations(df: pd.DataFrame, label: str) -> dict:
    pearson = stats.pearsonr(df["abs_baseline_bias"], df["abs_asym"])
    spearman = stats.spearmanr(df["abs_baseline_bias"], df["abs_asym"])

    pearson_signed = stats.pearsonr(df["f_0_B"] - 0.5, df["steerability_asym"])
    spearman_signed = stats.spearmanr(df["f_0_B"] - 0.5, df["steerability_asym"])
    return {
        "label": label,
        "n": int(len(df)),
        "pearson_abs": [float(pearson.statistic), float(pearson.pvalue)],
        "spearman_abs": [float(spearman.statistic), float(spearman.pvalue)],
        "pearson_signed": [
            float(pearson_signed.statistic),
            float(pearson_signed.pvalue),
        ],
        "spearman_signed": [
            float(spearman_signed.statistic),
            float(spearman_signed.pvalue),
        ],
    }


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    palette = {"trolley": "#1f77b4", "bbq": "#d62728"}
    for benchmark, sub in df.groupby("benchmark"):
        ax.scatter(
            sub["abs_baseline_bias"],
            sub["abs_asym"],
            s=14,
            alpha=0.4,
            color=palette.get(benchmark, "gray"),
            label=f"{benchmark} (n={len(sub)})",
        )
        # OLS regression line per benchmark for a quick visual.
        if len(sub) >= 2:
            slope, intercept = np.polyfit(sub["abs_baseline_bias"], sub["abs_asym"], 1)
            xs = np.linspace(0, sub["abs_baseline_bias"].max(), 50)
            ax.plot(
                xs,
                intercept + slope * xs,
                color=palette.get(benchmark, "gray"),
                linewidth=1.5,
            )

    ax.set_xlabel("|baseline bias|  =  |f_0(B) − 0.5|")
    ax.set_ylabel("|asymmetry|  =  |s(B) − s(A)|")
    ax.set_title("Per-condition asymmetry vs. baseline bias")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    df = load_long_table()

    # Persist the long-form per-condition table.
    keep_cols = [
        "benchmark",
        "model",
        "reasoning_condition",
        "factor",
        "nudge_type",
        "level_A",
        "level_B",
        "n_comparisons",
        "f_0_B",
        "abs_baseline_bias",
        "steerability_A",
        "steerability_B",
        "steerability_asym",
        "abs_asym",
        "normalized_steerability_asym",
        "abs_n_asym",
        "sig_baseline_B",
        "sig_asym",
    ]
    df[keep_cols].to_csv(DATA_DIR / "per_condition.csv", index=False)

    results = {"regressions": [], "correlations": []}

    # Per-benchmark fits
    for benchmark, sub in df.groupby("benchmark"):
        results["regressions"].append(
            fit_mixedlm(sub, benchmark, include_benchmark_fe=False)
        )
        results["correlations"].append(correlations(sub, benchmark))

    # Pooled fit with benchmark as a fixed effect
    results["regressions"].append(fit_mixedlm(df, "pooled", include_benchmark_fe=True))
    results["correlations"].append(correlations(df, "pooled"))

    # BBQ-specific signed Pearson sanity check (target: r ≈ -0.425 from main.tex).
    bbq = df[df["benchmark"] == "bbq"]
    bbq_pearson = stats.pearsonr(bbq["f_0_B"] - 0.5, bbq["steerability_asym"])
    results["sanity_check_bbq_signed_pearson"] = {
        "r": float(bbq_pearson.statistic),
        "p": float(bbq_pearson.pvalue),
        "n": int(len(bbq)),
        "draft_value": -0.425,
    }

    out_path = DATA_DIR / "regression_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    fig_path = FIG_DIR / "asym_vs_baseline_bias.png"
    make_scatter(df, fig_path)
    print(f"Wrote {fig_path}")

    # Print headline numbers.
    print("\nHeadline numbers:")
    for r in results["regressions"]:
        if "error" in r:
            print(f"  {r['label']}: ERROR — {r['error']}")
            continue
        print(
            f"  {r['label']:8s}  n={r['n']:4d}  "
            f"β={r['coef_abs_baseline_bias']:+.3f}  "
            f"95% CI [{r['ci95_abs_baseline_bias'][0]:+.3f}, {r['ci95_abs_baseline_bias'][1]:+.3f}]  "
            f"R²(marg)={r['r2_marginal']:.3f}  R²(cond)={r['r2_conditional']:.3f}  "
            f"OLS-R²={r['ols_r2']:.3f}"
        )

    print("\nSanity correlations (|baseline_bias|, |asym|):")
    for c in results["correlations"]:
        print(
            f"  {c['label']:8s}  n={c['n']:4d}  "
            f"pearson(|·|)={c['pearson_abs'][0]:+.3f} (p={c['pearson_abs'][1]:.2g})  "
            f"spearman(|·|)={c['spearman_abs'][0]:+.3f}  "
            f"pearson(signed)={c['pearson_signed'][0]:+.3f}"
        )
    sc = results["sanity_check_bbq_signed_pearson"]
    print(
        f"\nBBQ signed pearson r(f_0(B)-0.5, asym) = {sc['r']:+.3f} "
        f"(draft cites {sc['draft_value']})"
    )


if __name__ == "__main__":
    main()
