"""
Single-source-of-truth pipeline for the NeurIPS 2026 paper artifacts.

Computes every numeric claim cited by `main_phil.tex` plus a first pass at
the cross-benchmark headline figure, and writes them to one JSON the
paper rewrite can read from.

Why this exists:
  1. Paper had several internal number drifts (DailyDilemmas backfire 5.9%
     in headline vs 0.2% in appendix; trolley backfire 14.3% vs 17.7%
     when re-derived; DailyDilemmas headline 9pp vs 12.3pp from current
     data) caught by claim_verification.md and the 2026-05-02 rewrite.
  2. ChatGPT-style adversarial review (chatgpt_analysis.md in the latex
     repo) flagged that the paper's significance machinery was too simple
     for the design — IID Bernoulli z-tests across 1,498 conditions,
     no FDR, no cluster bootstrap.
  3. The existing analysis is split across `choices.analysis.create_summary`
     and several `experiments/2026-05-01-*/` directories with different
     output formats. There was no single place to look up "what does the
     paper claim and what number does the data support?".

This script:
  - Loads the per-benchmark summary CSVs maintained by
    `experiments/2026-05-01-asymmetry-baseline-regression/`.
  - Restricts to the paper's 9 model configs and 5 trolley factors;
    excludes DailyDilemmas's saturated `dishonesty` value.
  - Computes everything the paper cites: headline shifts under multiple
    subset definitions (all 7 / excl few-shot / single-sentence
    user-msg-only); cluster-bootstrap CIs over the factor/value axis;
    backfire rates under four definitions; baseline-undetected asymmetry
    rates under three neutrality definitions × three significance
    thresholds (with BH-FDR within each benchmark); per-benchmark
    breakdowns that map onto the appendix tables; pulls in the noise-
    floor and follow-up-probe headline numbers from sibling experiment
    dirs.

Outputs:
  data/paper_numbers.json    — one JSON, organized by paper section
  data/pvalues_by_condition.csv — exact Wald p-values per condition
                                  (needed for FDR; trolley + BBQ only)
  figures/cross_benchmark.png — first-pass headline figure

Run:
  cd Choices
  uv run --with pandas --with numpy --with scipy --with statsmodels \
         --with matplotlib python \
         experiments/2026-05-02-paper-artifacts/produce_paper_artifacts.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------- repository layout ------------------------------------------------

THIS = Path(__file__).resolve()
EXP_DIR = THIS.parent
DATA_OUT = EXP_DIR / "data"
FIG_OUT = EXP_DIR / "figures"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

CHOICES = EXP_DIR.parent.parent  # Choices/
REPO = CHOICES.parent  # values/

ASYM_DIR = CHOICES / "experiments/2026-05-01-asymmetry-baseline-regression"
NOISE_DIR = CHOICES / "experiments/2026-05-01-baseline-noise"
PROBE_DIR = CHOICES / "experiments/2026-05-01-followup-probe"
SUMMARY_DIR = ASYM_DIR / "data"

DATA_ROOTS = {
    "trolley": REPO / "moral-steerability-paper/google_drive/results_clean_arxiv",
    "bbq": REPO / "moral-steerability-paper/google_drive/results_bbq_v2",
    "dailydilemmas": REPO
    / "moral-steerability-paper/google_drive/results_dailydilemmas",
}

# ---------- paper scope ------------------------------------------------------

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

# Influence-type partition. These names cover trolley + BBQ + DailyDilemmas
# variants; DD uses survey/virtue/few_shot_value where trolley uses
# survey_preference/virtue_appeal/few_shot_3.
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

# ---------- helpers ----------------------------------------------------------


def _load_summary(name: str) -> pd.DataFrame:
    """Per-benchmark CSV restricted to the paper's model/factor scope."""
    df = pd.read_csv(SUMMARY_DIR / f"{name}_summary.csv")
    df = df[df.model.isin(PAPER_MODELS)]
    if name == "trolley":
        df = df[df.factor.isin(PAPER_FACTORS_TROLLEY)]
    elif name == "dailydilemmas":
        df = df[df.factor != "dishonesty"]
    df = df.copy()
    df["benchmark"] = name
    return df


def _cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """95% cluster-bootstrap CI for the mean of `values`, clustering on `clusters`."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    grouped = {c: values[clusters == c] for c in unique}
    point = float(np.mean(values))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot[i] = np.concatenate([grouped[c] for c in sampled]).mean()
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _wald_p_asym(c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B, asym):
    """Mirror choices.analysis.metrics.wald_test_steerability_asym."""

    def a(x):
        return x + 0.5

    var = (
        1.0 / a(c_A_A)
        + 1.0 / a(c_A_B)
        + 1.0 / a(c_B_A)
        + 1.0 / a(c_B_B)
        + 2 * (1.0 / a(c_0_A) + 1.0 / a(c_0_B))
    )
    se = math.sqrt(var) if var > 0 else 0.0
    if se <= 0:
        return 1.0, 0.0
    p = 2 * (1 - stats.norm.cdf(abs(asym / se)))
    return p, se


def _equivalence_neutrality_mask(
    f0: np.ndarray, n: np.ndarray, kind: str
) -> np.ndarray:
    if kind == "binom_05":
        c = np.round(f0 * n).astype(int)
        p = stats.binom.cdf(c, n, 0.5)
        return 2 * np.minimum(p, 1 - p) > 0.05
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


# ---------- p-value extraction (trolley + BBQ from raw counts) ---------------


def _find_graphs(root: Path):
    """Walk the simple_<factor>/<model>/<nudge>/<run>/preference_graph_*.json layout."""
    if not root.exists():
        return
    for factor_dir in root.glob("simple_*"):
        factor = factor_dir.name.replace("simple_", "")
        for model_dir in factor_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for nudge_dir in model_dir.iterdir():
                if not nudge_dir.is_dir():
                    continue
                nudge = nudge_dir.name
                for run_dir in nudge_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    parts = run_dir.name.split("_")
                    target = "_".join(parts[2:]) if len(parts) >= 3 else parts[-1]
                    for g in run_dir.glob("preference_graph_*.json"):
                        yield model, factor, nudge, target, g


def _counts_for_factor(graph: dict, factor_attr: str, level_A: str, level_B: str):
    options_by_id = {opt["id"]: opt for opt in graph.get("options", [])}
    wins_A = wins_B = total = 0
    for edge_key, edge_data in graph.get("edges", {}).items():
        try:
            ids = eval(edge_key)
            opt_a, opt_b = options_by_id.get(ids[0]), options_by_id.get(ids[1])
            if not opt_a or not opt_b:
                continue
            la, lb = opt_a.get(factor_attr), opt_b.get(factor_attr)
            if la == lb or la not in (level_A, level_B) or lb not in (level_A, level_B):
                continue
            aux = edge_data.get("aux_data", {})
            for resp in aux.get("original_parsed", []):
                if resp == "A":
                    if la == level_A:
                        wins_A += 1
                    elif la == level_B:
                        wins_B += 1
                    total += 1
                elif resp == "B":
                    if lb == level_A:
                        wins_A += 1
                    elif lb == level_B:
                        wins_B += 1
                    total += 1
            for resp in aux.get("flipped_parsed", []):
                if resp == "A":
                    if lb == level_A:
                        wins_A += 1
                    elif lb == level_B:
                        wins_B += 1
                    total += 1
                elif resp == "B":
                    if la == level_A:
                        wins_A += 1
                    elif la == level_B:
                        wins_B += 1
                    total += 1
        except Exception:
            continue
    return wins_A, wins_B, total


_TROLLEY_LEVELS = {
    "age_group": ("young", "old"),
    "gender": ("male", "female"),
    "handedness": ("left-handed", "right-handed"),
    "nationality": ("American", "Nigerian"),
    "wealth": ("poor", "rich"),
    "diet": ("vegan", "carnivore"),
    "tech_view": ("luddite", "technologist"),
}


def _bbq_levels(factor: str) -> tuple[str, str]:
    if factor.startswith("age"):
        return ("nonOld", "old")
    if factor.startswith("ses"):
        return ("highSES", "lowSES")
    raise ValueError(factor)


def _extract_pvalues_for(name: str) -> pd.DataFrame:
    """Compute exact Wald p-values for trolley and BBQ from preference graphs.

    DailyDilemmas uses a different format and is approximated downstream.
    """
    if name not in ("trolley", "bbq"):
        return pd.DataFrame()
    root = DATA_ROOTS[name]

    def _identity(f):
        return f

    if name == "trolley":

        def levels_for(f):
            return _TROLLEY_LEVELS[f]

        attr_for = _identity
    else:
        levels_for = _bbq_levels
        attr_for = _identity  # BBQ stores the polarity-suffixed name as attr

    by_cond: dict[tuple, dict] = defaultdict(dict)
    for model, factor, nudge, target, gpath in _find_graphs(root):
        try:
            la, lb = levels_for(factor)
        except (KeyError, ValueError):
            continue
        with open(gpath) as f:
            graph = json.load(f)
        c_A, c_B, n_total = _counts_for_factor(graph, attr_for(factor), la, lb)
        by_cond[(model, factor, nudge)][target] = (c_A, c_B, n_total)

    rows = []
    for (model, factor, nudge), targets in by_cond.items():
        try:
            la, lb = levels_for(factor)
        except (KeyError, ValueError):
            continue
        base = targets.get("base")
        nudge_A = nudge_B = None
        for tgt, vals in targets.items():
            if tgt == "base":
                continue
            t_norm = tgt.lower().replace("-", "")
            if t_norm == la.lower().replace("-", ""):
                nudge_A = vals
            elif t_norm == lb.lower().replace("-", ""):
                nudge_B = vals
        if base is None or nudge_A is None or nudge_B is None:
            continue
        c_0_A, c_0_B, _ = base
        c_A_A, c_A_B, _ = nudge_A
        c_B_A, c_B_B, _ = nudge_B

        def a(x):
            return x + 0.5

        s_A = math.log(a(c_A_A) / a(c_A_B)) - math.log(a(c_0_A) / a(c_0_B))
        s_B = math.log(a(c_B_B) / a(c_B_A)) - math.log(a(c_0_B) / a(c_0_A))
        asym = s_B - s_A
        p_asym, _ = _wald_p_asym(c_0_A, c_0_B, c_A_A, c_A_B, c_B_A, c_B_B, asym)
        rows.append(
            {
                "benchmark": name,
                "model": model,
                "factor": factor,
                "nudge_type": nudge,
                "level_A": la,
                "level_B": lb,
                "c_0_A": c_0_A,
                "c_0_B": c_0_B,
                "c_A_A": c_A_A,
                "c_A_B": c_A_B,
                "c_B_A": c_B_A,
                "c_B_B": c_B_B,
                "steerability_A": s_A,
                "steerability_B": s_B,
                "steerability_asym": asym,
                "p_asym": p_asym,
            }
        )
    return pd.DataFrame(rows)


def _reconstruct_dd_pvalues(df_dd: pd.DataFrame) -> pd.DataFrame:
    """Approximate DailyDilemmas Wald p-values from f_*_* and n_comparisons.

    DD doesn't use the preference_graph format the trolley/BBQ harnesses do,
    so we back-reconstruct counts assuming each (model, value, nudge,
    direction) condition has n_comparisons * 3 trials (k=3 per dilemma).
    """
    df = df_dd.copy()
    n = (df.n_comparisons.fillna(20).astype(int) * 3).values
    c0B = (df.f_0_B * n).round().astype(int).values
    c0A = (n - c0B).astype(int)
    cAB = (df.f_A_B * n).round().astype(int).values
    cAA = (n - cAB).astype(int)
    cBB = (df.f_B_B * n).round().astype(int).values
    cBA = (n - cBB).astype(int)

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
    df["p_asym"] = 2 * (1 - stats.norm.cdf(z))
    return df


# ---------- per-section computations ----------------------------------------


def _headline_table(summaries: dict[str, pd.DataFrame]) -> dict:
    """The cross-benchmark headline numbers, per benchmark."""
    out = {}
    for bench, df in summaries.items():
        rec = {}
        # Headlines under several subset definitions
        for name, subset in [
            ("all_7_influences", None),
            ("excl_few_shot", set(df.nudge_type.unique()) - MULTI_DEMONSTRATION),
            ("single_sentence_any", ONE_SENTENCE_USER_MSG | SYSTEM_REPLACE),
            ("single_sentence_user_msg_only", ONE_SENTENCE_USER_MSG),
        ]:
            sub = df if subset is None else df[df.nudge_type.isin(subset)]
            if len(sub) == 0:
                continue
            point, lo, hi = _cluster_bootstrap(sub.abs_effect.values, sub.factor.values)
            rec[name] = {
                "n_conditions": int(len(sub)),
                "mean_abs_effect": point,
                "boot95_lo": lo,
                "boot95_hi": hi,
            }
        out[bench] = rec
    return out


def _backfire_definitions(summaries: dict[str, pd.DataFrame]) -> dict:
    """Four definitions of the backfire rate, per benchmark, for reconciliation."""
    out = {}
    for bench, df in summaries.items():
        long_a = df[["sig_A", "backfire_A"]].rename(
            columns={"sig_A": "sig", "backfire_A": "bf"}
        )
        long_b = df[["sig_B", "backfire_B"]].rename(
            columns={"sig_B": "sig", "backfire_B": "bf"}
        )
        long = pd.concat([long_a, long_b], ignore_index=True)
        long["sig"] = long["sig"].fillna(False).astype(bool)
        long["bf"] = long["bf"].fillna(False).astype(bool)
        cond_sig_bf = (
            ((df.sig_A & df.backfire_A) | (df.sig_B & df.backfire_B))
            .fillna(False)
            .astype(bool)
        )
        cond_any_sig = (df.sig_A | df.sig_B).fillna(False).astype(bool)
        n_pairs = len(long)
        n_sig = int(long.sig.sum())
        n_sig_bf = int((long.sig & long.bf).sum())
        out[bench] = {
            "n_directed_pairs": n_pairs,
            "n_sig_directed": n_sig,
            "n_sig_bf_directed": n_sig_bf,
            "rate_sig_bf_among_sig_directed": n_sig_bf / n_sig
            if n_sig
            else float("nan"),
            "rate_bf_unconditional_directed": int(long.bf.sum()) / n_pairs,
            "n_conditions": int(len(df)),
            "n_cond_with_any_sig": int(cond_any_sig.sum()),
            "n_cond_with_sig_bf": int(cond_sig_bf.sum()),
            "rate_cond_sig_bf_total": float(cond_sig_bf.mean()),
            "rate_cond_sig_bf_among_sig": float(
                cond_sig_bf[cond_any_sig].mean() if cond_any_sig.any() else float("nan")
            ),
        }
    return out


def _asymmetry_with_fdr(
    summaries: dict[str, pd.DataFrame], pvals: pd.DataFrame
) -> dict:
    """Asymmetry rates under multiple neutrality definitions x significance thresholds."""
    pframes: dict[str, pd.DataFrame] = {}
    pframes["trolley"] = pvals[pvals.benchmark == "trolley"].copy()
    pframes["bbq"] = pvals[pvals.benchmark == "bbq"].copy()
    pframes["dailydilemmas"] = _reconstruct_dd_pvalues(summaries["dailydilemmas"])

    for bench in ["trolley", "bbq"]:
        s = summaries[bench]
        pf = pframes[bench]
        merge_cols = ["model", "factor", "nudge_type"]
        pf = pf.merge(
            s[merge_cols + ["f_0_B", "n_comparisons"]], on=merge_cols, how="left"
        )
        pframes[bench] = pf

    out = {}
    for bench, pf in pframes.items():
        if pf.empty:
            continue
        n_total_trials = pf["n_comparisons"].fillna(100).astype(int).values
        n_total_trials = n_total_trials * (3 if bench == "dailydilemmas" else 8)
        f0 = pf["f_0_B"].fillna(0.5).values
        p_asym = pf["p_asym"].fillna(1.0).values
        sig_alpha05 = p_asym < 0.05
        sig_alpha01 = p_asym < 0.01
        sig_bh, _, _, _ = multipletests(p_asym, alpha=0.05, method="fdr_bh")
        rec = {
            "n_conditions": int(len(pf)),
            "rate_sig_alpha05": float(sig_alpha05.mean()),
            "rate_sig_alpha01": float(sig_alpha01.mean()),
            "rate_sig_bh_fdr05": float(sig_bh.mean()),
            "by_neutrality": {},
        }
        for kind in ["binom_05", "binom_01", "margin_05", "margin_10", "ci_45_55"]:
            mask = _equivalence_neutrality_mask(f0, n_total_trials, kind)
            if mask.sum() == 0:
                rec["by_neutrality"][kind] = {
                    "n_neutral": 0,
                    "rate_alpha05": float("nan"),
                    "rate_alpha01": float("nan"),
                    "rate_bh_fdr05": float("nan"),
                }
                continue
            rec["by_neutrality"][kind] = {
                "n_neutral": int(mask.sum()),
                "rate_alpha05": float(sig_alpha05[mask].mean()),
                "rate_alpha01": float(sig_alpha01[mask].mean()),
                "rate_bh_fdr05": float(sig_bh[mask].mean()),
            }
        out[bench] = rec
    return out


# ---------- appendix tables --------------------------------------------------


def _table_by_group(df: pd.DataFrame, group_col: str) -> list[dict]:
    """Average summary stats grouped by group_col; mirrors create_summary tables."""
    agg = df.groupby(group_col).agg(
        n=("abs_effect", "size"),
        mean_abs_effect=("abs_effect", "mean"),
        mean_abs_steer=("abs_steerability", "mean"),
        mean_abs_asym=("steerability_asym", lambda s: s.abs().mean()),
        mean_abs_n_asym=("normalized_steerability_asym", lambda s: s.abs().mean()),
        sig_rate=(
            "sig_baseline_B",
            lambda s: float(
                (df.loc[s.index, "sig_A"] | df.loc[s.index, "sig_B"]).mean()
            ),
        ),
        sig_bf_rate=("sig_baseline_B", lambda s: _sig_bf_rate(df.loc[s.index])),
    )
    agg = agg.reset_index().to_dict(orient="records")
    return [
        {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in r.items()
        }
        for r in agg
    ]


def _sig_bf_rate(df: pd.DataFrame) -> float:
    """Backfire-among-sig-directed rate (matches paper's 'sig BF' columns)."""
    long_a = df[["sig_A", "backfire_A"]].rename(
        columns={"sig_A": "sig", "backfire_A": "bf"}
    )
    long_b = df[["sig_B", "backfire_B"]].rename(
        columns={"sig_B": "sig", "backfire_B": "bf"}
    )
    long = pd.concat([long_a, long_b], ignore_index=True)
    long["sig"] = long["sig"].fillna(False).astype(bool)
    long["bf"] = long["bf"].fillna(False).astype(bool)
    n_sig = int(long.sig.sum())
    if n_sig == 0:
        return float("nan")
    return float((long.sig & long.bf).sum() / n_sig)


def _table_per_model_reasoning(df: pd.DataFrame) -> list[dict]:
    """tab:reasoning-effects-models — one row per (model, reasoning)."""
    rows = []
    for (model, reasoning), sub in df.groupby(["model", "reasoning_condition"]):
        rows.append(
            {
                "model": model,
                "reasoning": reasoning,
                "n": int(len(sub)),
                "mean_abs_effect": float(sub.abs_effect.mean()),
                "mean_abs_steer": float(sub.abs_steerability.mean()),
                "mean_signed_steer": float(sub.avg_steerability.mean()),
                "mean_abs_asym": float(sub.steerability_asym.abs().mean()),
                "mean_abs_n_asym": float(sub.normalized_steerability_asym.abs().mean()),
                "sig_rate": float((sub.sig_A | sub.sig_B).mean()),
                "sig_bf_rate": _sig_bf_rate(sub),
            }
        )
    return rows


# ---------- experiment imports (noise floor, regression, follow-up) ---------


def _load_baseline_noise() -> dict:
    out: dict[str, Any] = {}
    for bench in ["trolley", "bbq"]:
        d = NOISE_DIR / f"analysis_{bench}"
        headline = d / "headline.json"
        if headline.exists():
            out[bench] = json.loads(headline.read_text())
        else:
            out[bench] = {"_note": f"missing {headline}"}
    out["dailydilemmas"] = {
        "_source": "earlier-work, ~2.5pp; not re-derived in this pipeline",
        "mean_abs_drift_pp_estimate": 2.5,
    }
    return out


def _load_asym_regression() -> dict:
    p = ASYM_DIR / "data/regression_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"_note": f"missing {p}"}


def _load_followup_probe() -> dict:
    p = PROBE_DIR / "analysis_out_v2/headline_numbers.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"_note": f"missing {p}"}


# ---------- figure: cross-benchmark headline --------------------------------


def _make_cross_benchmark_figure(out: dict, path: Path):
    """Three-panel headline figure: avg shift / asymmetry beyond baseline / backfire rate."""
    import matplotlib.pyplot as plt

    benches = ["trolley", "bbq", "dailydilemmas"]
    pretty = {"trolley": "Trolley", "bbq": "BBQ", "dailydilemmas": "DailyDilemmas"}

    shifts = {b: out["headline_table"][b]["all_7_influences"] for b in benches}
    asym = out["section_4_2_asymmetry"]
    bf = out["section_4_3_backfire"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Panel 1: avg choice-rate shift, with cluster-bootstrap CIs
    ax = axes[0]
    means = [shifts[b]["mean_abs_effect"] * 100 for b in benches]
    los = [
        (shifts[b]["mean_abs_effect"] - shifts[b]["boot95_lo"]) * 100 for b in benches
    ]
    his = [
        (shifts[b]["boot95_hi"] - shifts[b]["mean_abs_effect"]) * 100 for b in benches
    ]
    ax.bar(
        range(3),
        means,
        yerr=[los, his],
        capsize=4,
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels([pretty[b] for b in benches])
    ax.set_ylabel("Avg. choice-rate shift (pp)")
    ax.set_title("Score instability")
    for i, m in enumerate(means):
        ax.text(i, m + max(his) * 0.4, f"{m:.1f}", ha="center", fontsize=9)

    # Panel 2: asymmetry-beyond-baseline rate (binom α=.05)
    ax = axes[1]
    rates = []
    for b in benches:
        a = asym["rates_by_benchmark"][b]["by_neutrality"]["binom_05"]
        rates.append(a["rate_alpha05"] * 100)
    ax.bar(range(3), rates, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([pretty[b] for b in benches])
    ax.set_ylabel("Sig. asymmetry among baseline-undetected (%)")
    ax.set_title("Asymmetry beyond baseline")
    for i, r in enumerate(rates):
        ax.text(i, r + 1, f"{r:.1f}", ha="center", fontsize=9)

    # Panel 3: backfire among sig directed effects
    ax = axes[2]
    rates = [
        bf["definitions"][b]["rate_sig_bf_among_sig_directed"] * 100 for b in benches
    ]
    ax.bar(range(3), rates, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([pretty[b] for b in benches])
    ax.set_ylabel("Backfire rate among sig. effects (%)")
    ax.set_title("Backfiring with stated disclaiming")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.4, f"{r:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- main -------------------------------------------------------------


def main():
    summaries = {b: _load_summary(b) for b in ["trolley", "bbq", "dailydilemmas"]}

    # Extract or load p-values for trolley + BBQ from raw counts
    pvals_csv = DATA_OUT / "pvalues_by_condition.csv"
    if pvals_csv.exists():
        pvals = pd.read_csv(pvals_csv)
        print(f"  loaded existing p-values: {len(pvals)} rows")
    else:
        print("  extracting p-values from preference graphs (slow first run)...")
        parts = [_extract_pvalues_for(b) for b in ["trolley", "bbq"]]
        pvals = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        pvals.to_csv(pvals_csv, index=False)
        print(f"  wrote {pvals_csv}: {len(pvals)} rows")

    out: dict[str, Any] = {
        "_meta": {
            "produced_by": str(THIS.relative_to(REPO)),
            "source_summaries": str(SUMMARY_DIR.relative_to(REPO)),
            "paper_models": sorted(PAPER_MODELS),
            "trolley_factors": sorted(PAPER_FACTORS_TROLLEY),
            "dailydilemmas_excluded_values": ["dishonesty"],
            "influence_partition": {
                "one_sentence_user_msg": sorted(ONE_SENTENCE_USER_MSG),
                "system_replace": sorted(SYSTEM_REPLACE),
                "multi_demonstration": sorted(MULTI_DEMONSTRATION),
            },
            "ci_method": "cluster-bootstrap over the factor/value axis, B=5000",
            "fdr_method": "Benjamini–Hochberg within each benchmark",
        },
    }

    # Headline table — both 'all 7' and 'single-sentence user-msg only' framings
    out["headline_table"] = _headline_table(summaries)

    # Section 4.1 (instability): headline numbers + noise floor
    out["section_4_1_instability"] = {
        "shifts": out["headline_table"],
        "noise_floor_pp": _load_baseline_noise(),
    }

    # Section 4.2 (asymmetry beyond baseline): rates + regression + sensitivity
    out["section_4_2_asymmetry"] = {
        "rates_by_benchmark": _asymmetry_with_fdr(summaries, pvals),
        "regression": _load_asym_regression(),
    }

    # Section 4.3 (backfire): four-definition reconciliation + follow-up probe
    out["section_4_3_backfire"] = {
        "definitions": _backfire_definitions(summaries),
        "followup_probe_headline": _load_followup_probe(),
        "_paper_claims": {
            "main_text_pre_rewrite": {
                "trolley_pct": 14.3,
                "bbq_pct": 10.5,
                "dailydilemmas_pct": 5.9,
            },
            "post_rewrite_2026_05_02": {
                "trolley_pct": 17.7,
                "bbq_pct": 10.5,
                "dailydilemmas_pct": 0.2,
            },
        },
    }

    # Section 4.4 (reasoning reallocation): per-(model, reasoning) breakdown
    bbq_reas = _table_per_model_reasoning(summaries["bbq"])
    out["section_4_4_reasoning_reallocation"] = {
        "bbq_per_model_reasoning": bbq_reas,
        "trolley_per_model_reasoning": _table_per_model_reasoning(summaries["trolley"]),
        "trolley_baseline_p_larger_group": {
            "with_reasoning_paper_claim": 0.97,
            "without_reasoning_paper_claim": 0.82,
            "_note": "claim_verification flagged this as approx; condition-aggregated",
        },
        "trolley_avg_base_preference_paper_claim": {
            "with_reasoning": 0.55,
            "without_reasoning": 0.67,
        },
    }

    # Appendix tables
    out["appendix_tables"] = {
        "tab_nudge_type_effects_trolley": _table_by_group(
            summaries["trolley"], "nudge_type"
        ),
        "tab_factor_summary_trolley": _table_by_group(summaries["trolley"], "factor"),
        "tab_reasoning_effects_models_trolley": _table_per_model_reasoning(
            summaries["trolley"]
        ),
        "tab_reasoning_effects_conditions_trolley": _table_by_group(
            summaries["trolley"], "reasoning_condition"
        ),
        "tab_bbq_factor_details": _table_by_group(summaries["bbq"], "factor"),
        "tab_bbq_nudge_details": _table_by_group(summaries["bbq"], "nudge_type"),
        "tab_bbq_model_details": bbq_reas,
        "tab_dailydilemmas_values": _table_by_group(
            summaries["dailydilemmas"], "factor"
        ),
        "tab_baseline_noise": _load_baseline_noise(),
        "tab_asym_regression": _load_asym_regression(),
        "tab_followup_probe": _load_followup_probe(),
        "tab_realistic_scenarios": {
            "_status": "deferred",
            "_note": "raw data not in google_drive/; locate before camera-ready",
        },
        "tab_phrasing_young": {
            "_status": "deferred",
            "_note": "11 wording variants x 4 models grid; data not in google_drive/",
        },
    }

    # Robustness summary
    out["robustness"] = {
        "asym_sensitivity_table": out["section_4_2_asymmetry"]["rates_by_benchmark"],
        "headline_subset_sensitivity": out["headline_table"],
        "backfire_definition_sensitivity": out["section_4_3_backfire"]["definitions"],
    }

    json_path = DATA_OUT / "paper_numbers.json"
    json_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"  wrote {json_path}")

    # First-pass figure
    try:
        fig_path = FIG_OUT / "cross_benchmark.png"
        _make_cross_benchmark_figure(out, fig_path)
        print(f"  wrote {fig_path}")
    except Exception as e:
        print(f"  figure failed: {e}", file=sys.stderr)

    # Summary print
    print("\n=== headline shifts (single-sentence user-msg only) ===")
    for bench in ["trolley", "bbq", "dailydilemmas"]:
        h = out["headline_table"][bench]["single_sentence_user_msg_only"]
        print(
            f"  {bench:14s}: {h['mean_abs_effect']*100:5.1f}pp  "
            f"[{h['boot95_lo']*100:.1f}, {h['boot95_hi']*100:.1f}]"
            f"  n={h['n_conditions']}"
        )
    print("\n=== backfire among sig effects (sig-directed denom) ===")
    for bench in ["trolley", "bbq", "dailydilemmas"]:
        b = out["section_4_3_backfire"]["definitions"][bench]
        print(
            f"  {bench:14s}: {b['rate_sig_bf_among_sig_directed']*100:5.1f}%  "
            f"of {b['n_sig_directed']} sig directed effects"
        )
    print("\n=== baseline-undetected asym rate (binom_05) ===")
    for bench, a in out["section_4_2_asymmetry"]["rates_by_benchmark"].items():
        r = a["by_neutrality"]["binom_05"]
        print(
            f"  {bench:14s}: alpha=.05 {r['rate_alpha05']*100:5.1f}% | "
            f"BH-FDR {r['rate_bh_fdr05']*100:5.1f}%  (n={r['n_neutral']})"
        )


if __name__ == "__main__":
    main()
