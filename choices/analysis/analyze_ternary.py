#!/usr/bin/env python3
"""
Analysis and plotting for the ternary (K=3) direction-rotated influence audit.

Reads the tidy trial CSVs written by
``choices.experiments.nudging.ternary`` and reports, per
(factor, model, influence):

  * baseline neutrality against a uniform 1/3 split
  * one-vs-rest steerability toward each of the three groups  [the estimand]
  * pairwise asymmetries and a global test that they are all equal
  * displacement profile: where each cue's gain came from
  * the 1-df IIA test per cue                                  [the K=3 finding]
  * backfire classification: none / total / displaced

The ternary plot is the figure this whole experiment exists to produce. Under
IIA a cue toward group d must move the choice distribution along the *straight
line* from the baseline point to d's vertex, because the two rivals keep their
relative odds. Perpendicular departure from that line is selective
cannibalisation, and it is invisible in any binary audit.

Usage:
    uv run python -m choices.analysis.analyze_ternary --factor age3
    uv run python -m choices.analysis.analyze_ternary --factor age3 --model gpt-52-no-reasoning
    uv run python -m choices.analysis.analyze_ternary --all --out-csv ternary_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Dict, List, Optional

import pandas as pd

from choices.analysis.ternary_metrics import (
    BACKFIRE_DISPLACED,
    BACKFIRE_TOTAL,
    ConditionResult,
    analyze_rotation,
    backfire_destination,
    bh_fdr,
)
from choices.experiments.nudging.ternary import DEFAULT_OUT_DIR
from choices.experiments.nudging.ternary_factors import TERNARY_FACTORS


# ============================================================================
# Loading
# ============================================================================


def load_trials(
    results_dir: str,
    factor: Optional[str] = None,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and concatenate every trials.csv under the results tree.

    Separate invocations of the runner (e.g. a generic pilot followed by a
    ``--calibrate-from`` re-run) land side by side under the same
    factor/model path, distinguished only by their run_id directory. Without
    ``run_id`` this loads and merges all of them, which silently mixes
    incompatible size designs into one baseline/cue count. Pass ``run_id`` to
    restrict to a single audit run.
    """
    pattern = os.path.join(
        results_dir,
        factor or "*",
        model or "*",
        "*",
        "*",
        run_id or "*",
        "trials.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No trials.csv found matching:\n  {pattern}\n"
            "Run the experiment first (choices.experiments.nudging.ternary)."
        )

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} trials from {len(paths)} condition run(s).")
    return df


def counts_from(df: pd.DataFrame, groups: List[str]) -> Dict[str, float]:
    """Valid-response counts per group, zero-filled to the full group list."""
    valid = df[df["status"] == "ok"]
    vc = valid["chosen_group"].value_counts().to_dict()
    return {g: float(vc.get(g, 0)) for g in groups}


# ============================================================================
# Reporting
# ============================================================================


def print_report(res: ConditionResult) -> None:
    """Human-readable report for one rotation."""
    header = f" {res.factor} | {res.model} | influence = {res.influence} "
    print("\n" + "=" * 78)
    print(header.center(78, "="))
    print("=" * 78)

    n = res.neutrality
    print("\nBaseline")
    freq_str = "   ".join(
        f"{g}: {n.frequencies[g]:.3f}" for g in res.groups
    )
    print(f"  {freq_str}")
    print(
        f"  uniformity chi2({n.df}) = {n.chi2:.2f}, p = {n.p_value:.4f}"
        f"  -> {'NEUTRAL' if n.is_neutral else 'biased'}"
        f"   (max deviation from 1/3: {n.max_deviation:.3f})"
    )

    print("\nSteerability (one-vs-rest)")
    print(
        f"  {'cue toward':<16}{'f_base':>8}{'f_cue':>8}{'delta':>9}"
        f"{'s_ovr':>9}{'p':>10}   outcome"
    )
    for g in res.groups:
        s = res.steerabilities.get(g)
        if s is None:
            continue
        bf = res.backfire[g]
        if bf == BACKFIRE_TOTAL:
            dest = backfire_destination(res.displacements[g])
            outcome = f"BACKFIRE (total) -> {dest}"
        elif bf == BACKFIRE_DISPLACED:
            gainers = ", ".join(res.displacements[g].gaining_rivals)
            outcome = f"BACKFIRE (displaced) -> {gainers} gained"
        else:
            outcome = "complies" if s.s_ovr > 0 else "-"
        print(
            f"  {g:<16}{s.f_base:>8.3f}{s.f_cue:>8.3f}{s.delta:>+9.3f}"
            f"{s.s_ovr:>+9.3f}{s.p_value:>10.4f}   {outcome}"
        )

    print("\nAsymmetry  (Asym(a,b) = s(b) - s(a); positive = easier toward b)")
    for a in res.asymmetries:
        star = " *" if a.is_significant else ""
        print(
            f"  {a.target_a:>13} vs {a.target_b:<13}"
            f"{a.asym:>+8.3f}   N-Asym {a.normalized:>+6.3f}"
            f"   p = {a.p_value:.4f}{star}"
        )
    chi2, df_, p = res.homogeneity
    print(
        f"  all three equal?  chi2({df_}) = {chi2:.2f}, p = {p:.4f}"
        f"  -> {'no directional structure' if p >= 0.05 else 'DIRECTIONAL STRUCTURE'}"
    )

    print("\nDisplacement  (where each cue's gain came from)")
    for g in res.groups:
        d = res.displacements.get(g)
        if d is None:
            continue
        print(f"  cue -> {g}   (delta = {d.delta_target:+.3f})")
        if d.observed_shares:
            for rival in d.observed_shares:
                obs = d.observed_shares[rival]
                exp = d.expected_shares.get(rival, float("nan"))
                flag = "  <-- selective" if abs(obs - exp) > 0.15 else ""
                print(
                    f"      from {rival:<14} observed {obs:>+6.2f}"
                    f"   proportional-null {exp:>5.2f}{flag}"
                )
        else:
            for rival, delta in d.rival_deltas.items():
                print(f"      {rival:<14} changed {delta:+.3f}")

    print("\nIIA test  (did the cue change the *rivals'* relative odds?)")
    for g in res.groups:
        t = res.iia.get(g)
        if t is None:
            continue
        verdict = "IIA VIOLATED" if t.is_significant else "consistent with IIA"
        print(
            f"  cue -> {g:<14} log OR({t.rivals[1]}/{t.rivals[0]}) = "
            f"{t.log_or:>+7.3f}   z = {t.z:>+6.2f}   p = {t.p_value:.4f}"
            f"   -> {verdict}"
        )


# ============================================================================
# Summary table
# ============================================================================


def summary_rows(res: ConditionResult) -> List[dict]:
    """Flatten one rotation into one row per cue direction."""
    rows = []
    for g in res.groups:
        s = res.steerabilities.get(g)
        if s is None:
            continue
        d = res.displacements[g]
        t = res.iia.get(g)

        row = {
            "factor": res.factor,
            "model": res.model,
            "influence": res.influence,
            "cue_target": g,
            "baseline_neutral": res.neutrality.is_neutral,
            "baseline_p": res.neutrality.p_value,
            "f_base": s.f_base,
            "f_cue": s.f_cue,
            "delta": s.delta,
            "delta_p": s.delta_p_value,
            "s_ovr": s.s_ovr,
            "s_se": s.se,
            "s_p": s.p_value,
            "backfire": res.backfire[g],
            "backfire_destination": backfire_destination(d) or "",
            "n_base": s.n_base,
            "n_cue": s.n_cue,
            "iia_log_or": t.log_or if t else float("nan"),
            "iia_p": t.p_value if t else float("nan"),
            "iia_violated": bool(t.is_significant) if t else False,
        }
        for rival in d.rival_deltas:
            row[f"delta_{rival}"] = d.rival_deltas[rival]
            if d.observed_shares:
                row[f"share_{rival}"] = d.observed_shares[rival]
                row[f"share_null_{rival}"] = d.expected_shares[rival]
        rows.append(row)
    return rows


def print_headline(df: pd.DataFrame) -> None:
    """The numbers that go straight into the rebuttal."""
    print("\n" + "=" * 78)
    print(" HEADLINE RATES (K=3) ".center(78, "="))
    print("=" * 78)

    total = len(df)
    if total == 0:
        print("  no conditions")
        return

    mean_shift = df["delta"].abs().mean() * 100
    print(f"\n  mean |shift| under a cue:            {mean_shift:.1f} pp")

    sig = df[df["s_p"] < 0.05]
    print(f"  significant effects:                 {len(sig)}/{total}")

    if len(sig):
        n_total_bf = (sig["backfire"] == BACKFIRE_TOTAL).sum()
        n_disp_bf = (sig["backfire"] == BACKFIRE_DISPLACED).sum()
        print(
            f"  backfire rate among sig. effects:    "
            f"{100.0 * n_total_bf / len(sig):.1f}%  (total, binary-comparable)"
        )
        print(
            f"  displaced-backfire rate:             "
            f"{100.0 * n_disp_bf / len(sig):.1f}%  (K>=3 only)"
        )

    neutral = df[df["baseline_neutral"]]
    if len(neutral):
        n_asym = (neutral["s_p"] < 0.05).sum()
        print(
            f"  baseline-neutral cells with a\n"
            f"    resolved directional effect:       "
            f"{100.0 * n_asym / len(neutral):.1f}%  ({n_asym}/{len(neutral)})"
        )

    iia_rows = df.dropna(subset=["iia_p"])
    if len(iia_rows):
        n_viol = iia_rows["iia_violated"].sum()
        print(
            f"  IIA violations:                      "
            f"{100.0 * n_viol / len(iia_rows):.1f}%  ({n_viol}/{len(iia_rows)})"
        )
        print("     ^ mass does NOT disperse proportionally -- the K=3 result")

    # BH-FDR over the steerability tests.
    rejected = bh_fdr(df["s_p"].tolist(), q=0.05)
    print(
        f"\n  after BH-FDR (q=0.05):               "
        f"{sum(rejected)}/{total} steerability effects survive"
    )


# ============================================================================
# Ternary plot
# ============================================================================


def _bary_to_xy(f: List[float]) -> tuple:
    """Barycentric (f0, f1, f2) -> 2D cartesian on a unit triangle."""
    return f[1] + 0.5 * f[2], (math.sqrt(3) / 2.0) * f[2]


def plot_ternary(results: List[ConditionResult], out_path: str) -> None:
    """
    One simplex panel per influence type.

    Under IIA, a cue toward group d moves the distribution along the straight
    line from the baseline point to d's vertex (the rivals hold their relative
    odds, and that locus is exactly that line). Those null lines are drawn
    dashed; an observed arrow that leaves its line is an IIA violation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), squeeze=False)
    colors = ["#1b7837", "#762a83", "#d95f02"]

    for ax, res in zip(axes[0], results):
        groups = res.groups
        verts = [_bary_to_xy([1, 0, 0]), _bary_to_xy([0, 1, 0]), _bary_to_xy([0, 0, 1])]

        # triangle
        tri = verts + [verts[0]]
        ax.plot([p[0] for p in tri], [p[1] for p in tri], color="0.3", lw=1.2)

        offsets = [(-0.04, -0.05), (0.04, -0.05), (0.0, 0.035)]
        for v, g, off in zip(verts, groups, offsets):
            ax.annotate(
                g,
                (v[0] + off[0], v[1] + off[1]),
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        base_f = [res.neutrality.frequencies[g] for g in groups]
        base_xy = _bary_to_xy(base_f)

        for i, g in enumerate(groups):
            s = res.steerabilities.get(g)
            if s is None:
                continue

            # IIA null path: straight line baseline -> that group's vertex
            ax.plot(
                [base_xy[0], verts[i][0]],
                [base_xy[1], verts[i][1]],
                ls="--",
                lw=0.9,
                color=colors[i],
                alpha=0.45,
            )

            cue_f = [
                res.cue_counts[g][h] / max(sum(res.cue_counts[g].values()), 1)
                for h in groups
            ]
            cue_xy = _bary_to_xy(cue_f)

            # The IIA violation is already visible as departure from the
            # dashed null line, so arrows stay solid to avoid clashing with it.
            ax.annotate(
                "",
                xy=cue_xy,
                xytext=base_xy,
                arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.2),
            )
            ax.plot(*cue_xy, "o", color=colors[i], ms=8, zorder=5)

            bf = res.backfire[g]
            if bf in (BACKFIRE_TOTAL, BACKFIRE_DISPLACED):
                ax.plot(
                    *cue_xy,
                    "o",
                    mfc="none",
                    mec="crimson",
                    ms=15,
                    mew=2.0,
                    zorder=6,
                )

        ax.plot(*base_xy, "ko", ms=10, zorder=6)
        ax.annotate(
            "baseline",
            (base_xy[0], base_xy[1] - 0.045),
            ha="center",
            fontsize=9,
        )

        ax.set_title(f"{res.influence}\n{res.factor} | {res.model}", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, 1.0)

    fig.suptitle(
        "Direction-rotated influence on a 3-option choice.  "
        "Dashed line = IIA null path (proportional substitution);  "
        "red ring = backfire.",
        fontsize=10,
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot written to: {out_path}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ternary (K=3) influence audit results."
    )
    parser.add_argument("--factor", help="Ternary factor name (default: all)")
    parser.add_argument("--model", help="Model key (default: all)")
    parser.add_argument(
        "--run-id",
        help=(
            "Restrict to one audit run's timestamp directory (e.g. "
            "20260731_175508). Required when multiple runs of the same "
            "factor/model exist (e.g. a pilot plus a --calibrate-from "
            "re-run) and should not be merged together."
        ),
    )
    parser.add_argument("--results-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--all", action="store_true", help="Analyze everything found")
    parser.add_argument("--out-csv", default="ternary_summary.csv")
    parser.add_argument("--out-plot", default="ternary_simplex.png")
    parser.add_argument(
        "--no-significance-gate",
        action="store_true",
        help="Classify backfires regardless of statistical significance",
    )
    args = parser.parse_args()

    if not args.factor and not args.all:
        parser.error("pass --factor NAME or --all")

    df = load_trials(args.results_dir, args.factor, args.model, args.run_id)

    # Invalid-response accounting -- a fair question of any 3-option design.
    n_total = len(df)
    n_ok = (df["status"] == "ok").sum()
    print(f"Valid responses: {n_ok:,}/{n_total:,} ({100.0 * n_ok / n_total:.1f}%)")
    bad = df[df["status"] != "ok"]["status"].value_counts()
    if len(bad):
        print("  invalid breakdown: " + ", ".join(f"{k}={v}" for k, v in bad.items()))

    results: List[ConditionResult] = []
    all_rows: List[dict] = []

    for (factor_name, model), fm_df in df.groupby(["factor", "model"], sort=True):
        if factor_name not in TERNARY_FACTORS:
            print(f"  (skipping unknown factor {factor_name})")
            continue
        groups = TERNARY_FACTORS[factor_name].values

        base_df = fm_df[fm_df["influence"] == "base"]
        if base_df.empty:
            print(f"  (no baseline for {factor_name}/{model}; skipping)")
            continue
        base_counts = counts_from(base_df, groups)

        for influence, inf_df in fm_df.groupby("influence", sort=True):
            if influence == "base":
                continue

            cue_counts = {
                target: counts_from(t_df, groups)
                for target, t_df in inf_df.groupby("direction", sort=False)
                if target in groups
            }
            if not cue_counts:
                continue

            res = analyze_rotation(
                factor=factor_name,
                model=model,
                influence=influence,
                groups=groups,
                base_counts=base_counts,
                cue_counts=cue_counts,
                require_significance=not args.no_significance_gate,
            )
            print_report(res)
            results.append(res)
            all_rows.extend(summary_rows(res))

    if not all_rows:
        print("\nNothing to summarise.")
        return

    summary = pd.DataFrame(all_rows)
    summary["s_p_bh_significant"] = bh_fdr(summary["s_p"].tolist(), q=0.05)
    summary.to_csv(args.out_csv, index=False)
    print(f"\nSummary table written to: {args.out_csv}  ({len(summary)} rows)")

    print_headline(summary)
    plot_ternary(results, args.out_plot)


if __name__ == "__main__":
    main()
