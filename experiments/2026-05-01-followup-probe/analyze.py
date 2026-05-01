"""
Cross-tabulate follow-up classifications (acknowledged / denied / partial / unclear)
against significance and backfire status from the existing main-experiment summary.

Outputs the C2 headline number: among significant backfire conditions, what
fraction of trials had the model explicitly DENY being influenced when asked.

Usage:
    uv run python experiments/2026-05-01-followup-probe/analyze.py \
        --classified results/classified_*.jsonl \
        --summary-csvs ../2026-05-01-asymmetry-baseline-regression/data/{trolley,bbq}_summary.csv \
        --output-dir ./analysis_out
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_classified(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        for line in _open_text(Path(p)):
            if not line.strip():
                continue
            rec = json.loads(line)
            spec = rec.get("spec", {})
            row = {
                "benchmark": spec.get("benchmark"),
                "model": spec.get("model"),
                "factor": spec.get("factor"),
                "nudge_type": spec.get("nudge_type"),
                "target_group": spec.get("target_group"),
                "reasoning_condition": spec.get("reasoning_condition"),
                "edge_id": tuple(spec.get("edge_id", ())),
                "direction": spec.get("direction"),
                "saved_choice": spec.get("saved_choice"),
                "turn1_parsed": rec.get("turn1_parsed"),
                "judge_label": rec.get("judge_label"),
                "judge_reason": rec.get("judge_reason"),
                "error": rec.get("error"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def load_summary(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def annotate_status(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """For each (factor, model, nudge_type, target_group, reasoning_condition)
    cell, look up whether it was a significant backfire / significant
    compliance / no significant effect from the per-condition summary CSV."""
    summary = summary.copy()
    summary["target_a"] = summary["level_A"]
    summary["target_b"] = summary["level_B"]

    # The summary has one row per (model, reasoning, factor, nudge_type) with
    # both A-direction and B-direction stats. We need to match the trial's
    # target_group to either A-direction or B-direction.
    out_rows = []
    for _, t in df.iterrows():
        if t["nudge_type"] == "base" or t["target_group"] == "base":
            out_rows.append(
                {
                    "condition_kind": "base",
                    "is_significant": False,
                    "is_backfire": False,
                }
            )
            continue
        # The summary table uses 'off' for non-reasoning runs while saved
        # graphs store reasoning_mode='none'; treat them as equivalent.
        rc_trial = str(t["reasoning_condition"]).lower()
        rc_aliases = {rc_trial}
        if rc_trial in ("none", "off", "", "nan"):
            rc_aliases |= {"none", "off"}
        match = summary[
            (summary["model"] == t["model"])
            & (summary["reasoning_condition"].astype(str).str.lower().isin(rc_aliases))
            & (summary["factor"] == t["factor"])
            & (summary["nudge_type"] == t["nudge_type"])
        ]
        if len(match) == 0:
            out_rows.append(
                {
                    "condition_kind": "unknown",
                    "is_significant": False,
                    "is_backfire": False,
                }
            )
            continue
        m = match.iloc[0]
        # which side of the condition is this trial?
        if t["target_group"] == m["level_A"] or (
            isinstance(t["target_group"], str)
            and t["target_group"].lower() == str(m["level_A"]).lower()
        ):
            sig = bool(m["sig_A"])
            backfire = bool(m["backfire_A"])
        elif t["target_group"] == m["level_B"] or (
            isinstance(t["target_group"], str)
            and t["target_group"].lower() == str(m["level_B"]).lower()
        ):
            sig = bool(m["sig_B"])
            backfire = bool(m["backfire_B"])
        else:
            # path-level target may be lowercased / replaced underscores;
            # try a relaxed comparison
            tg_norm = str(t["target_group"]).lower().replace(" ", "_")
            if tg_norm == str(m["level_A"]).lower().replace(" ", "_"):
                sig = bool(m["sig_A"])
                backfire = bool(m["backfire_A"])
            elif tg_norm == str(m["level_B"]).lower().replace(" ", "_"):
                sig = bool(m["sig_B"])
                backfire = bool(m["backfire_B"])
            else:
                out_rows.append(
                    {
                        "condition_kind": "unknown_target",
                        "is_significant": False,
                        "is_backfire": False,
                    }
                )
                continue

        if sig and backfire:
            kind = "sig_backfire"
        elif sig and not backfire:
            kind = "sig_compliance"
        else:
            kind = "no_effect"
        out_rows.append(
            {
                "condition_kind": kind,
                "is_significant": sig,
                "is_backfire": backfire,
            }
        )
    annotated = pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)
    return annotated


def headline_numbers(annotated: pd.DataFrame) -> dict:
    """Compute the paper-headline % of significant backfires that DENIED."""
    out: dict = {"by_model_benchmark": {}, "overall": {}}
    valid = annotated[annotated["judge_label"].notna()]

    for (model, benchmark), sub in valid.groupby(["model", "benchmark"]):
        out["by_model_benchmark"][f"{model}__{benchmark}"] = _summarize(sub)
    out["overall"] = _summarize(valid)
    return out


def _summarize(df: pd.DataFrame) -> dict:
    total = len(df)
    backfires = df[df["condition_kind"] == "sig_backfire"]
    compliances = df[df["condition_kind"] == "sig_compliance"]
    no_effect = df[df["condition_kind"] == "no_effect"]
    base = df[df["condition_kind"] == "base"]

    def pct(sub, label):
        if len(sub) == 0:
            return None
        return float((sub["judge_label"] == label).sum()) / len(sub)

    LABELS = ("ACK_AFFECTED", "ACK_DISCLAIMED", "DENIED", "PARTIAL", "UNCLEAR")
    out = {
        "n_total": int(total),
        "n_sig_backfire": int(len(backfires)),
        "n_sig_compliance": int(len(compliances)),
        "n_no_effect": int(len(no_effect)),
        "n_base": int(len(base)),
    }
    for kind, sub in (
        ("backfires", backfires),
        ("compliances", compliances),
        ("no_effect", no_effect),
        ("base", base),
        ("overall", df),
    ):
        for label in LABELS:
            out[f"{label.lower()}_rate_in_{kind}"] = pct(sub, label)
    # The C2 paper-headline number we want: among significant backfires,
    # share that acknowledged the cue but disclaimed any effect on the
    # choice — i.e. the model says "the cue tried to nudge me, but it
    # didn't change my answer", while the data shows the answer DID
    # change (significantly opposite to the cue).
    out["headline_ack_disclaimed_in_backfires"] = pct(backfires, "ACK_DISCLAIMED")
    out["headline_denial_in_backfires"] = pct(backfires, "DENIED")
    out["headline_stated_no_effect_in_backfires"] = (
        out["headline_ack_disclaimed_in_backfires"] or 0.0
    ) + (out["headline_denial_in_backfires"] or 0.0)
    return out


def make_bar_plot(annotated: pd.DataFrame, out_path: Path) -> None:
    """Per-model stacked bar: ACK_AFFECTED / ACK_DISCLAIMED / DENIED / etc.,
    split by sig_backfire vs sig_compliance."""
    valid = annotated[annotated["judge_label"].notna()]

    LABELS = ("ACK_AFFECTED", "ACK_DISCLAIMED", "DENIED", "PARTIAL", "UNCLEAR")
    COLORS = {
        "ACK_AFFECTED": "#1f77b4",
        "ACK_DISCLAIMED": "#d62728",
        "DENIED": "#7f0000",
        "PARTIAL": "#ff7f0e",
        "UNCLEAR": "#888888",
    }

    rows = []
    for (model, benchmark), sub in valid.groupby(["model", "benchmark"]):
        for kind in ("sig_backfire", "sig_compliance"):
            cell = sub[sub["condition_kind"] == kind]
            if len(cell) == 0:
                continue
            for label in LABELS:
                rate = (cell["judge_label"] == label).sum() / len(cell)
                rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "kind": kind,
                        "label": label,
                        "rate": rate,
                        "n": len(cell),
                    }
                )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("No backfire/compliance trials with labels — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, ["sig_backfire", "sig_compliance"]):
        sub = plot_df[plot_df["kind"] == kind]
        groups = sub.groupby(["model", "benchmark"])
        x_labels = []
        bars = {label: [] for label in LABELS}
        ns = []
        for (model, benchmark), g in groups:
            short = f"{model.split('-')[0][:6]}\n{benchmark}"
            x_labels.append(short)
            for label in LABELS:
                bars[label].append(float(g[g["label"] == label]["rate"].sum()))
            ns.append(int(g["n"].iloc[0]))
        x = list(range(len(x_labels)))
        bottom = [0.0] * len(x_labels)
        for label in LABELS:
            ax.bar(x, bars[label], bottom=bottom, label=label, color=COLORS[label])
            bottom = [b + v for b, v in zip(bottom, bars[label])]
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{xl}\n(n={n})" for xl, n in zip(x_labels, ns)],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        ax.set_ylim(0, 1)
        ax.set_title(
            "significant backfires"
            if kind == "sig_backfire"
            else "significant compliances"
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of trials")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(
        "Turn-2 self-report: how the model frames the influence cue\n"
        "(ACK_DISCLAIMED is the stated-vs-revealed inconsistency in backfires)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified", nargs="+", required=True, help="One or more classified JSONL"
    )
    parser.add_argument(
        "--summary-csvs", nargs="+", required=True, help="Per-condition summary CSVs"
    )
    parser.add_argument(
        "--output-dir", default="experiments/2026-05-01-followup-probe/analysis_out"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_classified([Path(p) for p in args.classified])
    print(f"Loaded {len(df)} classified records")
    summary = load_summary([Path(p) for p in args.summary_csvs])
    print(f"Loaded {len(summary)} summary rows")

    annotated = annotate_status(df, summary)

    # Save the annotated long-form table.
    annotated.to_csv(out_dir / "trials_annotated.csv", index=False)

    # Headline numbers.
    h = headline_numbers(annotated)
    with (out_dir / "headline_numbers.json").open("w") as f:
        json.dump(h, f, indent=2, default=str)

    def _fmt_pct(x):
        return f"{x:.1%}" if x is not None else "N/A"

    print("\nHeadline numbers (overall):")
    o = h["overall"]
    print(f"  trials with judge label:        {o['n_total']}")
    print(f"  significant backfire trials:    {o['n_sig_backfire']}")
    print(f"  significant compliance trials:  {o['n_sig_compliance']}")
    print(f"  no-effect trials:               {o['n_no_effect']}")
    print(f"  base trials:                    {o['n_base']}")
    if o["n_sig_backfire"] > 0:
        print("\n  >>> Of significant BACKFIRES (paper headline):")
        print(f"    ACK_AFFECTED:    {_fmt_pct(o['ack_affected_rate_in_backfires'])}")
        print(
            f"    ACK_DISCLAIMED:  {_fmt_pct(o['ack_disclaimed_rate_in_backfires'])}  "
            f"<-- model acknowledges cue but denies the cue affected its choice"
        )
        print(f"    DENIED:          {_fmt_pct(o['denied_rate_in_backfires'])}")
        print(f"    PARTIAL:         {_fmt_pct(o['partial_rate_in_backfires'])}")
        print(f"    UNCLEAR:         {_fmt_pct(o['unclear_rate_in_backfires'])}")
        print(
            f"\n    'stated-no-effect' (ACK_DISCLAIMED + DENIED): "
            f"{_fmt_pct(o['headline_stated_no_effect_in_backfires'])}"
        )

    print("\nPer (model, benchmark):")
    for k, v in sorted(h["by_model_benchmark"].items()):
        print(f"\n  {k}:")
        if v["n_sig_backfire"] > 0:
            print(
                f"    sig_backfire   (n={v['n_sig_backfire']:5d}): "
                f"AFF={_fmt_pct(v['ack_affected_rate_in_backfires'])} "
                f"DIS={_fmt_pct(v['ack_disclaimed_rate_in_backfires'])} "
                f"DEN={_fmt_pct(v['denied_rate_in_backfires'])}"
            )
        if v["n_sig_compliance"] > 0:
            print(
                f"    sig_compliance (n={v['n_sig_compliance']:5d}): "
                f"AFF={_fmt_pct(v['ack_affected_rate_in_compliances'])} "
                f"DIS={_fmt_pct(v['ack_disclaimed_rate_in_compliances'])} "
                f"DEN={_fmt_pct(v['denied_rate_in_compliances'])}"
            )

    plot_path = out_dir / "denial_vs_ack_by_kind.png"
    make_bar_plot(annotated, plot_path)
    print(f"\nWrote {out_dir}/{{trials_annotated.csv, headline_numbers.json}}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
