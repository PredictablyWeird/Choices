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
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_classified(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
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

    return {
        "n_total": int(total),
        "n_sig_backfire": int(len(backfires)),
        "n_sig_compliance": int(len(compliances)),
        "n_no_effect": int(len(no_effect)),
        "n_base": int(len(base)),
        "denial_rate_in_backfires": pct(backfires, "DENIED"),
        "ack_rate_in_backfires": pct(backfires, "ACKNOWLEDGED"),
        "partial_rate_in_backfires": pct(backfires, "PARTIAL"),
        "denial_rate_in_compliances": pct(compliances, "DENIED"),
        "ack_rate_in_compliances": pct(compliances, "ACKNOWLEDGED"),
        "denial_rate_in_no_effect": pct(no_effect, "DENIED"),
        "ack_rate_in_no_effect": pct(no_effect, "ACKNOWLEDGED"),
        "denial_rate_overall": pct(df, "DENIED"),
        "ack_rate_overall": pct(df, "ACKNOWLEDGED"),
    }


def make_bar_plot(annotated: pd.DataFrame, out_path: Path) -> None:
    """Per-model bar chart: denial vs ack rate among backfires vs compliances."""
    valid = annotated[annotated["judge_label"].notna()]
    rows = []
    for (model, benchmark), sub in valid.groupby(["model", "benchmark"]):
        for kind in ("sig_backfire", "sig_compliance"):
            cell = sub[sub["condition_kind"] == kind]
            if len(cell) == 0:
                continue
            for label in ("ACKNOWLEDGED", "DENIED", "PARTIAL", "UNCLEAR"):
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind in zip(axes, ["sig_backfire", "sig_compliance"]):
        sub = plot_df[plot_df["kind"] == kind]
        groups = sub.groupby(["model", "benchmark"])
        x_labels = []
        denial = []
        ack = []
        partial = []
        for (model, benchmark), g in groups:
            label = f"{model.split('-')[0][:6]}/{benchmark}"
            x_labels.append(label)
            denial.append(float(g[g["label"] == "DENIED"]["rate"].sum()))
            ack.append(float(g[g["label"] == "ACKNOWLEDGED"]["rate"].sum()))
            partial.append(float(g[g["label"] == "PARTIAL"]["rate"].sum()))
        x = range(len(x_labels))
        ax.bar(x, ack, label="acknowledged", color="#1f77b4")
        ax.bar(x, partial, bottom=ack, label="partial", color="#ff7f0e")
        ax.bar(
            x,
            denial,
            bottom=[a + p for a, p in zip(ack, partial)],
            label="denied",
            color="#d62728",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(kind.replace("sig_", "significant "))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of trials")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Stated-vs-revealed: assistant's claim about influence (turn 2)")
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

    print("\nHeadline numbers (overall):")
    o = h["overall"]
    print(f"  trials with judge label: {o['n_total']}")
    print(f"  significant backfire trials: {o['n_sig_backfire']}")
    print(f"  significant compliance trials: {o['n_sig_compliance']}")
    print(f"  no-effect trials: {o['n_no_effect']}")
    print(f"  base trials: {o['n_base']}")
    if o["denial_rate_in_backfires"] is not None:
        print(
            f"  denial rate in backfires: {o['denial_rate_in_backfires']:.1%}  "
            f"(ack={o['ack_rate_in_backfires']:.1%}, partial={o['partial_rate_in_backfires']:.1%})"
        )

    print("\nPer (model, benchmark):")
    for k, v in sorted(h["by_model_benchmark"].items()):
        print(f"  {k}:")
        if v["denial_rate_in_backfires"] is not None:
            print(
                f"    sig_backfire: n={v['n_sig_backfire']} "
                f"denied={v['denial_rate_in_backfires']:.1%} "
                f"ack={v['ack_rate_in_backfires']:.1%}"
            )
        if v["denial_rate_in_compliances"] is not None:
            print(
                f"    sig_compliance: n={v['n_sig_compliance']} "
                f"denied={v['denial_rate_in_compliances']:.1%} "
                f"ack={v['ack_rate_in_compliances']:.1%}"
            )

    plot_path = out_dir / "denial_vs_ack_by_kind.png"
    make_bar_plot(annotated, plot_path)
    print(f"\nWrote {out_dir}/{{trials_annotated.csv, headline_numbers.json}}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
