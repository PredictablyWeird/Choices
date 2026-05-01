"""
Build the body figure for §4.3 (sec:claim-backfire).

Two panels:
  (a) stacked bars per (model × benchmark × kind={sig_compliance,
      sig_backfire}) showing the share of ack-affected /
      ack-disclaimed / denied / other (= partial + unclear).
  (b) the headline number isolated: share of ack-disclaimed among
      backfires vs among compliances, pooled across models +
      benchmarks. The "78%" is the dot the reader should remember.

Output: ~/code/values/moral-steerability-paper/figures/stated_vs_revealed.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
HEADLINE_JSON = EXP_DIR / "analysis_out_v2" / "headline_numbers.json"
TRIALS_CSV = EXP_DIR / "analysis_out_v2" / "trials_annotated.csv.gz"
PAPER_FIG_DIR = Path("~/code/values/moral-steerability-paper/figures").expanduser()

# Palette: red = ACK_DISCLAIMED ("claims no effect, but choice shifted"),
# blue = ACK_AFFECTED, dark grey = DENIED, light grey = other.
COLORS = {
    "ACK_DISCLAIMED": "#d62728",
    "ACK_AFFECTED": "#1f77b4",
    "DENIED": "#444444",
    "OTHER": "#bbbbbb",
}
LABEL_DISPLAY = {
    "ACK_DISCLAIMED": "ack-disclaimed",
    "ACK_AFFECTED": "ack-affected",
    "DENIED": "denied",
    "OTHER": "other",
}
ORDER = ["ACK_DISCLAIMED", "ACK_AFFECTED", "DENIED", "OTHER"]

MODEL_DISPLAY = {
    "gpt-5-2-non-reasoning": "GPT-5.2",
    "deepseek-v3-2-non-reasoning": "DeepSeek V3.2",
}
BENCH_DISPLAY = {"bbq": "BBQ", "trolley": "trolley"}
KIND_DISPLAY = {
    "sig_backfire": "sig.\nbackfire",
    "sig_compliance": "sig.\ncompliance",
}


def _shares(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {k: 0.0 for k in ORDER}
    counts = df["judge_label"].value_counts()
    out = {
        "ACK_DISCLAIMED": counts.get("ACK_DISCLAIMED", 0) / n,
        "ACK_AFFECTED": counts.get("ACK_AFFECTED", 0) / n,
        "DENIED": counts.get("DENIED", 0) / n,
    }
    out["OTHER"] = 1.0 - sum(out.values())
    return out


def main() -> None:
    df = pd.read_csv(TRIALS_CSV)
    df = df[df["judge_label"].notna()].copy()

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # -------- Panel (a) — per-cell stacked bars --------
    # Group bars by kind: backfires first (left block), compliances second
    # (right block), with a small visual gap between blocks.
    cells = []
    for kind in ("sig_backfire", "sig_compliance"):
        for model in ("gpt-5-2-non-reasoning", "deepseek-v3-2-non-reasoning"):
            for benchmark in ("bbq", "trolley"):
                sub = df[
                    (df["model"] == model)
                    & (df["benchmark"] == benchmark)
                    & (df["condition_kind"] == kind)
                ]
                if len(sub) == 0:
                    continue
                cells.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "kind": kind,
                        "n": len(sub),
                        "shares": _shares(sub),
                    }
                )

    # Insert a half-unit visual gap between the backfires and compliances blocks.
    x = []
    cursor = 0.0
    last_kind = None
    for c in cells:
        if last_kind is not None and c["kind"] != last_kind:
            cursor += 0.7
        x.append(cursor)
        cursor += 1.0
        last_kind = c["kind"]

    bottom = [0.0] * len(cells)
    for label in ORDER:
        heights = [c["shares"][label] for c in cells]
        ax_a.bar(
            x,
            heights,
            bottom=bottom,
            color=COLORS[label],
            label=LABEL_DISPLAY[label],
            width=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        bottom = [b + h for b, h in zip(bottom, heights)]

    # Bar labels: "MODEL / benchmark" rotated for readability + "n=..." on top.
    short_model = {
        "gpt-5-2-non-reasoning": "GPT-5.2",
        "deepseek-v3-2-non-reasoning": "DeepSeek",
    }
    xlabels = [
        f"{short_model[c['model']]} / {BENCH_DISPLAY[c['benchmark']]}" for c in cells
    ]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
    for xi, c in zip(x, cells):
        ax_a.text(
            xi,
            1.015,
            f"n={c['n']:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )

    # Block annotations under the panel.
    bf_xs = [xi for xi, c in zip(x, cells) if c["kind"] == "sig_backfire"]
    cm_xs = [xi for xi, c in zip(x, cells) if c["kind"] == "sig_compliance"]
    if bf_xs:
        ax_a.text(
            (min(bf_xs) + max(bf_xs)) / 2,
            -0.34,
            "significant backfires",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            transform=ax_a.get_xaxis_transform(),
        )
    if cm_xs:
        ax_a.text(
            (min(cm_xs) + max(cm_xs)) / 2,
            -0.34,
            "significant compliances",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            transform=ax_a.get_xaxis_transform(),
        )

    ax_a.set_xlim(min(x) - 0.6, max(x) + 0.6)
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel("share of trials", fontsize=9)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.tick_params(axis="y", labelsize=8)
    ax_a.set_title(
        "(a) Turn-2 self-report breakdown",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax_a.legend(
        ncol=4,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        handlelength=1.2,
        columnspacing=1.4,
    )

    # -------- Panel (b) — headline number isolated --------
    with HEADLINE_JSON.open() as f:
        h = json.load(f)
    o = h["overall"]
    bf_disclaimed = o["ack_disclaimed_rate_in_backfires"]
    cm_disclaimed = o["ack_disclaimed_rate_in_compliances"]
    n_bf = o["n_sig_backfire"]
    n_cm = o["n_sig_compliance"]

    # Lollipop: vertical lines + dots, two y positions.
    ys = [1.0, 0.0]
    rates = [bf_disclaimed, cm_disclaimed]
    ns = [n_bf, n_cm]
    labels = ["sig. backfires", "sig. compliances"]
    line_colors = ["#d62728", "#888888"]
    for y, rate, c in zip(ys, rates, line_colors):
        ax_b.hlines(y, 0, rate, color=c, lw=3)
    ax_b.scatter(
        rates,
        ys,
        s=[180, 100],
        color=line_colors,
        zorder=3,
        edgecolor="white",
        linewidth=1.2,
    )

    for y, rate, lbl, n in zip(ys, rates, labels, ns):
        ax_b.text(
            -0.04,
            y,
            f"{lbl}\n(n={n:,})",
            ha="right",
            va="center",
            fontsize=8.5,
            color="#222222" if y == 1 else "#555555",
        )
        # Place the percentage above the dot so it doesn't overlap.
        ax_b.text(
            rate,
            y + 0.22,
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=13 if y == 1 else 10,
            fontweight="bold" if y == 1 else "normal",
            color="#d62728" if y == 1 else "#444444",
        )

    ax_b.set_xlim(-0.55, 1.08)
    ax_b.set_ylim(-0.6, 1.6)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax_b.set_yticks([])
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.set_title(
        "(b) Share disclaiming any cue effect",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    ax_b.axvline(0, color="#cccccc", lw=0.7)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22)
    pdf_path = PAPER_FIG_DIR / "stated_vs_revealed.pdf"
    png_path = PAPER_FIG_DIR / "stated_vs_revealed.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
