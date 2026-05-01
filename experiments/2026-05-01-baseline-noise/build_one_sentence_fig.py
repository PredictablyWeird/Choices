"""
Build the §4.2 (sec:claim-instability) teaser figure.

A concrete (model × factor × influence) cell that puts the
single-sentence shift in front of the reader as a before / after
contrast. Three bars showing the rate at which the model picks
"option B" on a fixed BBQ comparison set:

  1. Baseline (no influence sentence)
  2. Replicate baseline (same prompts, fresh RNG draw — the noise
     floor calibrant from §4.6)
  3. Under one virtue-appeal sentence pushing toward "old"
  4. Under one virtue-appeal sentence pushing toward "non-old"

Cell: DeepSeek V3.2 (non-reasoning) on BBQ `age_pos` × `virtue_appeal`.
Baseline f_0(B) = 47%; under-influence shifts to 2% and 95% in the two
directions; baseline-replicate drift is only 1.2pp.

Output: ~/code/values/moral-steerability-paper/figures/one_sentence_shift.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_FIG_DIR = (
    Path("~/code/values/moral-steerability-paper/figures").expanduser()
)

# Cell: DeepSeek V3.2 (non-reasoning) — BBQ — age_pos — virtue_appeal.
# Pulled from the experiments artifacts:
#   f_0(B), f_A(B), f_B(B) → bbq_summary.csv (asymmetry-baseline-regression)
#   baseline-replicate drift → analysis_bbq/per_condition.csv (baseline-noise)
F_0 = 0.468  # original baseline P(option B = old)
F_REP = 0.488  # replicate baseline (drift = 1.19pp)
F_TOWARD_A = 0.020  # influence sentence pushes toward "non-old"
F_TOWARD_B = 0.952  # influence sentence pushes toward "old"

CELL_DESC = "DeepSeek V3.2 · BBQ · age (positive polarity) · virtue-appeal"


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    # Order: original baseline, replicate baseline, two influence directions.
    bars = [
        {"label": "baseline", "value": F_0, "color": "#888888"},
        {"label": "baseline\n(re-run)", "value": F_REP, "color": "#bbbbbb"},
        {
            "label": "+ virtue-appeal\ntoward non-old",
            "value": F_TOWARD_A,
            "color": "#1f77b4",
        },
        {
            "label": "+ virtue-appeal\ntoward old",
            "value": F_TOWARD_B,
            "color": "#d62728",
        },
    ]
    x = list(range(len(bars)))
    heights = [b["value"] for b in bars]
    colors = [b["color"] for b in bars]
    ax.bar(x, heights, color=colors, edgecolor="white", linewidth=0.6, width=0.7)

    # Reference 50/50 line.
    ax.axhline(0.5, color="#dddddd", lw=0.8, ls="--", zorder=0)

    # Annotate each bar with its value.
    for xi, b in zip(x, bars):
        ax.text(
            xi,
            b["value"] + 0.025,
            f"{b['value']:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Big delta annotations between baseline ↔ each influence direction.
    delta_a = (F_0 - F_TOWARD_A) * 100
    delta_b = (F_TOWARD_B - F_0) * 100
    drift_pp = abs(F_REP - F_0) * 100

    # baseline ↔ replicate: noise floor annotation
    y_anchor = max(F_0, F_REP) + 0.18
    ax.annotate(
        "",
        xy=(1, y_anchor),
        xytext=(0, y_anchor),
        arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.0),
    )
    ax.text(
        0.5,
        y_anchor + 0.018,
        f"re-run noise\n{drift_pp:.1f}pp",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    # baseline ↔ toward-A: large shift
    ax.annotate(
        "",
        xy=(2, F_TOWARD_A + 0.04),
        xytext=(0, F_TOWARD_A + 0.04),
        arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.4),
    )
    ax.text(
        1.0,
        F_TOWARD_A + 0.07,
        f"−{delta_a:.0f}pp",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#1f77b4",
    )

    # baseline ↔ toward-B: large shift
    ax.annotate(
        "",
        xy=(3, F_0 + 0.04),
        xytext=(0, F_0 + 0.04),
        arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.4),
    )
    ax.text(
        1.5,
        F_0 + 0.07,
        f"+{delta_b:.0f}pp",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#d62728",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in bars], fontsize=8.5)
    ax.set_ylabel("P(model picks the older person)", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"What one sentence does — {CELL_DESC}",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.20)
    pdf_path = PAPER_FIG_DIR / "one_sentence_shift.pdf"
    png_path = PAPER_FIG_DIR / "one_sentence_shift.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
