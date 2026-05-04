"""
Build the §4.2 (sec:claim-instability) teaser figure.

A concrete (model x factor x influence) cell that puts the
single-sentence shift in front of the reader as a before / after
contrast. Three bars:

  1. Baseline (no influence sentence), with an error bar showing the
     baseline-to-baseline noise floor measured by re-running the same
     baseline at the same temperature with a fresh RNG draw. This is
     the appendix-level noise calibrant; rendering it as an error bar
     instead of a separate bar keeps the visual focused on the size of
     the under-influence shift relative to natural variability.
  2. Under one virtue-appeal sentence pushing toward "non-old".
  3. Under one virtue-appeal sentence pushing toward "old".

Cell: DeepSeek V3.2 (non-reasoning) on BBQ `age_pos` x `virtue_appeal`.
Baseline f_0(B) = 47%; under-influence shifts to 2% and 95% in the two
directions; baseline-replicate drift is 2pp (rounded from 1.2pp on
this cell, which is in line with the BBQ benchmark mean of 1.7pp).

Output: ~/code/values/moral-steerability-paper/figures/one_sentence_shift.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_FIG_DIR = Path("~/code/values/moral-steerability-paper/figures").expanduser()

# Cell: DeepSeek V3.2 (non-reasoning) - BBQ - age_pos - virtue_appeal.
# Pulled from the experiments artifacts:
#   f_0(B), f_A(B), f_B(B) -> bbq_summary.csv (asymmetry-baseline-regression)
#   baseline-replicate drift -> analysis_bbq/per_condition.csv (baseline-noise)
F_0 = 0.468  # original baseline P(option B = old)
NOISE_PP = 2.0  # baseline-to-baseline drift on this cell, pp
F_TOWARD_A = 0.020  # influence sentence pushes toward "non-old"
F_TOWARD_B = 0.952  # influence sentence pushes toward "old"

CELL_DESC = "DeepSeek V3.2 · BBQ · age (positive polarity) · virtue-appeal"


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    bars = [
        {"label": "baseline", "value": F_0, "color": "#888888"},
        {
            "label": "+ virtue-appeal\ntoward non-old",
            "value": F_TOWARD_A,
            "color": "#0072B2",
        },
        {
            "label": "+ virtue-appeal\ntoward old",
            "value": F_TOWARD_B,
            "color": "#D55E00",
        },
    ]
    x = list(range(len(bars)))
    heights = [b["value"] for b in bars]
    colors = [b["color"] for b in bars]
    yerr = [NOISE_PP / 100.0, 0.0, 0.0]

    ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.66,
        yerr=yerr,
        capsize=8,
        error_kw=dict(ecolor="#222222", elinewidth=1.6, capthick=1.6),
    )

    # Reference 50/50 line.
    ax.axhline(0.5, color="#dddddd", lw=0.8, ls="--", zorder=0)

    # Annotate each bar with its value.
    for xi, b in zip(x, bars):
        ax.text(
            xi,
            b["value"] + 0.03,
            f"{b['value']:.0%}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )

    # Legend explaining the error bar.
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="|",
            color="#222222",
            markersize=14,
            markeredgewidth=1.8,
            linestyle="None",
            label=f"± baseline-to-baseline noise ({NOISE_PP:.0f}pp on this cell)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        fontsize=14,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in bars], fontsize=14)
    ax.set_ylabel("P(model picks the older person)", fontsize=15)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.26)
    pdf_path = PAPER_FIG_DIR / "one_sentence_shift.pdf"
    png_path = PAPER_FIG_DIR / "one_sentence_shift.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
