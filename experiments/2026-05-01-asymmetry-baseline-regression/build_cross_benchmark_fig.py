"""
Build the cross-benchmark generalization panel for §4 (after Table 1).

Three side-by-side panels (one per benchmark: trolley, BBQ,
DailyDilemmas) showing two pairs of bars per panel:

  Left pair (pp axis):   under-influence shift  vs  baseline noise
  Right pair (% axis):   asymmetry beyond base  vs  backfire rate

The figure visually re-presents Table 1 so a reader skimming §4 can
see at a glance that all three numbers behave the same way across all
three benchmarks. Two y-axes per panel keep the units honest.

Numbers come straight from Table 1 (`tab:headline` in main.tex).
DailyDilemmas's asymmetry-beyond-baseline cell is "n/a" in the table
because value-level baselines lack a clean neutrality interpretation;
we render it as a hatched empty bar.

Output: ~/code/values/moral-steerability-paper/figures/cross_benchmark.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_FIG_DIR = Path("~/code/values/moral-steerability-paper/figures").expanduser()

# Numbers from Table 1 (`tab:headline` in main.tex).
DATA = {
    "Trolley triage": {
        "shift_pp": 15.0,
        "noise_pp": 1.1,
        "asym_pct": 39.0,
        "backfire_pct": 14.3,
    },
    "BBQ": {
        "shift_pp": 18.0,
        "noise_pp": 1.7,
        "asym_pct": 34.3,
        "backfire_pct": 10.5,
    },
    "DailyDilemmas": {
        "shift_pp": 9.0,
        "noise_pp": 2.5,
        "asym_pct": None,  # n/a — see caption
        "backfire_pct": 5.9,
    },
}

# Per-bar formatting.
PP_COLOR_SHIFT = "#1f77b4"  # under-influence shift  (blue)
PP_COLOR_NOISE = "#888888"  # baseline noise floor   (grey)
PCT_COLOR_ASYM = "#7c3aed"  # asym beyond baseline   (purple)
PCT_COLOR_BACK = "#d62728"  # backfire of sig effs   (red)

PP_AX_MAX = 22.0
PCT_AX_MAX = 50.0


def _render_panel(
    ax_pp,
    ax_pct,
    benchmark: str,
    show_pp_ylabel: bool,
    show_pct_ylabel: bool,
) -> None:
    d = DATA[benchmark]

    # Left pair: pp metrics (shift, noise floor) on the left axis.
    pp_x = [0, 1]
    pp_h = [d["shift_pp"], d["noise_pp"]]
    pp_c = [PP_COLOR_SHIFT, PP_COLOR_NOISE]
    pp_lbl = ["avg shift", "noise floor"]
    bars_pp = ax_pp.bar(
        pp_x, pp_h, color=pp_c, edgecolor="white", linewidth=0.6, width=0.7
    )
    for xi, h in zip(pp_x, pp_h):
        ax_pp.text(
            xi,
            h + 0.4,
            f"{h:.1f}pp",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Right pair: % metrics on the secondary axis. Plot to the right of the
    # pp pair, on x positions [2.7, 3.7] so the unit split is visually obvious.
    pct_x = [2.7, 3.7]
    asym = d["asym_pct"]
    backfire = d["backfire_pct"]
    pct_h = [asym if asym is not None else 0, backfire]
    pct_c = [PCT_COLOR_ASYM, PCT_COLOR_BACK]
    bars_pct = ax_pct.bar(
        pct_x, pct_h, color=pct_c, edgecolor="white", linewidth=0.6, width=0.7
    )
    if asym is None:
        bars_pct[0].set_hatch("///")
        bars_pct[0].set_alpha(0.3)
        ax_pct.text(
            pct_x[0],
            2.0,
            "n/a",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#888888",
            fontstyle="italic",
        )
    else:
        ax_pct.text(
            pct_x[0],
            asym + 1.0,
            f"{asym:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_pct.text(
        pct_x[1],
        backfire + 1.0,
        f"{backfire:.0f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # Shared x-axis: 4 bars total, labelled below.
    all_x = pp_x + pct_x
    ax_pp.set_xticks(all_x)
    ax_pp.set_xticklabels(
        ["avg\nshift", "noise\nfloor", "asym\nbeyond", "sig.\nbackfire"],
        fontsize=8.5,
    )

    # Visual divider between the pp pair and the % pair.
    ax_pp.axvline(2.0, color="#dddddd", lw=0.7, ls="--", zorder=0)

    # Axis ranges + spines.
    ax_pp.set_ylim(0, PP_AX_MAX)
    ax_pct.set_ylim(0, PCT_AX_MAX)
    ax_pp.set_xlim(-0.6, 4.4)

    ax_pp.spines[["top"]].set_visible(False)
    ax_pct.spines[["top"]].set_visible(False)
    ax_pp.spines["right"].set_color("#cccccc")
    ax_pct.spines["right"].set_color("#cccccc")
    ax_pp.tick_params(axis="y", labelsize=8, colors="#444444")
    ax_pct.tick_params(axis="y", labelsize=8, colors="#444444")

    if show_pp_ylabel:
        ax_pp.set_ylabel("percentage points (pp)", fontsize=8.5, color="#444444")
    else:
        ax_pp.set_ylabel("")
        ax_pp.set_yticklabels([])
    if show_pct_ylabel:
        ax_pct.set_ylabel("percent of conditions (%)", fontsize=8.5, color="#444444")
    else:
        ax_pct.set_ylabel("")
        ax_pct.set_yticklabels([])

    ax_pp.set_title(benchmark, fontsize=10.5, fontweight="bold", loc="left", pad=6)


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    pct_axes = [ax.twinx() for ax in axes]

    for i, (ax_pp, ax_pct, bench) in enumerate(zip(axes, pct_axes, DATA.keys())):
        _render_panel(
            ax_pp,
            ax_pct,
            bench,
            show_pp_ylabel=(i == 0),
            show_pct_ylabel=(i == len(axes) - 1),
        )

    # Shared legend at top.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PP_COLOR_SHIFT),
        plt.Rectangle((0, 0), 1, 1, color=PP_COLOR_NOISE),
        plt.Rectangle((0, 0), 1, 1, color=PCT_COLOR_ASYM),
        plt.Rectangle((0, 0), 1, 1, color=PCT_COLOR_BACK),
    ]
    fig.legend(
        handles,
        [
            "avg under-influence shift (pp)",
            "baseline noise floor (pp)",
            "asymmetry beyond baseline (% of conditions)",
            "backfire rate of sig. effects (% of conditions)",
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.06, right=0.94, top=0.84, bottom=0.16, wspace=0.55)

    pdf_path = PAPER_FIG_DIR / "cross_benchmark.pdf"
    png_path = PAPER_FIG_DIR / "cross_benchmark.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
